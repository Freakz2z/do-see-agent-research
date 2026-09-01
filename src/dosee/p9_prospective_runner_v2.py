"""Provider-compatible P9 runner after the blinded operational-format stops.

Ollama's native agent-loop contract permits an assistant message to contain no
tool calls (loop termination) or multiple tool calls (a parallel batch).  The
sealed v1 runner required exactly one call and therefore treated a valid
provider response shape as a mechanical failure.  This version preserves the
scientific intervention while accepting both provider-native shapes.  The v5
protocol also returns one deterministic, body-free tool observation when a
workspace tool supplies an absolute path.  The frozen tool descriptions did
not expose their relative-path-only policy, so treating that provider-correctable
argument error as a fatal process exception was not a valid capability test.
The v6 protocol applies the same narrow treatment to a schema-valid
``read_file`` whose relative path does not exist: it returns a body-free,
retryable observation while leaving every other execution failure fatal.
The v7 protocol also treats provider HTTP 500 like the already-frozen
502/503/504 class: it retries the identical payload within the same physical
attempt ceiling and leaves every client, quota, and authentication status fatal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .p9_controller import BudgetLedger, P9ControllerError, ToolAction
from .p9_ollama_loop import OllamaToolLoop, RETRYABLE_HTTP
from .p9_prospective_runner import (
    ProspectiveForkBundle,
    ProspectiveRunner,
    ProvenanceObservationAdapter,
)
from .p9e_contract import canonical_json, sha256_json
from .p9_transport_resilience import FROZEN_TRANSPORT_RESILIENCE_POLICY


def parse_tool_actions(
    message: Mapping[str, object], request_id: str
) -> list[ToolAction]:
    """Parse a provider-native zero, single, or parallel tool-call batch."""

    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, list):
        raise P9ControllerError("prospective tool_calls is not a list")
    actions: list[ToolAction] = []
    for index, call in enumerate(raw_calls):
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping):
            raise P9ControllerError("prospective tool call lacks function data")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            raise P9ControllerError("prospective tool call is malformed")
        actions.append(
            ToolAction(
                action_id=f"provider.{request_id}.{index}",
                tool=name,
                arguments=dict(arguments),
            )
        )
    return actions


def _append_result(
    history: list[dict[str, object]], action: ToolAction, result: Mapping[str, object]
) -> None:
    history.append(
        {
            "role": "tool",
            "tool_name": action.tool,
            "content": canonical_json(result),
        }
    )


def recoverable_tool_validation_result(
    *, registry: Any, family: str, action: ToolAction
) -> dict[str, object] | None:
    """Return one safe tool observation for the undisclosed absolute-path rule.

    All other schema, containment, terminal, and runtime failures remain fatal.
    Validation happens before trigger evaluation, so a rejected action cannot
    mutate state, become the focal action, or enter a counterfactual fork.
    """

    try:
        registry.validate(family, action)
    except P9ControllerError as error:
        message = str(error)
        if message == "absolute workspace path rejected":
            reason = "workspace_path_must_be_relative"
        elif message.startswith(
            (
                "tool value at ",
                "tool string at ",
                "tool number at ",
                "tool array at ",
                "tool object at ",
            )
        ):
            reason = "arguments_do_not_match_schema"
        else:
            raise
        return {
            "error": "tool_arguments_rejected",
            "reason": reason,
            "retryable": True,
        }
    if action.tool == "run_command":
        argv = action.arguments.get("argv")
        shell_names = {"sh", "bash", "zsh", "dash", "fish", "ksh"}
        if (
            isinstance(argv, list)
            and argv
            and Path(str(argv[0])).name in shell_names
            and len(argv) > 1
            and argv[1] in {"-c", "-lc", "-ic"}
            and registry.path_policy.get("shell_string_execution") is False
        ):
            return {
                "error": "tool_arguments_rejected",
                "reason": "shell_string_execution_not_allowed",
                "retryable": True,
            }
    return None


def recoverable_tool_execution_result(
    *, action: ToolAction, error: Exception
) -> dict[str, object] | None:
    """Return body-free feedback for the one non-mutating missing-file case.

    The path is deliberately not echoed.  Permission failures, directory
    reads, decoding errors, command failures, and all mutating actions remain
    fatal, so this does not expand the provider's capabilities or weaken the
    frozen containment policy.
    """

    if action.tool == "read_file" and isinstance(error, FileNotFoundError):
        return {
            "error": "tool_execution_failed",
            "reason": "file_not_found",
            "retryable": True,
        }
    return None


class ProviderCompatibleProspectiveRunner(ProspectiveRunner):
    """P9 runner that follows Ollama's documented agent-loop semantics."""

    def _loop(self, adapter: Any) -> OllamaToolLoop:
        return OllamaToolLoop(
            plan=self.plan,
            family=self.family,
            model_id=self.model_id,
            provider_seed=self.provider_seed,
            tool_registry=self.registry,
            transport=self.transport,
            adapter=adapter,
            retryable_http_statuses=RETRYABLE_HTTP | {500},
        )

    def _execute_batch(
        self,
        *,
        adapter: Any,
        actions: Sequence[ToolAction],
        history: list[dict[str, object]],
    ) -> tuple[int, int]:
        executed = 0
        for action in actions:
            if adapter.terminal.decision is not None:
                break
            result = recoverable_tool_validation_result(
                registry=self.registry,
                family=self.family,
                action=action,
            )
            if result is None:
                try:
                    result = dict(adapter.execute(action))
                except Exception as error:
                    result = recoverable_tool_execution_result(
                        action=action, error=error
                    )
                    if result is None:
                        raise
            _append_result(history, action, result)
            executed += 1
        return executed, len(actions) - executed

    def _run_continuation_v2(
        self,
        *,
        adapter: ProvenanceObservationAdapter,
        messages: Sequence[Mapping[str, object]],
        pending_actions: Sequence[ToolAction],
        logical_turn_ceiling: int,
    ) -> tuple[dict[str, object], list[dict[str, object]], Any]:
        loop = self._loop(adapter)
        budget = BudgetLedger(
            logical_ceiling=logical_turn_ceiling,
            physical_ceiling=logical_turn_ceiling
            * (1 + int(self.plan["turn_limits"]["transport_retries_per_logical_turn"])),
        )
        history = [dict(row) for row in messages]
        pending_executed, pending_ignored = self._execute_batch(
            adapter=adapter, actions=pending_actions, history=history
        )
        termination_mode = (
            "explicit_finalize" if adapter.terminal.decision is not None else None
        )
        ignored_after_terminal = pending_ignored
        while adapter.terminal.decision is None and termination_mode is None:
            if budget.logical_turns >= budget.logical_ceiling:
                termination_mode = "turn_ceiling_hold"
                break
            assistant, attempts = loop._one_turn(history, timeout_seconds=300)
            budget.charge(physical_attempts=attempts)
            history.append(assistant)
            actions = parse_tool_actions(assistant, loop.request_ids[-1])
            if not actions:
                termination_mode = "assistant_stop_hold"
                break
            _, ignored = self._execute_batch(
                adapter=adapter, actions=actions, history=history
            )
            ignored_after_terminal += ignored
            if adapter.terminal.decision is not None:
                termination_mode = "explicit_finalize"
        decision = adapter.terminal.decision
        if decision is None:
            decision = "hold"
        return (
            {
                "terminal_decision": decision,
                "termination_mode": termination_mode,
                "pending_tool_calls_executed": pending_executed,
                "tool_calls_ignored_after_terminal": ignored_after_terminal,
                "logical_turns": budget.logical_turns,
                "physical_attempts": budget.physical_attempts,
                "request_ids": list(loop.request_ids),
                "transcript_digest": sha256_json(history),
            },
            history,
            loop,
        )

    def run(self, messages: Sequence[Mapping[str, object]]) -> dict[str, object]:
        limits = self.plan["turn_limits"]
        prefork = BudgetLedger(
            logical_ceiling=int(limits["prefork_max"]),
            physical_ceiling=int(limits["prefork_max"])
            * (1 + int(limits["transport_retries_per_logical_turn"])),
        )
        history = [dict(row) for row in messages]
        common_loop = self._loop(self.backend.common_adapter)
        trigger_audits: list[dict[str, object]] = []
        bundle: ProspectiveForkBundle | None = None
        focal_action: ToolAction | None = None
        common_prefix_digest: str | None = None
        pending_actions: list[ToolAction] = []
        common_termination_mode: str | None = None
        common_ignored_after_terminal = 0

        while self.backend.common_adapter.terminal.decision is None:
            if prefork.logical_turns >= prefork.logical_ceiling:
                common_termination_mode = "turn_ceiling_hold"
                break
            assistant, attempts = common_loop._one_turn(history, timeout_seconds=300)
            prefork.charge(physical_attempts=attempts)
            history.append(assistant)
            actions = parse_tool_actions(assistant, common_loop.request_ids[-1])
            if not actions:
                common_termination_mode = "assistant_stop_hold"
                break
            for index, action in enumerate(actions):
                if self.backend.common_adapter.terminal.decision is not None:
                    common_ignored_after_terminal += len(actions) - index
                    break
                validation_result = recoverable_tool_validation_result(
                    registry=self.registry,
                    family=self.family,
                    action=action,
                )
                if validation_result is not None:
                    trigger_audits.append(
                        {
                            "action_digest": action.digest,
                            "eligible_mutation": False,
                            "proxy_positive_after_action": False,
                            "target_true_after_action": False,
                            "triggered": False,
                            "counterfactual_prewrite_digest": None,
                            "counterfactual_postwrite_digest": None,
                        }
                    )
                    _append_result(history, action, validation_result)
                    continue
                trigger = self.backend.evaluate_trigger(action)
                trigger_audits.append(
                    {
                        "action_digest": action.digest,
                        "eligible_mutation": trigger.eligible_mutation,
                        "proxy_positive_after_action": trigger.proxy_positive_after_action,
                        "target_true_after_action": trigger.target_true_after_action,
                        "triggered": trigger.triggered,
                        "counterfactual_prewrite_digest": trigger.counterfactual_prewrite_digest,
                        "counterfactual_postwrite_digest": trigger.counterfactual_postwrite_digest,
                    }
                )
                if trigger.triggered:
                    focal_action = action
                    common_prefix_digest = sha256_json(history)
                    bundle = self.backend.fork(action)
                    pending_actions = list(actions[index + 1 :])
                    break
                try:
                    result = dict(self.backend.common_adapter.execute(action))
                except Exception as error:
                    result = recoverable_tool_execution_result(
                        action=action, error=error
                    )
                    if result is None:
                        raise
                _append_result(history, action, result)
            if bundle is not None:
                break

        if bundle is None or focal_action is None or common_prefix_digest is None:
            return {
                "format": "dosee.p9-prospective-prefix-run.v1",
                "runner_protocol": "provider_compatible_v7",
                "family": self.family,
                "model_id": self.model_id,
                "provider_seed": self.provider_seed,
                "opportunity_triggered": False,
                "natural_echo_exposure": False,
                "prefork_logical_turns": prefork.logical_turns,
                "prefork_physical_attempts": prefork.physical_attempts,
                "common_terminal_decision": (
                    self.backend.common_adapter.terminal.decision or "hold"
                ),
                "common_termination_mode": (
                    common_termination_mode
                    or (
                        "explicit_finalize"
                        if self.backend.common_adapter.terminal.decision is not None
                        else "turn_ceiling_hold"
                    )
                ),
                "common_tool_calls_ignored_after_terminal": common_ignored_after_terminal,
                "trigger_audits": trigger_audits,
                "provider_request_ids": list(common_loop.request_ids),
                "behavioral_estimate_computed": False,
            }

        if bundle.proposed_action_digest != focal_action.digest:
            raise P9ControllerError("forked action differs from intercepted proposal")
        receipt_message = {
            "role": "tool",
            "tool_name": focal_action.tool,
            "content": canonical_json(bundle.immediate_receipt),
        }
        continuation_prefix = [*history, receipt_message]
        continuation_prefix_digest = sha256_json(continuation_prefix)
        rows: list[dict[str, object]] = []
        all_request_ids = list(common_loop.request_ids)
        for name in self.continuation_bindings:
            runtime = bundle.continuations[name]
            if runtime.continuation != name:
                raise P9ControllerError("continuation runtime order drift")
            runtime.activate()
            adapter = ProvenanceObservationAdapter(runtime)
            result, continuation_history, loop = self._run_continuation_v2(
                adapter=adapter,
                messages=continuation_prefix,
                pending_actions=pending_actions,
                logical_turn_ceiling=int(limits["continuation_max_each"]),
            )
            all_request_ids.extend(loop.request_ids)
            decision = str(result["terminal_decision"])
            post_probe = None
            if name == "echo_L2_causal_lineage":
                post_probe = self._post_decision_probe(
                    history=continuation_history,
                    record=runtime.provenance_record,
                    terminal_decision=decision,
                )
                all_request_ids.append(str(post_probe["request_id"]))
            rows.append(
                {
                    "continuation": name,
                    "source_arm": runtime.source_arm,
                    "prefix_digest": continuation_prefix_digest,
                    "prewrite_state_digest": runtime.prewrite_state_digest,
                    "postwrite_state_digest": runtime.postwrite_state_digest,
                    "proposed_action_digest": focal_action.digest,
                    "immediate_receipt_digest": bundle.immediate_receipt_digest,
                    "evidence_observation_count": adapter.observation_count,
                    "last_rendered_observation": adapter.last_rendered_observation,
                    "target_truth": runtime.target_truth,
                    "terminal_decision": decision,
                    "termination_mode": result["termination_mode"],
                    "false_irreversible_approval": (
                        decision == "submit" and runtime.target_truth is False
                    ),
                    "logical_turns": result["logical_turns"],
                    "physical_attempts": result["physical_attempts"],
                    "request_ids": result["request_ids"],
                    "pending_tool_calls_executed": result[
                        "pending_tool_calls_executed"
                    ],
                    "tool_calls_ignored_after_terminal": result[
                        "tool_calls_ignored_after_terminal"
                    ],
                    "post_decision_probe": post_probe,
                }
            )

        if len(all_request_ids) != len(set(all_request_ids)):
            raise P9ControllerError("provider request IDs collide across prefix forks")
        invariant_sets = (
            {row["prefix_digest"] for row in rows},
            {row["prewrite_state_digest"] for row in rows},
            {row["proposed_action_digest"] for row in rows},
            {row["immediate_receipt_digest"] for row in rows},
        )
        if not all(len(values) == 1 for values in invariant_sets):
            raise P9ControllerError("same-prefix fork integrity failed")
        echo_exposure = next(
            row for row in rows if row["continuation"] == "echo_L0_value_only"
        )["evidence_observation_count"] > 0
        probe_logical = sum(
            int(row["post_decision_probe"]["logical_turns"])
            for row in rows
            if row["post_decision_probe"] is not None
        )
        probe_physical = sum(
            int(row["post_decision_probe"]["physical_attempts"])
            for row in rows
            if row["post_decision_probe"] is not None
        )
        total_logical = (
            prefork.logical_turns
            + sum(int(row["logical_turns"]) for row in rows)
            + probe_logical
        )
        total_physical = prefork.physical_attempts + sum(
            int(row["physical_attempts"]) for row in rows
        ) + probe_physical
        if total_logical > int(limits["max_per_prefix"]):
            raise P9ControllerError("full prospective prefix logical budget exceeded")
        if total_physical > int(limits["max_per_prefix"]) * (
            1 + int(limits["transport_retries_per_logical_turn"])
        ):
            raise P9ControllerError("full prospective prefix physical budget exceeded")
        return {
            "format": "dosee.p9-prospective-prefix-run.v1",
            "runner_protocol": "provider_compatible_v7",
            "family": self.family,
            "model_id": self.model_id,
            "provider_seed": self.provider_seed,
            "opportunity_triggered": True,
            "natural_echo_exposure": echo_exposure,
            "common_prefix_digest": common_prefix_digest,
            "continuation_prefix_digest": continuation_prefix_digest,
            "prewrite_state_digest": bundle.prewrite_state_digest,
            "proposed_action_digest": focal_action.digest,
            "immediate_receipt_digest": bundle.immediate_receipt_digest,
            "prefork_logical_turns": prefork.logical_turns,
            "prefork_physical_attempts": prefork.physical_attempts,
            "pending_parallel_tool_calls": len(pending_actions),
            "continuations": rows,
            "all_same_prefix_integrity": True,
            "all_request_ids_unique": True,
            "logical_turns": total_logical,
            "physical_attempts": total_physical,
            "provider_request_ids": all_request_ids,
            "behavioral_estimate_computed": False,
        }


class ResilientProviderCompatibleProspectiveRunner(
    ProviderCompatibleProspectiveRunner
):
    """V8 runner: V7 semantics plus content-blind journaled backoff."""

    def _loop(self, adapter: Any) -> OllamaToolLoop:
        journal = getattr(self.transport, "journal", None)
        if (
            journal is None
            or journal.policy.digest != FROZEN_TRANSPORT_RESILIENCE_POLICY.digest
        ):
            raise P9ControllerError(
                "provider-compatible V8 requires the frozen transport journal"
            )
        return OllamaToolLoop(
            plan=self.plan,
            family=self.family,
            model_id=self.model_id,
            provider_seed=self.provider_seed,
            tool_registry=self.registry,
            transport=self.transport,
            adapter=adapter,
            retryable_http_statuses=RETRYABLE_HTTP | {500},
            resilience_policy=FROZEN_TRANSPORT_RESILIENCE_POLICY,
        )

    def run(self, messages: Sequence[Mapping[str, object]]) -> dict[str, object]:
        result = dict(super().run(messages))
        if result.get("runner_protocol") != "provider_compatible_v7":
            raise P9ControllerError("provider-compatible V8 parent protocol drifted")
        result["runner_protocol"] = "provider_compatible_v8"
        result["transport_resilience_policy_digest"] = (
            FROZEN_TRANSPORT_RESILIENCE_POLICY.digest
        )
        result["transport_content_blind_journaled"] = True
        return result
