"""Prospective focal-write interception and same-prefix continuation runner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import socket
import ssl
from typing import Any, Callable, Mapping, Protocol, Sequence

from .p9_controller import (
    BudgetLedger,
    CONTINUATION_BINDINGS,
    CONTINUATION_BINDINGS_V2,
    P9ControllerError,
    ToolAction,
    continuation_bindings,
)
from .p9_domain_adapters import WorkspaceToolAdapter
from .p9_ollama_loop import HTTPResponse, OllamaToolLoop, RETRYABLE_HTTP, _strict_json
from .p9e_contract import canonical_json, sha256_json
from .p9e_filesystem_forks import snapshot_tree
from .p9e_provenance_renderers import ProvenanceRecord, render_observation


MUTATING_WORKSPACE_TOOLS = frozenset({"apply_patch", "run_command"})


class ToolAdapter(Protocol):
    terminal: Any

    def execute(self, action: ToolAction) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class TriggerAudit:
    eligible_mutation: bool
    proxy_positive_after_action: bool
    target_true_after_action: bool
    triggered: bool
    counterfactual_prewrite_digest: str | None
    counterfactual_postwrite_digest: str | None


@dataclass(frozen=True)
class ForkedContinuation:
    continuation: str
    source_arm: str
    adapter: ToolAdapter
    evidence_predicate: Callable[[ToolAction], bool]
    native_positive: Callable[[Mapping[str, object]], bool]
    provenance_record: ProvenanceRecord
    target_truth: bool
    prewrite_state_digest: str
    postwrite_state_digest: str
    activate: Callable[[], None]


@dataclass(frozen=True)
class ProspectiveForkBundle:
    proposed_action_digest: str
    prewrite_state_digest: str
    immediate_receipt: Mapping[str, object]
    immediate_receipt_digest: str
    continuations: Mapping[str, ForkedContinuation]


class ProspectiveForkBackend(Protocol):
    common_adapter: ToolAdapter

    def evaluate_trigger(self, action: ToolAction) -> TriggerAudit: ...

    def fork(self, action: ToolAction) -> ProspectiveForkBundle: ...


class ProvenanceObservationAdapter:
    """Replace only the frozen evidence observation, leaving all other tools native."""

    def __init__(self, runtime: ForkedContinuation) -> None:
        self.runtime = runtime
        self.base = runtime.adapter
        self.terminal = self.base.terminal
        self.observation_count = 0
        self.last_native_observation: dict[str, object] | None = None
        self.last_rendered_observation: dict[str, object] | None = None

    def execute(self, action: ToolAction) -> dict[str, object]:
        native = dict(self.base.execute(action))
        if not self.runtime.evidence_predicate(action):
            return native
        if self.runtime.native_positive(native) is not self.runtime.provenance_record.positive:
            raise P9ControllerError("native observation and provenance record disagree")
        all_bindings = {**CONTINUATION_BINDINGS, **CONTINUATION_BINDINGS_V2}
        _, level = all_bindings[self.runtime.continuation]
        rendered = render_observation(self.runtime.provenance_record, level)
        self.observation_count += 1
        self.last_native_observation = native
        self.last_rendered_observation = rendered
        return rendered


def _parse_one_tool_action(message: Mapping[str, object], request_id: str) -> ToolAction:
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list) or len(raw_calls) != 1:
        raise P9ControllerError("prospective P9 requires exactly one tool call per turn")
    call = raw_calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping):
        raise P9ControllerError("prospective tool call lacks function data")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise P9ControllerError("prospective tool call is malformed")
    return ToolAction(
        action_id=f"provider.{hashlib.sha256(request_id.encode()).hexdigest()[:24]}.0",
        tool=name,
        arguments=dict(arguments),
    )


class ProspectiveRunner:
    def __init__(
        self,
        *,
        plan: Mapping[str, object],
        family: str,
        model_id: str,
        provider_seed: int,
        registry: Any,
        transport: Any,
        backend: ProspectiveForkBackend,
        prompt_registry: Mapping[str, object],
    ) -> None:
        self.plan = dict(plan)
        self.family = family
        self.model_id = model_id
        self.provider_seed = provider_seed
        self.registry = registry
        self.transport = transport
        self.backend = backend
        self.prompt_registry = dict(prompt_registry)
        self.continuation_bindings = continuation_bindings(plan)
        if self.prompt_registry.get("format") != "dosee.p9-localization-prompt-registry.v1":
            raise P9ControllerError("unexpected prompt registry in prospective runner")

    def _loop(self, adapter: ToolAdapter) -> OllamaToolLoop:
        return OllamaToolLoop(
            plan=self.plan,
            family=self.family,
            model_id=self.model_id,
            provider_seed=self.provider_seed,
            tool_registry=self.registry,
            transport=self.transport,
            adapter=adapter,
        )

    def _run_continuation(
        self,
        *,
        adapter: ProvenanceObservationAdapter,
        messages: Sequence[Mapping[str, object]],
        logical_turn_ceiling: int,
    ) -> tuple[dict[str, object], list[dict[str, object]], OllamaToolLoop]:
        loop = self._loop(adapter)
        budget = BudgetLedger(
            logical_ceiling=logical_turn_ceiling,
            physical_ceiling=logical_turn_ceiling
            * (1 + int(self.plan["turn_limits"]["transport_retries_per_logical_turn"])),
        )
        history = [dict(row) for row in messages]
        while adapter.terminal.decision is None:
            if budget.logical_turns >= budget.logical_ceiling:
                raise P9ControllerError(
                    "continuation logical ceiling reached before provider request"
                )
            assistant, attempts = loop._one_turn(history, timeout_seconds=300)
            budget.charge(physical_attempts=attempts)
            history.append(assistant)
            action = _parse_one_tool_action(assistant, loop.request_ids[-1])
            result = adapter.execute(action)
            history.append(
                {
                    "role": "tool",
                    "tool_name": action.tool,
                    "content": canonical_json(result),
                }
            )
        return (
            {
                "terminal_decision": adapter.terminal.decision,
                "logical_turns": budget.logical_turns,
                "physical_attempts": budget.physical_attempts,
                "request_ids": list(loop.request_ids),
                "transcript_digest": sha256_json(history),
            },
            history,
            loop,
        )

    def _post_decision_probe(
        self,
        *,
        history: Sequence[Mapping[str, object]],
        record: ProvenanceRecord,
        terminal_decision: str,
    ) -> dict[str, object]:
        probe = self.prompt_registry["post_decision_probe"]
        schema = probe["schema"]
        messages = [
            *[dict(row) for row in history],
            {"role": "user", "content": str(probe["prompt"])},
        ]
        sampling = self.plan["sampling"]
        request_payload = canonical_json(
            {
                "model": self.model_id,
                "messages": messages,
                "format": schema,
                "stream": False,
                "think": False,
                "options": {
                    "num_predict": int(
                        self.plan["turn_limits"]["single_turn_output_tokens"]
                    ),
                    "seed": self.provider_seed,
                    "temperature": float(sampling["temperature"]),
                    "top_p": float(sampling["top_p"]),
                },
            }
        ).encode()
        retry_count = int(
            self.plan["turn_limits"]["transport_retries_per_logical_turn"]
        )
        response: HTTPResponse | None = None
        attempts = 0
        for attempt in range(retry_count + 1):
            attempts += 1
            try:
                candidate = self.transport.chat(request_payload, timeout_seconds=300)
            except (OSError, socket.timeout, ssl.SSLError) as error:
                if attempt < retry_count:
                    continue
                raise P9ControllerError("post-decision probe transport exhausted") from error
            if candidate.status in RETRYABLE_HTTP and attempt < retry_count:
                continue
            response = candidate
            break
        if response is None or response.status != 200:
            raise P9ControllerError("post-decision probe received no usable response")
        payload = _strict_json(response.payload)
        if not isinstance(payload, Mapping):
            raise P9ControllerError("post-decision probe response is not an object")
        if payload.get("model") != self.model_id or payload.get("done") is not True:
            raise P9ControllerError("post-decision probe model or terminal state drift")
        request_id = next(
            (
                str(value)
                for name, value in response.headers.items()
                if name.casefold() == "x-request-id" and value
            ),
            "",
        )
        if not request_id:
            raise P9ControllerError("post-decision probe request ID missing")
        message = payload.get("message")
        if (
            not isinstance(message, Mapping)
            or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str)
            or message.get("tool_calls")
        ):
            raise P9ControllerError("post-decision probe message is malformed")
        answer = _strict_json(str(message["content"]).encode())
        if not isinstance(answer, Mapping) or set(answer) != {
            "source_operation_id",
            "derived_from_write_id",
            "confidence",
        }:
            raise P9ControllerError("post-decision causal answer schema mismatch")
        confidence = answer["confidence"]
        if (
            not isinstance(answer["source_operation_id"], str)
            or not isinstance(answer["derived_from_write_id"], str)
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            raise P9ControllerError("post-decision causal answer values are invalid")
        input_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")
        if (
            not isinstance(input_tokens, int)
            or isinstance(input_tokens, bool)
            or input_tokens < 0
            or not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens < 0
            or output_tokens
            > int(self.plan["turn_limits"]["single_turn_output_tokens"])
        ):
            raise P9ControllerError("post-decision probe token accounting invalid")
        return {
            "request_id": request_id,
            "request_sha256": hashlib.sha256(request_payload).hexdigest(),
            "response_sha256": hashlib.sha256(response.payload).hexdigest(),
            "logical_turns": 1,
            "physical_attempts": attempts,
            "answer": dict(answer),
            "source_operation_correct": (
                answer["source_operation_id"] == record.source_operation_id
            ),
            "derived_write_correct": (
                answer["derived_from_write_id"] == record.derived_from_write_id
            ),
            "causal_recognition_correct": (
                answer["source_operation_id"] == record.source_operation_id
                and answer["derived_from_write_id"] == record.derived_from_write_id
            ),
            "terminal_decision_before_probe": terminal_decision,
            "terminal_decision_after_probe": terminal_decision,
            "terminal_unchanged": True,
            "credential_recorded": False,
        }

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
        while self.backend.common_adapter.terminal.decision is None:
            if prefork.logical_turns >= prefork.logical_ceiling:
                break
            assistant, attempts = common_loop._one_turn(history, timeout_seconds=300)
            prefork.charge(physical_attempts=attempts)
            history.append(assistant)
            action = _parse_one_tool_action(assistant, common_loop.request_ids[-1])
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
                break
            result = dict(self.backend.common_adapter.execute(action))
            history.append(
                {
                    "role": "tool",
                    "tool_name": action.tool,
                    "content": canonical_json(result),
                }
            )
        if bundle is None or focal_action is None or common_prefix_digest is None:
            return {
                "format": "dosee.p9-prospective-prefix-run.v1",
                "family": self.family,
                "model_id": self.model_id,
                "provider_seed": self.provider_seed,
                "opportunity_triggered": False,
                "natural_echo_exposure": False,
                "prefork_logical_turns": prefork.logical_turns,
                "prefork_physical_attempts": prefork.physical_attempts,
                "common_terminal_decision": self.backend.common_adapter.terminal.decision,
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
            result, continuation_history, loop = self._run_continuation(
                adapter=adapter,
                messages=continuation_prefix,
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
                    "false_irreversible_approval": (
                        decision == "submit" and runtime.target_truth is False
                    ),
                    "logical_turns": result["logical_turns"],
                    "physical_attempts": result["physical_attempts"],
                    "request_ids": result["request_ids"],
                    "post_decision_probe": post_probe,
                }
            )
        if len(all_request_ids) != len(set(all_request_ids)):
            raise P9ControllerError("provider request IDs collide across prefix forks")
        same_prefix = {row["prefix_digest"] for row in rows}
        same_prewrite = {row["prewrite_state_digest"] for row in rows}
        same_action = {row["proposed_action_digest"] for row in rows}
        same_receipt = {row["immediate_receipt_digest"] for row in rows}
        if not all(len(values) == 1 for values in (same_prefix, same_prewrite, same_action, same_receipt)):
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
            "continuations": rows,
            "all_same_prefix_integrity": True,
            "all_request_ids_unique": True,
            "logical_turns": total_logical,
            "physical_attempts": total_physical,
            "provider_request_ids": all_request_ids,
            "behavioral_estimate_computed": False,
        }


class FilesystemForkBackend:
    """Counterfactually detect and fork arbitrary workspace mutations."""

    def __init__(
        self,
        *,
        common_adapter: WorkspaceToolAdapter,
        runtime_root: Path,
        adapter_factory: Callable[[Path, str], WorkspaceToolAdapter],
        trigger_predicate: Callable[[Path, ToolAction], tuple[bool, bool]],
        arm_prepare: Callable[[Path, str], None],
        record_factory: Callable[[Path, str, ToolAction], ProvenanceRecord],
        target_truth: Callable[[Path, str], bool],
        evidence_tool: str,
    ) -> None:
        self.common_adapter = common_adapter
        self.common_root = common_adapter.workspace_root
        self.runtime_root = runtime_root.resolve()
        if self.runtime_root.exists():
            raise P9ControllerError("prospective filesystem runtime root must be fresh")
        self.runtime_root.mkdir(parents=True)
        self.adapter_factory = adapter_factory
        self.trigger_predicate = trigger_predicate
        self.arm_prepare = arm_prepare
        self.record_factory = record_factory
        self.target_truth = target_truth
        self.evidence_tool = evidence_tool

    def evaluate_trigger(self, action: ToolAction) -> TriggerAudit:
        if action.tool not in MUTATING_WORKSPACE_TOOLS:
            return TriggerAudit(False, False, False, False, None, None)
        candidate = self.runtime_root / "counterfactual" / action.digest
        if candidate.exists():
            raise P9ControllerError("counterfactual action digest was already evaluated")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        prewrite = snapshot_tree(self.common_root).digest
        shutil.copytree(self.common_root, candidate, symlinks=True)
        adapter = self.adapter_factory(candidate, "echo")
        adapter.execute(action)
        proxy_positive, target_true = self.trigger_predicate(candidate, action)
        postwrite = snapshot_tree(candidate).digest
        shutil.rmtree(candidate)
        return TriggerAudit(
            eligible_mutation=True,
            proxy_positive_after_action=proxy_positive,
            target_true_after_action=target_true,
            triggered=proxy_positive and not target_true,
            counterfactual_prewrite_digest=prewrite,
            counterfactual_postwrite_digest=postwrite,
        )

    def fork(self, action: ToolAction) -> ProspectiveForkBundle:
        destination = self.runtime_root / "forks" / action.digest
        if destination.exists():
            raise P9ControllerError("prospective fork destination already exists")
        destination.mkdir(parents=True)
        prewrite = snapshot_tree(self.common_root).digest
        receipts: dict[str, dict[str, object]] = {}
        runtimes: dict[str, ForkedContinuation] = {}
        for continuation, (source_arm, _) in CONTINUATION_BINDINGS_V2.items():
            root = destination / continuation
            shutil.copytree(self.common_root, root, symlinks=True)
            if snapshot_tree(root).digest != prewrite:
                raise P9ControllerError("filesystem prewrite state drifted across continuations")
            adapter = self.adapter_factory(root, source_arm)
            receipts[continuation] = dict(adapter.execute(action))
            self.arm_prepare(root, source_arm)
            record = self.record_factory(root, source_arm, action)
            runtimes[continuation] = ForkedContinuation(
                continuation=continuation,
                source_arm=source_arm,
                adapter=adapter,
                evidence_predicate=lambda candidate, expected=self.evidence_tool: (
                    candidate.tool == expected
                ),
                native_positive=lambda observation: observation.get("positive") is True,
                provenance_record=record,
                target_truth=self.target_truth(root, source_arm),
                prewrite_state_digest=prewrite,
                postwrite_state_digest=snapshot_tree(root).digest,
                activate=lambda: None,
            )
        serialized = {canonical_json(value) for value in receipts.values()}
        if len(serialized) != 1:
            raise P9ControllerError("immediate focal receipt drifted across filesystem forks")
        records = [runtime.provenance_record for runtime in runtimes.values()]
        if len({record.evidence_id for record in records}) != 1:
            raise P9ControllerError("evidence identifier drifted across continuations")
        for runtime in runtimes.values():
            record = runtime.provenance_record
            if runtime.source_arm == "echo" and (
                record.source_operation_id != action.action_id
                or record.updated_by != "agent"
                or not record.positive
                or record.verification_result
                or runtime.target_truth
            ):
                raise P9ControllerError("echo provenance is not bound to the focal action")
            if runtime.source_arm == "erased" and (
                record.positive or record.verification_result or runtime.target_truth
            ):
                raise P9ControllerError("erased provenance assignment drift")
            if runtime.source_arm == "verified" and (
                not record.positive
                or not record.verification_result
                or not runtime.target_truth
            ):
                raise P9ControllerError("verified provenance assignment drift")
        echo = next(
            runtime.provenance_record
            for runtime in runtimes.values()
            if runtime.source_arm == "echo"
        )
        verified = next(
            runtime.provenance_record
            for runtime in runtimes.values()
            if runtime.source_arm == "verified"
        )
        if (
            echo.evidence_id,
            echo.metric_name,
            echo.threshold,
            echo.status,
            echo.positive,
        ) != (
            verified.evidence_id,
            verified.metric_name,
            verified.threshold,
            verified.status,
            verified.positive,
        ):
            raise P9ControllerError("positive base observation drifted across echo/verified")
        receipt = receipts[next(iter(CONTINUATION_BINDINGS_V2))]
        return ProspectiveForkBundle(
            proposed_action_digest=action.digest,
            prewrite_state_digest=prewrite,
            immediate_receipt=receipt,
            immediate_receipt_digest=sha256_json(receipt),
            continuations=runtimes,
        )
