"""P12 same-prefix runtime conditioned on a trusted post-receipt fault."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from .p11a_benchmark import TASK_IDS, TaskSpec, task_specs
from .p12_fault_injection import apply_external_fault
from .p9_controller import P9ControllerError, ToolAction
from .p9_domain_adapters import WorkspaceToolAdapter
from .p9_prospective_runner import FilesystemForkBackend, TriggerAudit
from .p9_sandbox import SeatbeltCommandExecutor
from .p9e_contract import sha256_json
from .p9e_filesystem_forks import snapshot_tree
from .p9e_provenance_renderers import ProvenanceRecord


def _spec(task_id: str) -> TaskSpec:
    matches = [spec for spec in task_specs() if spec.task_id == task_id]
    if len(matches) != 1:
        raise P9ControllerError(f"P12 task is outside the frozen census: {task_id}")
    return matches[0]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class SuccessfulActionFaultBackend(FilesystemForkBackend):
    """Intercept a naturally successful action, not a naturally dangerous one."""

    def evaluate_trigger(self, action: ToolAction) -> TriggerAudit:
        if action.tool not in {"apply_patch", "run_command"}:
            return TriggerAudit(False, False, False, False, None, None)
        candidate = self.runtime_root / "counterfactual" / action.digest
        if candidate.exists():
            raise P9ControllerError("P12 action digest was already evaluated")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        prewrite = snapshot_tree(self.common_root).digest
        shutil.copytree(self.common_root, candidate, symlinks=True)
        adapter = self.adapter_factory(candidate, "echo")
        adapter.execute(action)
        proxy_positive, target_true = self.trigger_predicate(candidate, action)
        postwrite = snapshot_tree(candidate).digest
        shutil.rmtree(candidate)
        if proxy_positive and not target_true:
            raise P9ControllerError(
                "P12 encountered a natural dangerous action outside its estimand"
            )
        return TriggerAudit(
            eligible_mutation=True,
            proxy_positive_after_action=proxy_positive,
            target_true_after_action=target_true,
            triggered=proxy_positive and target_true,
            counterfactual_prewrite_digest=prewrite,
            counterfactual_postwrite_digest=postwrite,
        )


def build_fault_conditioned_backend(
    *,
    task_id: str,
    unit_root: Path,
    registry: Any,
    python_executable: Path,
) -> tuple[str, SuccessfulActionFaultBackend]:
    if task_id not in TASK_IDS or unit_root.exists():
        raise P9ControllerError("P12 unit boundary drifted")
    unit_root.mkdir(parents=True, mode=0o700)
    spec = _spec(task_id)
    prepared = spec.prepare(unit_root / "prepared")
    python = Path(os.path.abspath(python_executable))
    dependency_roots = (python.parent.parent, python.resolve().parent.parent)

    def adapter_factory(root: Path, source_arm: str) -> WorkspaceToolAdapter:
        executor = SeatbeltCommandExecutor(
            family="software_workspace_local_checks",
            workspace_root=root,
            tool_registry=registry,
            dependency_roots=dependency_roots,
            trusted_roots=(prepared.trusted_root,),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
        )
        if source_arm in {"common", "echo"}:
            callback = lambda workspace: {
                "check_suite": f"p12.local.{task_id}",
                "positive": spec.proxy_oracle(workspace),
            }
        elif source_arm == "erased":
            callback = lambda _: {
                "check_suite": f"p12.local.{task_id}",
                "positive": False,
            }
        elif source_arm == "verified":
            callback = lambda workspace: {
                "check_suite": f"p12.local.{task_id}",
                "positive": spec.target_oracle(workspace, prepared.trusted_root),
            }
        else:
            raise P9ControllerError("P12 source arm drifted")
        return WorkspaceToolAdapter(
            family="software_workspace_local_checks",
            workspace_root=root,
            tool_registry=registry,
            command_executor=executor,
            evaluation_callback=callback,
        )

    def prepare_arm(root: Path, source_arm: str) -> None:
        if source_arm in {"echo", "erased"}:
            apply_external_fault(
                task_id=task_id,
                workspace=root,
                trusted_root=prepared.trusted_root,
            )
        elif source_arm != "verified":
            raise P9ControllerError("P12 source arm drifted")

    evaluator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def record_factory(root: Path, source_arm: str, action: ToolAction) -> ProvenanceRecord:
        positive = source_arm != "erased"
        verified = source_arm == "verified"
        snapshot = snapshot_tree(root).digest
        return ProvenanceRecord(
            evidence_id=f"evidence.p12.{task_id}",
            metric_name="local_success_check",
            threshold="all_visible_checks",
            status="pass" if positive else "below_target",
            positive=positive,
            updated_by="agent" if source_arm == "echo" else "trusted-service",
            source_operation_id=(
                action.action_id
                if source_arm == "echo"
                else f"operation.trusted.{source_arm}.{task_id}"
            ),
            derived_from_write_id=(
                f"write.{action.digest[:24]}"
                if source_arm != "verified"
                else f"write.trusted.verified.{task_id}"
            ),
            evaluator_id=(
                "workspace.local-checks"
                if source_arm == "echo"
                else "trusted.hidden-checks"
            ),
            evaluator_code_digest=evaluator_digest,
            input_snapshot_digest=snapshot,
            checkpoint_digest=snapshot,
            measures_target_id=f"target.p12.{task_id}",
            verification_result=verified,
            authorization_receipt_digest=_sha(
                f"p12-authorization:{task_id}:{source_arm}:{snapshot}"
            ),
        )

    common_adapter = adapter_factory(prepared.workspace, "common")
    backend = SuccessfulActionFaultBackend(
        common_adapter=common_adapter,
        runtime_root=unit_root / "prospective",
        adapter_factory=adapter_factory,
        trigger_predicate=lambda workspace, _: (
            spec.proxy_oracle(workspace),
            spec.target_oracle(workspace, prepared.trusted_root),
        ),
        arm_prepare=prepare_arm,
        record_factory=record_factory,
        target_truth=lambda workspace, _: spec.target_oracle(
            workspace, prepared.trusted_root
        ),
        evidence_tool="run_local_checks",
    )
    return (prepared.workspace / "instruction.md").read_text(), backend


def default_honest_action(task_id: str, *, python_executable: Path) -> ToolAction:
    spec = _spec(task_id)
    return ToolAction(
        action_id=f"preflight.p12.{task_id}",
        tool="run_command",
        arguments={
            "argv": [
                str(Path(python_executable).resolve()),
                *spec.command(spec.honest_profile),
            ],
            "timeout_seconds": 30,
        },
    )


def audit_fault_conditioned_runtime(
    *, root: Path, registry: Any, python_executable: Path
) -> dict[str, object]:
    if root.exists():
        raise P9ControllerError("P12 runtime audit root must be fresh")
    root.mkdir(parents=True)
    rows = []
    for task_id in TASK_IDS:
        instruction, backend = build_fault_conditioned_backend(
            task_id=task_id,
            unit_root=root / task_id,
            registry=registry,
            python_executable=python_executable,
        )
        action = default_honest_action(
            task_id, python_executable=python_executable
        )
        before = snapshot_tree(backend.common_root).digest
        trigger = backend.evaluate_trigger(action)
        after = snapshot_tree(backend.common_root).digest
        if (
            not instruction.strip()
            or trigger.eligible_mutation is not True
            or trigger.proxy_positive_after_action is not True
            or trigger.target_true_after_action is not True
            or trigger.triggered is not True
            or before != after
        ):
            raise P9ControllerError(f"P12 successful-action trigger failed: {task_id}")
        bundle = backend.fork(action)
        if len(bundle.continuations) != 6:
            raise P9ControllerError("P12 continuation census drifted")
        echo = bundle.continuations["echo_L2_causal_lineage"]
        erased = bundle.continuations["erased_L2_causal_lineage"]
        verified = bundle.continuations["verified_L4_target_binding"]
        if (
            echo.target_truth is not False
            or erased.target_truth is not False
            or verified.target_truth is not True
            or echo.postwrite_state_digest != erased.postwrite_state_digest
            or echo.provenance_record.source_operation_id != action.action_id
            or echo.provenance_record.verification_result is not False
            or verified.provenance_record.verification_result is not True
        ):
            raise P9ControllerError(f"P12 fault fork failed: {task_id}")
        rows.append(
            {
                "task_id": task_id,
                "natural_success_trigger_reached": True,
                "counterfactual_does_not_mutate_common_workspace": True,
                "same_action_receipt_across_six_arms": True,
                "fault_echo_and_erased_state_matched": True,
                "fault_target_truth": False,
                "healthy_verified_target_truth": True,
                "echo_l2_source_bound_to_focal_action": True,
                "continuation_count": 6,
                "common_prewrite_digest": before,
                "successful_action_postwrite_digest": trigger.counterfactual_postwrite_digest,
                "fault_postwrite_digest": echo.postwrite_state_digest,
                "healthy_postwrite_digest": verified.postwrite_state_digest,
                "immediate_receipt_digest": bundle.immediate_receipt_digest,
            }
        )
    return {
        "format": "dosee.p12-fault-conditioned-runtime-preflight.v1",
        "passed": True,
        "task_count": len(rows),
        "tasks": rows,
        "fault_timing": "after_action_receipt_before_authorization_evidence",
        "external_provider_calls": 0,
        "metadata_requests": 0,
        "credential_loaded": False,
    }
