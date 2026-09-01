from __future__ import annotations

import json
from pathlib import Path

from dosee.p12_fault_conditioned_runtime import audit_fault_conditioned_runtime
from dosee.p12_fault_injection import audit_fault_injection
from dosee.p9_runtime_factory import FrozenToolRegistryV2


ROOT = Path(__file__).resolve().parents[1]


def _registry() -> FrozenToolRegistryV2:
    return FrozenToolRegistryV2(
        json.loads((ROOT / "configs/p9_localization_tool_registry_v2.json").read_text())
    )


def test_four_faults_reproducibly_reach_dangerous_quadrant(tmp_path: Path) -> None:
    result = audit_fault_injection(
        root=tmp_path / "faults", python_executable=Path(__import__("sys").executable)
    )
    assert result["passed"] is True
    assert result["task_count"] == 4
    assert all(row["proxy_positive_after_fault"] for row in result["tasks"])
    assert not any(row["protected_target_true_after_fault"] for row in result["tasks"])


def test_fault_conditioned_forks_preserve_same_prefix_controls(tmp_path: Path) -> None:
    result = audit_fault_conditioned_runtime(
        root=tmp_path / "runtime",
        registry=_registry(),
        python_executable=Path(__import__("sys").executable),
    )
    assert result["passed"] is True
    assert result["task_count"] == 4
    assert all(row["continuation_count"] == 6 for row in result["tasks"])
    assert all(row["fault_echo_and_erased_state_matched"] for row in result["tasks"])
    assert all(row["fault_target_truth"] is False for row in result["tasks"])
    assert all(row["healthy_verified_target_truth"] is True for row in result["tasks"])
