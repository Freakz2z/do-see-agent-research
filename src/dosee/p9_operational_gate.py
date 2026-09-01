"""Science-blind operational gate for a future authorized P9 model slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .p9e_contract import P9EContractError, sha256_json


UNIT_FIELDS = frozenset(
    {
        "unit_id",
        "model_registry_id",
        "task_id",
        "prompt_digest",
        "tool_schema_digest",
        "request_ids",
        "logical_turns",
        "physical_attempts",
        "parseable",
        "tool_compatible",
        "complete_logging",
        "snapshot_integrity",
        "fork_reachability",
        "retry_accounting_complete",
        "quota_exhausted",
        "transport_stop",
    }
)
MANIFEST_FIELDS = frozenset(
    {
        "format",
        "plan_digest",
        "planned_unit_ids",
        "logical_turn_ceiling",
        "physical_attempt_ceiling",
        "model_registry_digest",
        "prompt_registry_digest",
        "tool_schema_registry_digest",
    }
)


@dataclass(frozen=True)
class OperationalUnit:
    unit_id: str
    model_registry_id: str
    task_id: str
    prompt_digest: str
    tool_schema_digest: str
    request_ids: tuple[str, ...]
    logical_turns: int
    physical_attempts: int
    parseable: bool
    tool_compatible: bool
    complete_logging: bool
    snapshot_integrity: bool
    fork_reachability: bool
    retry_accounting_complete: bool
    quota_exhausted: bool
    transport_stop: bool


def _exact_fields(row: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    actual = frozenset(row)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise P9EContractError(
            f"science-blind {label} schema mismatch; unknown={unknown}, missing={missing}"
        )


def load_operational_unit(row: Mapping[str, object]) -> OperationalUnit:
    _exact_fields(row, UNIT_FIELDS, "unit")
    request_ids_raw = row["request_ids"]
    if not isinstance(request_ids_raw, list) or not all(
        isinstance(value, str) and value for value in request_ids_raw
    ):
        raise P9EContractError("request_ids must be a non-empty-string list")
    logical_turns = int(row["logical_turns"])
    physical_attempts = int(row["physical_attempts"])
    if logical_turns < 0 or physical_attempts < logical_turns:
        raise P9EContractError("invalid logical/physical attempt accounting")
    boolean_names = UNIT_FIELDS - {
        "unit_id",
        "model_registry_id",
        "task_id",
        "prompt_digest",
        "tool_schema_digest",
        "request_ids",
        "logical_turns",
        "physical_attempts",
    }
    if not all(type(row[name]) is bool for name in boolean_names):
        raise P9EContractError("operational flags must be booleans")
    return OperationalUnit(
        unit_id=str(row["unit_id"]),
        model_registry_id=str(row["model_registry_id"]),
        task_id=str(row["task_id"]),
        prompt_digest=str(row["prompt_digest"]),
        tool_schema_digest=str(row["tool_schema_digest"]),
        request_ids=tuple(request_ids_raw),
        logical_turns=logical_turns,
        physical_attempts=physical_attempts,
        parseable=bool(row["parseable"]),
        tool_compatible=bool(row["tool_compatible"]),
        complete_logging=bool(row["complete_logging"]),
        snapshot_integrity=bool(row["snapshot_integrity"]),
        fork_reachability=bool(row["fork_reachability"]),
        retry_accounting_complete=bool(row["retry_accounting_complete"]),
        quota_exhausted=bool(row["quota_exhausted"]),
        transport_stop=bool(row["transport_stop"]),
    )


def audit_operational_slice(
    manifest: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Audit mechanics while making behavioral fields unrepresentable."""

    _exact_fields(manifest, MANIFEST_FIELDS, "manifest")
    if manifest["format"] != "dosee.p9-operational-plan.v1":
        raise P9EContractError("unexpected P9 operational plan format")
    planned_raw = manifest["planned_unit_ids"]
    if not isinstance(planned_raw, list) or not all(
        isinstance(value, str) and value for value in planned_raw
    ):
        raise P9EContractError("planned_unit_ids must be a string list")
    planned = tuple(planned_raw)
    if len(set(planned)) != len(planned):
        raise P9EContractError("planned operational unit IDs are not unique")
    units = tuple(load_operational_unit(row) for row in rows)
    observed_ids = tuple(unit.unit_id for unit in units)
    if observed_ids != planned:
        raise P9EContractError("operational unit census/order differs from frozen plan")
    request_ids = [request_id for unit in units for request_id in unit.request_ids]
    if len(request_ids) != len(set(request_ids)):
        raise P9EContractError("provider request IDs are not globally unique")
    logical = sum(unit.logical_turns for unit in units)
    physical = sum(unit.physical_attempts for unit in units)
    if logical > int(manifest["logical_turn_ceiling"]):
        raise P9EContractError("logical turn ceiling exceeded")
    if physical > int(manifest["physical_attempt_ceiling"]):
        raise P9EContractError("physical attempt ceiling exceeded")
    quota_stopped = any(unit.quota_exhausted for unit in units)
    transport_stopped = any(unit.transport_stop for unit in units)
    integrity = all(
        unit.parseable
        and unit.tool_compatible
        and unit.complete_logging
        and unit.snapshot_integrity
        and unit.fork_reachability
        and unit.retry_accounting_complete
        for unit in units
    )
    passed = bool(units) and integrity and not quota_stopped and not transport_stopped
    return {
        "format": "dosee.p9-science-blind-operational-gate.v1",
        "plan_digest": manifest["plan_digest"],
        "unit_count": len(units),
        "logical_turns": logical,
        "physical_attempts": physical,
        "all_request_ids_unique": True,
        "quota_stopped": quota_stopped,
        "transport_stopped": transport_stopped,
        "operational_integrity_passed": integrity,
        "science_blind_gate_passed": passed,
        "input_digest": sha256_json(
            {"manifest": dict(manifest), "units": [dict(row) for row in rows]}
        ),
        "behavioral_fields_read": [],
        "behavioral_estimate_computed": False,
    }
