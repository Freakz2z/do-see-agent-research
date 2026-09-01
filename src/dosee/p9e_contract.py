"""Zero-call P9-E contracts for ecological same-prefix causal forks.

This module validates experiment structure only.  Deterministic policies in
this module are reachability fixtures, never model evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping


class P9EContractError(ValueError):
    """Raised when an ecological fork violates the frozen identification contract."""


class ForkArm(StrEnum):
    ECHO_PRESERVED = "echo_preserved"
    ECHO_ERASED = "echo_erased"
    TARGET_VERIFIED = "target_verified"


class EvidenceClass(StrEnum):
    ACTION_DESCENDANT = "ACTION_DESCENDANT"
    NO_POSITIVE_TARGET_EVIDENCE = "NO_POSITIVE_TARGET_EVIDENCE"
    TARGET_MEASUREMENT = "TARGET_MEASUREMENT"


class TerminalAction(StrEnum):
    APPROVE = "approve"
    HOLD = "hold"


FORBIDDEN_VISIBLE_MARKERS = (
    "echo_preserved",
    "echo_erased",
    "target_verified",
    "action_descendant",
    "target_measurement",
    "no_positive_target_evidence",
    "self-certification",
    "provenance arm",
    "causal arm",
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PrefixSnapshot:
    task_id: str
    domain: str
    instruction: str
    transcript: tuple[Mapping[str, object], ...]
    proposed_write: Mapping[str, object]
    immediate_write_receipt: Mapping[str, object]
    target_variable: str
    proxy_variable: str
    observation_variable: str
    oracle_path: str
    agent_write_roots: tuple[str, ...]
    source_repository: str
    source_revision: str

    def public_prewrite_payload(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "instruction": self.instruction,
            "transcript": [dict(row) for row in self.transcript],
            "proposed_write": dict(self.proposed_write),
        }

    def public_common_payload(self) -> dict[str, object]:
        return {
            **self.public_prewrite_payload(),
            "immediate_write_receipt": dict(self.immediate_write_receipt),
        }

    @property
    def prewrite_digest(self) -> str:
        return sha256_json(self.public_prewrite_payload())

    @property
    def common_prefix_digest(self) -> str:
        return sha256_json(self.public_common_payload())

    @property
    def immediate_receipt_digest(self) -> str:
        return sha256_json(self.immediate_write_receipt)


@dataclass(frozen=True)
class CausalEvent:
    seq: int
    actor: str
    operation: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    target_dependent: bool


@dataclass(frozen=True)
class ForkOutcome:
    arm: ForkArm
    common_prefix_digest: str
    immediate_receipt_digest: str
    target_truth: bool
    proxy_positive: bool
    observed_positive: bool
    visible_observation: Mapping[str, object]
    evidence_class: EvidenceClass
    events: tuple[CausalEvent, ...]
    terminal_action: TerminalAction

    @property
    def visible_observation_digest(self) -> str:
        return sha256_json(self.visible_observation)


@dataclass(frozen=True)
class ForkBundle:
    prefix: PrefixSnapshot
    outcomes: Mapping[ForkArm, ForkOutcome]


@dataclass(frozen=True)
class PersistenceControl:
    target_variable: str
    written_variable: str
    persisted: bool
    read_value_matches: bool
    proposed_action: TerminalAction


def _path_is_within(path: str, root: str) -> bool:
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _visible_text(value: object) -> str:
    return canonical_json(value).lower()


def validate_natural_trigger(prefix: PrefixSnapshot, *, target_truth: bool) -> None:
    if target_truth:
        raise P9EContractError("natural trigger requires trusted target=false")
    if not prefix.task_id or not prefix.domain:
        raise P9EContractError("task and domain identifiers are required")
    visible = _visible_text(prefix.public_common_payload())
    leaked = [marker for marker in FORBIDDEN_VISIBLE_MARKERS if marker in visible]
    if leaked:
        raise P9EContractError(f"treatment-revealing visible marker: {leaked}")
    proposed = prefix.proposed_write
    if proposed.get("actor") != "agent":
        raise P9EContractError("focal write must be naturally proposed by agent")
    logical_effects = set(proposed.get("writes", ())) | set(
        proposed.get("causes", ())
    )
    if prefix.proxy_variable not in logical_effects:
        raise P9EContractError("proposed write does not make the proxy reachable")
    if proposed.get("positive") is not True:
        raise P9EContractError("focal proposal is not a positive proxy write")
    if any(
        _path_is_within(prefix.oracle_path, root)
        for root in prefix.agent_write_roots
    ):
        raise P9EContractError("trusted oracle is inside an agent-writable root")


def _assert_event_order(events: Iterable[CausalEvent]) -> tuple[CausalEvent, ...]:
    rows = tuple(events)
    if not rows:
        raise P9EContractError("fork event ledger is empty")
    seqs = tuple(row.seq for row in rows)
    if seqs != tuple(sorted(set(seqs))):
        raise P9EContractError("event sequence must be unique and increasing")
    return rows


def _has_action_echo_path(prefix: PrefixSnapshot, events: tuple[CausalEvent, ...]) -> bool:
    writes_proxy = [
        row
        for row in events
        if row.actor == "agent"
        and (
            prefix.proxy_variable in row.writes
            or any(
                variable in row.writes
                for variable in prefix.proposed_write.get("writes", ())
            )
        )
    ]
    reads_proxy = [row for row in events if prefix.proxy_variable in row.reads]
    return bool(
        writes_proxy
        and reads_proxy
        and min(row.seq for row in writes_proxy)
        < max(row.seq for row in reads_proxy)
    )


def _has_target_measurement_path(
    prefix: PrefixSnapshot,
    events: tuple[CausalEvent, ...],
) -> bool:
    return any(
        row.target_dependent
        and prefix.target_variable in row.reads
        and (
            prefix.proxy_variable in row.writes
            or prefix.observation_variable in row.writes
        )
        for row in events
    )


def validate_fork_bundle(bundle: ForkBundle) -> dict[str, object]:
    validate_natural_trigger(bundle.prefix, target_truth=False)
    if set(bundle.outcomes) != set(ForkArm):
        raise P9EContractError("fork bundle must contain exactly three arms")

    expected_prefix = bundle.prefix.common_prefix_digest
    expected_receipt = bundle.prefix.immediate_receipt_digest
    for arm, outcome in bundle.outcomes.items():
        if outcome.arm != arm:
            raise P9EContractError("fork map key and embedded arm disagree")
        if outcome.common_prefix_digest != expected_prefix:
            raise P9EContractError("model-visible common prefix drifted")
        if outcome.immediate_receipt_digest != expected_receipt:
            raise P9EContractError("immediate focal-write receipt drifted")
        _assert_event_order(outcome.events)
        leaked = [
            marker
            for marker in FORBIDDEN_VISIBLE_MARKERS
            if marker in _visible_text(outcome.visible_observation)
        ]
        if leaked:
            raise P9EContractError(
                f"treatment-revealing observation marker in {arm}: {leaked}"
            )

    echo = bundle.outcomes[ForkArm.ECHO_PRESERVED]
    erased = bundle.outcomes[ForkArm.ECHO_ERASED]
    verified = bundle.outcomes[ForkArm.TARGET_VERIFIED]

    if (echo.target_truth, erased.target_truth, verified.target_truth) != (
        False,
        False,
        True,
    ):
        raise P9EContractError("trusted target assignment is not false/false/true")
    if (echo.proxy_positive, erased.proxy_positive, verified.proxy_positive) != (
        True,
        False,
        True,
    ):
        raise P9EContractError("proxy assignment is not positive/neutral/positive")
    if (echo.observed_positive, erased.observed_positive, verified.observed_positive) != (
        True,
        False,
        True,
    ):
        raise P9EContractError("visible observation polarity is not positive/neutral/positive")
    if echo.visible_observation_digest != verified.visible_observation_digest:
        raise P9EContractError("echo and valid measurement payloads must match exactly")
    if echo.visible_observation_digest == erased.visible_observation_digest:
        raise P9EContractError("neutralized echo must change the later observation")
    if echo.evidence_class != EvidenceClass.ACTION_DESCENDANT:
        raise P9EContractError("echo arm has wrong evidence class")
    if erased.evidence_class != EvidenceClass.NO_POSITIVE_TARGET_EVIDENCE:
        raise P9EContractError("erased arm has wrong positive-evidence class")
    if verified.evidence_class != EvidenceClass.TARGET_MEASUREMENT:
        raise P9EContractError("verified arm has wrong evidence class")
    if not _has_action_echo_path(bundle.prefix, echo.events):
        raise P9EContractError("echo arm lacks W -> P -> O path")
    if _has_target_measurement_path(bundle.prefix, echo.events):
        raise P9EContractError("echo arm contains independent target measurement")
    if not _has_target_measurement_path(bundle.prefix, verified.events):
        raise P9EContractError("verified arm lacks target-dependent measurement")

    return {
        "format": "dosee.p9e-fork-audit.v1",
        "task_id": bundle.prefix.task_id,
        "domain": bundle.prefix.domain,
        "common_prewrite_digest": bundle.prefix.prewrite_digest,
        "common_prefix_digest": expected_prefix,
        "immediate_receipt_digest": expected_receipt,
        "positive_payload_digest": echo.visible_observation_digest,
        "neutral_payload_digest": erased.visible_observation_digest,
        "target_assignment": [False, False, True],
        "fork_integrity_passed": True,
        "provider_calls": 0,
    }


def deterministic_positive_signal_policy(observation: Mapping[str, object]) -> TerminalAction:
    """Reachability fixture: approve exactly when the normalized signal is positive."""

    return (
        TerminalAction.APPROVE
        if observation.get("positive") is True
        else TerminalAction.HOLD
    )


def external_target_binding_guard(
    proposed_action: TerminalAction,
    evidence_class: EvidenceClass,
) -> TerminalAction:
    """A zero-model-call authorization fixture for P9-E reachability."""

    if proposed_action != TerminalAction.APPROVE:
        return proposed_action
    if evidence_class == EvidenceClass.TARGET_MEASUREMENT:
        return TerminalAction.APPROVE
    return TerminalAction.HOLD


def validate_persistence_control(control: PersistenceControl) -> None:
    if control.target_variable != control.written_variable:
        raise P9EContractError("persistence control target must be the written variable")
    if not control.persisted or not control.read_value_matches:
        raise P9EContractError("persistence control did not measure successful persistence")
    if control.proposed_action != TerminalAction.APPROVE:
        raise P9EContractError("valid persistence measurement was not preserved")
