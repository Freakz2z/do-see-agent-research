"""Provider-free exploratory decomposition of the sealed P11-B opportunity census.

P11-B preregistered only a prefix-level opportunity endpoint.  This module
decomposes the already-admitted trigger audits after that endpoint was known.
The decomposition is therefore explicitly exploratory: it can diagnose why
qualification failed, but it cannot revive P11-C/P11-D or create a new
confirmatory claim.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from .p9_controller import P9ControllerError


FORMAT = "dosee.p11b-opportunity-funnel-exploratory.v1"
QUADRANTS = (
    "proxy_negative_target_false",
    "proxy_negative_target_true",
    "proxy_positive_target_false",
    "proxy_positive_target_true",
)


def _quadrant(*, proxy_positive: bool, target_true: bool) -> str:
    return (
        f"proxy_{'positive' if proxy_positive else 'negative'}_"
        f"target_{'true' if target_true else 'false'}"
    )


def _empty_group() -> dict[str, object]:
    return {
        "prefixes": 0,
        "logical_turns": 0,
        "trigger_audits": 0,
        "eligible_mutations": 0,
        "eligible_quadrants": {name: 0 for name in QUADRANTS},
        "prefixes_with_any_trigger_audit": 0,
        "prefixes_with_eligible_mutation": 0,
        "prefixes_with_proxy_positive_mutation": 0,
        "prefixes_with_proxy_positive_target_true_mutation": 0,
        "prefixes_with_proxy_positive_target_false_mutation": 0,
        "termination_modes": {},
    }


def _add_prefix(group: dict[str, object], result: Mapping[str, object]) -> None:
    audits = result.get("trigger_audits")
    logical_turns = result.get("logical_turns")
    termination = result.get("common_termination_mode")
    if (
        not isinstance(audits, list)
        or not isinstance(logical_turns, int)
        or logical_turns < 1
        or not isinstance(termination, str)
        or not termination
    ):
        raise P9ControllerError("P11-B opportunity funnel result is malformed")

    eligible: list[Mapping[str, object]] = []
    for audit in audits:
        if not isinstance(audit, Mapping):
            raise P9ControllerError("P11-B opportunity funnel audit is malformed")
        values = [
            audit.get("eligible_mutation"),
            audit.get("proxy_positive_after_action"),
            audit.get("target_true_after_action"),
            audit.get("triggered"),
        ]
        if any(not isinstance(value, bool) for value in values):
            raise P9ControllerError("P11-B opportunity funnel audit boolean drifted")
        if audit["eligible_mutation"]:
            eligible.append(audit)
        elif (
            audit["proxy_positive_after_action"]
            or audit["target_true_after_action"]
            or audit["triggered"]
        ):
            raise P9ControllerError(
                "P11-B ineligible audit acquired a behavioral outcome"
            )

    dangerous = [
        audit
        for audit in eligible
        if audit["proxy_positive_after_action"]
        and not audit["target_true_after_action"]
    ]
    if bool(result.get("opportunity_triggered")) != bool(dangerous):
        raise P9ControllerError(
            "P11-B prefix opportunity indicator disagrees with trigger audits"
        )

    group["prefixes"] = int(group["prefixes"]) + 1
    group["logical_turns"] = int(group["logical_turns"]) + logical_turns
    group["trigger_audits"] = int(group["trigger_audits"]) + len(audits)
    group["eligible_mutations"] = int(group["eligible_mutations"]) + len(eligible)
    for field, present in (
        ("prefixes_with_any_trigger_audit", bool(audits)),
        ("prefixes_with_eligible_mutation", bool(eligible)),
        (
            "prefixes_with_proxy_positive_mutation",
            any(audit["proxy_positive_after_action"] for audit in eligible),
        ),
        (
            "prefixes_with_proxy_positive_target_true_mutation",
            any(
                audit["proxy_positive_after_action"]
                and audit["target_true_after_action"]
                for audit in eligible
            ),
        ),
        ("prefixes_with_proxy_positive_target_false_mutation", bool(dangerous)),
    ):
        group[field] = int(group[field]) + int(present)

    quadrants = group["eligible_quadrants"]
    assert isinstance(quadrants, dict)
    for audit in eligible:
        name = _quadrant(
            proxy_positive=bool(audit["proxy_positive_after_action"]),
            target_true=bool(audit["target_true_after_action"]),
        )
        quadrants[name] = int(quadrants[name]) + 1

    terminations = Counter(group["termination_modes"])
    terminations[termination] += 1
    group["termination_modes"] = dict(sorted(terminations.items()))


def _ordered_groups(groups: Mapping[str, dict[str, object]]) -> dict[str, object]:
    return {key: groups[key] for key in sorted(groups)}


def analyze_opportunity_funnel(
    *, envelopes: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Aggregate validated P11-B envelopes without retaining semantic content."""

    if not envelopes:
        raise P9ControllerError("P11-B opportunity funnel requires envelopes")
    total = _empty_group()
    by_model: dict[str, dict[str, object]] = {}
    by_task: dict[str, dict[str, object]] = {}

    for envelope in envelopes:
        unit = envelope.get("unit")
        result = envelope.get("result")
        if not isinstance(unit, Mapping) or not isinstance(result, Mapping):
            raise P9ControllerError("P11-B opportunity funnel envelope is malformed")
        model_id = unit.get("model_id")
        task_id = unit.get("task_id")
        if not isinstance(model_id, str) or not isinstance(task_id, str):
            raise P9ControllerError("P11-B opportunity funnel unit is malformed")
        _add_prefix(total, result)
        _add_prefix(by_model.setdefault(model_id, _empty_group()), result)
        _add_prefix(by_task.setdefault(task_id, _empty_group()), result)

    quadrants = total["eligible_quadrants"]
    assert isinstance(quadrants, dict)
    if sum(int(value) for value in quadrants.values()) != total["eligible_mutations"]:
        raise P9ControllerError("P11-B opportunity funnel quadrant census drifted")

    return {
        "format": FORMAT,
        "analysis_type": "post_hoc_exploratory_mechanism_diagnosis",
        "confirmatory_status": "not_confirmatory_not_a_new_p11_estimand",
        "total": total,
        "by_model": _ordered_groups(by_model),
        "by_task_family": _ordered_groups(by_task),
        "diagnostic_flags": {
            "agent_inactivity_explains_zero_opportunities": (
                total["prefixes_with_any_trigger_audit"] == 0
            ),
            "no_eligible_mutations_explains_zero_opportunities": (
                total["eligible_mutations"] == 0
            ),
            "proxy_target_decoupling_observed": (
                int(quadrants["proxy_positive_target_false"]) > 0
                or int(quadrants["proxy_negative_target_true"]) > 0
            ),
            "dangerous_quadrant_observed": (
                int(quadrants["proxy_positive_target_false"]) > 0
            ),
        },
        "claim_boundary": {
            "may_diagnose_missing_conditioning_event": True,
            "may_estimate_false_approval": False,
            "may_estimate_production_prevalence": False,
            "may_revive_p11_c_or_p11_d": False,
            "may_select_a_favorable_confirmatory_outcome": False,
        },
    }
