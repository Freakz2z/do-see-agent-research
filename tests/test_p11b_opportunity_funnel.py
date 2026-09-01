from __future__ import annotations

import pytest

from dosee.p11b_opportunity_funnel import analyze_opportunity_funnel
from dosee.p9_controller import P9ControllerError


def _audit(*, eligible: bool, proxy: bool, target: bool, triggered: bool = False):
    return {
        "eligible_mutation": eligible,
        "proxy_positive_after_action": proxy,
        "target_true_after_action": target,
        "triggered": triggered,
    }


def _envelope(model: str, task: str, audits: list[dict[str, bool]]):
    return {
        "unit": {"model_id": model, "task_id": task},
        "result": {
            "logical_turns": 3,
            "common_termination_mode": (
                "natural_opportunity_intercept"
                if any(audit["triggered"] for audit in audits)
                else "assistant_stop_without_tool"
            ),
            "trigger_audits": audits,
            "opportunity_triggered": any(audit["triggered"] for audit in audits),
        },
    }


def test_funnel_separates_activity_from_dangerous_quadrant() -> None:
    result = analyze_opportunity_funnel(
        envelopes=[
            _envelope(
                "m1",
                "t1",
                [
                    _audit(eligible=False, proxy=False, target=False),
                    _audit(eligible=True, proxy=False, target=False),
                    _audit(eligible=True, proxy=True, target=True),
                ],
            ),
            _envelope(
                "m2",
                "t1",
                [_audit(eligible=True, proxy=True, target=False, triggered=True)],
            ),
        ]
    )
    total = result["total"]
    assert total["prefixes"] == 2
    assert total["trigger_audits"] == 4
    assert total["eligible_mutations"] == 3
    assert total["eligible_quadrants"] == {
        "proxy_negative_target_false": 1,
        "proxy_negative_target_true": 0,
        "proxy_positive_target_false": 1,
        "proxy_positive_target_true": 1,
    }
    assert total["prefixes_with_any_trigger_audit"] == 2
    assert total["prefixes_with_proxy_positive_target_false_mutation"] == 1
    assert result["diagnostic_flags"]["agent_inactivity_explains_zero_opportunities"] is False
    assert result["diagnostic_flags"]["dangerous_quadrant_observed"] is True


def test_funnel_rejects_indicator_audit_disagreement() -> None:
    envelope = _envelope(
        "m1", "t1", [_audit(eligible=True, proxy=True, target=False, triggered=False)]
    )
    with pytest.raises(P9ControllerError, match="disagrees"):
        analyze_opportunity_funnel(envelopes=[envelope])


def test_funnel_rejects_outcome_on_ineligible_action() -> None:
    envelope = _envelope(
        "m1", "t1", [_audit(eligible=False, proxy=True, target=False)]
    )
    with pytest.raises(P9ControllerError, match="ineligible"):
        analyze_opportunity_funnel(envelopes=[envelope])
