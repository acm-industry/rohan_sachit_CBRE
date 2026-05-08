"""Tests for `agent.nodes.validator`.

Validates the HITL gate logic: needs_human_review decisions,
emergency dispatch conservatism, and edge cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data.hitl import _reset_cache_for_tests  # noqa: E402
from agent.nodes.validator import validate, ValidatorResult  # noqa: E402


def _reset():
    _reset_cache_for_tests()


# ─── Required tests per acceptance criteria ───────────────────────────


def test_routine_call_no_review():
    """LOW risk, non-trap subcategory, good confidence → auto-resolve."""
    _reset()
    result = validate(
        subcategory="door_mechanical",
        risk_level="LOW",
        classification_confidence=0.9,
        fallback_invoked=False,
    )
    assert result.needs_human_review is False
    assert result.dispatched_emergency_services is False
    assert "auto_resolve" in result.reasons[0]


def test_high_risk_needs_review():
    """HIGH risk → always pause for human review."""
    _reset()
    result = validate(
        subcategory="unauthorized_access",
        risk_level="HIGH",
        classification_confidence=0.85,
        fallback_invoked=False,
    )
    assert result.needs_human_review is True
    assert result.dispatched_emergency_services is False
    assert "high_band" in result.reasons[0]


def test_fire_emergency_dispatches_911():
    """EMERGENCY + life-safety subcategory → 911 + review."""
    _reset()
    result = validate(
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        classification_confidence=0.95,
        fallback_invoked=False,
    )
    assert result.needs_human_review is True
    assert result.dispatched_emergency_services is True


def test_trap_subcategory_triggers_review():
    """LOW risk but trap-prone subcategory (air_quality, >15% OER) → review, no 911."""
    _reset()
    result = validate(
        subcategory="air_quality",
        risk_level="LOW",
        classification_confidence=0.8,
        fallback_invoked=False,
    )
    assert result.needs_human_review is True
    assert result.dispatched_emergency_services is False
    assert "trap_prone" in result.reasons[0]


# ─── Additional edge cases ────────────────────────────────────────────


def test_low_confidence_triggers_review():
    """Low classification confidence → pause regardless of risk."""
    _reset()
    result = validate(
        subcategory="lighting",
        risk_level="LOW",
        classification_confidence=0.3,
        fallback_invoked=False,
    )
    assert result.needs_human_review is True
    assert "low_confidence" in result.reasons[0]


def test_fallback_invoked_triggers_review():
    """Classifier fallback path → always pause."""
    _reset()
    result = validate(
        subcategory="lighting",
        risk_level="LOW",
        classification_confidence=0.9,
        fallback_invoked=True,
    )
    assert result.needs_human_review is True
    assert "fallback" in result.reasons[0]


def test_emergency_non_life_safety_no_911():
    """EMERGENCY but non-life-safety subcategory → review but no 911."""
    _reset()
    result = validate(
        subcategory="pipe_leak",
        risk_level="EMERGENCY",
        classification_confidence=0.9,
        fallback_invoked=False,
    )
    assert result.needs_human_review is True
    assert result.dispatched_emergency_services is False


def test_full_dev_set_hitl_accuracy():
    """Validate HITL decisions against 200-row dev set ground truth."""
    import json

    _reset()
    labels_path = Path(__file__).resolve().parents[1] / "evaluation" / "dev_labels.json"
    labels = json.loads(labels_path.read_text())["by_id"]

    tp = fp = fn = 0
    false_911 = 0

    for tid, l in labels.items():
        gt_review = bool(l.get("true_needs_human_review"))
        subcategory = l.get("true_subcategory", "")
        risk_level = l.get("true_risk_level", "LOW")

        result = validate(
            subcategory=subcategory,
            risk_level=risk_level,
            classification_confidence=0.9,
            fallback_invoked=False,
        )

        if result.needs_human_review and gt_review:
            tp += 1
        elif result.needs_human_review and not gt_review:
            fp += 1
        elif not result.needs_human_review and gt_review:
            fn += 1

        if l.get("is_over_escalation_trap") and result.dispatched_emergency_services:
            false_911 += 1

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0

    assert false_911 == 0, f"False 911 dispatches: {false_911}"
    assert f1 >= 0.70, f"HITL F1 = {f1:.3f}, expected >= 0.70"


if __name__ == "__main__":
    import inspect

    tests = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__])
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
