"""Tests for `agent.nodes.validator`.

Validates the HITL gate logic: needs_human_review decisions,
emergency-dispatch conservatism (the hazard-cue / fallback / confidence
preconditions that prevent a false −5 911), and edge cases.
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


def test_fire_emergency_with_hazard_cue_dispatches_911():
    """EMERGENCY + life-safety subcategory + extracted hazard cue +
    confident, non-fallback classification → 911 + review."""
    _reset()
    result = validate(
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        classification_confidence=0.95,
        fallback_invoked=False,
        extracted_urgency_cues=["smoke filling the lobby", "I can see flames"],
    )
    assert result.needs_human_review is True
    assert result.dispatched_emergency_services is True
    assert any("emergency_dispatch:fire_smoke" in r for r in result.reasons)


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


def test_waste_odor_benign_smoke_trap_reviews_without_911():
    """LOW waste_odor normally auto-resolves, but benign smoke/fire
    language should still get a human review while preserving no-911."""
    _reset()
    transcript = (
        "[CALLER] Smoke alarm beeped once and there's smoke, but it was just burnt toast.\n"
        "[AGENT] Can you confirm there's no actual fire or injury right now?\n"
        "[CALLER] No fire, just the burnt smell."
    )
    result = validate(
        subcategory="waste_odor",
        risk_level="LOW",
        classification_confidence=0.95,
        fallback_invoked=False,
        extracted_urgency_cues=["smoke"],
        transcript_text=transcript,
    )
    assert result.needs_human_review is True
    assert result.dispatched_emergency_services is False
    assert any("benign_smoke_odor_review" in r for r in result.reasons)


def test_routine_waste_odor_still_auto_resolves():
    _reset()
    result = validate(
        subcategory="waste_odor",
        risk_level="LOW",
        classification_confidence=0.95,
        fallback_invoked=False,
        extracted_urgency_cues=[],
        transcript_text="[CALLER] Trash room odor near the service hallway.",
    )
    assert result.needs_human_review is False
    assert result.dispatched_emergency_services is False


# ─── False-911 prevention (the −5 penalty path) ───────────────────────


def test_misclassified_no_cue_does_not_dispatch_911():
    """A benign call *misclassified* as a life-safety EMERGENCY on the
    low-confidence fallback path with NO extracted hazard cue must never
    auto-dispatch 911 — it escalates to a human instead."""
    _reset()
    for sub in ("fire_smoke", "active_threat", "gas_chemical", "entrapment", "panel_hazard"):
        result = validate(
            subcategory=sub,
            risk_level="EMERGENCY",
            classification_confidence=0.2,
            fallback_invoked=True,
            extracted_urgency_cues=[],
        )
        assert result.dispatched_emergency_services is False, sub
        assert result.needs_human_review is True, sub
        assert any("life_safety_no_autodispatch" in r for r in result.reasons), sub


def test_life_safety_emergency_with_cue_but_fallback_no_911():
    """Hazard cue present but the classifier fell back → label is a guess;
    do not auto-dispatch, escalate instead."""
    _reset()
    result = validate(
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        classification_confidence=0.9,
        fallback_invoked=True,
        extracted_urgency_cues=["smoke everywhere"],
    )
    assert result.dispatched_emergency_services is False
    assert result.needs_human_review is True
    assert any("life_safety_no_autodispatch:classifier_fallback" in r for r in result.reasons)


def test_life_safety_emergency_with_cue_but_low_confidence_no_911():
    """Hazard cue present but classifier confidence below the dispatch
    floor → escalate to a human, do not auto-dispatch."""
    _reset()
    result = validate(
        subcategory="gas_chemical",
        risk_level="EMERGENCY",
        classification_confidence=0.35,
        fallback_invoked=False,
        extracted_urgency_cues=["strong gas leak smell"],
    )
    assert result.dispatched_emergency_services is False
    assert result.needs_human_review is True
    assert any("life_safety_no_autodispatch:low_confidence" in r for r in result.reasons)


def test_burnt_popcorn_benign_override_suppresses_911():
    """The test-set scripted over-escalation trap: caller reports the
    floor 'smells smoky' (matches the hard-hazard `smoke` cue), but the
    agent confirms 'no actual fire' and the caller agrees it's 'just
    burnt popcorn in the microwave'. The benign-context override
    suppresses 911 even though every other precondition holds — caught
    9 false-911s on the 800-row test eval (each −5 rubric)."""
    _reset()
    transcript = (
        "[AGENT] What's going on?\n"
        "[CALLER] Someone burned popcorn in the microwave — the whole floor smells smoky.\n"
        "[AGENT] Can you confirm there's no actual fire or injury right now?\n"
        "[CALLER] No fire, just the burnt smell."
    )
    result = validate(
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        classification_confidence=0.95,
        fallback_invoked=False,
        extracted_urgency_cues=["smoky", "smells smoky"],
        transcript_text=transcript,
    )
    assert result.dispatched_emergency_services is False
    assert result.needs_human_review is True
    assert any("benign_context" in r for r in result.reasons), result.reasons


def test_genuine_electrical_smoke_still_dispatches_911():
    """The benign-override must not suppress a genuine emergency that
    happens to share negation-looking phrases. 'Smoke coming from the
    electrical room' with no benign-context phrase → still dispatches.
    Regression guard against an over-aggressive negation gate."""
    _reset()
    transcript = (
        "[AGENT] What do you need?\n"
        "[CALLER] There's smoke coming from the electrical room, I can see it.\n"
        "[AGENT] Where exactly — which suite?\n"
        "[CALLER] Metro Center Offices, Floor 1, Suite 105."
    )
    result = validate(
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        classification_confidence=0.95,
        fallback_invoked=False,
        extracted_urgency_cues=["smoke coming from the electrical room"],
        transcript_text=transcript,
    )
    assert result.dispatched_emergency_services is True


def test_no_flames_alone_does_not_trigger_benign_override():
    """'No flames visible' on its own is NOT a benign-context signal —
    a smoke-only fire is still a real emergency. The override only
    fires on stronger specific phrases (no-actual-fire / burnt-popcorn /
    false-alarm / fire-drill / etc.)."""
    _reset()
    transcript = (
        "[CALLER] Hello, There's smoke coming from the electrical room, I can see it.\n"
        "[AGENT] Have you pulled the fire alarm?\n"
        "[CALLER] Just smoke right now, no flames visible.\n"
        "[AGENT] Where exactly — which suite?"
    )
    result = validate(
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        classification_confidence=0.95,
        fallback_invoked=False,
        extracted_urgency_cues=["smoke from the electrical room"],
        transcript_text=transcript,
    )
    assert result.dispatched_emergency_services is True


def test_emergency_non_life_safety_no_911():
    """EMERGENCY but non-life-safety subcategory → review but no 911,
    even with a hazard-sounding cue."""
    _reset()
    result = validate(
        subcategory="pipe_leak",
        risk_level="EMERGENCY",
        classification_confidence=0.9,
        fallback_invoked=False,
        extracted_urgency_cues=["water everywhere", "flooding fast"],
    )
    assert result.needs_human_review is True
    assert result.dispatched_emergency_services is False


def test_should_pause_method_mirrors_needs_review():
    """`ValidatorResult.should_pause()` is the boolean the graph reads
    for `interrupt()` — it must mirror needs_human_review and be
    decoupled from the 911 flag."""
    _reset()
    paused = validate(subcategory="unauthorized_access", risk_level="HIGH")
    assert paused.should_pause() is True
    assert paused.should_pause() == paused.needs_human_review
    auto = validate(
        subcategory="door_mechanical", risk_level="LOW",
        classification_confidence=0.9,
    )
    assert auto.should_pause() is False


# ─── Additional edge cases ────────────────────────────────────────────


def test_low_confidence_low_risk_auto_resolves():
    """Risk-weighted: LOW risk + low confidence = negligible cost of error."""
    _reset()
    result = validate(
        subcategory="lighting",
        risk_level="LOW",
        classification_confidence=0.3,
        fallback_invoked=False,
    )
    assert result.needs_human_review is False


def test_low_confidence_medium_risk_triggers_review():
    """Risk-weighted: MEDIUM risk + low confidence = non-trivial cost."""
    _reset()
    result = validate(
        subcategory="lighting",
        risk_level="MEDIUM",
        classification_confidence=0.3,
        fallback_invoked=False,
    )
    assert result.needs_human_review is True
    assert any("low_confidence_high_cost" in r for r in result.reasons)


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


def test_full_dev_set_hitl_accuracy():
    """Validate HITL decisions against 200-row dev set ground truth.

    No urgency cues are passed (oracle labels only), so this is also a
    strong false-911 guard: with no hazard cue, dispatch can never fire.
    """
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
