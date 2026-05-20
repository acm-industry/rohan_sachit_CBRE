"""Tests for `agent.nodes.risk.assign_risk`.

The AC requires four scenario tests — routine-LOW, after-hours-MEDIUM,
sensitive-building-HIGH, life-safety-EMERGENCY — plus the EMERGENCY-cap
guard that prevents non-life-safety subcategories from emitting EMERGENCY.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.nodes.extract import Extraction, FieldConfidence  # noqa: E402
from agent.nodes.risk import (  # noqa: E402
    _LIFE_SAFETY_SUBCATEGORIES,
    RiskAssignment,
    assign_risk,
)


def _extraction(urgency_cues=None, **overrides) -> Extraction:
    base = {
        "problem_summary": "test problem",
        "building_name": "Test Building",
        "floor": "Floor 1",
        "suite": "Suite 100",
        "urgency_cues": urgency_cues or [],
        "caller_role": "tenant",
        "language": "en",
        "confidence": {
            "problem_summary": 0.9, "building_name": 0.9,
            "floor": 0.9, "suite": 0.9, "caller_role": 0.7,
        },
    }
    base.update(overrides)
    return Extraction(**base)


# ─── The four AC-required scenarios ─────────────────────────────────────


def test_routine_low_no_cues_no_context():
    """A routine LOW-base subcategory with no urgency cues → LOW band."""
    e = _extraction(urgency_cues=[])
    r = assign_risk(e, "lighting")
    assert r.band == "LOW"
    assert r.base_risk == "LOW"
    assert r.reasons == ("base_risk:LOW",)
    assert r.score == 0.0


def test_after_hours_lifts_routine_to_medium():
    """A routine LOW-base call after hours → MEDIUM (one rank up)."""
    e = _extraction(urgency_cues=[])
    r = assign_risk(e, "no_cooling", after_hours=True)
    assert r.band == "MEDIUM"
    assert "after_hours" in r.reasons


def test_sensitive_building_plus_soft_cue_on_routine_base():
    """A glass_damage call (MEDIUM base, MEDIUM/HIGH historical max) in a
    medical building plus a soft cue → HIGH. glass_damage's historical
    distribution is 58% MEDIUM / 41% HIGH / 2% EMERGENCY (P(EMERGENCY) <
    threshold) so the cap is HIGH even with stacked modifiers."""
    e = _extraction(urgency_cues=["spreading"])
    r = assign_risk(e, "glass_damage", building_type="medical")
    assert r.band == "HIGH"
    assert "sensitive_building:medical" in r.reasons
    assert any("soft_cue" in reason for reason in r.reasons)


def test_life_safety_subcategory_emerges_at_emergency():
    """fire_smoke is a documented EMERGENCY-base subcategory."""
    e = _extraction(urgency_cues=["flames", "smoke"])
    r = assign_risk(e, "fire_smoke")
    assert r.band == "EMERGENCY"
    assert r.base_risk == "EMERGENCY"
    assert r.score == 1.0


# ─── EMERGENCY cap on non-life-safety subcategories ─────────────────────


def test_historical_max_cap_holds_for_routine_subcategory():
    """`lighting` is 100% LOW in history. Even with hard emergency cues +
    sensitive building + after-hours stacked, the data-driven cap pins
    band at LOW. This is the over-escalation-trap defense at the risk
    layer: alarming-sounding cues on inherently-routine subcategories
    don't escalate the band, because the subcategory's historical
    distribution says it's never severe."""
    e = _extraction(urgency_cues=["flames", "trapped"])
    r = assign_risk(e, "lighting", building_type="medical", after_hours=True)
    assert r.band == "LOW"
    assert any("capped_at_LOW" in reason for reason in r.reasons)


def test_historical_max_cap_holds_for_waste_odor_trap():
    """The over-escalation trap pattern: caller mentions 'smoke' but
    classifier correctly routes to waste_odor. waste_odor is 100% LOW
    historically — risk band must stay at LOW even with the 'smoke' cue."""
    e = _extraction(urgency_cues=["smoke"])
    r = assign_risk(e, "waste_odor")
    assert r.band == "LOW"


def test_documented_life_safety_set_matches_emergency_modal_subcategories():
    """Drift guard: the LIFE_SAFETY_SUBCATEGORIES set must stay in sync
    with the historical EMERGENCY-modal subcategories from issue #13."""
    from agent.data import risk as risk_data
    expected = set()
    for sub in risk_data.all_subcategories_in_table():
        if risk_data.base_risk_for(sub) == "EMERGENCY":
            expected.add(sub)
    assert _LIFE_SAFETY_SUBCATEGORIES == expected, (
        f"life-safety set drifted from historical EMERGENCY-modal set: "
        f"got {_LIFE_SAFETY_SUBCATEGORIES}, expected {expected}"
    )


# ─── Cue classification (the regex tier system) ─────────────────────────


def test_hard_cues_keep_emergency_subcategory_at_emergency():
    """active_threat is EMERGENCY base; a hard cue keeps it there (cap is also EMERGENCY)."""
    e = _extraction(urgency_cues=["unconscious"])
    r = assign_risk(e, "active_threat")
    assert r.band == "EMERGENCY"


def test_soft_cues_capped_at_high_on_routable_base():
    """Soft cues (flooding, getting-worse, …) may push the band UP but
    never beyond HIGH — only hard life-safety cues are permitted to
    reach EMERGENCY. pipe_leak (base HIGH) + "flooding" (soft +1) now
    stays at HIGH instead of cascading to EMERGENCY, which prevents
    false-emergency dispatches on routable-base subcategories."""
    e = _extraction(urgency_cues=["flooding"])  # soft cue: +1
    r = assign_risk(e, "pipe_leak")
    assert r.band == "HIGH"
    assert any("soft_cue:flooding" in reason for reason in r.reasons)


def test_hard_cues_uncapped_can_promote_pipe_leak_to_emergency():
    """The soft-cue cap is one-sided: hard life-safety cues (gas leak,
    fire, trapped, …) remain uncapped, so a genuine emergency on a
    routable base still promotes correctly. pipe_leak (base HIGH) +
    "gas leak" (hard +2) → EMERGENCY (within pipe_leak's historical
    max of 11% EMERGENCY)."""
    e = _extraction(urgency_cues=["gas leak"])  # hard cue: +2
    r = assign_risk(e, "pipe_leak")
    assert r.band == "EMERGENCY"
    assert any("hard_cue:gas leak" in reason for reason in r.reasons)


def test_soft_cues_bump_by_one():
    """Soft cues lift no_heating (base LOW) to MEDIUM."""
    e = _extraction(urgency_cues=["overnight"])
    r = assign_risk(e, "no_heating")
    assert r.band == "MEDIUM"
    assert any("soft_cue:overnight" in reason for reason in r.reasons)


def test_noise_phrases_dont_bump():
    """A cue with no known emergency or escalation keywords → no bump."""
    e = _extraction(urgency_cues=["the doorknob is loose"])
    r = assign_risk(e, "door_mechanical")
    assert r.band == "LOW"
    assert "the doorknob is loose" not in " ".join(r.reasons)


def test_fire_exit_does_not_trigger_fire_cue():
    """Phrase 'fire exit' is a door type, not a fire — must not match the
    hard 'fire' pattern. Validates the negative lookahead in the regex."""
    e = _extraction(urgency_cues=["it's a fire exit"])
    r = assign_risk(e, "auto_door")
    assert r.band == "LOW"


def test_smoke_alarm_does_not_trigger_smoke_cue():
    """'Smoke alarm' (the device) is not the same as ongoing smoke."""
    e = _extraction(urgency_cues=["smoke alarm beeped"])
    r = assign_risk(e, "waste_odor")
    assert r.band == "LOW"


# ─── Confidence reason surfacing ────────────────────────────────────────


def test_low_classifier_confidence_appears_in_reasons_but_doesnt_bump():
    e = _extraction(urgency_cues=[])
    r = assign_risk(e, "lighting", classification_confidence=0.3)
    # Band is unchanged — confidence is recorded for #16, not used as a bump.
    assert r.band == "LOW"
    assert any("low_classifier_confidence" in reason for reason in r.reasons)


def test_high_classifier_confidence_does_not_add_reason():
    e = _extraction(urgency_cues=[])
    r = assign_risk(e, "lighting", classification_confidence=0.95)
    assert all("low_classifier_confidence" not in reason for reason in r.reasons)


# ─── Ranking + score arithmetic ─────────────────────────────────────────


def test_score_normalized_to_zero_one():
    e = _extraction(urgency_cues=[])
    assert assign_risk(e, "lighting").score == 0.0
    assert assign_risk(e, "fire_smoke").score == 1.0


def test_unknown_subcategory_falls_back_to_medium_base():
    e = _extraction()
    r = assign_risk(e, "definitely_not_a_real_subcategory")
    assert r.base_risk == "MEDIUM"
    assert r.band == "MEDIUM"


# ─── Inline runner ─────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect
    tests = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__])
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
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
