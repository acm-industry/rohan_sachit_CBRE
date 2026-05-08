"""Tests for `agent.nodes.trainer_log`.

Validates trainer log assembly, schema conformance, and edge cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.nodes.trainer_log import assemble_trainer_log, build_ai_prediction  # noqa: E402


def test_basic_assembly_has_all_keys():
    """Assembled log must contain all four required keys."""
    ai = build_ai_prediction(
        category="PLUMBING",
        subcategory="pipe_leak",
        risk_level="MEDIUM",
        dispatched_vendor_id="v_002",
    )
    log = assemble_trainer_log(
        full_transcript="Caller reports a pipe leak on floor 3.",
        ai_prediction=ai,
    )
    assert "full_transcript" in log
    assert "ai_prediction" in log
    assert "human_override" in log
    assert "final_decision" in log


def test_no_override_final_equals_ai():
    """When no override, final_decision == ai_prediction."""
    ai = build_ai_prediction(
        category="HVAC",
        subcategory="no_cooling",
        risk_level="LOW",
    )
    log = assemble_trainer_log(
        full_transcript="AC not working.",
        ai_prediction=ai,
    )
    assert log["human_override"] is None
    assert log["final_decision"] == log["ai_prediction"]


def test_override_captured():
    """When human overrides, final_decision reflects the override."""
    ai = build_ai_prediction(
        category="SECURITY",
        subcategory="suspicious_person",
        risk_level="HIGH",
        needs_human_review=True,
    )
    override = {"risk_level": "MEDIUM", "needs_human_review": False}
    final = {**ai, **override}

    log = assemble_trainer_log(
        full_transcript="Someone loitering in parking lot.",
        ai_prediction=ai,
        human_override=override,
        final_decision=final,
    )
    assert log["human_override"] == override
    assert log["final_decision"]["risk_level"] == "MEDIUM"
    assert log["ai_prediction"]["risk_level"] == "HIGH"


def test_full_transcript_preserved():
    """full_transcript must be the exact string passed in."""
    transcript = "This is a long transcript with special chars: é, ñ, 日本語"
    ai = build_ai_prediction(category="OTHER", subcategory="minor_issue", risk_level="LOW")
    log = assemble_trainer_log(full_transcript=transcript, ai_prediction=ai)
    assert log["full_transcript"] == transcript


def test_ai_prediction_snapshot_fields():
    """ai_prediction contains all expected fields."""
    ai = build_ai_prediction(
        category="FIRE_LIFE_SAFETY",
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        dispatched_vendor_id="v_021",
        dispatched_emergency_services=True,
        needs_human_review=True,
        confidence_category=0.95,
        confidence_subcategory=0.92,
        validator_reasons=["emergency_band:always_pause"],
        risk_reasons=["base_risk=EMERGENCY"],
        retrieved_record_ids=["REC-001", "REC-002"],
    )
    assert ai["category"] == "FIRE_LIFE_SAFETY"
    assert ai["dispatched_emergency_services"] is True
    assert ai["confidence_category"] == 0.95
    assert len(ai["retrieved_record_ids"]) == 2
    assert ai["validator_reasons"] == ["emergency_band:always_pause"]


def test_scoring_conformance():
    """Log passes the scoring.py check: isinstance(dict) with full_transcript and final_decision."""
    ai = build_ai_prediction(category="PLUMBING", subcategory="pipe_leak", risk_level="LOW")
    log = assemble_trainer_log(full_transcript="Test.", ai_prediction=ai)
    assert isinstance(log, dict)
    assert "full_transcript" in log
    assert "final_decision" in log


def test_empty_transcript_still_valid():
    """Even with empty transcript, structure is preserved."""
    ai = build_ai_prediction(category="OTHER", subcategory="minor_issue", risk_level="LOW")
    log = assemble_trainer_log(full_transcript="", ai_prediction=ai)
    assert log["full_transcript"] == ""
    assert isinstance(log["final_decision"], dict)


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
