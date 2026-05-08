"""Tests for `agent.nodes.clarify`.

Three layers:
1. Pure unit tests on each signal function.
2. Integration tests on `needs_clarification()` over synthetic turn lists.
3. Real-data tests on the dev set's three trigger patterns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.nodes.classify import CategoryEnum, Classification, SubcategoryEnum  # noqa: E402
from agent.nodes.clarify import (  # noqa: E402
    SPARSE_FIRST_TURN_WORDS,
    SPARSE_TOTAL_WORDS,
    ClarificationDecision,
    _floor_self_correction,
    _generic_opening,
    _sparse_caller,
    needs_clarification,
)
from agent.nodes.extract import Extraction  # noqa: E402


def _stub_extraction(**overrides) -> Extraction:
    base = {
        "problem_summary": "stub",
        "building_name": "Stub Building",
        "floor": "Floor 1",
        "suite": "Suite 100",
        "urgency_cues": [],
        "caller_role": "tenant",
        "language": "en",
        "confidence": {
            "problem_summary": 0.9, "building_name": 0.9,
            "floor": 0.9, "suite": 0.9, "caller_role": 0.7,
        },
    }
    base.update(overrides)
    return Extraction(**base)


def _stub_classification(**overrides) -> Classification:
    return Classification(
        category=overrides.get("category", CategoryEnum("PLUMBING")),
        subcategory=overrides.get("subcategory", SubcategoryEnum("pipe_leak")),
        confidence_category=overrides.get("confidence_category", 0.95),
        confidence_subcategory=overrides.get("confidence_subcategory", 0.95),
        reasoning="stub",
    )


# ─── _sparse_caller ─────────────────────────────────────────────────────


def test_sparse_caller_fires_on_short_first_and_short_total():
    turns = [
        {"speaker": "agent", "text": "Hi, what's going on?"},
        {"speaker": "caller", "text": "Lights are off."},  # 3 words
        {"speaker": "agent", "text": "Where?"},
        {"speaker": "caller", "text": "Floor 8 Suite 803."},  # 3 words; total 6
    ]
    assert _sparse_caller(turns) is True


def test_sparse_caller_does_not_fire_when_total_speech_is_high():
    turns = [
        {"speaker": "caller", "text": "Lights are off."},  # 3 words (short)
        {"speaker": "caller", "text": " ".join(["word"] * 30)},  # 30 words → total 33
    ]
    assert _sparse_caller(turns) is False


def test_sparse_caller_does_not_fire_on_detailed_first_turn():
    turns = [
        {"speaker": "caller", "text": "Half the suite at Oakwood Corporate Campus on Floor 6 has no power."}
    ]
    assert _sparse_caller(turns) is False


# ─── _generic_opening ───────────────────────────────────────────────────


def test_generic_opening_catches_issue_with_pattern():
    turns = [{"speaker": "caller", "text": "There's an issue with parking lighting."}]
    assert _generic_opening(turns) is True


def test_generic_opening_catches_being_weird_pattern():
    turns = [{"speaker": "caller", "text": "The main doors are being weird."}]
    assert _generic_opening(turns) is True


def test_generic_opening_does_not_fire_on_specific_complaint():
    turns = [{"speaker": "caller", "text": "Toilet clogged in restroom on Floor 4."}]
    assert _generic_opening(turns) is False


# ─── _floor_self_correction ────────────────────────────────────────────


def test_floor_self_correction_catches_two_distinct_floors():
    turns = [
        {"speaker": "caller", "text": "Soap dispenser on Floor 8 is empty."},
        {"speaker": "caller", "text": "Sorry, I meant Floor 2."},
    ]
    out = _floor_self_correction(turns)
    assert out == (8, 2)


def test_floor_self_correction_does_not_fire_on_single_floor():
    turns = [{"speaker": "caller", "text": "Stuck on Floor 5, breaker tripped."}]
    assert _floor_self_correction(turns) is None


def test_floor_self_correction_ignores_agent_turns():
    """Agent might mention different floors when probing — only caller's
    own floor mentions count."""
    turns = [
        {"speaker": "agent", "text": "Are you on Floor 3 or Floor 4?"},
        {"speaker": "caller", "text": "Floor 4."},
    ]
    assert _floor_self_correction(turns) is None


def test_floor_self_correction_dedupes_repeat_mentions():
    turns = [
        {"speaker": "caller", "text": "Floor 7 has the leak."},
        {"speaker": "caller", "text": "Yeah, Floor 7 again."},
    ]
    assert _floor_self_correction(turns) is None  # same floor mentioned twice


# ─── Integration: needs_clarification ──────────────────────────────────


def test_needs_clarification_fires_on_sparse_caller():
    turns = [
        {"speaker": "agent", "text": "How can I help?"},
        {"speaker": "caller", "text": "Lights are off."},
        {"speaker": "agent", "text": "Where?"},
        {"speaker": "caller", "text": "Floor 8 Suite 803."},
    ]
    d = needs_clarification(turns, _stub_extraction(), _stub_classification())
    assert d.needs_clarification is True
    assert any("sparse_caller" in r for r in d.reasons)
    assert d.question is not None


def test_needs_clarification_fires_on_generic_opening():
    turns = [
        {"speaker": "caller", "text": "There's an issue with parking lighting."},
        {"speaker": "caller", "text": "Lakeview Executive Suites, Floor 4, Suite 408."},
    ]
    d = needs_clarification(turns, _stub_extraction(), _stub_classification())
    assert d.needs_clarification is True
    assert any("generic_opening" in r for r in d.reasons)


def test_needs_clarification_fires_on_floor_self_correction():
    turns = [
        {"speaker": "caller", "text": "Soap dispenser on Floor 8 is empty."},
        {"speaker": "agent", "text": "Eastlake Medical Plaza only has 6 floors."},
        {"speaker": "caller", "text": "Sorry, I meant Floor 2. I'm flustered."},
    ]
    d = needs_clarification(turns, _stub_extraction(), _stub_classification())
    assert d.needs_clarification is True
    assert any("floor_self_correction" in r for r in d.reasons)
    assert "Floor 8" in d.question or "Floor 2" in d.question


def test_needs_clarification_does_not_fire_on_clear_routine_call():
    turns = [
        {"speaker": "caller", "text":
            "Half the suite at Oakwood Corporate Campus on Floor 6 in Suite 605 "
            "has no power, and the breaker keeps tripping."}
    ]
    d = needs_clarification(turns, _stub_extraction(), _stub_classification())
    assert d.needs_clarification is False
    assert d.question is None


def test_needs_clarification_returns_decision_dataclass():
    turns = [{"speaker": "caller", "text": "Working fine."}]
    d = needs_clarification(turns, _stub_extraction(), _stub_classification())
    assert isinstance(d, ClarificationDecision)
    assert isinstance(d.needs_clarification, bool)


# ─── Real-data fixture tests ───────────────────────────────────────────


def _load_dev(tid: str) -> dict:
    rows = json.loads(open("evaluation/eval_transcripts_dev.json").read())
    return next(r for r in rows if r["transcript_id"] == tid)


def test_real_clarification_case_eval_0274_pauses():
    """EVAL-0274 ('Lights are off.') is a true clarification case."""
    row = _load_dev("EVAL-0274")
    d = needs_clarification(row["turns"], _stub_extraction(),
                             _stub_classification(),
                             caller_phone=row["caller_phone"])
    assert d.needs_clarification is True


def test_real_location_conflict_case_eval_0668_pauses():
    """EVAL-0668: caller says Floor 8 then corrects to Floor 2."""
    row = _load_dev("EVAL-0668")
    d = needs_clarification(row["turns"], _stub_extraction(),
                             _stub_classification(),
                             caller_phone=row["caller_phone"])
    assert d.needs_clarification is True


def test_real_normal_case_eval_0019_does_not_pause():
    """EVAL-0019 is a routine power-outage call with a detailed first
    utterance — should auto-resolve, not pause."""
    row = _load_dev("EVAL-0019")
    d = needs_clarification(row["turns"], _stub_extraction(),
                             _stub_classification(),
                             caller_phone=row["caller_phone"])
    assert d.needs_clarification is False


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
