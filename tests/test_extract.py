"""Tests for `agent.nodes.extract`.

Layers:

1. **Pure-function tests** — turn flattening, profile block formatting,
   prompt assembly, Pydantic validation. No LLM calls, no API key.

2. **Stub-LLM integration** — verify `extract()` wires profile lookup,
   prompt construction, and `with_structured_output` correctly. Captures
   the rendered prompt for inspection.

3. **Real-LLM E2E** — the AC's three fixtures from
   `evaluation/eval_transcripts_dev.json`:
     - EVAL-0027: known caller, transcript+profile aligned (clear case)
     - EVAL-0006: anonymous caller, transcript explicit on location
     - EVAL-0004: known caller, transcript-overrides-profile (Floor 9 vs Floor 12)
   Gated on `OPENAI_API_KEY` so the suite still runs cleanly without a key.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data import profiles  # noqa: E402
from agent.data.profiles import Profile  # noqa: E402
from agent.nodes import extract as extract_mod  # noqa: E402
from agent.nodes.extract import (  # noqa: E402
    Extraction,
    FieldConfidence,
    _flatten_turns,
    _format_profile_block,
    build_prompt,
    extract,
)


# ─── Test doubles ──────────────────────────────────────────────────────


class _StubStructured:
    def __init__(self, parent):
        self.parent = parent

    def invoke(self, prompt):
        self.parent.last_prompt = prompt
        return self.parent.response


class _StubLLM:
    """Mimics the langchain ChatOpenAI surface that `extract()` calls.

    Records the prompt that was rendered so tests can assert on its
    contents (profile block present, transcript flattened, etc.).
    """

    def __init__(self, response: Extraction):
        self.response = response
        self.last_prompt: Optional[str] = None

    def with_structured_output(self, schema):
        return _StubStructured(self)


def _stub_extraction(**overrides) -> Extraction:
    base = {
        "problem_summary": "stub problem",
        "building_name": None,
        "floor": None,
        "suite": None,
        "urgency_cues": [],
        "caller_role": None,
        "language": "en",
        "confidence": {
            "problem_summary": 0.9,
            "building_name": 0.0,
            "floor": 0.0,
            "suite": 0.0,
            "caller_role": 0.0,
        },
    }
    base.update(overrides)
    return Extraction(**base)


# ─── _flatten_turns ────────────────────────────────────────────────────


def test_flatten_turns_basic():
    turns = [
        {"speaker": "agent", "text": "Hi."},
        {"speaker": "caller", "text": "Sink leaking."},
    ]
    out = _flatten_turns(turns)
    assert out == "[AGENT] Hi.\n[CALLER] Sink leaking."


def test_flatten_turns_handles_missing_fields():
    turns = [{"text": "no speaker"}, {"speaker": "caller"}]
    out = _flatten_turns(turns)
    assert "[?]" in out
    assert "[CALLER]" in out


def test_flatten_turns_uppercases_speaker_tag():
    turns = [{"speaker": "Agent", "text": "x"}]
    assert "[AGENT]" in _flatten_turns(turns)


# ─── _format_profile_block ─────────────────────────────────────────────


def test_format_profile_block_anonymous_returns_sentinel():
    out = _format_profile_block(None)
    assert "no caller profile" in out


def test_format_profile_block_includes_key_fields():
    p = profiles.lookup("+1-310-555-0142")  # Leo Green
    assert p is not None
    out = _format_profile_block(p)
    assert "Leo Green" in out
    assert "Westfield Commerce Center" in out
    assert "Floor 5" in out
    assert "office_manager" in out


def test_format_profile_block_is_valid_json():
    p = profiles.lookup("+1-310-555-0142")
    out = _format_profile_block(p)
    parsed = json.loads(out)
    assert parsed["primary_building_name"] == "Westfield Commerce Center"


# ─── build_prompt ──────────────────────────────────────────────────────


def test_build_prompt_includes_transcript_and_profile_block():
    turns = [{"speaker": "caller", "text": "leak in suite 902"}]
    p = profiles.lookup("+1-310-555-0142")
    prompt = build_prompt(turns, p)
    assert "[CALLER] leak in suite 902" in prompt
    assert "Westfield Commerce Center" in prompt
    assert "TRANSCRIPT ALWAYS WINS" in prompt


def test_build_prompt_anonymous_caller_renders_sentinel():
    prompt = build_prompt([{"speaker": "caller", "text": "x"}], None)
    assert "no caller profile" in prompt


# ─── Extraction Pydantic schema ────────────────────────────────────────


def test_extraction_validates_confidence_bounds():
    raised = False
    try:
        FieldConfidence(
            problem_summary=1.5, building_name=0.0,
            floor=0.0, suite=0.0, caller_role=0.0,
        )
    except Exception:
        raised = True
    assert raised, "FieldConfidence should reject values outside [0, 1]"


def test_extraction_strips_empty_urgency_cues():
    e = Extraction(
        problem_summary="x",
        urgency_cues=["flooding", "", "  ", "smoke"],
        confidence={
            "problem_summary": 0.9, "building_name": 0.0,
            "floor": 0.0, "suite": 0.0, "caller_role": 0.0,
        },
    )
    assert e.urgency_cues == ["flooding", "smoke"]


def test_extraction_minimal_valid_payload():
    # The minimum Pydantic accepts: problem_summary + confidence (others Optional).
    e = Extraction(
        problem_summary="x",
        confidence={
            "problem_summary": 0.9, "building_name": 0.0,
            "floor": 0.0, "suite": 0.0, "caller_role": 0.0,
        },
    )
    assert e.urgency_cues == []
    assert e.building_name is None


# ─── extract() with stub LLM ───────────────────────────────────────────


def test_extract_with_stub_llm_returns_response():
    canned = _stub_extraction(problem_summary="stubbed result")
    stub = _StubLLM(canned)
    result = extract([{"speaker": "caller", "text": "hi"}], None, llm=stub)
    assert result.problem_summary == "stubbed result"


def test_extract_passes_profile_block_into_prompt_for_known_caller():
    canned = _stub_extraction()
    stub = _StubLLM(canned)
    extract(
        [{"speaker": "caller", "text": "leak"}],
        "+1-310-555-0142",
        llm=stub,
    )
    assert stub.last_prompt is not None
    assert "Leo Green" in stub.last_prompt
    assert "Westfield Commerce Center" in stub.last_prompt


def test_extract_passes_anonymous_block_for_unknown_phone():
    canned = _stub_extraction()
    stub = _StubLLM(canned)
    extract([{"speaker": "caller", "text": "x"}], None, llm=stub)
    assert "no caller profile" in stub.last_prompt


def test_extract_passes_anonymous_block_for_inactive_caller():
    # Inactive profiles return None from lookup() — extraction should
    # treat them as anonymous, NOT silently use stale data.
    canned = _stub_extraction()
    stub = _StubLLM(canned)
    extract(
        [{"speaker": "caller", "text": "x"}],
        "+1-562-555-0198",  # Jasmine Clark, active=False
        llm=stub,
    )
    assert "no caller profile" in stub.last_prompt
    assert "Jasmine Clark" not in stub.last_prompt


# ─── Real LLM E2E (the AC's three fixtures) ────────────────────────────


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _load_dev_row(transcript_id: str) -> dict:
    rows = json.loads(open("evaluation/eval_transcripts_dev.json").read())
    return next(r for r in rows if r["transcript_id"] == transcript_id)


def test_e2e_known_caller_clear_location():
    """EVAL-0027: known caller, transcript and profile aligned on building."""
    if not _has_openai_key():
        print("    SKIP (no OPENAI_API_KEY)")
        return
    row = _load_dev_row("EVAL-0027")  # Redwood Corporate Plaza, Floor 4, Suite 408
    e = extract(row["turns"], row["caller_phone"])

    assert e.building_name and "Redwood" in e.building_name, e.building_name
    assert e.floor == "Floor 4", e.floor
    assert e.suite == "Suite 408", e.suite
    # Profile says Suite 405; transcript says Suite 408 — transcript wins.
    # Confidence on transcript-stated fields must be high.
    assert e.confidence.floor >= 0.7, e.confidence.floor
    assert e.confidence.suite >= 0.7, e.confidence.suite


def test_e2e_unknown_caller_explicit_location():
    """EVAL-0006: anonymous caller; transcript states full location explicitly."""
    if not _has_openai_key():
        print("    SKIP (no OPENAI_API_KEY)")
        return
    row = _load_dev_row("EVAL-0006")  # Torrance Gateway Center, Floor 2, Suite 205
    e = extract(row["turns"], row["caller_phone"])

    assert e.building_name and "Torrance" in e.building_name, e.building_name
    assert e.floor == "Floor 2", e.floor
    assert e.suite == "Suite 205", e.suite
    # All location fields are transcript-stated → high confidence.
    assert e.confidence.building_name >= 0.7
    assert e.confidence.floor >= 0.7


def test_e2e_known_caller_transcript_overrides_profile_floor():
    """EVAL-0004: profile says Floor 12; transcript says Floor 9.

    The AC's central assertion: transcript wins over profile. The floor
    in the extraction MUST be Floor 9, not Floor 12. Building name is
    silent in the transcript — should fall back to the profile (and
    confidence on building_name should be lower than confidence on floor).
    """
    if not _has_openai_key():
        print("    SKIP (no OPENAI_API_KEY)")
        return
    row = _load_dev_row("EVAL-0004")
    e = extract(row["turns"], row["caller_phone"])

    assert e.floor == "Floor 9", (
        f"transcript says Floor 9 (breakroom sink); profile says Floor 12. "
        f"transcript MUST win. got {e.floor!r}"
    )
    # Building isn't stated in the transcript — profile's "Westfield Commerce
    # Center" is the only source. Acceptable to either return it (with lower
    # confidence) or return None.
    if e.building_name is not None:
        assert "Westfield" in e.building_name, e.building_name
        # Profile-derived → confidence should be moderate, not high.
        assert e.confidence.building_name <= e.confidence.floor + 0.05, (
            f"profile-derived building confidence ({e.confidence.building_name}) "
            f"should be <= transcript-stated floor confidence ({e.confidence.floor})"
        )

    # Urgency cues: caller mentions "overflowing" and "smell" — at least
    # one of these recognizable cues should surface.
    cues_lower = [c.lower() for c in e.urgency_cues]
    has_relevant_cue = any("overflow" in c or "smell" in c for c in cues_lower)
    assert has_relevant_cue, f"expected an urgency cue from transcript; got {e.urgency_cues}"


# ─── Inline runner ─────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect, traceback
    tests = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__])
             if n.startswith("test_") and callable(f)]
    failed, skipped = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            if not _has_openai_key() and name.startswith("test_e2e_"):
                print(f"  SKIP  {name} (no OPENAI_API_KEY)")
                skipped += 1
            else:
                print(f"  FAIL  {name}: {e}")
                failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    total = len(tests)
    passed = total - failed - skipped
    print(f"\n{passed}/{total} passed, {skipped} skipped, {failed} failed")
    sys.exit(1 if failed else 0)
