"""Tests for `agent.nodes.classify`.

Layers (same shape as test_extract.py):

1. Pure-function tests on prompt assembly + Pydantic schema. No LLM, no API.
2. Stub-LLM integration: profile-/retrieval-wired prompt, retry path on
   schema validation failure, fallback path.
3. Real-LLM E2E on 5 hand-picked dev fixtures (the AC's requirement),
   each chosen to exercise a different classifier failure mode the
   prompt-engineering work targeted.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import pytest

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import logging  # noqa: E402
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError  # noqa: E402

from agent.data import taxonomy  # noqa: E402
from agent.nodes.classify import (  # noqa: E402
    CategoryEnum,
    Classification,
    SubcategoryEnum,
    _build_query,
    _category_only_fallback,
    _format_retrieval_block,
    _format_taxonomy_block,
    build_prompt,
    classify,
)
from agent.nodes.extract import Extraction, FieldConfidence  # noqa: E402
from agent.rag.retriever import RetrievedRecord  # noqa: E402


# ─── Fixtures ──────────────────────────────────────────────────────────


def _stub_extraction(**overrides) -> Extraction:
    base = {
        "problem_summary": "stub problem",
        "building_name": "Stub Building",
        "floor": "Floor 1",
        "suite": "Suite 100",
        "urgency_cues": [],
        "caller_role": "tenant",
        "language": "en",
        "confidence": {
            "problem_summary": 0.9,
            "building_name": 0.8,
            "floor": 0.9,
            "suite": 0.7,
            "caller_role": 0.6,
        },
    }
    base.update(overrides)
    return Extraction(**base)


def _stub_record(**md) -> RetrievedRecord:
    md.setdefault("ticket_id", "TKT-stub")
    md.setdefault("intake_category", "PLUMBING")
    md.setdefault("intake_subcategory", "pipe_leak")
    md.setdefault("final_category", "PLUMBING")
    md.setdefault("final_subcategory", "pipe_leak")
    md.setdefault("was_audit_flagged", False)
    md.setdefault("audit_reclassified", False)
    return RetrievedRecord(text="(stub doc)", metadata=md, distance=0.5)


# ─── Pure helpers ──────────────────────────────────────────────────────


def test_format_taxonomy_block_includes_all_10_categories():
    block = _format_taxonomy_block()
    for cat in taxonomy.categories():
        assert cat in block, f"missing {cat} in rendered taxonomy"
    # Every subcategory should also appear
    for sub in taxonomy.all_subcategories():
        assert sub in block


def test_format_retrieval_block_marks_qa_reclassified_records():
    flagged = _stub_record(
        ticket_id="TKT-flagged",
        intake_subcategory="roof_leak",
        final_subcategory="structural",
        intake_category="PLUMBING",
        final_category="LIFE_SAFETY",
        was_audit_flagged=True,
        audit_reclassified=True,
    )
    plain = _stub_record(ticket_id="TKT-plain")
    block = _format_retrieval_block([flagged, plain])
    assert "QA-RECLASSIFIED" in block
    assert "intake=roof_leak" in block
    # The corrected (final) labels appear in the rendered text
    assert "structural" in block
    assert "LIFE_SAFETY" in block


def test_format_retrieval_block_handles_empty():
    assert "no retrieved tickets" in _format_retrieval_block([])


def test_build_query_combines_problem_and_urgency():
    e = _stub_extraction(
        problem_summary="ceiling tile fell",
        urgency_cues=["water damaged", "nobody hurt"],
    )
    q = _build_query(e)
    assert "ceiling tile fell" in q
    assert "water damaged" in q
    assert "nobody hurt" in q


def test_build_query_handles_no_urgency_cues():
    e = _stub_extraction(problem_summary="thing broke", urgency_cues=[])
    q = _build_query(e)
    assert q == "thing broke"


def test_build_prompt_renders_full_context():
    e = _stub_extraction(problem_summary="kitchen sink leak")
    rec = _stub_record(intake_subcategory="pipe_leak")
    prompt = build_prompt(e, [rec])
    assert "kitchen sink leak" in prompt
    assert "pipe_leak" in prompt
    # Anti-over-escalation guidance must always appear
    assert "OVER-ESCALATION GUARD" in prompt
    # Taxonomy must be present
    assert "PLUMBING" in prompt and "ELECTRICAL" in prompt


# ─── Pydantic schema ───────────────────────────────────────────────────


def test_classification_valid_pair_passes():
    c = Classification(
        category=CategoryEnum("PLUMBING"),
        subcategory=SubcategoryEnum("pipe_leak"),
        confidence_category=0.9,
        confidence_subcategory=0.85,
        reasoning="test",
    )
    assert c.category_str == "PLUMBING"
    assert c.subcategory_str == "pipe_leak"


def test_classification_rejects_invalid_pair():
    raised = False
    try:
        Classification(
            category=CategoryEnum("PLUMBING"),
            subcategory=SubcategoryEnum("lighting"),  # ELECTRICAL, not PLUMBING
            confidence_category=0.9,
            confidence_subcategory=0.9,
            reasoning="test",
        )
    except ValidationError:
        raised = True
    assert raised, "model_validator should reject mismatched (category, subcategory)"


def test_classification_rejects_off_taxonomy_strings():
    # The Enum constraint catches these before model_validator runs.
    raised = False
    try:
        Classification(
            category="NOT_A_REAL_CATEGORY",
            subcategory="pipe_leak",
            confidence_category=0.9,
            confidence_subcategory=0.9,
            reasoning="test",
        )
    except (ValidationError, ValueError):
        raised = True
    assert raised


def test_classification_rejects_confidence_out_of_range():
    raised = False
    try:
        Classification(
            category=CategoryEnum("PLUMBING"),
            subcategory=SubcategoryEnum("pipe_leak"),
            confidence_category=1.5,  # > 1.0
            confidence_subcategory=0.9,
            reasoning="test",
        )
    except ValidationError:
        raised = True
    assert raised


def test_category_only_fallback_uses_retrieval_consensus():
    records = [
        _stub_record(intake_category="HVAC", final_category="HVAC"),
        _stub_record(intake_category="HVAC", final_category="HVAC"),
        _stub_record(intake_category="HVAC", final_category="HVAC"),
        _stub_record(intake_category="PLUMBING", final_category="PLUMBING"),
    ]
    c = _category_only_fallback(records)
    assert c.category_str == "HVAC"
    assert c.confidence_category < 0.5  # forced low so #16 escalates
    assert "fallback" in c.reasoning.lower()


def test_category_only_fallback_handles_empty_retrieval():
    c = _category_only_fallback([])
    # Per the comment: PEST_SPECIALTY is the catch-all bucket.
    assert c.category_str == "PEST_SPECIALTY"


# ─── Stub-LLM integration ──────────────────────────────────────────────


class _StubStructured:
    def __init__(self, parent):
        self.parent = parent

    def invoke(self, prompt):
        self.parent.last_prompt = prompt
        if self.parent.errors:
            err = self.parent.errors.pop(0)
            raise err
        return self.parent.responses.pop(0)


class _StubLLM:
    def __init__(self, responses=None, errors=None):
        self.responses = list(responses or [])
        self.errors = list(errors or [])
        self.last_prompt: Optional[str] = None

    def with_structured_output(self, schema):
        return _StubStructured(self)


def _good_classification() -> Classification:
    return Classification(
        category=CategoryEnum("PLUMBING"),
        subcategory=SubcategoryEnum("pipe_leak"),
        confidence_category=0.9,
        confidence_subcategory=0.85,
        reasoning="stub",
    )


def test_classify_stub_returns_response():
    e = _stub_extraction()
    stub = _StubLLM(responses=[_good_classification()])
    c = classify(e, llm=stub, records=[])
    assert c.category_str == "PLUMBING"
    assert c.subcategory_str == "pipe_leak"


def test_classify_stub_renders_taxonomy_in_prompt():
    e = _stub_extraction()
    stub = _StubLLM(responses=[_good_classification()])
    classify(e, llm=stub, records=[])
    assert stub.last_prompt is not None
    assert "PLUMBING" in stub.last_prompt
    assert "OVER-ESCALATION GUARD" in stub.last_prompt


def test_classify_retries_on_first_validation_error():
    # First call raises ValidationError; second call returns a valid Classification.
    err = ValidationError.from_exception_data(
        title="Classification",
        line_errors=[],
    )
    stub = _StubLLM(
        responses=[_good_classification()],
        errors=[err],
    )
    e = _stub_extraction()
    c = classify(e, llm=stub, records=[])
    assert c.category_str == "PLUMBING"
    # Both calls happened
    assert "RETRY GUIDANCE" in stub.last_prompt


def test_classify_falls_back_when_both_attempts_fail():
    err1 = ValidationError.from_exception_data(title="Classification", line_errors=[])
    err2 = ValidationError.from_exception_data(title="Classification", line_errors=[])
    stub = _StubLLM(errors=[err1, err2])
    e = _stub_extraction()
    records = [
        _stub_record(intake_category="HVAC", final_category="HVAC"),
        _stub_record(intake_category="HVAC", final_category="HVAC"),
    ]
    c = classify(e, llm=stub, records=records)
    # Fallback returned a low-confidence Classification (based on retrieval consensus).
    assert c.category_str == "HVAC"
    assert c.confidence_category < 0.5
    assert "fallback" in c.reasoning.lower()


# ─── Real-LLM E2E (the AC's 5 fixtures) ────────────────────────────────


def _has_openai_key() -> bool:
    return (
        os.environ.get("OPENAI_LIVE_TEST") == "1"
        and bool(os.environ.get("OPENAI_API_KEY"))
    )


def _openai_skip_reason() -> str:
    if os.environ.get("OPENAI_LIVE_TEST") != "1":
        return "live OpenAI tests skipped — set OPENAI_LIVE_TEST=1 to run"
    return "live OpenAI tests skipped — missing OPENAI_API_KEY"


def _load_dev(transcript_id: str) -> dict:
    rows = json.loads(open("evaluation/eval_transcripts_dev.json").read())
    return next(r for r in rows if r["transcript_id"] == transcript_id)


def _classify_dev(transcript_id: str) -> tuple[Classification, dict]:
    """Run extract → classify on a dev fixture."""
    from agent.nodes.extract import extract
    row = _load_dev(transcript_id)
    e = extract(row["turns"], row["caller_phone"])
    return classify(e), row


def test_e2e_normal_drainage_backup():
    """EVAL-0004: breakroom sink not draining → PLUMBING/drainage_backup."""
    if not _has_openai_key():
        pytest.skip(_openai_skip_reason())
    c, _ = _classify_dev("EVAL-0004")
    assert c.category_str == "PLUMBING"
    assert c.subcategory_str == "drainage_backup"
    assert c.confidence_subcategory >= 0.7


def test_e2e_over_escalation_trap_burnt_toast():
    """EVAL-0788: smoke alarm beeped but burnt toast → JANITORIAL/waste_odor.

    The deliberately-seeded over-escalation trap. The classifier must not
    take the alarming opening at face value and route to LIFE_SAFETY/fire_smoke
    when the caller explicitly clarifies it's just burnt food.
    """
    if not _has_openai_key():
        pytest.skip(_openai_skip_reason())
    c, _ = _classify_dev("EVAL-0788")
    assert c.category_str == "JANITORIAL", f"expected JANITORIAL, got {c.category_str}"
    assert c.subcategory_str == "waste_odor", f"expected waste_odor, got {c.subcategory_str}"


def test_e2e_over_escalation_trap_drop_ceiling_water_leak():
    """EVAL-0319: drop-ceiling tile fell from water leak → PLUMBING/roof_leak.

    Trap: caller says 'ceiling fell' but explicitly says 'nothing structural,
    nobody hurt'. Root cause is water; tile is a symptom.
    """
    if not _has_openai_key():
        pytest.skip(_openai_skip_reason())
    c, _ = _classify_dev("EVAL-0319")
    assert c.category_str == "PLUMBING"
    assert c.subcategory_str == "roof_leak"


def test_e2e_genuine_life_safety_active_threat():
    """EVAL-0727: real active-threat call → SECURITY/active_threat.

    The other side of the over-escalation guard: when there IS a genuine
    safety incident, the classifier still escalates correctly.
    """
    if not _has_openai_key():
        pytest.skip(_openai_skip_reason())
    c, _ = _classify_dev("EVAL-0727")
    assert c.category_str == "SECURITY"
    assert c.subcategory_str == "active_threat"


def test_e2e_taxonomy_disambiguation_sprinkler_irrigation():
    """EVAL-0161: sprinkler head stuck on → PEST_SPECIALTY/landscaping.

    Tests the taxonomy disambiguation rule that sprinkler/irrigation
    belongs to landscaping, not plumbing.
    """
    if not _has_openai_key():
        pytest.skip(_openai_skip_reason())
    c, _ = _classify_dev("EVAL-0161")
    assert c.category_str == "PEST_SPECIALTY"
    assert c.subcategory_str == "landscaping"


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
        except pytest.skip.Exception as e:
            print(f"  SKIP  {name}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
            failed += 1
    total = len(tests)
    print(f"\n{total - failed - skipped}/{total} passed, {skipped} skipped, {failed} failed")
    sys.exit(1 if failed else 0)
