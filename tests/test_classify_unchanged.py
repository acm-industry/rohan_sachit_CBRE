"""Regression guard for the grader path (`agent.classify:classify`).

The demo backend in `backend/` adds a sibling `classify_with_events()` to
`agent/classify.py`. This test makes sure the original `classify()`
function body never gets touched accidentally — its score (composite
91.94, 0 false-911s) is locked at tag `submission-v1` on main.

Two layers of guard:

  1. **Source pin** — the AST of `classify()` is extracted and hashed.
     If anyone edits the function body the hash diverges and this test
     fails loudly. To intentionally update the grader path you must
     bump `EXPECTED_CLASSIFY_HASH` and re-run the eval.

  2. **Mocked smoke test** — runs `classify()` with every node patched
     to a deterministic fixture (same pattern as `tests/test_pipeline.py`)
     and snapshots the full output dict. Catches any orchestration-level
     drift that wouldn't show up in the source hash (e.g. a change to a
     helper inside `agent/classify.py` like `_safe_fallback`).
"""
from __future__ import annotations

import ast
import hashlib
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import classify as orchestrator  # noqa: E402
from agent.nodes.classify import (  # noqa: E402
    CategoryEnum, Classification, SubcategoryEnum,
)
from agent.nodes.clarify import ClarificationDecision  # noqa: E402
from agent.nodes.extract import Extraction, FieldConfidence  # noqa: E402
from agent.nodes.location import ResolvedLocation  # noqa: E402
from agent.nodes.risk import RiskAssignment  # noqa: E402
from agent.nodes.validator import ValidatorResult  # noqa: E402
from agent.nodes.vendor_select import VendorSelection  # noqa: E402


# ─── Layer 1: source pin ───────────────────────────────────────────────

EXPECTED_CLASSIFY_HASH = (
    "5c053c326218b4a984ae38202762429418b3a35d9e200b9ea7bc7d3910c01033"
)


def _classify_source_hash() -> str:
    src = (Path(__file__).resolve().parents[1] / "agent" / "classify.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "classify":
            body = ast.get_source_segment(src, node)
            assert body is not None
            return hashlib.sha256(body.encode()).hexdigest()
    raise AssertionError("classify() function not found in agent/classify.py")


def test_classify_source_pinned():
    actual = _classify_source_hash()
    assert actual == EXPECTED_CLASSIFY_HASH, (
        f"classify() function body changed.\n"
        f"  expected: {EXPECTED_CLASSIFY_HASH}\n"
        f"  actual:   {actual}\n"
        "If this change is intentional, re-run evaluation and bump "
        "EXPECTED_CLASSIFY_HASH."
    )


# ─── Layer 2: mocked smoke snapshot ────────────────────────────────────

SAMPLE_TURNS = [
    {"speaker": "agent", "text": "CBRE maintenance, this is Dana."},
    {"speaker": "caller", "text": "The breakroom sink on Floor 9 won't drain."},
]
SAMPLE_PHONE = "+15551234567"


def _good_extraction() -> Extraction:
    return Extraction(
        problem_summary="sink won't drain",
        building_name="Pacific Ridge Medical Plaza",
        floor="Floor 9",
        suite=None,
        urgency_cues=[],
        caller_role="tenant",
        language="en",
        confidence=FieldConfidence(
            problem_summary=0.9, building_name=0.8, floor=0.9,
            suite=0.0, caller_role=0.6,
        ),
    )


def _good_classification() -> Classification:
    return Classification(
        category=CategoryEnum("PLUMBING"),
        subcategory=SubcategoryEnum("drainage_backup"),
        confidence_category=0.9,
        confidence_subcategory=0.9,
        reasoning="drainage blocked",
        is_fallback=False,
        retrieved_record_ids=["TKT-2024-00001", "TKT-2024-00002"],
    )


def _good_location() -> ResolvedLocation:
    return ResolvedLocation(
        building_name="Pacific Ridge Medical Plaza",
        address="3200 Pacific Coast Hwy",
        floor="Floor 9",
        city="Long Beach",
        building_type="medical",
        floor_check="ok",
        source_building="transcript",
        source_floor="transcript",
    )


def _good_risk() -> RiskAssignment:
    return RiskAssignment(
        band="MEDIUM",
        score=0.33,
        base_risk="MEDIUM",
        reasons=("base_risk_for_subcategory",),
    )


def _validator() -> ValidatorResult:
    return ValidatorResult(
        needs_human_review=False,
        dispatched_emergency_services=False,
        reasons=("auto_resolve:no_flags_fired",),
    )


def _vendor() -> VendorSelection:
    return VendorSelection(
        vendor_id="v_002",
        vendor_name="AquaFix Plumbing",
        reason="selected",
    )


def _clarification() -> ClarificationDecision:
    return ClarificationDecision(needs_clarification=False, question=None, reasons=())


def test_classify_mocked_snapshot():
    """Output dict from `classify()` matches a frozen snapshot.

    Catches orchestration-level regressions even if the source hash
    happens to collide (which it shouldn't, but defence in depth).
    """
    with ExitStack() as stack:
        stack.enter_context(patch.object(orchestrator, "extract",
                                         return_value=_good_extraction()))
        # Issue #27: retrieve is now called by the orchestrator (lifted out
        # of classify_call so it can be timed separately). Patch to an empty
        # list — classify_call is also mocked, so it never reads from records.
        stack.enter_context(patch.object(orchestrator, "retrieve",
                                         return_value=[]))
        stack.enter_context(patch.object(orchestrator, "classify_call",
                                         return_value=_good_classification()))
        stack.enter_context(patch.object(orchestrator, "reconcile",
                                         return_value=_good_location()))
        stack.enter_context(patch.object(orchestrator, "assign_risk",
                                         return_value=_good_risk()))
        stack.enter_context(patch.object(orchestrator, "validate",
                                         return_value=_validator()))
        stack.enter_context(patch.object(orchestrator, "select_vendor",
                                         return_value=_vendor()))
        stack.enter_context(patch.object(orchestrator, "needs_clarification",
                                         return_value=_clarification()))

        result = orchestrator.classify(SAMPLE_TURNS, SAMPLE_PHONE)

    # Top-level keys
    assert result["category"] == "PLUMBING"
    assert result["subcategory"] == "drainage_backup"
    assert result["risk_level"] == "MEDIUM"
    assert result["needs_human_review"] is False
    assert result["needs_clarification"] is False
    assert result["building_name"] == "Pacific Ridge Medical Plaza"
    assert result["address"] == "3200 Pacific Coast Hwy"
    assert result["floor"] == "Floor 9"
    assert result["dispatched_vendor_id"] == "v_002"
    assert result["dispatched_emergency_services"] is False
    assert isinstance(result["call_summary"], str) and result["call_summary"]

    # Trainer log shape unchanged
    tl = result["trainer_log"]
    assert set(tl.keys()) == {"full_transcript", "ai_prediction",
                              "human_override", "final_decision"}
    ai = tl["ai_prediction"]
    assert ai["category"] == "PLUMBING"
    assert ai["subcategory"] == "drainage_backup"
    assert ai["building_name"] == "Pacific Ridge Medical Plaza"
    assert ai["address"] == "3200 Pacific Coast Hwy"
    assert ai["floor"] == "Floor 9"
    assert ai["dispatched_vendor_id"] == "v_002"
    assert ai["call_summary"] == result["call_summary"]
    assert ai["reasoning"] == "drainage blocked"
    assert ai["classification_reasoning"] == "drainage blocked"
    assert ai["hitl_reasons"] == ["auto_resolve:no_flags_fired"]
    assert ai["clarification_reasons"] == []
    assert ai["retrieved_record_ids"] == ["TKT-2024-00001", "TKT-2024-00002"]
    assert tl["human_override"] is None
    assert tl["final_decision"]["building_name"] == "Pacific Ridge Medical Plaza"
    assert tl["final_decision"]["address"] == "3200 Pacific Coast Hwy"
    assert tl["final_decision"]["floor"] == "Floor 9"
    assert tl["final_decision"]["call_summary"] == result["call_summary"]


if __name__ == "__main__":
    test_classify_source_pinned()
    test_classify_mocked_snapshot()
    print("classify() grader path intact")
