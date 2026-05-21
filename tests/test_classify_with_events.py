"""Event-emission tests for `agent.classify.classify_with_events`.

We patch every node module-level import on `agent.classify` (same pattern
as `tests/test_pipeline.py`) so the pipeline runs without LLM or RAG I/O.
We then assert:

  1. Every stage emits both a `started` and a `complete` (or `gate_open`/
     `gate_resumed`/`failed`) event, in the canonical pipeline order.
  2. When `pause_at_validator=True` and the validator decides
     `needs_human_review`, the gate opens, an override decision is
     applied, and downstream state (vendor, trainer_log) reflects it.
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import classify as orchestrator  # noqa: E402
from agent.nodes.classify import (  # noqa: E402
    CategoryEnum,
    Classification,
    SubcategoryEnum,
)
from agent.nodes.clarify import ClarificationDecision  # noqa: E402
from agent.nodes.extract import Extraction, FieldConfidence  # noqa: E402
from agent.nodes.location import ResolvedLocation  # noqa: E402
from agent.nodes.risk import RiskAssignment  # noqa: E402
from agent.nodes.validator import ValidatorResult  # noqa: E402
from agent.nodes.vendor_select import VendorSelection  # noqa: E402


SAMPLE_TURNS = [
    {"speaker": "agent", "text": "CBRE maintenance, this is Dana."},
    {"speaker": "caller", "text": "Sink won't drain."},
]


def _good_extraction() -> Extraction:
    return Extraction(
        problem_summary="sink won't drain",
        building_name="Pacific Ridge",
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
        reasoning="drain blocked",
        is_fallback=False,
        retrieved_record_ids=["TKT-2024-00001"],
    )


def _good_location() -> ResolvedLocation:
    return ResolvedLocation(
        building_name="Pacific Ridge",
        address="3200 Pacific Coast Hwy",
        floor="Floor 9",
        city="Long Beach",
        building_type="medical",
        floor_check="ok",
        source_building="transcript",
        source_floor="transcript",
    )


def _good_risk(band: str = "MEDIUM") -> RiskAssignment:
    scores = {"LOW": 0.0, "MEDIUM": 0.33, "HIGH": 0.67, "EMERGENCY": 1.0}
    return RiskAssignment(
        band=band, score=scores[band], base_risk=band,
        reasons=("base_risk_for_subcategory",),
    )


def _validator(*, needs_human_review: bool = False) -> ValidatorResult:
    return ValidatorResult(
        needs_human_review=needs_human_review,
        dispatched_emergency_services=False,
        reasons=(
            ["high_band:always_pause"] if needs_human_review
            else ["auto_resolve:no_flags_fired"]
        ),
    )


def _vendor(vendor_id: Optional[str] = "v_002") -> VendorSelection:
    return VendorSelection(
        vendor_id=vendor_id,
        vendor_name="AquaFix Plumbing" if vendor_id else None,
        reason="selected" if vendor_id else "no_match",
    )


def _clarification() -> ClarificationDecision:
    return ClarificationDecision(needs_clarification=False, question=None, reasons=())


def _run_with_events(
    *,
    pause: bool = False,
    review_decision: Optional[Dict[str, Any]] = None,
    validator: Optional[ValidatorResult] = None,
    vendor: Optional[VendorSelection] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    async def emitter(event: Dict[str, Any]) -> None:
        events.append(event)

    async def waiter() -> Dict[str, Any]:
        return review_decision or {"decision": "approve"}

    patches = {
        "extract": _good_extraction(),
        "classify_call": _good_classification(),
        "reconcile": _good_location(),
        "assign_risk": _good_risk(),
        "validate": validator or _validator(),
        "select_vendor": vendor or _vendor(),
        "needs_clarification": _clarification(),
    }

    async def _go():
        with ExitStack() as stack:
            for name, val in patches.items():
                stack.enter_context(
                    patch.object(orchestrator, name, return_value=val)
                )
            return await orchestrator.classify_with_events(
                SAMPLE_TURNS,
                "+15551234567",
                emitter=emitter,
                pause_at_validator=pause,
                wait_for_resume=waiter,
            )

    result = asyncio.run(_go())
    return events, result


def _stages_in_order(events: List[Dict[str, Any]]) -> List[tuple[str, str]]:
    return [(e["stage"], e["status"]) for e in events]


def test_happy_path_emits_all_stages():
    events, result = _run_with_events()
    pairs = _stages_in_order(events)

    # Expect started+complete for every stage in canonical order.
    expected_order = [
        "extract", "retrieve", "classify", "location",
        "risk", "validate", "vendor", "clarify", "summary", "trainer_log",
    ]
    started = [s for s, st in pairs if st == "started"]
    completed = [s for s, st in pairs if st == "complete"]
    assert started == expected_order, f"started order wrong: {started}"
    for stage in expected_order:
        assert stage in completed, f"missing complete for {stage}"

    # Payload sanity
    by_stage = {(e["stage"], e["status"]): e for e in events}
    assert by_stage[("classify", "complete")]["payload"]["category"] == "PLUMBING"
    assert by_stage[("risk", "complete")]["payload"]["risk_level"] == "MEDIUM"
    assert by_stage[("vendor", "complete")]["payload"]["vendor_id"] == "v_002"

    # Final prediction
    assert result["category"] == "PLUMBING"
    assert result["dispatched_vendor_id"] == "v_002"
    assert result["needs_human_review"] is False


def test_gate_opens_and_resumes_on_approve():
    events, result = _run_with_events(
        pause=True,
        validator=_validator(needs_human_review=True),
        review_decision={"decision": "approve"},
    )
    pairs = _stages_in_order(events)
    assert ("validate", "gate_open") in pairs
    assert ("validate", "gate_resumed") in pairs
    # Approve preserves AI prediction
    assert result["needs_human_review"] is True
    assert result["dispatched_vendor_id"] == "v_002"


def test_override_replaces_subcategory_and_risk():
    override = {
        "subcategory": "minor_issue",
        "risk_level": "LOW",
    }
    events, result = _run_with_events(
        pause=True,
        validator=_validator(needs_human_review=True),
        review_decision={"decision": "override", "override": override},
    )
    # Vendor selection runs after the override, so it sees the new subcategory.
    # Downstream final state reflects the overridden values.
    assert result["subcategory"] == "minor_issue"
    assert result["risk_level"] == "LOW"
    # trainer_log human_override is recorded
    assert result["trainer_log"]["human_override"] == override


def test_pause_disabled_skips_gate():
    events, _ = _run_with_events(
        pause=False,
        validator=_validator(needs_human_review=True),
    )
    pairs = _stages_in_order(events)
    assert ("validate", "gate_open") not in pairs
    assert ("validate", "gate_resumed") not in pairs


if __name__ == "__main__":
    test_happy_path_emits_all_stages()
    test_gate_opens_and_resumes_on_approve()
    test_override_replaces_subcategory_and_risk()
    test_pause_disabled_skips_gate()
    print("all event tests passed")
