"""Integration tests for the agent orchestrator (`agent.classify:classify`).

These tests patch every node import on `agent.classify` so the orchestrator's
glue logic — fault isolation, fallback-flag propagation, unroutable
escalation, retrieval provenance — can be exercised without touching the
LLM, chroma, or any I/O. The helper-only path (`_safe_fallback`, `_flatten`)
is also covered, but the bulk of the surface area is happy-path glue.

Patches target module-level imports on `agent.classify`, e.g.
`agent.classify.extract`, `agent.classify.classify_call`, etc. Python's
import system has already bound those names in the orchestrator's
namespace by the time the test runs, so patching the module attribute is
both correct and faster than patching the underlying definition.
"""
from __future__ import annotations

import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import classify as orchestrator  # noqa: E402
from agent.data import taxonomy as taxonomy_mod  # noqa: E402
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
    {"speaker": "agent", "text": "CBRE maintenance, this is Dana. What's going on?"},
    {"speaker": "caller", "text": "The breakroom sink on Floor 9 won't drain."},
]
SAMPLE_PHONE = "+15551234567"

REQUIRED_KEYS = {
    "category", "subcategory", "risk_level", "needs_human_review",
    "needs_clarification", "building_name", "address", "floor",
    "dispatched_vendor_id", "dispatched_emergency_services",
    "call_summary", "trainer_log",
}
TRAINER_LOG_REQUIRED_KEYS = {
    "full_transcript", "ai_prediction", "human_override", "final_decision",
}


# ─── Fixture builders ──────────────────────────────────────────────────


def _good_extraction(**overrides: Any) -> Extraction:
    base = dict(
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
    base.update(overrides)
    return Extraction(**base)


def _good_classification(
    *,
    is_fallback: bool = False,
    retrieved_record_ids: Optional[list] = None,
    category: str = "PLUMBING",
    subcategory: str = "drainage_backup",
    confidence_category: float = 0.9,
    confidence_subcategory: float = 0.9,
) -> Classification:
    return Classification(
        category=CategoryEnum(category),
        subcategory=SubcategoryEnum(subcategory),
        confidence_category=confidence_category,
        confidence_subcategory=confidence_subcategory,
        reasoning="stub",
        is_fallback=is_fallback,
        retrieved_record_ids=(
            ["TKT-2024-00001", "TKT-2024-00002"]
            if retrieved_record_ids is None
            else retrieved_record_ids
        ),
    )


def _good_location(**overrides: Any) -> ResolvedLocation:
    base = dict(
        building_name="Pacific Ridge Medical Plaza",
        address="3200 Pacific Coast Hwy",
        floor="Floor 9",
        city="Long Beach",
        building_type="medical",
        floor_check="ok",
        source_building="transcript",
        source_floor="transcript",
    )
    base.update(overrides)
    return ResolvedLocation(**base)


_RISK_SCORE = {"LOW": 0.0, "MEDIUM": 0.33, "HIGH": 0.67, "EMERGENCY": 1.0}


def _good_risk(band: str = "MEDIUM") -> RiskAssignment:
    return RiskAssignment(
        band=band,
        score=_RISK_SCORE[band],
        base_risk=band,
        reasons=("base_risk_for_subcategory",),
    )


def _validator(
    *, needs_human_review: bool = False, dispatched_emergency_services: bool = False
) -> ValidatorResult:
    reasons = (
        ["auto_resolve:no_flags_fired"]
        if not needs_human_review
        else ["high_band:always_pause"]
    )
    return ValidatorResult(
        needs_human_review=needs_human_review,
        dispatched_emergency_services=dispatched_emergency_services,
        reasons=reasons,
    )


def _vendor(vendor_id: Optional[str] = "v_002") -> VendorSelection:
    return VendorSelection(
        vendor_id=vendor_id,
        vendor_name="AquaFix Plumbing" if vendor_id else None,
        reason="selected" if vendor_id else "no_match",
    )


def _clarification(needs: bool = False) -> ClarificationDecision:
    return ClarificationDecision(
        needs_clarification=needs, question=None, reasons=(),
    )


# ─── Patch harness ─────────────────────────────────────────────────────


_DEFAULTS = {
    "extract": _good_extraction,
    "classify_call": _good_classification,
    "reconcile": _good_location,
    "assign_risk": _good_risk,
    "validate": _validator,
    "select_vendor": _vendor,
    "needs_clarification": _clarification,
    # Issue #27: retrieve is now invoked by the orchestrator (lifted out
    # of classify_call so its cost can be timed separately). Patch it to
    # an empty list so tests don't hit chroma; classify_call is also
    # mocked, so the empty records list never actually drives anything.
    "retrieve": lambda *_a, **_kw: [],
}


def _run(**overrides: Any) -> dict:
    """Run the orchestrator with each node patched.

    For each node name in `_DEFAULTS`, pass the override value (an instance,
    a callable, or an `Exception` instance/class) or omit to use the default
    happy-path fixture. Returns the orchestrator's output dict.
    """
    with ExitStack() as stack:
        for name, default_factory in _DEFAULTS.items():
            override = overrides.get(name)
            if override is None:
                value = default_factory()
                stack.enter_context(
                    patch.object(orchestrator, name, return_value=value)
                )
            elif isinstance(override, Exception) or (
                isinstance(override, type) and issubclass(override, BaseException)
            ):
                stack.enter_context(
                    patch.object(orchestrator, name, side_effect=override)
                )
            elif callable(override) and not isinstance(override, type):
                # Callable spies — let them observe kwargs.
                stack.enter_context(
                    patch.object(orchestrator, name, side_effect=override)
                )
            else:
                stack.enter_context(
                    patch.object(orchestrator, name, return_value=override)
                )
        return orchestrator.classify(SAMPLE_TURNS, SAMPLE_PHONE)


# ─── Helper-only coverage ──────────────────────────────────────────────


def test_flatten_preserves_speaker_tags():
    text = orchestrator._flatten(SAMPLE_TURNS)
    assert "[AGENT]" in text
    assert "[CALLER]" in text
    assert "breakroom sink" in text


def test_safe_fallback_schema():
    result = orchestrator._safe_fallback(SAMPLE_TURNS, "test error")
    assert REQUIRED_KEYS.issubset(result.keys())
    assert TRAINER_LOG_REQUIRED_KEYS.issubset(result["trainer_log"].keys())
    assert result["dispatched_emergency_services"] is False
    assert result["needs_human_review"] is True
    assert len(result["call_summary"]) >= 30
    assert "breakroom sink" in result["trainer_log"]["full_transcript"]


def test_safe_fallback_emits_fully_shaped_latency_with_no_timings():
    """Contract: ai_prediction.latency_ms is always a fully-shaped dict.

    Even when `_safe_fallback` is called with no timings at all (e.g. an
    upstream code path that never threaded the timings dict through),
    every STAGE_NAMES key must be present with 0.0 — downstream
    consumers can assume the shape is fixed.
    """
    result = orchestrator._safe_fallback(SAMPLE_TURNS, "test error")
    latency = result["trainer_log"]["ai_prediction"]["latency_ms"]
    assert set(latency.keys()) == set(orchestrator.STAGE_NAMES)
    assert all(v == 0.0 for v in latency.values())


def test_safe_fallback_uses_valid_taxonomy_pair():
    """H1: fallback must produce a (category, subcategory) the scorer can match."""
    result = orchestrator._safe_fallback(SAMPLE_TURNS, "x")
    assert taxonomy_mod.is_valid_category(result["category"]), (
        f"fallback category {result['category']!r} not in canonical taxonomy"
    )
    assert taxonomy_mod.is_valid_pair(result["category"], result["subcategory"]), (
        f"fallback ({result['category']}, {result['subcategory']}) not a valid pair"
    )


def test_safe_fallback_final_decision_mirrors_base():
    result = orchestrator._safe_fallback(SAMPLE_TURNS, "x")
    tl = result["trainer_log"]
    assert tl["human_override"] is None
    assert tl["final_decision"]["needs_human_review"] is True
    assert tl["final_decision"]["dispatched_emergency_services"] is False
    assert tl["final_decision"]["call_summary"] == result["call_summary"]
    assert tl["final_decision"]["building_name"] == result["building_name"]


# ─── Happy-path orchestrator glue ──────────────────────────────────────


def test_happy_path_returns_schema_valid_dict():
    result = _run()
    assert REQUIRED_KEYS.issubset(result.keys()), (
        f"missing keys: {REQUIRED_KEYS - result.keys()}"
    )
    assert TRAINER_LOG_REQUIRED_KEYS.issubset(result["trainer_log"].keys())


def test_happy_path_dispatches_vendor():
    result = _run()
    assert result["dispatched_vendor_id"] == "v_002"
    assert result["needs_human_review"] is False
    assert result["dispatched_emergency_services"] is False


def test_happy_path_carries_location_fields():
    result = _run()
    assert result["building_name"] == "Pacific Ridge Medical Plaza"
    assert result["floor"] == "Floor 9"
    assert result["address"] == "3200 Pacific Coast Hwy"


def test_happy_path_call_summary_is_nonempty():
    result = _run()
    assert isinstance(result["call_summary"], str)
    assert len(result["call_summary"]) >= 30


# ─── Fault isolation ───────────────────────────────────────────────────


def test_extract_failure_returns_safe_fallback():
    result = _run(extract=RuntimeError("extract boom"))
    assert result["needs_human_review"] is True
    assert result["dispatched_emergency_services"] is False
    assert result["dispatched_vendor_id"] is None
    assert "extract" in result["call_summary"].lower()


def test_classify_failure_returns_safe_fallback():
    result = _run(classify_call=RuntimeError("classify boom"))
    assert result["needs_human_review"] is True
    assert result["dispatched_emergency_services"] is False
    assert result["dispatched_vendor_id"] is None


def test_risk_failure_returns_safe_fallback():
    result = _run(assign_risk=RuntimeError("risk boom"))
    assert result["needs_human_review"] is True
    assert result["dispatched_emergency_services"] is False


def test_location_failure_soft_degrades():
    """Location is non-load-bearing: pipeline continues with raw extraction."""
    result = _run(reconcile=RuntimeError("location boom"))
    assert REQUIRED_KEYS.issubset(result.keys())
    assert result["building_name"] == "Pacific Ridge Medical Plaza"
    assert result["address"] is None


def test_vendor_failure_soft_degrades_and_escalates():
    """Vendor exception → no dispatch + B2 promotion to human review."""
    result = _run(select_vendor=RuntimeError("vendor boom"))
    assert result["dispatched_vendor_id"] is None
    assert result["needs_human_review"] is True


def test_clarification_failure_soft_degrades():
    result = _run(needs_clarification=RuntimeError("clarify boom"))
    assert result["needs_clarification"] is False
    assert REQUIRED_KEYS.issubset(result.keys())


# ─── B2: unroutable promotion ──────────────────────────────────────────


def test_unroutable_promotes_needs_human_review_on_routine_call():
    """B2: vendor returns None on a call the validator auto-resolved.

    Without the promotion, the orchestrator would emit
    `dispatched_vendor_id=None` AND `needs_human_review=False`, which
    fails the scorer's unroutable rule on 2/6 dev cases (EVAL-0162,
    EVAL-0415) and silently degrades 3 axes (vendor, HITL-F1, auto-res).
    """
    result = _run(
        validate=_validator(needs_human_review=False),
        select_vendor=_vendor(vendor_id=None),
    )
    assert result["dispatched_vendor_id"] is None
    assert result["needs_human_review"] is True, (
        "no qualified vendor must promote needs_human_review"
    )
    reasons = result["trainer_log"]["ai_prediction"]["validator_reasons"]
    assert any("vendor_escalation" in r for r in reasons), (
        f"expected vendor_escalation reason, got {reasons}"
    )


def test_unroutable_does_not_double_flag_when_validator_already_paused():
    """Validator already paused (HIGH risk) + vendor None: no duplicate reason."""
    result = _run(
        validate=_validator(needs_human_review=True),
        select_vendor=_vendor(vendor_id=None),
    )
    reasons = result["trainer_log"]["ai_prediction"]["validator_reasons"]
    vendor_reasons = [r for r in reasons if "vendor_escalation" in r]
    assert len(vendor_reasons) == 0, (
        "vendor_escalation should not fire when validator already paused"
    )
    assert result["needs_human_review"] is True


# ─── H2: is_fallback propagation ──────────────────────────────────────


def test_is_fallback_propagates_to_validator():
    """H2: fallback flag is read directly from Classification.is_fallback.

    Before the fix, the orchestrator substring-matched the reasoning
    string. A reword would silently break the validator-gate's
    fallback-pause rule.
    """
    captured: dict = {}

    def _spy_validate(**kwargs):
        captured.update(kwargs)
        return _validator(needs_human_review=True)

    _run(
        classify_call=_good_classification(is_fallback=True),
        validate=_spy_validate,
    )
    assert captured.get("fallback_invoked") is True


def test_is_fallback_false_propagates_to_validator():
    captured: dict = {}

    def _spy_validate(**kwargs):
        captured.update(kwargs)
        return _validator(needs_human_review=False)

    _run(
        classify_call=_good_classification(is_fallback=False),
        validate=_spy_validate,
    )
    assert captured.get("fallback_invoked") is False


# ─── H3: retrieval provenance ──────────────────────────────────────────


def test_trainer_log_carries_retrieved_record_ids():
    """H3: classifier-side record IDs flow into ai_prediction."""
    result = _run(
        classify_call=_good_classification(
            retrieved_record_ids=["TKT-A", "TKT-B", "TKT-C"],
        ),
    )
    ai_pred = result["trainer_log"]["ai_prediction"]
    assert ai_pred["retrieved_record_ids"] == ["TKT-A", "TKT-B", "TKT-C"]


def test_trainer_log_final_decision_is_complete_prediction_snapshot():
    """The final decision carries the full top-level prediction, not only model internals."""
    result = _run()
    final_decision = result["trainer_log"]["final_decision"]
    for key in REQUIRED_KEYS - {"trainer_log"}:
        assert final_decision[key] == result[key]


def test_trainer_log_ai_prediction_carries_training_context():
    result = _run()
    ai_pred = result["trainer_log"]["ai_prediction"]
    assert ai_pred["building_name"] == result["building_name"]
    assert ai_pred["address"] == result["address"]
    assert ai_pred["floor"] == result["floor"]
    assert ai_pred["call_summary"] == result["call_summary"]
    assert ai_pred["reasoning"] == "stub"
    assert ai_pred["classification_reasoning"] == "stub"
    assert ai_pred["hitl_reasons"] == ai_pred["validator_reasons"]
    assert "clarification_reasons" in ai_pred


def test_trainer_log_handles_empty_record_ids():
    """Empty retrieval (the no-records-found edge case) still produces a list."""
    result = _run(
        classify_call=_good_classification(retrieved_record_ids=[]),
    )
    assert result["trainer_log"]["ai_prediction"]["retrieved_record_ids"] == []


# ─── Issue #27: latency profile ────────────────────────────────────────


def test_trainer_log_carries_latency_breakdown():
    """Every per-stage label in STAGE_NAMES surfaces in ai_prediction.latency_ms.

    Pinning the stage set guards against silent drift: if a future
    refactor renames a stage or drops it from the orchestrator, the
    eval_runs latency report would suddenly become inconsistent. The
    test fails loudly instead.
    """
    result = _run()
    latency = result["trainer_log"]["ai_prediction"]["latency_ms"]
    assert set(latency.keys()) == set(orchestrator.STAGE_NAMES), (
        f"latency_ms keys diverged from STAGE_NAMES: "
        f"missing={set(orchestrator.STAGE_NAMES) - latency.keys()}, "
        f"extra={latency.keys() - set(orchestrator.STAGE_NAMES)}"
    )
    for name, ms in latency.items():
        assert isinstance(ms, (int, float)), f"{name} latency is not numeric"
        assert ms >= 0.0, f"{name} latency is negative"


def test_latency_total_at_least_sums_subtotals():
    """`total` is the wall-clock from first-stage start to summary-stage end —
    so it should be ≥ the sum of measured per-node subtotals (modulo
    tiny non-stage glue like `_flatten` and `profiles.lookup`)."""
    result = _run()
    latency = result["trainer_log"]["ai_prediction"]["latency_ms"]
    sub = sum(v for k, v in latency.items() if k != "total")
    # Allow a small tolerance: total can be marginally less than sub if
    # perf_counter resolution rounds unfavourably on a sub-millisecond
    # stage, but it should never be drastically less.
    assert latency["total"] + 0.5 >= sub, (
        f"total {latency['total']} unexpectedly less than subtotal sum {sub}"
    )


def test_latency_breakdown_on_extract_failure_fallback():
    """Fault-isolation path still emits the breakdown (with zeros for
    stages that never ran). Without this, a slow extract followed by a
    fallback would silently lose the cost in the latency report."""
    result = _run(extract=RuntimeError("extract boom"))
    latency = result["trainer_log"]["ai_prediction"]["latency_ms"]
    assert set(latency.keys()) == set(orchestrator.STAGE_NAMES)
    # extract attempted but raised; total stamped at fallback time.
    assert latency["retrieve"] == 0.0
    assert latency["classify"] == 0.0
    assert latency["total"] >= 0.0


# ─── No false-911 guarantee ────────────────────────────────────────────


def test_no_false_911_on_routine_dispatch():
    result = _run()
    assert result["dispatched_emergency_services"] is False


def test_911_only_when_validator_dispatches():
    """If validator says dispatch_911, the orchestrator surfaces it."""
    result = _run(
        validate=ValidatorResult(
            needs_human_review=True,
            dispatched_emergency_services=True,
            reasons=["emergency_dispatch:fire_smoke"],
        ),
    )
    assert result["dispatched_emergency_services"] is True


# ─── Inline runner ─────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect
    import traceback

    tests = [
        (n, f) for n, f in inspect.getmembers(sys.modules[__name__])
        if n.startswith("test_") and callable(f)
    ]
    failed = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
            failed += 1
    total = len(tests)
    print(f"\n{total - failed}/{total} passed, {failed} failed")
    sys.exit(1 if failed else 0)
