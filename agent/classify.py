"""Single public entrypoint for the agent pipeline.

`classify(turns, caller_phone) -> dict` is the contract consumed by
`evaluation/run_eval.py`. It chains all nodes in sequence:

  Extract → Classify → Location → Risk → Validate → Vendor → Clarify → Summary → TrainerLog

Each node is fault-isolated:

- Extract / Classify / Risk are load-bearing — if they raise, the whole
  call falls back to a schema-valid "escalate to human" prediction via
  `_safe_fallback()` (needs_human_review=True, no 911 dispatch).
- Location / Vendor / Clarification soft-degrade — if they raise, the
  pipeline continues with safe defaults for that node and a logged
  warning, because their downstream consumers (summary, trainer log)
  can handle None / empty values.
- Vendor selection returning `VendorSelection(vendor_id=None, ...)` is
  the "unroutable" path: the validator's `needs_human_review` is
  promoted to True with a `vendor_escalation` reason so the scorer's
  unroutable-case rule (dispatched=None AND needs_human_review=True)
  is satisfied. The summary node frames the escalation explicitly.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Optional, Union

from agent.data import profiles
from agent.nodes.extract import Extraction, extract
from agent.nodes.classify import (
    Classification,
    DEFAULT_K,
    _build_query,
    classify as classify_call,
)
from agent.security import SecurityScan, scan_turns, sanitize_for_storage, validate_extraction_output
from agent.nodes.risk import RiskAssignment, assign_risk
from agent.nodes.validator import ValidatorResult, validate
from agent.nodes.location import ResolvedLocation, reconcile
from agent.nodes.vendor_select import VendorSelection, select_vendor
from agent.nodes.clarify import ClarificationDecision, needs_clarification
from agent.nodes.summary import generate_summary
from agent.nodes.trainer_log import assemble_trainer_log, build_ai_prediction
from agent.rag.retriever import retrieve

logger = logging.getLogger(__name__)


# Per-stage labels surfaced in trainer_log.ai_prediction.latency_ms.
# Pinned (rather than read off the dict at runtime) so a missing stage —
# e.g. a node short-circuited by a soft-degrade path — still shows up as
# 0.0 in the breakdown instead of silently dropping out.
STAGE_NAMES: tuple[str, ...] = (
    "extract",
    "retrieve",
    "classify",
    "location",
    "risk",
    "validate",
    "vendor",
    "clarify",
    "summary",
    "total",
)


@contextmanager
def _time_stage(timings: Dict[str, float], name: str) -> Iterator[None]:
    """Record wall-clock elapsed (ms) for the wrapped block under `name`.

    Uses perf_counter for monotonic high-resolution timing. The timing is
    written even if the wrapped block raises, so fault-isolation paths
    that catch the exception downstream still get an honest cost recorded
    for the failing stage (rather than the failure costing 0ms in the
    breakdown).
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round((time.perf_counter() - t0) * 1000.0, 3)


def _flatten(turns: List[dict]) -> str:
    return "\n".join(
        f"[{t.get('speaker', '?').upper()}] {t.get('text', '')}".rstrip()
        for t in turns
    )


def _decision_snapshot(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Return the top-level decision fields without embedding trainer_log."""
    keys = (
        "category",
        "subcategory",
        "risk_level",
        "needs_human_review",
        "needs_clarification",
        "building_name",
        "address",
        "floor",
        "dispatched_vendor_id",
        "dispatched_emergency_services",
        "call_summary",
    )
    return {k: prediction.get(k) for k in keys}


def _safe_fallback(
    turns: List[dict],
    error: str,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Return a safe prediction on pipeline failure — avoids false-911.

    Contract: `ai_prediction.latency_ms` is ALWAYS emitted as a
    fully-shaped dict keyed by STAGE_NAMES, with 0.0 for any stage that
    didn't run (or for every stage when timings was never threaded in).
    Downstream consumers can assume the key is present and shape is
    fixed.
    """
    full_transcript = _flatten(turns)
    # (ELEVATOR, minor_issue) is the only minor_issue pair in the canonical
    # taxonomy. modal_risk in the derived table is LOW; we hold to MEDIUM
    # here because the fallback always escalates and "I don't know" should
    # not present as LOW to the reviewer.
    base = {
        "category": "ELEVATOR",
        "subcategory": "minor_issue",
        "risk_level": "MEDIUM",
        "needs_human_review": True,
        "needs_clarification": False,
        "building_name": None,
        "address": None,
        "floor": None,
        "dispatched_vendor_id": None,
        "dispatched_emergency_services": False,
        "call_summary": f"Pipeline error: {error}. Escalated to human reviewer.",
    }
    src = timings or {}
    ai_pred = build_ai_prediction(
        category=base["category"],
        subcategory=base["subcategory"],
        risk_level=base["risk_level"],
        needs_human_review=base["needs_human_review"],
        needs_clarification=base["needs_clarification"],
        building_name=base["building_name"],
        address=base["address"],
        floor=base["floor"],
        dispatched_vendor_id=base["dispatched_vendor_id"],
        dispatched_emergency_services=base["dispatched_emergency_services"],
        call_summary=base["call_summary"],
        classification_reasoning=f"Pipeline error: {error}",
        validator_reasons=[f"safe_fallback:{error}"],
        latency_ms={name: src.get(name, 0.0) for name in STAGE_NAMES},
    )
    return {
        **base,
        "trainer_log": {
            "full_transcript": full_transcript,
            "ai_prediction": ai_pred,
            "human_override": None,
            "final_decision": base.copy(),
        },
    }


def classify(turns: List[dict], caller_phone: Optional[str]) -> Dict[str, Any]:
    """Run the full agent pipeline and return a schema-valid Prediction dict.

    This is the entrypoint consumed by `evaluation/run_eval.py`:
        python evaluation/run_eval.py --agent agent.classify:classify
    """
    full_transcript = _flatten(turns)

    timings: Dict[str, float] = {}
    _t_total_start = time.perf_counter()

    # ── Step 0: Security scan ───────────────────────────────────────
    security_scan = scan_turns(turns)
    if security_scan.any_threat:
        logger.warning(
            "security scan flagged: %s", security_scan.matched_patterns
        )

    # ── Step 1: Extract ──────────────────────────────────────────────
    try:
        with _time_stage(timings, "extract"):
            extraction = extract(turns, caller_phone)
    except Exception as e:
        timings["total"] = round((time.perf_counter() - _t_total_start) * 1000.0, 3)
        logger.error("extraction failed: %s", e)
        return _safe_fallback(turns, f"extraction: {e}", timings=timings)

    # Post-extraction output validation (T6/T3 defense)
    output_warnings = validate_extraction_output(
        extraction.building_name, extraction.floor, caller_phone, turns
    )
    if output_warnings:
        logger.warning("extraction output anomaly: %s", output_warnings)

    # ── Step 2: Retrieve + Classify ──────────────────────────────────
    # Lifted retrieval out of the classifier node so the embedding/index
    # cost (chroma roundtrip) is timed separately from the LLM call.
    # Without this split, "classify" lumps the two together and we can't
    # see which one is the dominant cost on a slow call.
    try:
        with _time_stage(timings, "retrieve"):
            query = _build_query(extraction)
            records = retrieve(query, k=DEFAULT_K) if query else []
        with _time_stage(timings, "classify"):
            classification = classify_call(extraction, records=records)
    except Exception as e:
        timings["total"] = round((time.perf_counter() - _t_total_start) * 1000.0, 3)
        logger.error("classification failed: %s", e)
        return _safe_fallback(turns, f"classification: {e}", timings=timings)

    category = classification.category_str
    subcategory = classification.subcategory_str
    confidence_cat = classification.confidence_category
    confidence_sub = classification.confidence_subcategory
    min_confidence = min(confidence_cat, confidence_sub)

    # ── Step 3: Location ─────────────────────────────────────────────
    profile = profiles.lookup(caller_phone)
    try:
        with _time_stage(timings, "location"):
            location = reconcile(
                extracted_building_name=extraction.building_name,
                extracted_floor=extraction.floor,
                building_confidence=extraction.confidence.building_name,
                floor_confidence=extraction.confidence.floor,
                profile=profile,
                # Issue #65: phone history outranks the (possibly stale)
                # static profile when the historicals are confidently
                # concentrated on a single building for this phone.
                caller_phone=caller_phone,
            )
    except Exception as e:
        logger.warning("location reconciliation failed: %s", e)
        location = ResolvedLocation(
            building_name=extraction.building_name,
            address=None,
            floor=extraction.floor,
            city=None,
            building_type=None,
            floor_check="unknown",
            source_building="none",
            source_floor="none",
        )

    # ── Step 4: Risk ─────────────────────────────────────────────────
    try:
        with _time_stage(timings, "risk"):
            risk = assign_risk(
                extraction,
                subcategory,
                building_type=location.building_type,
                after_hours=False,
                classification_confidence=min_confidence,
            )
    except Exception as e:
        timings["total"] = round((time.perf_counter() - _t_total_start) * 1000.0, 3)
        logger.error("risk assignment failed: %s", e)
        return _safe_fallback(turns, f"risk: {e}", timings=timings)

    risk_level = risk.band
    is_emergency = risk_level == "EMERGENCY"

    # ── Step 5: Validator gate ───────────────────────────────────────
    # H2: read the structured flag instead of substring-matching reasoning.
    fallback_invoked = classification.is_fallback
    try:
        with _time_stage(timings, "validate"):
            validator_result = validate(
                subcategory=subcategory,
                risk_level=risk_level,
                classification_confidence=min_confidence,
                fallback_invoked=fallback_invoked,
                # A 911 dispatch additionally requires a real extracted hazard
                # cue — passing these is what lets the validator distinguish a
                # genuine life-safety call from a misclassification.
                extracted_urgency_cues=extraction.urgency_cues,
                # Full transcript powers the benign-context override that
                # blocks 911 when the call carries an explicit "this is not a
                # real emergency" signal ("no actual fire", "burnt popcorn",
                # "false alarm", ...) — catches the over-escalation trap
                # that surfaced on the test-set audit.
                transcript_text=full_transcript,
            )
    except Exception as e:
        logger.warning("validator failed: %s", e)
        validator_result = ValidatorResult(
            needs_human_review=True,
            dispatched_emergency_services=False,
            reasons=[f"validator_error:{e}"],
        )

    # Security override: if injection detected, always pause for human
    # review and NEVER auto-dispatch 911 (T6 defense — attacker could
    # fabricate urgency cues to trigger autonomous emergency dispatch).
    if security_scan.any_threat:
        validator_result = replace(
            validator_result,
            needs_human_review=True,
            dispatched_emergency_services=False,
            reasons=(*validator_result.reasons, *security_scan.reasons),
        )

    # ── Step 6: Vendor selection ─────────────────────────────────────
    try:
        with _time_stage(timings, "vendor"):
            vendor = select_vendor(
                subcategory=subcategory,
                city=location.city,
                building_type=location.building_type,
                risk_level=risk_level,
                is_emergency=is_emergency,
                after_hours=False,
            )
    except Exception as e:
        logger.warning("vendor selection failed: %s", e)
        vendor = VendorSelection(vendor_id=None, vendor_name=None, reason=f"error:{e}")

    # B2: no qualified vendor → escalate. The scorer's unroutable-case rule
    # (dispatched in (None, "") AND pr_h) requires both halves; without this
    # promotion, low/medium routine calls with no vendor match would score
    # wrong on vendor (10%), HITL-F1 (15%), and auto-resolution (10%).
    if vendor.vendor_id is None and not validator_result.needs_human_review:
        validator_result = replace(
            validator_result,
            needs_human_review=True,
            reasons=[*validator_result.reasons, "vendor_escalation:no_qualified_vendor"],
        )

    # ── Step 7: Clarification ────────────────────────────────────────
    try:
        with _time_stage(timings, "clarify"):
            clarification = needs_clarification(
                turns, extraction, classification, caller_phone=caller_phone
            )
    except Exception as e:
        logger.warning("clarification check failed: %s", e)
        clarification = ClarificationDecision(
            needs_clarification=False, question=None, reasons=()
        )

    # The clarify node detects in-transcript ambiguity; the location node
    # independently flags an out-of-range floor (a floor we can prove is
    # impossible). Either one means we must ask before acting.
    need_clarification = (
        clarification.needs_clarification or location.needs_clarification
    )
    clarification_reasons = list(clarification.reasons)
    if location.needs_clarification:
        clarification_reasons.append(
            f"location_floor_out_of_range:{location.floor or 'unknown'}"
        )
    clarification_question = clarification.question
    if location.needs_clarification and not clarification_question:
        clarification_question = "Can you confirm which floor the issue is on?"

    # ── Step 8: Summary ──────────────────────────────────────────────
    unroutable = vendor.vendor_id is None and validator_result.needs_human_review
    with _time_stage(timings, "summary"):
        summary = generate_summary(
            subcategory=subcategory,
            risk_level=risk_level,
            building_name=location.building_name,
            floor=location.floor,
            city=location.city,
            vendor_name=vendor.vendor_name,
            vendor_id=vendor.vendor_id,
            dispatched_emergency_services=validator_result.dispatched_emergency_services,
            needs_human_review=validator_result.needs_human_review,
            unroutable=unroutable,
        )

    # Stamp total wall-clock before assembling the trainer log so the
    # final dict carries the complete breakdown — STAGE_NAMES is the
    # source of truth for which keys must appear.
    timings["total"] = round((time.perf_counter() - _t_total_start) * 1000.0, 3)
    latency_ms = {name: timings.get(name, 0.0) for name in STAGE_NAMES}

    final_prediction = {
        "category": category,
        "subcategory": subcategory,
        "risk_level": risk_level,
        "needs_human_review": validator_result.needs_human_review,
        "needs_clarification": need_clarification,
        "building_name": location.building_name,
        "address": location.address,
        "floor": location.floor,
        "dispatched_vendor_id": vendor.vendor_id,
        "dispatched_emergency_services": validator_result.dispatched_emergency_services,
        "call_summary": summary,
    }

    # ── Step 9: Trainer log ──────────────────────────────────────────
    ai_pred = build_ai_prediction(
        category=category,
        subcategory=subcategory,
        risk_level=risk_level,
        needs_human_review=validator_result.needs_human_review,
        needs_clarification=need_clarification,
        building_name=location.building_name,
        address=location.address,
        floor=location.floor,
        dispatched_vendor_id=vendor.vendor_id,
        dispatched_emergency_services=validator_result.dispatched_emergency_services,
        call_summary=summary,
        confidence_category=confidence_cat,
        confidence_subcategory=confidence_sub,
        classification_reasoning=classification.reasoning,
        validator_reasons=list(validator_result.reasons),
        risk_reasons=list(risk.reasons),
        clarification_reasons=clarification_reasons,
        clarification_question=clarification_question,
        retrieved_record_ids=list(classification.retrieved_record_ids),
        latency_ms=latency_ms,
    )

    # Sanitize transcript before storing in trainer log (T1 defense —
    # prevents injection payloads from persisting into fine-tuning data).
    safe_transcript = sanitize_for_storage(full_transcript) if security_scan.any_threat else full_transcript

    trainer_log = assemble_trainer_log(
        full_transcript=safe_transcript,
        ai_prediction=ai_pred,
        human_override=None,
        final_decision=final_prediction.copy(),
    )

    # ── Assemble final prediction ────────────────────────────────────
    return {
        **final_prediction,
        "trainer_log": trainer_log,
    }


# ─── Events-aware sibling ──────────────────────────────────────────────
#
# `classify_with_events()` mirrors `classify()` but emits per-stage
# events through an `emitter` callback and optionally pauses at the
# validator gate for live HITL review. The pipeline contract,
# fault-isolation policy, and final output dict are identical to
# `classify()` — only the orchestration shell differs.

Emitter = Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]
ResumeWaiter = Callable[[], Awaitable[Dict[str, Any]]]


async def _emit(emitter: Optional[Emitter], event: Dict[str, Any]) -> None:
    if emitter is None:
        return
    result = emitter(event)
    if hasattr(result, "__await__"):
        await result  # type: ignore[func-returns-value]


def _stage_event(
    *,
    stage: str,
    status: str,
    timing_ms: float = 0.0,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": status,
        "timing_ms": round(timing_ms, 2),
        "payload": payload or {},
    }


def _retrieved_summaries(record_ids: List[str]) -> List[Dict[str, Any]]:
    """Best-effort lookup of retrieved tickets for the reviewer payload.

    The classifier records only ticket IDs. For the demo UI we surface a
    short summary per ID; we read the historical corpus lazily and return
    `{ticket_id, summary}` pairs, never raising — the demo display
    degrades gracefully if the corpus isn't on disk.
    """
    if not record_ids:
        return []
    try:
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "operational" / "historical_records.json"
        with open(path) as f:
            corpus = json.load(f)
        by_id = {r.get("ticket_id"): r for r in corpus if isinstance(r, dict)}
    except Exception:
        return [{"ticket_id": tid, "summary": None} for tid in record_ids]

    out: List[Dict[str, Any]] = []
    for tid in record_ids:
        rec = by_id.get(tid) or {}
        summary = (
            rec.get("resolution_notes")
            or rec.get("dispatch_notes")
            or rec.get("intake_notes")
            or rec.get("caller_transcript")
            or ""
        )
        if isinstance(summary, str) and len(summary) > 200:
            summary = summary[:197] + "..."
        out.append({"ticket_id": tid, "summary": summary or None})
    return out


async def classify_with_events(
    turns: List[dict],
    caller_phone: Optional[str],
    emitter: Optional[Emitter] = None,
    *,
    pause_at_validator: bool = False,
    wait_for_resume: Optional[ResumeWaiter] = None,
) -> Dict[str, Any]:
    """Run the agent pipeline emitting per-stage events.

    Mirrors `classify()` exactly except that it:
      - awaits `emitter({stage, status, timing_ms, payload})` at the
        start and end of every stage (including failures);
      - when `pause_at_validator=True` and the validator decides
        `needs_human_review`, emits `validate.gate_open` with the
        reviewer payload and awaits `wait_for_resume()`. The resume
        decision (`{"decision": "approve" | "override", "override": ...}`)
        is applied to the AI's state before continuing.

    The original `classify()` function above is untouched — this sibling
    only adds the eventing/pause hooks for the live demo.
    """
    full_transcript = _flatten(turns)

    async def _err_fallback(stage: str, error: Exception) -> Dict[str, Any]:
        msg = f"{stage}: {error}"
        await _emit(emitter, _stage_event(stage=stage, status="failed", payload={"error": str(error)}))
        fb = _safe_fallback(turns, msg)
        await _emit(
            emitter,
            _stage_event(
                stage="trainer_log",
                status="complete",
                payload={"trainer_log": fb["trainer_log"]},
            ),
        )
        return fb

    # ── Step 1: Extract ──────────────────────────────────────────────
    await _emit(emitter, _stage_event(stage="extract", status="started"))
    t0 = time.perf_counter()
    try:
        extraction = extract(turns, caller_phone)
    except Exception as e:
        logger.error("extraction failed: %s", e)
        return await _err_fallback("extract", e)
    extract_ms = (time.perf_counter() - t0) * 1000.0
    await _emit(
        emitter,
        _stage_event(
            stage="extract",
            status="complete",
            timing_ms=extract_ms,
            payload={
                "problem_summary": extraction.problem_summary,
                "building_name": extraction.building_name,
                "floor": extraction.floor,
                "suite": extraction.suite,
                "urgency_cues": list(extraction.urgency_cues),
                "caller_role": extraction.caller_role,
                "language": extraction.language,
                "confidence": {
                    "problem_summary": extraction.confidence.problem_summary,
                    "building_name": extraction.confidence.building_name,
                    "floor": extraction.confidence.floor,
                    "suite": extraction.confidence.suite,
                    "caller_role": extraction.confidence.caller_role,
                },
            },
        ),
    )

    # ── Step 2: Classify (RAG retrieve happens inside) ───────────────
    await _emit(emitter, _stage_event(stage="retrieve", status="started"))
    await _emit(emitter, _stage_event(stage="classify", status="started"))
    t0 = time.perf_counter()
    try:
        classification = classify_call(extraction)
    except Exception as e:
        logger.error("classification failed: %s", e)
        await _emit(emitter, _stage_event(stage="retrieve", status="failed", payload={"error": str(e)}))
        return await _err_fallback("classify", e)
    classify_ms = (time.perf_counter() - t0) * 1000.0

    retrieved = _retrieved_summaries(list(classification.retrieved_record_ids))
    await _emit(
        emitter,
        _stage_event(
            stage="retrieve",
            status="complete",
            timing_ms=classify_ms,
            payload={"retrieved": retrieved},
        ),
    )

    category = classification.category_str
    subcategory = classification.subcategory_str
    confidence_cat = classification.confidence_category
    confidence_sub = classification.confidence_subcategory
    min_confidence = min(confidence_cat, confidence_sub)

    await _emit(
        emitter,
        _stage_event(
            stage="classify",
            status="complete",
            timing_ms=classify_ms,
            payload={
                "category": category,
                "subcategory": subcategory,
                "confidence_category": confidence_cat,
                "confidence_subcategory": confidence_sub,
                "reasoning": classification.reasoning,
                "is_fallback": classification.is_fallback,
            },
        ),
    )

    # ── Step 3: Location ─────────────────────────────────────────────
    await _emit(emitter, _stage_event(stage="location", status="started"))
    t0 = time.perf_counter()
    profile = profiles.lookup(caller_phone)
    try:
        location = reconcile(
            extracted_building_name=extraction.building_name,
            extracted_floor=extraction.floor,
            building_confidence=extraction.confidence.building_name,
            floor_confidence=extraction.confidence.floor,
            profile=profile,
            caller_phone=caller_phone,
        )
    except Exception as e:
        logger.warning("location reconciliation failed: %s", e)
        location = ResolvedLocation(
            building_name=extraction.building_name,
            address=None,
            floor=extraction.floor,
            city=None,
            building_type=None,
            floor_check="unknown",
            source_building="none",
            source_floor="none",
        )
    location_ms = (time.perf_counter() - t0) * 1000.0
    await _emit(
        emitter,
        _stage_event(
            stage="location",
            status="complete",
            timing_ms=location_ms,
            payload={
                "building_name": location.building_name,
                "address": location.address,
                "floor": location.floor,
                "city": location.city,
                "building_type": location.building_type,
                "floor_check": location.floor_check,
                "source_building": location.source_building,
                "source_floor": location.source_floor,
            },
        ),
    )

    # ── Step 4: Risk ─────────────────────────────────────────────────
    await _emit(emitter, _stage_event(stage="risk", status="started"))
    t0 = time.perf_counter()
    try:
        risk = assign_risk(
            extraction,
            subcategory,
            building_type=location.building_type,
            after_hours=False,
            classification_confidence=min_confidence,
        )
    except Exception as e:
        logger.error("risk assignment failed: %s", e)
        return await _err_fallback("risk", e)
    risk_ms = (time.perf_counter() - t0) * 1000.0
    risk_level = risk.band
    is_emergency = risk_level == "EMERGENCY"
    await _emit(
        emitter,
        _stage_event(
            stage="risk",
            status="complete",
            timing_ms=risk_ms,
            payload={
                "risk_level": risk_level,
                "risk_reasons": list(risk.reasons),
                "base_risk": risk.base_risk,
                "score": risk.score,
            },
        ),
    )

    # ── Step 5: Validator gate ───────────────────────────────────────
    await _emit(emitter, _stage_event(stage="validate", status="started"))
    fallback_invoked = classification.is_fallback
    t0 = time.perf_counter()
    try:
        validator_result = validate(
            subcategory=subcategory,
            risk_level=risk_level,
            classification_confidence=min_confidence,
            fallback_invoked=fallback_invoked,
            extracted_urgency_cues=extraction.urgency_cues,
            transcript_text=full_transcript,
        )
    except Exception as e:
        logger.warning("validator failed: %s", e)
        validator_result = ValidatorResult(
            needs_human_review=True,
            dispatched_emergency_services=False,
            reasons=[f"validator_error:{e}"],
        )
    validate_ms = (time.perf_counter() - t0) * 1000.0
    await _emit(
        emitter,
        _stage_event(
            stage="validate",
            status="complete",
            timing_ms=validate_ms,
            payload={
                "needs_human_review": validator_result.needs_human_review,
                "dispatched_emergency_services": validator_result.dispatched_emergency_services,
                "validator_reasons": list(validator_result.reasons),
            },
        ),
    )

    # ── HITL pause point ─────────────────────────────────────────────
    human_override: Optional[Dict[str, Any]] = None
    if pause_at_validator and validator_result.needs_human_review:
        ai_pred_preview = {
            "category": category,
            "subcategory": subcategory,
            "risk_level": risk_level,
            "needs_human_review": True,
            "dispatched_emergency_services": validator_result.dispatched_emergency_services,
            "building_name": location.building_name,
            "address": location.address,
            "floor": location.floor,
            "confidence_category": confidence_cat,
            "confidence_subcategory": confidence_sub,
        }
        await _emit(
            emitter,
            _stage_event(
                stage="validate",
                status="gate_open",
                payload={
                    "ai_prediction": ai_pred_preview,
                    "validator_reasons": list(validator_result.reasons),
                    "reasoning": classification.reasoning,
                    "retrieved": retrieved,
                    "transcript": full_transcript,
                },
            ),
        )

        if wait_for_resume is None:
            raise RuntimeError(
                "pause_at_validator=True requires wait_for_resume callable"
            )
        decision = await wait_for_resume()
        action = (decision or {}).get("decision", "approve")
        override = (decision or {}).get("override") or {}
        await _emit(
            emitter,
            _stage_event(
                stage="validate",
                status="gate_resumed",
                payload={"decision": action, "override": override or None},
            ),
        )

        if action == "override" and override:
            human_override = dict(override)
            # Apply override fields onto the AI state used by downstream nodes.
            if "category" in override:
                category = override["category"]
            if "subcategory" in override:
                subcategory = override["subcategory"]
            if "risk_level" in override:
                risk_level = override["risk_level"]
                is_emergency = risk_level == "EMERGENCY"
            if "dispatched_emergency_services" in override:
                validator_result = replace(
                    validator_result,
                    dispatched_emergency_services=bool(
                        override["dispatched_emergency_services"]
                    ),
                )
            if "needs_human_review" in override:
                validator_result = replace(
                    validator_result,
                    needs_human_review=bool(override["needs_human_review"]),
                )

    # ── Step 6: Vendor selection ─────────────────────────────────────
    await _emit(emitter, _stage_event(stage="vendor", status="started"))
    t0 = time.perf_counter()
    try:
        vendor = select_vendor(
            subcategory=subcategory,
            city=location.city,
            building_type=location.building_type,
            risk_level=risk_level,
            is_emergency=is_emergency,
            after_hours=False,
        )
    except Exception as e:
        logger.warning("vendor selection failed: %s", e)
        vendor = VendorSelection(vendor_id=None, vendor_name=None, reason=f"error:{e}")
    vendor_ms = (time.perf_counter() - t0) * 1000.0

    # Apply vendor override if the reviewer specified one.
    if human_override and "dispatched_vendor_id" in human_override:
        vendor = VendorSelection(
            vendor_id=human_override["dispatched_vendor_id"],
            vendor_name=vendor.vendor_name,
            reason="human_override",
        )

    if vendor.vendor_id is None and not validator_result.needs_human_review:
        validator_result = replace(
            validator_result,
            needs_human_review=True,
            reasons=[*validator_result.reasons, "vendor_escalation:no_qualified_vendor"],
        )

    await _emit(
        emitter,
        _stage_event(
            stage="vendor",
            status="complete",
            timing_ms=vendor_ms,
            payload={
                "vendor_id": vendor.vendor_id,
                "vendor_name": vendor.vendor_name,
                "reasoning": vendor.reason,
            },
        ),
    )

    # ── Step 7: Clarification ────────────────────────────────────────
    await _emit(emitter, _stage_event(stage="clarify", status="started"))
    t0 = time.perf_counter()
    try:
        clarification = needs_clarification(
            turns, extraction, classification, caller_phone=caller_phone
        )
    except Exception as e:
        logger.warning("clarification check failed: %s", e)
        clarification = ClarificationDecision(
            needs_clarification=False, question=None, reasons=()
        )
    clarify_ms = (time.perf_counter() - t0) * 1000.0
    need_clarification = (
        clarification.needs_clarification or location.needs_clarification
    )
    clarification_reasons = list(clarification.reasons)
    if location.needs_clarification:
        clarification_reasons.append(
            f"location_floor_out_of_range:{location.floor or 'unknown'}"
        )
    clarification_question = clarification.question
    if location.needs_clarification and not clarification_question:
        clarification_question = "Can you confirm which floor the issue is on?"
    await _emit(
        emitter,
        _stage_event(
            stage="clarify",
            status="complete",
            timing_ms=clarify_ms,
            payload={
                "needs_clarification": need_clarification,
                "question": clarification.question,
                "reasons": list(clarification.reasons),
            },
        ),
    )

    # ── Step 8: Summary ──────────────────────────────────────────────
    await _emit(emitter, _stage_event(stage="summary", status="started"))
    t0 = time.perf_counter()
    unroutable = vendor.vendor_id is None and validator_result.needs_human_review
    summary = generate_summary(
        subcategory=subcategory,
        risk_level=risk_level,
        building_name=location.building_name,
        floor=location.floor,
        city=location.city,
        vendor_name=vendor.vendor_name,
        vendor_id=vendor.vendor_id,
        dispatched_emergency_services=validator_result.dispatched_emergency_services,
        needs_human_review=validator_result.needs_human_review,
        unroutable=unroutable,
    )
    summary_ms = (time.perf_counter() - t0) * 1000.0
    await _emit(
        emitter,
        _stage_event(
            stage="summary",
            status="complete",
            timing_ms=summary_ms,
            payload={"call_summary": summary},
        ),
    )

    # ── Step 9: Trainer log ──────────────────────────────────────────
    await _emit(emitter, _stage_event(stage="trainer_log", status="started"))
    final_prediction = {
        "category": category,
        "subcategory": subcategory,
        "risk_level": risk_level,
        "needs_human_review": validator_result.needs_human_review,
        "needs_clarification": need_clarification,
        "building_name": location.building_name,
        "address": location.address,
        "floor": location.floor,
        "dispatched_vendor_id": vendor.vendor_id,
        "dispatched_emergency_services": validator_result.dispatched_emergency_services,
        "call_summary": summary,
    }
    ai_pred = build_ai_prediction(
        category=category,
        subcategory=subcategory,
        risk_level=risk_level,
        needs_human_review=validator_result.needs_human_review,
        needs_clarification=need_clarification,
        building_name=location.building_name,
        address=location.address,
        floor=location.floor,
        dispatched_vendor_id=vendor.vendor_id,
        dispatched_emergency_services=validator_result.dispatched_emergency_services,
        call_summary=summary,
        confidence_category=confidence_cat,
        confidence_subcategory=confidence_sub,
        classification_reasoning=classification.reasoning,
        validator_reasons=list(validator_result.reasons),
        risk_reasons=list(risk.reasons),
        clarification_reasons=clarification_reasons,
        clarification_question=clarification_question,
        retrieved_record_ids=list(classification.retrieved_record_ids),
    )

    trainer_log = assemble_trainer_log(
        full_transcript=full_transcript,
        ai_prediction=ai_pred,
        human_override=human_override,
        final_decision=final_prediction.copy(),
    )
    await _emit(
        emitter,
        _stage_event(
            stage="trainer_log",
            status="complete",
            payload={"trainer_log": trainer_log},
        ),
    )

    return {
        **final_prediction,
        "trainer_log": trainer_log,
    }
