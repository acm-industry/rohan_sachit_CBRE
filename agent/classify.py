"""Single public entrypoint for the agent pipeline.

`classify(turns, caller_phone) -> dict` is the contract consumed by
`evaluation/run_eval.py`. It chains all nodes in sequence:

  Extract → Classify → Risk → Validate → Location → Vendor → Clarify → Summary → TrainerLog

Each node is fault-isolated: if an LLM-backed node raises, the pipeline
falls back to safe defaults (needs_human_review=True, no 911 dispatch)
and still returns a schema-valid prediction dict.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.data import profiles
from agent.nodes.extract import Extraction, extract
from agent.nodes.classify import Classification, classify as classify_call
from agent.nodes.risk import RiskAssignment, assign_risk
from agent.nodes.validator import ValidatorResult, validate
from agent.nodes.location import ResolvedLocation, reconcile
from agent.nodes.vendor_select import VendorSelection, select_vendor
from agent.nodes.clarify import ClarificationDecision, needs_clarification
from agent.nodes.summary import generate_summary
from agent.nodes.trainer_log import assemble_trainer_log, build_ai_prediction

logger = logging.getLogger(__name__)


def _flatten(turns: List[dict]) -> str:
    return "\n".join(
        f"[{t.get('speaker', '?').upper()}] {t.get('text', '')}".rstrip()
        for t in turns
    )


def _safe_fallback(turns: List[dict], error: str) -> Dict[str, Any]:
    """Return a safe prediction on pipeline failure — avoids false-911."""
    full_transcript = _flatten(turns)
    base = {
        "category": "OTHER",
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
    return {
        **base,
        "trainer_log": {
            "full_transcript": full_transcript,
            "ai_prediction": base.copy(),
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

    # ── Step 1: Extract ──────────────────────────────────────────────
    try:
        extraction = extract(turns, caller_phone)
    except Exception as e:
        logger.error("extraction failed: %s", e)
        return _safe_fallback(turns, f"extraction: {e}")

    # ── Step 2: Classify ─────────────────────────────────────────────
    try:
        classification = classify_call(extraction)
    except Exception as e:
        logger.error("classification failed: %s", e)
        return _safe_fallback(turns, f"classification: {e}")

    category = classification.category_str
    subcategory = classification.subcategory_str
    confidence_cat = classification.confidence_category
    confidence_sub = classification.confidence_subcategory
    min_confidence = min(confidence_cat, confidence_sub)

    # ── Step 3: Location ─────────────────────────────────────────────
    profile = profiles.lookup(caller_phone)
    try:
        location = reconcile(
            extracted_building_name=extraction.building_name,
            extracted_floor=extraction.floor,
            building_confidence=extraction.confidence.building_name,
            floor_confidence=extraction.confidence.floor,
            profile=profile,
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
        risk = assign_risk(
            extraction,
            subcategory,
            building_type=location.building_type,
            after_hours=False,
            classification_confidence=min_confidence,
        )
    except Exception as e:
        logger.error("risk assignment failed: %s", e)
        return _safe_fallback(turns, f"risk: {e}")

    risk_level = risk.band
    is_emergency = risk_level == "EMERGENCY"

    # ── Step 5: Validator gate ───────────────────────────────────────
    fallback_invoked = (
        classification.reasoning is not None
        and "fallback" in classification.reasoning.lower()
    )
    try:
        validator_result = validate(
            subcategory=subcategory,
            risk_level=risk_level,
            classification_confidence=min_confidence,
            fallback_invoked=fallback_invoked,
        )
    except Exception as e:
        logger.warning("validator failed: %s", e)
        validator_result = ValidatorResult(
            needs_human_review=True,
            dispatched_emergency_services=False,
            reasons=[f"validator_error:{e}"],
        )

    # ── Step 6: Vendor selection ─────────────────────────────────────
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

    # ── Step 7: Clarification ────────────────────────────────────────
    try:
        clarification = needs_clarification(
            turns, extraction, classification, caller_phone=caller_phone
        )
    except Exception as e:
        logger.warning("clarification check failed: %s", e)
        clarification = ClarificationDecision(
            needs_clarification=False, question=None, reasons=()
        )

    # ── Step 8: Summary ──────────────────────────────────────────────
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

    # ── Step 9: Trainer log ──────────────────────────────────────────
    ai_pred = build_ai_prediction(
        category=category,
        subcategory=subcategory,
        risk_level=risk_level,
        dispatched_vendor_id=vendor.vendor_id,
        dispatched_emergency_services=validator_result.dispatched_emergency_services,
        needs_human_review=validator_result.needs_human_review,
        needs_clarification=clarification.needs_clarification,
        confidence_category=confidence_cat,
        confidence_subcategory=confidence_sub,
        validator_reasons=list(validator_result.reasons),
        risk_reasons=list(risk.reasons),
        retrieved_record_ids=[],
    )

    trainer_log = assemble_trainer_log(
        full_transcript=full_transcript,
        ai_prediction=ai_pred,
        human_override=None,
        final_decision=None,
    )

    # ── Assemble final prediction ────────────────────────────────────
    return {
        "category": category,
        "subcategory": subcategory,
        "risk_level": risk_level,
        "needs_human_review": validator_result.needs_human_review,
        "needs_clarification": clarification.needs_clarification,
        "building_name": location.building_name,
        "address": location.address,
        "floor": location.floor,
        "dispatched_vendor_id": vendor.vendor_id,
        "dispatched_emergency_services": validator_result.dispatched_emergency_services,
        "call_summary": summary,
        "trainer_log": trainer_log,
    }
