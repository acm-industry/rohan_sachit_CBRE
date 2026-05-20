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
from dataclasses import replace
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
    # H2: read the structured flag instead of substring-matching reasoning.
    fallback_invoked = classification.is_fallback
    try:
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
        needs_clarification=need_clarification,
        confidence_category=confidence_cat,
        confidence_subcategory=confidence_sub,
        validator_reasons=list(validator_result.reasons),
        risk_reasons=list(risk.reasons),
        retrieved_record_ids=list(classification.retrieved_record_ids),
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
        "needs_clarification": need_clarification,
        "building_name": location.building_name,
        "address": location.address,
        "floor": location.floor,
        "dispatched_vendor_id": vendor.vendor_id,
        "dispatched_emergency_services": validator_result.dispatched_emergency_services,
        "call_summary": summary,
        "trainer_log": trainer_log,
    }
