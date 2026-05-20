"""Trainer log assembly node.

Runs as the final node of the graph on every call. Produces the
supervised-training signal: a frozen snapshot of the AI's pre-override
prediction, any human override, and the final decision.

Output schema (matches scoring.py expectations):
    {
        "full_transcript": str,
        "ai_prediction": { ... },
        "human_override": None | { field: new_value, ... },
        "final_decision": { ... }
    }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def assemble_trainer_log(
    *,
    full_transcript: str,
    ai_prediction: Dict[str, Any],
    human_override: Optional[Dict[str, Any]] = None,
    final_decision: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the trainer log dict.

    Args:
        full_transcript: the raw transcript text.
        ai_prediction: frozen snapshot of pre-validator AI outputs —
            category, subcategory, risk_level, vendor, confidence,
            validator_reasons, retrieved_record_ids, etc.
        human_override: None if no override happened; otherwise a dict
            of {field_name: overridden_value} for fields the reviewer changed.
        final_decision: post-override result that becomes the top-level
            prediction. If None (no override), uses ai_prediction.

    Returns:
        Complete trainer_log dict with all four required keys.
    """
    decision = final_decision if final_decision is not None else ai_prediction

    return {
        "full_transcript": full_transcript,
        "ai_prediction": ai_prediction,
        "human_override": human_override,
        "final_decision": decision,
    }


def build_ai_prediction(
    *,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    risk_level: Optional[str] = None,
    dispatched_vendor_id: Optional[str] = None,
    dispatched_emergency_services: bool = False,
    needs_human_review: bool = False,
    needs_clarification: bool = False,
    confidence_category: Optional[float] = None,
    confidence_subcategory: Optional[float] = None,
    validator_reasons: Optional[List[str]] = None,
    risk_reasons: Optional[List[str]] = None,
    retrieved_record_ids: Optional[List[str]] = None,
    latency_ms: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build the ai_prediction snapshot from node outputs."""
    pred: Dict[str, Any] = {
        "category": category,
        "subcategory": subcategory,
        "risk_level": risk_level,
        "dispatched_vendor_id": dispatched_vendor_id,
        "dispatched_emergency_services": dispatched_emergency_services,
        "needs_human_review": needs_human_review,
        "needs_clarification": needs_clarification,
        "confidence_category": confidence_category,
        "confidence_subcategory": confidence_subcategory,
        "validator_reasons": validator_reasons or [],
        "risk_reasons": risk_reasons or [],
        "retrieved_record_ids": retrieved_record_ids or [],
    }
    # Only emit latency_ms when the orchestrator measured it. Older
    # callers (tests, harness replays) that don't pass it stay unchanged
    # so the scorer's strict-shape check on existing trainer_log fields
    # isn't disturbed.
    if latency_ms is not None:
        pred["latency_ms"] = latency_ms
    return pred
