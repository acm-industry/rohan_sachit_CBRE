"""Single public entrypoint for the agent.

This is the stub scaffolding for issue #1: it returns a schema-valid
`Prediction` dict so the eval harness runs end-to-end. Real classification,
retrieval, risk scoring, HITL gating, and vendor selection land in later
issues. The goal here is wire compatibility, not score.

Contract reference: `evaluation/prediction.py::Prediction` + `TrainerLog`.
"""
from __future__ import annotations

from typing import Any, Optional


def _flatten(turns: list[dict]) -> str:
    return "\n".join(
        f"[{t.get('speaker', '?').upper()}] {t.get('text', '')}".rstrip()
        for t in turns
    )


def classify(turns: list[dict], caller_phone: Optional[str]) -> dict[str, Any]:
    """Return a schema-valid Prediction dict.

    Stub behavior: emit safe, conservative defaults that pass the schema check
    in `evaluation/run_eval.py` and avoid the −5 false-911 penalty. Downstream
    issues replace this with the real LangGraph pipeline.
    """
    base = {
        "category": "OTHER",
        "subcategory": "other",
        "risk_level": "MEDIUM",
        "needs_human_review": True,
        "needs_clarification": False,
        "building_name": None,
        "address": None,
        "floor": None,
        "dispatched_vendor_id": None,
        "dispatched_emergency_services": False,
        "call_summary": (
            "Stub prediction from agent scaffold (issue #1). "
            "Call routed to human reviewer pending full pipeline implementation."
        ),
    }

    full_transcript = _flatten(turns)
    return {
        **base,
        "trainer_log": {
            "full_transcript": full_transcript,
            "ai_prediction": dict(base),
            "human_override": None,
            "final_decision": dict(base),
        },
    }
