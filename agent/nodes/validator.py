"""Validator gate node — decides needs_human_review and emergency dispatch.

Runs after risk assignment. Outputs:
  - needs_human_review: bool (drives LangGraph interrupt)
  - validator_reasons: list of strings explaining why
  - dispatched_emergency_services: bool (911 — conservative default False)

Emergency dispatch rule (conservative, avoids −5 penalty):
  dispatched_emergency_services = True ONLY when:
    1. risk_level == "EMERGENCY", AND
    2. subcategory is on the life-safety whitelist

The HITL trigger logic is delegated to `agent.data.hitl.should_pause()`,
which encodes the derived policy (HIGH/EMERGENCY bands, trap-prone
subcategories, low confidence, fallback path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from agent.data.hitl import should_pause


_LIFE_SAFETY_SUBCATEGORIES = frozenset(
    {"active_threat", "entrapment", "fire_smoke", "gas_chemical", "panel_hazard"}
)


@dataclass(frozen=True)
class ValidatorResult:
    needs_human_review: bool
    dispatched_emergency_services: bool
    reasons: List[str] = field(default_factory=list)


def validate(
    *,
    subcategory: str,
    risk_level: str,
    classification_confidence: Optional[float] = None,
    fallback_invoked: bool = False,
) -> ValidatorResult:
    """Run the validator gate.

    Args:
        subcategory: classified subcategory from the classify node.
        risk_level: risk band from the risk node (LOW/MEDIUM/HIGH/EMERGENCY).
        classification_confidence: min confidence across category/subcategory
            axes. Pass None when unavailable (will skip the low-conf rule).
        fallback_invoked: True when the classifier used its fallback path.

    Returns:
        ValidatorResult with needs_human_review, dispatched_emergency_services,
        and the reasons trace for trainer_log.
    """
    pause, reasons = should_pause(
        subcategory=subcategory,
        predicted_risk=risk_level,
        classification_confidence=classification_confidence,
        fallback_invoked=fallback_invoked,
    )

    dispatch_911 = (
        risk_level == "EMERGENCY"
        and subcategory in _LIFE_SAFETY_SUBCATEGORIES
    )

    if dispatch_911 and not pause:
        pause = True
        reasons = [f"emergency_dispatch:{subcategory}"]

    return ValidatorResult(
        needs_human_review=pause,
        dispatched_emergency_services=dispatch_911,
        reasons=reasons,
    )
