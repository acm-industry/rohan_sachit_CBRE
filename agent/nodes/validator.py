"""Validator gate node — decides needs_human_review and emergency dispatch.

Runs after risk assignment. Outputs:
  - needs_human_review: bool (drives LangGraph interrupt)
  - reasons: tuple of strings explaining why (captured into trainer_log)
  - dispatched_emergency_services: bool (911 — conservative default False)

Emergency dispatch rule (conservative — a false 911 is a hard −5 rubric
penalty, so every precondition below must hold):

  dispatched_emergency_services = True ONLY when ALL of:
    1. risk_level == "EMERGENCY", AND
    2. subcategory is on the life-safety whitelist, AND
    3. at least one *extracted* urgency cue matches the hard-hazard
       lexicon (a present, active life-safety hazard — fire, smoke,
       trapped, gas leak, weapon, ...), AND
    4. the classifier did NOT fall back (a fallback label is a guess we
       must never autonomously act on), AND
    5. classification confidence is not below `_DISPATCH_MIN_CONFIDENCE`.

A life-safety EMERGENCY that fails 3/4/5 is never auto-dispatched but is
always escalated to a human immediately (`needs_human_review=True` with a
`life_safety_no_autodispatch:*` reason). This is the failure mode the
hazard-cue clause exists to prevent: a benign call *misclassified* as
`fire_smoke` on the low-confidence fallback path must not call 911.

The HITL trigger logic is delegated to `agent.data.hitl.should_pause()`,
which encodes the derived policy (HIGH/EMERGENCY bands, trap-prone
subcategories, low confidence, fallback path). This module never
duplicates that policy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from agent.data.hitl import should_pause as _hitl_should_pause


_LIFE_SAFETY_SUBCATEGORIES = frozenset(
    {"active_threat", "entrapment", "fire_smoke", "gas_chemical", "panel_hazard"}
)

# Hard-hazard lexicon. A 911 dispatch additionally requires at least one
# *extracted* urgency cue matching one of these — i.e. the caller actually
# described a present, active life-safety hazard. Mirrors the "+2 hard
# emergency" tier in agent/nodes/risk.py, kept as an explicit auditable
# list here because this is the single point that can autonomously call
# 911. Word-boundary anchored to avoid substring false-positives.
_HAZARD_CUE_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bflame[s]?\b",
        r"\bfire\b(?!\s*exit)",          # "fire" but not "fire exit" (a door)
        r"\bsmoke\b",
        r"\bburning\b",
        r"\bgas\s+leak\b",
        r"\bsmell(?:s|ing)?\s+(?:of\s+)?gas\b",
        r"\bchemical\s+(?:spill|leak|smell)\b",
        r"\btrapped\b",
        r"\bstuck\s+in\s+(?:the\s+)?elevator\b",
        r"\bunconscious\b",
        r"\bnot\s+breathing\b",
        r"\bcollaps(?:e|ed|ing)\b",
        r"\bweapon[s]?\b",
        r"\bgun\b",
        r"\bknife\b",
        r"\bshot[s]?\b",
        r"\bshooter\b",
        r"\bintruder\b",
        r"\bexplosion\b",
        r"\bbomb\b",
    )
)

# Confidence floor for autonomous 911. Even a life-safety subcategory with
# a hazard cue is escalated to a human (not auto-dispatched) when the
# classifier is below this — mirrors the HITL low-confidence pause intent.
_DISPATCH_MIN_CONFIDENCE = 0.5


def _has_hazard_cue(cues: Optional[Sequence[str]]) -> bool:
    """True iff any extracted urgency cue matches the hard-hazard lexicon."""
    if not cues:
        return False
    for c in cues:
        if not c:
            continue
        for pat in _HAZARD_CUE_PATTERNS:
            if pat.search(c):
                return True
    return False


@dataclass(frozen=True)
class ValidatorResult:
    needs_human_review: bool
    dispatched_emergency_services: bool
    reasons: Tuple[str, ...] = ()

    def should_pause(self) -> bool:
        """Boolean the LangGraph graph reads to decide whether to
        `interrupt()` for human review. Exposed per issue #16 AC —
        the gate's pause signal, decoupled from the 911 flag."""
        return self.needs_human_review


def validate(
    *,
    subcategory: str,
    risk_level: str,
    classification_confidence: Optional[float] = None,
    fallback_invoked: bool = False,
    extracted_urgency_cues: Optional[Sequence[str]] = None,
) -> ValidatorResult:
    """Run the validator gate.

    Args:
        subcategory: classified subcategory from the classify node.
        risk_level: risk band from the risk node (LOW/MEDIUM/HIGH/EMERGENCY).
        classification_confidence: min confidence across category/subcategory
            axes. Pass None when unavailable (will skip the low-conf rule).
        fallback_invoked: True when the classifier used its fallback path.
        extracted_urgency_cues: verbatim urgency phrases from the extraction
            node (`Extraction.urgency_cues`). Required for a 911 dispatch —
            omitted/empty means "no hazard cue", which conservatively
            *blocks* autonomous dispatch (escalates to a human instead).

    Returns:
        ValidatorResult with needs_human_review, dispatched_emergency_services,
        and the reasons trace for trainer_log.
    """
    pause, reasons = _hitl_should_pause(
        subcategory=subcategory,
        predicted_risk=risk_level,
        classification_confidence=classification_confidence,
        fallback_invoked=fallback_invoked,
    )
    reasons = list(reasons)

    is_life_safety_emergency = (
        risk_level == "EMERGENCY" and subcategory in _LIFE_SAFETY_SUBCATEGORIES
    )
    hazard_cue = _has_hazard_cue(extracted_urgency_cues)
    low_confidence = (
        classification_confidence is not None
        and classification_confidence < _DISPATCH_MIN_CONFIDENCE
    )

    dispatch_911 = (
        is_life_safety_emergency
        and hazard_cue
        and not fallback_invoked
        and not low_confidence
    )

    if dispatch_911:
        # A 911 dispatch always also goes to a human.
        pause = True
        reasons.append(f"emergency_dispatch:{subcategory}:hazard_cue_confirmed")
    elif is_life_safety_emergency:
        # Life-safety EMERGENCY but a 911 precondition failed: never
        # auto-dispatch — escalate to a human fast instead.
        if not hazard_cue:
            blocker = "no_hazard_cue"
        elif fallback_invoked:
            blocker = "classifier_fallback"
        else:
            blocker = "low_confidence"
        pause = True
        reasons.append(f"life_safety_no_autodispatch:{blocker}")

    return ValidatorResult(
        needs_human_review=pause,
        dispatched_emergency_services=dispatch_911,
        reasons=tuple(reasons),
    )
