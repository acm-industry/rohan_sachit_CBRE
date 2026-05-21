"""Risk-band assignment node.

Hybrid scoring: start from the per-subcategory historical base risk
(`agent.data.risk.base_risk_for`, derived in #13) and nudge the band up
based on caller-cue and context modifiers. Output is a `RiskAssignment`
the validator gate (#16) consumes to decide whether to interrupt for
human review.

API: `assign_risk(extraction, subcategory, *, building_type, after_hours,
classification_confidence) -> RiskAssignment`. The reasons list is the
inspection trace captured into `trainer_log.ai_prediction.risk_reasons`.

Hard rule: EMERGENCY band only fires on subcategories whose **historical
base risk is EMERGENCY** (`active_threat`, `entrapment`, `fire_smoke`,
`gas_chemical`, `panel_hazard`). Caller cues + sensitive context can push
a routine-base call up to HIGH but not beyond. The validator gate
re-enforces this on dispatch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from agent.data import risk as risk_data
from agent.data.risk import RISK_LEVEL_RANK, RISK_LEVELS
from agent.nodes.extract import Extraction


# Subcategories where the historical modal risk is EMERGENCY. Used by the
# drift-guard test only — the runtime cap is data-driven via
# `_historical_max_rank` so a corpus refresh updates behavior automatically.
_LIFE_SAFETY_SUBCATEGORIES = frozenset(
    {"active_threat", "entrapment", "fire_smoke", "gas_chemical", "panel_hazard"}
)

# Probability threshold for "historically observed at this risk band".
# Subcategories whose `P(band) >= HISTORICAL_BAND_THRESHOLD` for some band
# `B` are allowed to be assigned `B` after modifiers; bands above the
# observed max are capped. 5% is a deliberately permissive cut — we want
# to allow rare-but-real escalations (e.g. pipe_leak EMERGENCY at 11%) but
# block modifier-driven escalation of subcategories that never appear at
# higher bands (e.g. waste_odor 100% LOW — even a "smoke" cue mustn't
# push it past LOW because the historical data says it's always routine).
HISTORICAL_BAND_THRESHOLD = 0.05

# Word-boundary patterns for cue matching. Order matters: the first
# matching tier sets the bump amount per cue.
#
# +2: caller language describing an active, present hazard.
# +1: caller language describing escalating-but-not-emergency conditions
#     (overnight outages, deterioration, prolonged failure).
_HARD_EMERGENCY_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bflame[s]?\b",
    r"\bfire\b(?!\s*exit)",        # "fire" but not "fire exit" (a door type)
    r"\btrapped\b",
    r"\bunconscious\b",
    r"\bweapon[s]?\b",
    r"\bgun\b",
    r"\bknife\b",
    r"\bgas leak\b",
    r"\bstabbing\b",
    r"\bbleeding\b",
    r"\bsmoke\b(?!\s*alarm)",      # ongoing smoke, not just "smoke alarm beeped"
))

_SOFT_ESCALATION_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bovernight\b",
    r"\bgetting worse\b",
    r"\bspreading\b",
    r"\bcan'?t wait\b",
    r"\bhours\b",
    r"\bpeople stuck\b",
    r"\bfreezing\b",               # no_heating becomes more acute when explicit
    r"\bflood(ing)?\b",            # active flooding, not just water
))

# Building types where occupant safety / 24-hour presence elevates risk.
_SENSITIVE_BUILDING_TYPES = frozenset({"medical", "residential"})

# Near-tie base demotion (Lever A from the post-submission audit).
# Two subcategories — pipe_leak and power_outage — have nearly-equal
# historical distributions between MEDIUM and HIGH (pipe_leak: 43.3%
# MEDIUM vs 46.0% HIGH; power_outage: 46.9% vs 47.7%). The modal-risk
# rule blindly picks HIGH and over-classifies the ~half of these calls
# that are routine MEDIUM. Without any extracted urgency cue from the
# transcript, default to the cautious MEDIUM band; cues (soft or hard)
# can still push it back up via the modifier loop below. The thresholds
# are data-driven — any subcategory with HIGH and MEDIUM both ≥ 40%
# and within 10 percentage points of each other qualifies, so a corpus
# refresh updates the policy automatically.
_NEAR_TIE_MIN_SHARE = 0.40
_NEAR_TIE_TOLERANCE = 0.10


@dataclass(frozen=True)
class RiskAssignment:
    band: str
    score: float           # 0..1 normalized rank (LOW=0.0, MEDIUM=.33, HIGH=.67, EMERGENCY=1.0)
    base_risk: str         # the historical-prior band before modifiers
    reasons: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def rank(self) -> int:
        return RISK_LEVEL_RANK[self.band]


def _match_any(patterns: Tuple[re.Pattern, ...], text: str) -> Optional[str]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


def _classify_cue(cue: str) -> Tuple[int, str]:
    """Return (bump, tier) for a single urgency cue string.

    `bump` is the rank delta this cue contributes; `tier` is one of
    'hard', 'soft', or 'noise' (added to the reasons trace for inspection).
    """
    if (m := _match_any(_HARD_EMERGENCY_PATTERNS, cue)):
        return 2, "hard"
    if (m := _match_any(_SOFT_ESCALATION_PATTERNS, cue)):
        return 1, "soft"
    return 0, "noise"


def _historical_max_rank(subcategory: str) -> int:
    """Highest risk-level rank where this subcategory's historical
    distribution meets `HISTORICAL_BAND_THRESHOLD`.

    Drives the modifier cap: a subcategory whose historical distribution
    is 100% LOW gets a cap at LOW — modifier bumps don't promote it
    even on alarming cues, because the corpus tells us this issue type
    is genuinely routine and the alarming language is the
    over-escalation trap pattern.

    Returns EMERGENCY rank for unknown subcategories (be permissive on
    out-of-distribution inputs; the model_validator in the classifier
    keeps these vanishingly rare).
    """
    dist = risk_data.risk_distribution(subcategory)
    if not dist:
        return RISK_LEVEL_RANK["EMERGENCY"]
    max_rank = RISK_LEVEL_RANK["LOW"]
    for level in RISK_LEVELS:
        if dist.get(level, 0.0) >= HISTORICAL_BAND_THRESHOLD:
            max_rank = max(max_rank, RISK_LEVEL_RANK[level])
    return max_rank


def _cap_to_historical_max(rank: int, subcategory: str, reasons: List[str]) -> int:
    """Apply the historical-max cap. Records the cap reason when it fires."""
    max_rank = _historical_max_rank(subcategory)
    if rank > max_rank:
        capped_band = RISK_LEVELS[max_rank]
        reasons.append(
            f"capped_at_{capped_band}:historical_max for subcategory={subcategory}"
        )
        return max_rank
    return rank


def assign_risk(
    extraction: Extraction,
    subcategory: str,
    *,
    building_type: Optional[str] = None,
    after_hours: bool = False,
    classification_confidence: Optional[float] = None,
) -> RiskAssignment:
    """Assign a risk band to a classified call.

    Args:
        extraction: structured fields from `agent.nodes.extract.extract`.
        subcategory: the leaf label from `agent.nodes.classify.classify`.
        building_type: from `agent.data.buildings` lookup (issue #5),
            populated when known. None when the building registry has
            no entry for this address.
        after_hours: True when the call is outside business hours.
            Wired by the orchestrator from a runtime clock; defaults
            False so unit tests are deterministic.
        classification_confidence: optional 0..1 confidence from the
            classifier (#11). Currently surfaced as a reason at low
            values (<0.5) but does NOT change the band — the validator
            gate (#16) handles confidence-driven escalation to human
            review. This module is concerned with severity, not certainty.

    Returns:
        `RiskAssignment(band, score, base_risk, reasons)`. The reasons
        tuple is the inspection trace; downstream nodes capture it into
        `trainer_log.ai_prediction.risk_reasons` for error analysis (#25).
    """
    base = risk_data.base_risk_for(subcategory) or "MEDIUM"
    reasons: List[str] = [f"base_risk:{base}"]

    # Near-tie demote: when the corpus is split between MEDIUM and HIGH
    # for this subcategory AND the caller offered no urgency cue, fall
    # to the cautious side. Cues below can still push back up. Targets
    # the pipe_leak / power_outage HITL-FP cluster from §9.7's audit.
    if base == "HIGH" and not extraction.urgency_cues:
        _dist = risk_data.risk_distribution(subcategory) or {}
        _h, _m = _dist.get("HIGH", 0.0), _dist.get("MEDIUM", 0.0)
        if (
            _h >= _NEAR_TIE_MIN_SHARE
            and _m >= _NEAR_TIE_MIN_SHARE
            and abs(_h - _m) < _NEAR_TIE_TOLERANCE
        ):
            base = "MEDIUM"
            reasons.append(
                f"near_tie_no_cue_demote:HIGH->MEDIUM(HIGH={_h:.0%},MEDIUM={_m:.0%})"
            )

    rank = RISK_LEVEL_RANK[base]

    # Modifier 1: caller's verbatim urgency cues.
    # Soft cues may push up to HIGH but never to EMERGENCY — only hard
    # (life-safety) cues can reach EMERGENCY. This prevents "flooding"
    # on a pipe_leak (base=HIGH) from triggering a false emergency.
    for cue in extraction.urgency_cues:
        bump, tier = _classify_cue(cue)
        if bump > 0:
            if tier == "soft":
                rank = min(rank + bump, RISK_LEVEL_RANK["HIGH"])
            else:
                rank += bump
            reasons.append(f"{tier}_cue:{cue}")

    # Modifier 2: sensitive building types (medical / residential).
    if building_type and building_type.lower() in _SENSITIVE_BUILDING_TYPES:
        rank += 1
        reasons.append(f"sensitive_building:{building_type.lower()}")

    # Modifier 3: after-hours.
    if after_hours:
        rank += 1
        reasons.append("after_hours")

    # Modifier 4 (no rank change): low classifier confidence is recorded
    # as a reason for the trainer log + validator gate, but does NOT
    # itself bump severity. Confidence-driven HITL is #16's job.
    if classification_confidence is not None and classification_confidence < 0.5:
        reasons.append(f"low_classifier_confidence:{classification_confidence:.2f}")

    # Data-driven cap: never exceed the historical-max band for this
    # subcategory. Ensures modifiers can't push a routine-only subcategory
    # past its observed range (the over-escalation trap defense at the
    # risk-band layer).
    rank = _cap_to_historical_max(rank, subcategory, reasons)

    # Bound to [0, 3].
    rank = max(0, min(RISK_LEVEL_RANK["EMERGENCY"], rank))
    band = RISK_LEVELS[rank]

    return RiskAssignment(
        band=band,
        score=rank / 3.0,
        base_risk=base,
        reasons=tuple(reasons),
    )
