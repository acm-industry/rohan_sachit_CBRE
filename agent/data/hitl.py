"""Runtime loader for the HITL trigger policy.

Loads `agent/data/derived/hitl_policy.json` (produced by
`scripts/derive_hitl_policy.py`) and exposes `should_pause()` — the
validator gate (#16) calls this for every classified call to decide
whether to interrupt for human review.

The rule (v2):
  1. predicted_risk in (HIGH, EMERGENCY)             → pause
  2. subcategory.over_escalation_rate >= 15%         → pause
     (catches `air_quality` at 19.4%)
  3. subcategory is "trap-cascade"                   → pause
     (issue #63: pred LOW/MEDIUM but the cells table shows ≥1 cell at
     HIGH or EMERGENCY for this subcategory with audit_rate ≥
     `cascade_audit_rate_pct` and n ≥ `cascade_min_n`, MINUS any
     `cascade_excludes` overrides. Catches under-classification on
     `malfunction`, `roof_leak`, `suspicious_person` — all sit below
     the 15% OER bar but have overwhelming audit evidence at higher
     bands. `waste_odor` qualifies structurally but is excluded — see
     the dev-sweep table in `scripts/derive_hitl_policy.py`.)
  4. classifier_confidence < `low_confidence` AND
     predicted_risk != LOW                            → pause
     (risk-weighted: low confidence on a LOW-risk call
     is acceptable because cost of error is negligible)
  5. classifier fallback path invoked                → pause
  6. otherwise                                       → auto-resolve

Each pause path returns the matching reason in the trace, captured into
`trainer_log.ai_prediction.hitl_reasons` for error analysis (#25) and
the reviewer payload.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import FrozenSet, List, Optional, Tuple

from agent.data import risk as risk_data

_POLICY_PATH: Path = (
    Path(__file__).resolve().parent / "derived" / "hitl_policy.json"
)

_POLICY_CACHE: Optional[dict] = None
_CASCADE_SUBS_CACHE: Optional[FrozenSet[str]] = None


def _load() -> dict:
    if not _POLICY_PATH.exists():
        raise RuntimeError(
            f"HITL policy not found at {_POLICY_PATH}. "
            "Run `python scripts/derive_hitl_policy.py` to generate it."
        )
    return json.loads(_POLICY_PATH.read_text())


def _policy() -> dict:
    global _POLICY_CACHE
    if _POLICY_CACHE is None:
        _POLICY_CACHE = _load()
    return _POLICY_CACHE


def trap_threshold() -> float:
    """The trap-prone over-escalation-rate threshold from the policy
    file, as a 0..1 fraction."""
    return _policy()["thresholds"]["trap_over_escalation_rate_pct"] / 100.0


def low_confidence_threshold() -> float:
    return _policy()["thresholds"]["low_confidence"]


def cascade_audit_rate_threshold() -> float:
    """Audit-rate threshold (0..1) for the trap-cascade rule (issue #63)."""
    return _policy()["thresholds"].get("cascade_audit_rate_pct", 50.0) / 100.0


def cascade_min_n() -> int:
    """Minimum cell `n` for the trap-cascade rule — avoids small-sample
    cells (e.g. n=1 audit_rate=100%) firing spuriously."""
    return int(_policy()["thresholds"].get("cascade_min_n", 5))


def cascade_excludes() -> FrozenSet[str]:
    """Subcategories that meet the cascade structural criteria but are
    explicitly excluded because the dev sweep showed they cost more on
    `auto_resolution` than they gain on `hitl_f1` (see the table in
    `scripts/derive_hitl_policy.py`)."""
    return frozenset(_policy()["thresholds"].get("cascade_excludes", []))


def cascade_subcategories() -> FrozenSet[str]:
    """The set of subcategories that should pause on any LOW/MEDIUM
    prediction because the cells table shows overwhelming audit evidence
    at HIGH or EMERGENCY (under-classification trap — issue #63).

    Derived once per process from the loaded policy, minus any
    `cascade_excludes` overrides, so a corpus refresh via
    `scripts/derive_hitl_policy.py` updates the rule automatically while
    preserving the documented dev-tuned exclusions.
    """
    global _CASCADE_SUBS_CACHE
    if _CASCADE_SUBS_CACHE is None:
        ar_min = cascade_audit_rate_threshold()
        n_min = cascade_min_n()
        cells = _policy().get("cells", {})
        subs = {
            sub for sub, bands in cells.items()
            if any(
                b in ("HIGH", "EMERGENCY")
                and c.get("n", 0) >= n_min
                and c.get("audit_rate", 0.0) >= ar_min
                for b, c in bands.items()
            )
        }
        _CASCADE_SUBS_CACHE = frozenset(subs - cascade_excludes())
    return _CASCADE_SUBS_CACHE


def should_pause(
    subcategory: str,
    predicted_risk: str,
    *,
    classification_confidence: Optional[float] = None,
    fallback_invoked: bool = False,
) -> Tuple[bool, List[str]]:
    """Decide whether to interrupt for human review.

    Args:
        subcategory: classifier output (#11).
        predicted_risk: risk-band output from the risk node (#14).
        classification_confidence: per-axis confidence from the
            classifier; pass `min(confidence_category, confidence_subcategory)`
            for a conservative read.
        fallback_invoked: True when the classifier had to use the
            category-only fallback path (both LLM attempts failed
            schema validation). Always pause in that case.

    Returns:
        `(pause, reasons)` tuple. `reasons` is a list of strings tagging
        which rule branch fired; passed through to the trainer log and
        the reviewer payload.
    """
    reasons: List[str] = []

    if predicted_risk in ("HIGH", "EMERGENCY"):
        reasons.append(f"{predicted_risk.lower()}_band:always_pause")
        return True, reasons

    oer = risk_data.intake_over_escalation_rate(subcategory)
    if oer >= trap_threshold():
        reasons.append(
            f"trap_prone_subcategory:{subcategory}(over_escalation_rate={oer:.0%})"
        )
        return True, reasons

    if subcategory in cascade_subcategories():
        reasons.append(
            f"trap_cascade_subcategory:{subcategory}"
            "(high-audit cell at HIGH/EMERGENCY — possible under-classification)"
        )
        return True, reasons

    if classification_confidence is not None and classification_confidence < low_confidence_threshold():
        # Risk-weighted threshold: low confidence on a LOW-risk call is
        # acceptable because cost of error is negligible (minor mis-routing
        # at worst). Only pause when the cost of a wrong decision is
        # non-trivial (MEDIUM+). Implements the brief's Stage 4 guidance:
        # "Risk = P(error) × Cost(error)".
        if predicted_risk != "LOW":
            reasons.append(f"low_confidence_high_cost:{classification_confidence:.2f}")
            return True, reasons

    if fallback_invoked:
        reasons.append("classifier_fallback_invoked")
        return True, reasons

    reasons.append("auto_resolve:no_flags_fired")
    return False, reasons


def _reset_cache_for_tests() -> None:
    global _POLICY_CACHE, _CASCADE_SUBS_CACHE
    _POLICY_CACHE = None
    _CASCADE_SUBS_CACHE = None
