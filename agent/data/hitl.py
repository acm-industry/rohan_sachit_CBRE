"""Runtime loader for the HITL trigger policy.

Loads `agent/data/derived/hitl_policy.json` (produced by
`scripts/derive_hitl_policy.py`) and exposes `should_pause()` — the
validator gate (#16) calls this for every classified call to decide
whether to interrupt for human review.

The rule (v1):
  1. predicted_risk in (HIGH, EMERGENCY)             → pause
  2. subcategory.over_escalation_rate >= 15%         → pause
     (trap-prone subcategories: air_quality, waste_odor,
     suspicious_person, roof_leak, malfunction, panel_hazard)
  3. classifier_confidence < 0.5                     → pause
  4. classifier fallback path invoked                → pause
  5. otherwise                                       → auto-resolve

Each pause path returns the matching reason in the trace, captured into
`trainer_log.ai_prediction.hitl_reasons` for error analysis (#25) and
the reviewer payload.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from agent.data import risk as risk_data

_POLICY_PATH: Path = (
    Path(__file__).resolve().parent / "derived" / "hitl_policy.json"
)

_POLICY_CACHE: Optional[dict] = None


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

    if classification_confidence is not None and classification_confidence < low_confidence_threshold():
        reasons.append(f"low_confidence:{classification_confidence:.2f}")
        return True, reasons

    if fallback_invoked:
        reasons.append("classifier_fallback_invoked")
        return True, reasons

    reasons.append("auto_resolve:no_flags_fired")
    return False, reasons


def _reset_cache_for_tests() -> None:
    global _POLICY_CACHE
    _POLICY_CACHE = None
