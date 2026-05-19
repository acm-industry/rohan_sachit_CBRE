"""Vendor selection and dispatch node.

Given a classified call (subcategory, city, building_type, risk band),
selects one vendor from the qualified pool or returns None to signal
escalation (unroutable).

Selection pipeline:
  1. Hard-constraint filter via `qualify()` (specialty, city, cert, 24/7)
  2. Availability filter (the stale-cache trust model — see below)
  3. SLA filter: vendor's response SLA must be <= the risk-based max
  4. Tie-break (risk-dependent ordering — see `_sort_key`)

Stale-availability trust model
------------------------------
The brief notes the availability cache "can be hours or days old; how
much you trust it is your call." Our documented model:

  - `offline`     — trusted ONLY when recently confirmed. An offline
                    vendor whose `last_status_confirmed_at` is within
                    `_FRESH_HOURS` is skipped; an offline vendor with a
                    stale confirmation is treated as *unknown* (kept —
                    we don't trust a day-old "offline" to still hold).
  - `at_capacity` — skipped ONLY on emergencies (don't pile an urgent
                    life-safety job on a saturated crew); allowed for
                    routine work where a short queue is acceptable.
  - `available`   — never blocked. A stale "available" is downgraded to
                    unknown confidence but remains a candidate.
  - missing/None  — unknown; kept.

"Recent" is measured against a *deterministic* reference: the newest
`last_status_confirmed_at` in the catalog (data-driven, so a catalog
refresh moves the clock automatically and `classify()` stays
reproducible across runs — issue #28). Tests may inject `now=`.

When nobody qualifies, returns `VendorSelection(vendor_id=None,
needs_human_review=True, reason="no qualifying vendor — manual
reroute …")` so the orchestrator escalates rather than dispatching.

SLA caps by risk level (derived from dev labels — deterministic):
    EMERGENCY → 30 min (uses emergency_response_sla_minutes)
    HIGH      → 120 min
    MEDIUM    → 240 min
    LOW       → 480 min
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from agent.data.vendors import Vendor, all_vendors, qualify


_RISK_SLA_CAP = {
    "EMERGENCY": 30,
    "HIGH": 120,
    "MEDIUM": 240,
    "LOW": 480,
}

_COST_RANK = {"budget": 0, "standard": 1, "premium": 2}

# A status confirmation older than this (relative to the deterministic
# catalog reference time) is "stale": we no longer trust an `offline`,
# and `available` is downgraded to unknown confidence.
_FRESH_HOURS = 24

_CATALOG_NOW_CACHE: Optional[datetime] = None


@dataclass(frozen=True)
class VendorSelection:
    vendor_id: Optional[str]
    vendor_name: Optional[str]
    reason: str
    # True on the unroutable path so the orchestrator/scorer's
    # unroutable rule (dispatched=None AND needs_human_review=True) holds
    # even when the validator gate would otherwise auto-resolve.
    needs_human_review: bool = False
    # 'confirmed' when the chosen vendor's status was freshly confirmed,
    # 'unknown' when it was stale (degraded confidence, not blocked).
    availability_confidence: str = "unknown"


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _catalog_now() -> Optional[datetime]:
    """Deterministic 'now' = newest last_status_confirmed_at in the
    catalog. Cached; reset via `agent.data.vendors._reset_caches_for_tests`
    is not needed (pure function of the static catalog)."""
    global _CATALOG_NOW_CACHE
    if _CATALOG_NOW_CACHE is None:
        stamps = [
            _parse_ts(v.last_status_confirmed_at) for v in all_vendors()
        ]
        stamps = [s for s in stamps if s is not None]
        _CATALOG_NOW_CACHE = max(stamps) if stamps else None
    return _CATALOG_NOW_CACHE


def _reset_catalog_now_cache() -> None:
    global _CATALOG_NOW_CACHE
    _CATALOG_NOW_CACHE = None


def _is_recent(v: Vendor, now: Optional[datetime]) -> bool:
    """True when v's status was confirmed within `_FRESH_HOURS` of `now`."""
    if now is None:
        return False
    confirmed = _parse_ts(v.last_status_confirmed_at)
    if confirmed is None:
        return False
    return (now - confirmed) <= timedelta(hours=_FRESH_HOURS)


def _availability_ok(v: Vendor, *, is_emergency: bool, now: Optional[datetime]) -> bool:
    """Apply the documented stale-cache trust model. True = keep."""
    status = v.status_at_last_check
    if status == "offline":
        # Trust an offline only if it was recently confirmed.
        return not _is_recent(v, now)
    if status == "at_capacity":
        # Saturated crews are skipped for emergencies only.
        return not is_emergency
    # 'available', None, or any unknown status → keep.
    return True


def _vendor_sla(v: Vendor, is_emergency: bool) -> int:
    """Effective SLA for this vendor given the call type."""
    if is_emergency:
        return v.emergency_response_sla_minutes or 999
    return v.response_sla_minutes or 999


def _sort_key(v: Vendor, is_emergency: bool) -> Tuple:
    """Tie-break key (lower is better).

    Documented ordering (issue #21 AC):
      EMERGENCY: 24/7 first → fastest emergency SLA → rating → cost
      ROUTINE  : rating → cost → response SLA
    """
    rating = -(v.rating or 0.0)
    cost_rank = _COST_RANK.get(v.cost_tier or "premium", 2)
    sla = _vendor_sla(v, is_emergency)
    if is_emergency:
        not_247 = 0 if v.available_24_7 else 1
        return (not_247, sla, rating, cost_rank)
    return (rating, cost_rank, sla)


def select_vendor(
    *,
    subcategory: Optional[str] = None,
    city: Optional[str] = None,
    building_type: Optional[str] = None,
    risk_level: str = "MEDIUM",
    is_emergency: bool = False,
    after_hours: bool = False,
    now: Optional[datetime] = None,
) -> VendorSelection:
    """Select the best vendor for a classified call.

    Args:
        subcategory: classified subcategory (drives specialty matching).
        city: resolved city for the call location.
        building_type: from building registry.
        risk_level: one of LOW/MEDIUM/HIGH/EMERGENCY — drives SLA cap.
        is_emergency: True for EMERGENCY band.
        after_hours: True when outside business hours.
        now: reference time for the staleness window. Defaults to the
            deterministic catalog reference (newest confirmation).

    Returns:
        VendorSelection with vendor_id (or None if unroutable, in which
        case `needs_human_review` is True).
    """
    ref_now = now if now is not None else _catalog_now()

    candidates = qualify(
        subcategory=subcategory,
        city=city,
        building_type=building_type,
        is_emergency=is_emergency,
        after_hours=after_hours,
    )

    # Availability filter (stale-cache trust model)
    candidates = [
        v for v in candidates
        if _availability_ok(v, is_emergency=is_emergency, now=ref_now)
    ]

    # SLA filter: vendor must meet the risk-level cap
    sla_cap = _RISK_SLA_CAP.get(risk_level, 480)
    candidates = [
        v for v in candidates
        if _vendor_sla(v, is_emergency) <= sla_cap
    ]

    if not candidates:
        return VendorSelection(
            vendor_id=None,
            vendor_name=None,
            needs_human_review=True,
            reason=_no_vendor_reason(
                subcategory, city, building_type, risk_level,
                is_emergency, after_hours,
            ),
        )

    ranked = sorted(candidates, key=lambda v: _sort_key(v, is_emergency))
    best = ranked[0]
    avail_conf = "confirmed" if _is_recent(best, ref_now) else "unknown"

    return VendorSelection(
        vendor_id=best.vendor_id,
        vendor_name=best.name,
        reason=_pick_reason(best, len(candidates), is_emergency, avail_conf),
        needs_human_review=False,
        availability_confidence=avail_conf,
    )


def _pick_reason(v: Vendor, pool_size: int, is_emergency: bool, avail_conf: str) -> str:
    parts = [f"selected from {pool_size} qualified"]
    parts.append(f"rating={v.rating}")
    parts.append(f"cost={v.cost_tier}")
    sla = _vendor_sla(v, is_emergency)
    parts.append(f"sla={sla}min")
    parts.append(f"status={v.status_at_last_check or 'unknown'}({avail_conf})")
    if is_emergency:
        parts.append(f"24_7={v.available_24_7}")
    return "; ".join(parts)


def _no_vendor_reason(
    subcategory: Optional[str],
    city: Optional[str],
    building_type: Optional[str],
    risk_level: str,
    is_emergency: bool,
    after_hours: bool,
) -> str:
    constraints = []
    if subcategory:
        constraints.append(f"subcategory={subcategory}")
    if city:
        constraints.append(f"city={city}")
    if building_type:
        constraints.append(f"building_type={building_type}")
    constraints.append(f"sla<={_RISK_SLA_CAP.get(risk_level, 480)}min")
    if is_emergency:
        constraints.append("emergency")
    if is_emergency and after_hours:
        constraints.append("requires_24_7=True")
    # The AC-mandated phrase ("manual reroute") and the legacy
    # "escalate to human" phrasing are both kept for downstream matching.
    return (
        "no qualifying vendor — manual reroute (escalate to human) "
        f"[{', '.join(constraints)}]"
    )
