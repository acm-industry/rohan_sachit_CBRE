"""Vendor selection and dispatch node.

Given a classified call (subcategory, city, building_type, risk band),
selects one vendor from the qualified pool or returns None to signal
escalation (unroutable).

Selection pipeline:
1. Hard-constraint filter via `qualify()` (specialty, city, cert, 24/7)
2. SLA filter: vendor's response SLA must be <= the risk-based max
3. Tie-breaking: rating > cost tier > response SLA

When no vendor qualifies, returns None — the orchestrator should set
`dispatched_vendor_id=None` and ensure `needs_human_review=True`.

SLA caps by risk level (derived from dev labels — deterministic):
    EMERGENCY → 30 min (uses emergency_response_sla_minutes)
    HIGH      → 120 min
    MEDIUM    → 240 min
    LOW       → 480 min
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent.data.vendors import Vendor, qualify


_RISK_SLA_CAP = {
    "EMERGENCY": 30,
    "HIGH": 120,
    "MEDIUM": 240,
    "LOW": 480,
}

_COST_RANK = {"budget": 0, "standard": 1, "premium": 2}


@dataclass(frozen=True)
class VendorSelection:
    vendor_id: Optional[str]
    vendor_name: Optional[str]
    reason: str


def _vendor_sla(v: Vendor, is_emergency: bool) -> int:
    """Effective SLA for this vendor given the call type."""
    if is_emergency:
        return v.emergency_response_sla_minutes or 999
    return v.response_sla_minutes or 999


def _sort_key(v: Vendor, is_emergency: bool) -> Tuple:
    """Sort key for tie-breaking. Lower is better.

    Priority: higher rating, then lower cost, then shorter SLA.
    Availability status is NOT a hard filter — the spec says it's a
    stale cache. We use it only as a final tie-breaker.
    """
    rating = -(v.rating or 0.0)
    cost_rank = _COST_RANK.get(v.cost_tier or "premium", 2)
    sla = _vendor_sla(v, is_emergency)
    status_rank = {"available": 0, "at_capacity": 1, "offline": 2}.get(
        v.status_at_last_check or "offline", 2
    )
    return (rating, cost_rank, sla, status_rank)


def select_vendor(
    *,
    subcategory: Optional[str] = None,
    city: Optional[str] = None,
    building_type: Optional[str] = None,
    risk_level: str = "MEDIUM",
    is_emergency: bool = False,
    after_hours: bool = False,
) -> VendorSelection:
    """Select the best vendor for a classified call.

    Args:
        subcategory: classified subcategory (drives specialty matching).
        city: resolved city for the call location.
        building_type: from building registry.
        risk_level: one of LOW/MEDIUM/HIGH/EMERGENCY — drives SLA cap.
        is_emergency: True for EMERGENCY band.
        after_hours: True when outside business hours.

    Returns:
        VendorSelection with vendor_id (or None if unroutable).
    """
    candidates = qualify(
        subcategory=subcategory,
        city=city,
        building_type=building_type,
        is_emergency=is_emergency,
        after_hours=after_hours,
    )

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
            reason=_no_vendor_reason(subcategory, city, building_type, risk_level, is_emergency, after_hours),
        )

    ranked = sorted(candidates, key=lambda v: _sort_key(v, is_emergency))
    best = ranked[0]

    return VendorSelection(
        vendor_id=best.vendor_id,
        vendor_name=best.name,
        reason=_pick_reason(best, len(candidates), is_emergency),
    )


def _pick_reason(v: Vendor, pool_size: int, is_emergency: bool) -> str:
    parts = [f"selected from {pool_size} qualified"]
    parts.append(f"rating={v.rating}")
    parts.append(f"cost={v.cost_tier}")
    sla = _vendor_sla(v, is_emergency)
    parts.append(f"sla={sla}min")
    parts.append(f"status={v.status_at_last_check or 'unknown'}")
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
    if is_emergency and after_hours:
        constraints.append("requires_24_7=True")
    return f"no vendor qualifies for [{', '.join(constraints)}] — escalate to human"
