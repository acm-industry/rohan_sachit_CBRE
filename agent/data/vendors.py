"""Vendor catalog over `operational/vendors.json` plus a qualification API.

`qualify(subcategory, city, building_type, is_emergency, after_hours)`
returns the candidate set — vendors that *could* take this job — *before*
any ranking or availability-cache filtering. Issue #21 picks one vendor out
of this set; this module just narrows the field.

Filter rules:

- **specialty / vendor_type** — strict match first: a vendor qualifies if
  its `specialties` list contains the subcategory string. Loose fallback:
  the vendor's `vendor_type` appears in the subcategory→vendor-type map
  loaded from `agent/data/derived/subcategory_to_vendor_type.json` once
  issue #20 ships. Until then, the loose mapping is empty and only the
  strict specialty match applies — that's already strong because the
  dataset's `specialties` field contains subcategory-level strings (e.g.
  `"pipe_leak"`).

- **city** — exact match against `coverage_cities`. Soft-pass when the
  vendor's coverage list is empty/missing — we don't know, so we don't
  exclude.

- **building_type** — exact match against `building_types_certified`.
  Soft-pass when the cert list is empty/missing.

- **available_24_7** — only enforced when `is_emergency AND after_hours`.
  During business hours or on routine calls, every vendor is in scope
  regardless of the 24/7 flag.

This module deliberately ignores `status_at_last_check` /
`last_status_confirmed_at` — those drive the tie-breaker and stale-
availability handling in issue #21, not eligibility.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

_VENDOR_PATH = Path(__file__).resolve().parents[2] / "operational" / "vendors.json"
_DERIVED_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "derived" / "subcategory_to_vendor_type.json"
)


@dataclass(frozen=True)
class Vendor:
    vendor_id: str
    name: str
    vendor_type: str
    specialties: Tuple[str, ...] = field(default_factory=tuple)
    coverage_cities: Tuple[str, ...] = field(default_factory=tuple)
    building_types_certified: Tuple[str, ...] = field(default_factory=tuple)
    certifications: Tuple[str, ...] = field(default_factory=tuple)
    response_sla_minutes: Optional[int] = None
    emergency_response_sla_minutes: Optional[int] = None
    available_24_7: bool = False
    cost_tier: Optional[str] = None
    rating: Optional[float] = None
    phone: Optional[str] = None
    status_at_last_check: Optional[str] = None  # 'available' | 'at_capacity' | 'offline' | None
    last_status_confirmed_at: Optional[str] = None


_ALL_CACHE: Optional[List[Vendor]] = None
_BY_ID_CACHE: Optional[Dict[str, Vendor]] = None
_SUBCAT_MAPPING_CACHE: Optional[Mapping[str, Tuple[str, ...]]] = None


def _load_all() -> List[Vendor]:
    raw = json.loads(_VENDOR_PATH.read_text())
    out: List[Vendor] = []
    for r in raw:
        out.append(
            Vendor(
                vendor_id=r["vendor_id"],
                name=r.get("name", ""),
                vendor_type=r.get("vendor_type", ""),
                specialties=tuple(r.get("specialties") or ()),
                coverage_cities=tuple(r.get("coverage_cities") or ()),
                building_types_certified=tuple(r.get("building_types_certified") or ()),
                certifications=tuple(r.get("certifications") or ()),
                response_sla_minutes=r.get("response_sla_minutes"),
                emergency_response_sla_minutes=r.get("emergency_response_sla_minutes"),
                available_24_7=bool(r.get("available_24_7")),
                cost_tier=r.get("cost_tier"),
                rating=r.get("rating"),
                phone=r.get("phone"),
                status_at_last_check=r.get("status_at_last_check"),
                last_status_confirmed_at=r.get("last_status_confirmed_at"),
            )
        )
    return out


def _all_vendors() -> List[Vendor]:
    global _ALL_CACHE
    if _ALL_CACHE is None:
        _ALL_CACHE = _load_all()
    return _ALL_CACHE


def _by_id() -> Dict[str, Vendor]:
    global _BY_ID_CACHE
    if _BY_ID_CACHE is None:
        _BY_ID_CACHE = {v.vendor_id: v for v in _all_vendors()}
    return _BY_ID_CACHE


def _subcat_to_vendor_types() -> Mapping[str, Tuple[str, ...]]:
    """Loose subcategory→vendor_type mapping, populated by issue #20.

    Returns an empty mapping until that derived artifact lands; the strict
    `specialties` filter still does the heavy lifting in the meantime.
    """
    global _SUBCAT_MAPPING_CACHE
    if _SUBCAT_MAPPING_CACHE is not None:
        return _SUBCAT_MAPPING_CACHE
    if _DERIVED_PATH.exists():
        raw = json.loads(_DERIVED_PATH.read_text())
        _SUBCAT_MAPPING_CACHE = {k: tuple(v) for k, v in raw.items()}
    else:
        _SUBCAT_MAPPING_CACHE = {}
    return _SUBCAT_MAPPING_CACHE


def get_by_id(vendor_id: Optional[str]) -> Optional[Vendor]:
    if not vendor_id:
        return None
    return _by_id().get(vendor_id)


def all_vendors() -> List[Vendor]:
    return list(_all_vendors())


def _matches_subcategory(v: Vendor, subcategory: str) -> bool:
    if subcategory in v.specialties:
        return True
    types_for_subcat = _subcat_to_vendor_types().get(subcategory)
    if types_for_subcat and v.vendor_type in types_for_subcat:
        return True
    return False


def qualify(
    subcategory: Optional[str] = None,
    city: Optional[str] = None,
    building_type: Optional[str] = None,
    is_emergency: bool = False,
    after_hours: bool = False,
) -> List[Vendor]:
    """Return vendors that satisfy every applicable hard constraint.

    Empty list = nobody qualifies → caller (issue #21) should escalate
    rather than dispatch.
    """
    out: List[Vendor] = []
    enforce_247 = is_emergency and after_hours
    for v in _all_vendors():
        if subcategory and not _matches_subcategory(v, subcategory):
            continue
        if city and v.coverage_cities and city not in v.coverage_cities:
            continue
        if (
            building_type
            and v.building_types_certified
            and building_type not in v.building_types_certified
        ):
            continue
        if enforce_247 and not v.available_24_7:
            continue
        out.append(v)
    return out


def _reset_caches_for_tests() -> None:
    """Drop the in-process caches so tests can swap data files mid-run."""
    global _ALL_CACHE, _BY_ID_CACHE, _SUBCAT_MAPPING_CACHE
    _ALL_CACHE = None
    _BY_ID_CACHE = None
    _SUBCAT_MAPPING_CACHE = None
