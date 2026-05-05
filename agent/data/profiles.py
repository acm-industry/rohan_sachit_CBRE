"""Caller-profile lookup against `operational/caller_profiles.json`.

Profiles are a CRM-style "last-known-default" snapshot — a prior, not the
source of truth (see `assignment_brief.md`). The transcript always wins on
conflict; this module just gives the agent a typed default to start from.

Behaviour summary:
- `lookup(None)` and `lookup("")` return None (anonymous caller).
- An unknown phone returns None.
- An inactive profile (`active=false`) returns None by default; pass
  `include_inactive=True` to surface it (e.g. for the trainer log).
- `is_stale()` flags a profile whose `last_verified_at` is older than
  `max_age_days` (default 180). The cutoff is documented + tunable; the
  design-doc justifies the 180-day choice from the data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

_PROFILE_PATH = Path(__file__).resolve().parents[2] / "operational" / "caller_profiles.json"

DEFAULT_STALE_DAYS = 180


@dataclass(frozen=True)
class Profile:
    """Subset of caller-profile fields the agent actually consumes.

    Field set is exhaustive of `caller_profiles.json` so downstream nodes
    (location reconciliation, vendor selection) don't have to round-trip
    back to the raw dict.
    """
    caller_name: str
    phone_number: str
    email: Optional[str]
    tenant_company: Optional[str]
    tenant_id: Optional[str]
    contact_role: Optional[str]
    preferred_language: Optional[str]
    primary_building_id: Optional[str]
    primary_building_name: Optional[str]
    primary_address: Optional[str]
    primary_city: Optional[str]
    primary_floor: Optional[str]
    primary_suite: Optional[str]
    primary_building_type: Optional[str]
    last_verified_at: Optional[str]
    active: bool


_INDEX_CACHE: Optional[Dict[str, Profile]] = None


def _normalize_phone(phone: str) -> str:
    """Strip whitespace; keep the canonical '+1-NNN-NNN-NNNN' shape used
    consistently across `caller_profiles.json` and the eval transcripts."""
    return phone.strip()


def _build_index() -> Dict[str, Profile]:
    raw = json.loads(_PROFILE_PATH.read_text())
    out: Dict[str, Profile] = {}
    for r in raw:
        phone = r.get("phone_number")
        if not phone:
            continue
        out[_normalize_phone(phone)] = Profile(
            caller_name=r.get("caller_name", ""),
            phone_number=phone,
            email=r.get("email"),
            tenant_company=r.get("tenant_company"),
            tenant_id=r.get("tenant_id"),
            contact_role=r.get("contact_role"),
            preferred_language=r.get("preferred_language"),
            primary_building_id=r.get("primary_building_id"),
            primary_building_name=r.get("primary_building_name"),
            primary_address=r.get("primary_address"),
            primary_city=r.get("primary_city"),
            primary_floor=r.get("primary_floor"),
            primary_suite=r.get("primary_suite"),
            primary_building_type=r.get("primary_building_type"),
            last_verified_at=r.get("last_verified_at"),
            active=bool(r.get("active", True)),
        )
    return out


def _index() -> Dict[str, Profile]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = _build_index()
    return _INDEX_CACHE


def lookup(phone: Optional[str], *, include_inactive: bool = False) -> Optional[Profile]:
    """Resolve a phone number to a Profile.

    Returns None for anonymous callers (`phone in (None, "")`), unknown
    numbers, and inactive profiles unless `include_inactive=True`.
    """
    if not phone:
        return None
    p = _index().get(_normalize_phone(phone))
    if p is None:
        return None
    if not p.active and not include_inactive:
        return None
    return p


def is_stale(profile: Profile,
             *,
             max_age_days: int = DEFAULT_STALE_DAYS,
             now: Optional[datetime] = None) -> bool:
    """True if `profile.last_verified_at` is older than `max_age_days`.

    Treats a missing or unparseable `last_verified_at` as stale — the
    agent should not trust a profile we can't date.
    """
    if not profile.last_verified_at:
        return True
    try:
        verified = datetime.fromisoformat(profile.last_verified_at)
    except ValueError:
        return True
    if verified.tzinfo is None:
        verified = verified.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - verified).days > max_age_days
