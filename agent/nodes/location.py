"""Location reconciliation node.

Resolves the final (building_name, address, floor, city) tuple from three
sources, in precedence order:

    1. Transcript-derived fields (from extraction) — highest authority
    2. Building registry lookup (canonical address/city for a matched name)
    3. Caller profile defaults — **only when the profile is active and not
       stale**. An inactive (`active=False`) or stale profile
       (`last_verified_at` older than `profiles.DEFAULT_STALE_DAYS`) is a
       prior we do not trust; precedence falls through to the
       anonymous-caller fallback (None) instead.

The spec: "transcript always wins on conflict; registry is canonical for
address/city; profiles are a stale prior, not source of truth."

Floor sanity: a floor that resolves outside the building's `floor_count`
is **not silently corrected**. The reported floor is kept verbatim and
`needs_clarification` is set so the orchestrator asks the caller rather
than guessing (e.g. "Floor 25 in an 18-floor building").

Consumed by the orchestrator to populate prediction fields. The scorer
checks all three (building_name, address, floor) case-insensitively.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from agent.data import profiles
from agent.data.buildings import Building, FloorCheck, get_by_name, validate_floor
from agent.data.profiles import Profile


@dataclass(frozen=True)
class ResolvedLocation:
    building_name: Optional[str]
    address: Optional[str]
    floor: Optional[str]
    city: Optional[str]
    building_type: Optional[str]
    floor_check: FloorCheck
    source_building: str  # 'transcript' | 'profile' | 'none'
    source_floor: str     # 'transcript' | 'profile' | 'none'
    # True when the resolved floor is out of range for the building — the
    # orchestrator must raise a clarification rather than dispatch on a
    # floor we know is impossible. Defaulted so existing constructors
    # (e.g. the orchestrator's exception fallback) stay valid.
    needs_clarification: bool = False


def _usable_profile(
    profile: Optional[Profile], now: Optional[datetime]
) -> Optional[Profile]:
    """Return the profile only if it is active and not stale, else None.

    Precedence per issue #19: caller profile is consulted *only* when
    active and not stale; otherwise we fall through to the anonymous
    fallback rather than trust a profile we can't date or that the
    tenant has deactivated.
    """
    if profile is None:
        return None
    if not profile.active:
        return None
    if profiles.is_stale(profile, now=now):
        return None
    return profile


def reconcile(
    *,
    extracted_building_name: Optional[str] = None,
    extracted_floor: Optional[str] = None,
    building_confidence: float = 0.0,
    floor_confidence: float = 0.0,
    profile: Optional[Profile] = None,
    now: Optional[datetime] = None,
) -> ResolvedLocation:
    """Reconcile location from extraction + profile + registry.

    Strategy:
    - The profile is first gated through `_usable_profile`: an inactive
      or stale profile is dropped (treated as no profile at all).
    - If extraction has a building name (confidence > 0.3), try to resolve
      it against the registry. This catches both transcript-stated names
      and profile-derived names the LLM echoed.
    - If extraction has no building name or the registry lookup fails,
      fall back to the (usable) profile's primary_building_name → registry.
    - Address and city always come from the registry when we have a match;
      the registry is the canonical source for those.
    - Floor: extraction wins when confidence is high (> 0.3); otherwise
      fall back to the usable profile. Validate against floor_count; an
      out-of-range floor sets `needs_clarification` (kept, not corrected).

    Args:
        now: reference time for the profile-staleness check. Defaults to
            wall-clock now (production); tests pin it for determinism.
    """
    profile = _usable_profile(profile, now)

    building: Optional[Building] = None
    source_building = "none"
    resolved_name: Optional[str] = None

    # Try extraction's building name first
    if extracted_building_name and building_confidence > 0.3:
        building = get_by_name(extracted_building_name)
        if building:
            resolved_name = building.name
            source_building = "transcript"

    # Fall back to profile's building
    if building is None and profile and profile.primary_building_name:
        building = get_by_name(profile.primary_building_name)
        if building:
            resolved_name = building.name
            source_building = "profile"
        elif profile.primary_building_name:
            resolved_name = profile.primary_building_name
            source_building = "profile"

    # If extraction had a name but it didn't resolve, still use it raw
    if resolved_name is None and extracted_building_name and building_confidence > 0.3:
        resolved_name = extracted_building_name
        source_building = "transcript"

    # Address and city: registry wins, then profile
    address: Optional[str] = None
    city: Optional[str] = None
    building_type: Optional[str] = None

    if building:
        address = building.address
        city = building.city
        building_type = building.building_type
    if not address and profile:
        address = profile.primary_address
    if not city and profile:
        city = profile.primary_city
    if not building_type and profile:
        building_type = profile.primary_building_type

    # Floor: extraction wins when confident, otherwise profile
    floor: Optional[str] = None
    source_floor = "none"

    if extracted_floor and floor_confidence > 0.3:
        floor = extracted_floor
        source_floor = "transcript"
    elif profile and profile.primary_floor:
        floor = profile.primary_floor
        source_floor = "profile"

    # Validate floor against building. An out-of-range floor is kept
    # verbatim (never silently corrected) but flags clarification.
    floor_check = validate_floor(building, floor)
    needs_clarification = floor_check == "out_of_range"

    return ResolvedLocation(
        building_name=resolved_name,
        address=address,
        floor=floor,
        city=city,
        building_type=building_type,
        floor_check=floor_check,
        source_building=source_building,
        source_floor=source_floor,
        needs_clarification=needs_clarification,
    )
