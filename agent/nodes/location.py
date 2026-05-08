"""Location reconciliation node.

Resolves the final (building_name, address, floor, city) tuple from three
sources, in precedence order:

    1. Transcript-derived fields (from extraction) — highest authority
    2. Building registry lookup (canonical address/city for a matched name)
    3. Caller profile defaults (last-known prior — lowest authority)

The spec: "transcript always wins on conflict; registry is canonical for
address/city; profiles are a stale prior, not source of truth."

Consumed by the orchestrator to populate prediction fields. The scorer
checks all three (building_name, address, floor) case-insensitively.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


def reconcile(
    *,
    extracted_building_name: Optional[str] = None,
    extracted_floor: Optional[str] = None,
    building_confidence: float = 0.0,
    floor_confidence: float = 0.0,
    profile: Optional[Profile] = None,
) -> ResolvedLocation:
    """Reconcile location from extraction + profile + registry.

    Strategy:
    - If extraction has a building name (confidence > 0.3), try to resolve
      it against the registry. This catches both transcript-stated names
      and profile-derived names the LLM echoed.
    - If extraction has no building name or the registry lookup fails,
      fall back to the profile's primary_building_name → registry lookup.
    - Address and city always come from the registry when we have a match;
      the registry is the canonical source for those.
    - Floor: extraction wins when confidence is high (> 0.3); otherwise
      fall back to profile. Validate against building floor_count.
    """
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

    # Validate floor against building
    floor_check = validate_floor(building, floor)

    return ResolvedLocation(
        building_name=resolved_name,
        address=address,
        floor=floor,
        city=city,
        building_type=building_type,
        floor_check=floor_check,
        source_building=source_building,
        source_floor=source_floor,
    )
