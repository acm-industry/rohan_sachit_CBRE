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

from agent.data import phone_history, profiles
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
    source_building: str  # 'transcript' | 'phone_history' | 'profile' | 'none'
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
    caller_phone: Optional[str] = None,
    now: Optional[datetime] = None,
) -> ResolvedLocation:
    """Reconcile location from extraction + phone history + profile + registry.

    Precedence (highest first):
    - **Explicit transcript** building (extraction confidence > 0.3) —
      always wins (AC #19).
    - **Caller phone history** — when the historicals corpus has ≥
      `phone_history.MIN_N` past tickets for this phone, all but
      ≥ `MIN_SHARE` concentrated on one building, use it (issue #65).
      Outranks the profile because historical tickets are the recency
      signal the static profile lacks (catches callers who moved
      buildings: the 5 wrong-value dev rows all matched this pattern
      with 160–190 historical tickets, all on the GT building).
    - **Active + non-stale profile** (`_usable_profile`) — inactive or
      stale profiles are dropped and precedence falls through.
    - **Anonymous** fallback (None).

    Address / city / building_type come from the registry when a name
    matched; from the chosen-source's fields otherwise.

    Floor: extraction wins when confidence is high (> 0.3); otherwise
    fall back to the usable profile. An out-of-range floor sets
    `needs_clarification` (kept verbatim, not silently corrected).

    Args:
        caller_phone: caller's phone number (for the historicals lookup).
        now: reference time for the profile-staleness check. Defaults to
            wall-clock now (production); tests pin it for determinism.
    """
    raw_profile = profile  # keep for the profile-echo detection below
    profile = _usable_profile(profile, now)

    building: Optional[Building] = None
    source_building = "none"
    resolved_name: Optional[str] = None
    # phone_history-derived facts: when the historicals corpus is
    # confidently concentrated on one building for this phone, it's the
    # recency signal the static profile lacks.
    history_match = phone_history.recent_building(caller_phone)

    # Profile-echo detection: agent/nodes/extract.py merges the caller
    # profile into the LLM prompt as a prior, and on transcripts where
    # the caller doesn't restate the building the LLM frequently emits
    # the profile-derived building back at high confidence. We can spot
    # that pattern (extraction value literally matches the raw profile
    # building) and let phone_history override even though `extraction`
    # nominally "won" branch 1 — this isn't violating the AC #19
    # transcript-wins rule because the value didn't come from the
    # transcript.
    def _eqci(a, b):
        return bool(a and b and a.strip().lower() == b.strip().lower())

    profile_echo = (
        extracted_building_name is not None
        and raw_profile is not None
        and _eqci(extracted_building_name, raw_profile.primary_building_name)
    )
    history_contradicts_extraction = (
        history_match is not None
        and extracted_building_name is not None
        and not _eqci(extracted_building_name, history_match.building_name)
    )
    override_with_history = (
        history_match is not None
        and (profile_echo and history_contradicts_extraction
             or (extracted_building_name is None or building_confidence <= 0.3))
    )

    # 1. Try extraction's building name first (transcript wins), unless
    #    we've detected the profile-echo / phone-history-overrides case.
    if (
        extracted_building_name
        and building_confidence > 0.3
        and not (profile_echo and history_contradicts_extraction)
    ):
        building = get_by_name(extracted_building_name)
        if building:
            resolved_name = building.name
            source_building = "transcript"

    # 2. Phone-history majority — outranks profile (#65), and overrides a
    #    high-confidence-but-profile-echoed extraction (issue #65 #2).
    if building is None and history_match is not None and override_with_history:
        building = get_by_name(history_match.building_name)
        if building:
            resolved_name = building.name
        else:
            resolved_name = history_match.building_name
        source_building = "phone_history"

    # 3. Fall back to profile's building.
    if building is None and resolved_name is None and profile and profile.primary_building_name:
        building = get_by_name(profile.primary_building_name)
        if building:
            resolved_name = building.name
            source_building = "profile"
        elif profile.primary_building_name:
            resolved_name = profile.primary_building_name
            source_building = "profile"

    # 4. If extraction had a name but it didn't resolve, still use it raw.
    if resolved_name is None and extracted_building_name and building_confidence > 0.3:
        resolved_name = extracted_building_name
        source_building = "transcript"

    # Address / city / building_type: registry wins; then the chosen
    # source (phone_history when we picked it, else profile).
    address: Optional[str] = None
    city: Optional[str] = None
    building_type: Optional[str] = None

    if building:
        address = building.address
        city = building.city
        building_type = building.building_type
    if source_building == "phone_history" and history_match is not None:
        # Even if the registry resolved, phone-history's fields are
        # consistent (same building); they only matter when the registry
        # missed (history names a building outside our registry).
        if not address:
            address = history_match.address
        if not city:
            city = history_match.city
        if not building_type:
            building_type = history_match.building_type
    elif profile:
        if not address:
            address = profile.primary_address
        if not city:
            city = profile.primary_city
        if not building_type:
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
