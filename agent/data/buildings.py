"""Building registry over `operational/buildings.json`.

Behaviour summary:
- `get_by_id` is exact-match on `building_id`.
- `get_by_name` does a normalized lookup (case + whitespace insensitive)
  followed by a substring match — so a transcript "Pacific Ridge"
  resolves to the canonical "Pacific Ridge Medical Plaza".
- `validate_floor` is a three-state checker (`ok` / `out_of_range` /
  `unknown`). It returns `unknown` whenever the building lacks
  `floor_count` rather than rejecting — the buildings.json file is
  intentionally sparse for warehouses / older properties (~8% of records).
- `resolve_address_city` returns the canonical `(address, city)` tuple
  preferring the building registry over a caller profile, per the brief's
  "transcript wins on conflict; registry is canonical" rule.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

_BUILDING_PATH = Path(__file__).resolve().parents[2] / "operational" / "buildings.json"

FloorCheck = Literal["ok", "out_of_range", "unknown"]

# Recognised floor strings beyond plain "Floor N" — pulled from the floor
# value distributions in profiles + historicals (Ground Floor and Mezzanine
# show up frequently; Basement / Rooftop appear in the historicals).
_FLOOR_NUM_RX = re.compile(r"^\s*floor\s+(\d+)\s*$", re.IGNORECASE)
_BASEMENT_RX = re.compile(r"^\s*(basement|b\s*\d+)\s*$", re.IGNORECASE)
_ROOFTOP_RX = re.compile(r"^\s*(roof(top)?)\s*$", re.IGNORECASE)
_GROUND_RX = re.compile(r"^\s*ground(\s+floor)?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Building:
    building_id: str
    name: str
    address: Optional[str]
    city: Optional[str]
    building_type: Optional[str]
    floor_count: Optional[int]
    has_basement: Optional[bool]
    has_rooftop: Optional[bool]
    special_floors: Tuple[str, ...] = field(default_factory=tuple)
    suites_per_floor: Optional[int] = None


_INDEX_BY_ID: Optional[Dict[str, Building]] = None
_INDEX_BY_NAME: Optional[Dict[str, Building]] = None  # normalized name → Building


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _build_indexes() -> Tuple[Dict[str, Building], Dict[str, Building]]:
    raw = json.loads(_BUILDING_PATH.read_text())
    by_id: Dict[str, Building] = {}
    by_name: Dict[str, Building] = {}
    for r in raw:
        bid = r.get("building_id")
        if not bid:
            continue
        b = Building(
            building_id=bid,
            name=r.get("name", ""),
            address=r.get("address"),
            city=r.get("city"),
            building_type=r.get("building_type"),
            floor_count=r.get("floor_count"),
            has_basement=r.get("has_basement"),
            has_rooftop=r.get("has_rooftop"),
            special_floors=tuple(r.get("special_floors") or ()),
            suites_per_floor=r.get("suites_per_floor"),
        )
        by_id[bid] = b
        if b.name:
            by_name[_normalize(b.name)] = b
    return by_id, by_name


def _indexes() -> Tuple[Dict[str, Building], Dict[str, Building]]:
    global _INDEX_BY_ID, _INDEX_BY_NAME
    if _INDEX_BY_ID is None or _INDEX_BY_NAME is None:
        _INDEX_BY_ID, _INDEX_BY_NAME = _build_indexes()
    return _INDEX_BY_ID, _INDEX_BY_NAME


def get_by_id(building_id: Optional[str]) -> Optional[Building]:
    if not building_id:
        return None
    by_id, _ = _indexes()
    return by_id.get(building_id)


def get_by_name(name: Optional[str]) -> Optional[Building]:
    """Resolve a free-text building name to the canonical record.

    Match strategy: normalized exact → unique substring (caller said the
    short-form "Pacific Ridge", catalog has "Pacific Ridge Medical Plaza").
    Returns None when the candidate is ambiguous (multiple substring hits)
    rather than guessing.
    """
    if not name:
        return None
    _, by_name = _indexes()
    key = _normalize(name)
    if key in by_name:
        return by_name[key]
    candidates: List[Building] = [
        b for n, b in by_name.items() if key in n or n in key
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def validate_floor(building: Optional[Building], floor_str: Optional[str]) -> FloorCheck:
    """Check whether `floor_str` is plausible for this building.

    Returns:
        - 'unknown' when the building is missing or has no `floor_count`
          (we can't evaluate; downstream should treat this as soft-pass)
        - 'ok' when the floor parses and falls in range / matches a
          documented special floor / matches has_basement / has_rooftop
        - 'out_of_range' when the floor parses to a number outside
          [1, floor_count] OR references basement/rooftop/special on a
          building documented to lack them
    """
    if building is None or building.floor_count is None:
        return "unknown"
    if not floor_str:
        return "unknown"

    s = floor_str.strip()

    # Plain "Floor N"
    m = _FLOOR_NUM_RX.match(s)
    if m:
        n = int(m.group(1))
        return "ok" if 1 <= n <= building.floor_count else "out_of_range"

    # "Ground Floor" — treat as floor 1 (always valid for a building that
    # exists at all).
    if _GROUND_RX.match(s):
        return "ok"

    # Basement variants — valid only if the building documents one.
    if _BASEMENT_RX.match(s):
        return "ok" if building.has_basement else "out_of_range"

    # Rooftop variants
    if _ROOFTOP_RX.match(s):
        return "ok" if building.has_rooftop else "out_of_range"

    # Special floors (Mezzanine, Loading Dock N, etc.) — case-insensitive
    # match against the documented list.
    if any(_normalize(s) == _normalize(sp) for sp in building.special_floors):
        return "ok"

    # Unparseable floor string + we have floor_count → can't confirm → unknown
    # (caller said something we don't understand, not necessarily wrong).
    return "unknown"


def resolve_address_city(
    building: Optional[Building] = None,
    profile_address: Optional[str] = None,
    profile_city: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (address, city) preferring the registry over the profile.

    The brief: the building registry is the canonical source for address /
    city; the caller profile is a last-known default. When both exist and
    disagree, the registry wins.
    """
    if building is not None:
        return (
            building.address or profile_address,
            building.city or profile_city,
        )
    return (profile_address, profile_city)
