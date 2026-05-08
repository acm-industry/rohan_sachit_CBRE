"""Call summary generator — 2-3 sentence operator narrative.

Produces a terse, factual, action-oriented summary suitable for
shift handoff. No first-person, no LLM filler, no apologies.

Style target: "Pipe leak reported at Pacific Ridge Medical Plaza,
Floor 3. MEDIUM risk. Dispatched AquaFix Plumbing (v_002)."

Length cap: ~280 characters. If an LLM-generated summary exceeds
the cap, it gets truncated at the last sentence boundary under limit.
"""
from __future__ import annotations

from typing import Optional


_MAX_CHARS = 280

_SUBCATEGORY_DESCRIPTIONS = {
    "access_control": "access control issue",
    "active_threat": "active threat",
    "air_quality": "air quality concern",
    "appliance_kitchen": "kitchen appliance issue",
    "auto_door": "automatic door malfunction",
    "carpet_floor": "carpet/flooring damage",
    "controls_bms": "building controls issue",
    "door_mechanical": "mechanical door issue",
    "drainage_backup": "drainage backup",
    "entrapment": "person trapped",
    "fire_smoke": "fire/smoke detected",
    "gas_chemical": "gas or chemical hazard",
    "glass_damage": "glass damage",
    "infestation": "pest infestation",
    "landscaping": "landscaping issue",
    "lighting": "lighting failure",
    "low_voltage_data": "low-voltage/data issue",
    "malfunction": "equipment malfunction",
    "minor_issue": "minor maintenance issue",
    "no_cooling": "no cooling",
    "no_heating": "no heating",
    "panel_hazard": "electrical panel hazard",
    "parking_lighting": "parking area lighting issue",
    "pavement_damage": "pavement damage",
    "pipe_leak": "pipe leak",
    "power_outage": "power outage",
    "refrigerant": "refrigerant issue",
    "restroom_fixture": "restroom fixture issue",
    "restroom_supplies": "restroom supply issue",
    "roof_leak": "roof leak",
    "signage_fencing": "signage/fencing issue",
    "slip_trip": "slip/trip hazard",
    "structural": "structural concern",
    "suspicious_person": "suspicious person reported",
    "unauthorized_access": "unauthorized access",
    "waste_odor": "waste/odor issue",
}


def generate_summary(
    *,
    subcategory: Optional[str] = None,
    risk_level: Optional[str] = None,
    building_name: Optional[str] = None,
    floor: Optional[str] = None,
    city: Optional[str] = None,
    vendor_name: Optional[str] = None,
    vendor_id: Optional[str] = None,
    dispatched_emergency_services: bool = False,
    needs_human_review: bool = False,
    unroutable: bool = False,
) -> str:
    """Generate a 2-3 sentence operator-style call summary.

    Returns:
        Summary string, capped at ~280 characters.
    """
    problem = _SUBCATEGORY_DESCRIPTIONS.get(subcategory or "", subcategory or "maintenance issue")
    problem = problem[0].upper() + problem[1:]

    location_parts = []
    if building_name:
        location_parts.append(building_name)
    if floor:
        location_parts.append(floor)
    if city and not building_name:
        location_parts.append(city)

    if location_parts:
        sentence1 = f"{problem} reported at {', '.join(location_parts)}."
    else:
        sentence1 = f"{problem} reported."

    sentence2 = f"{risk_level or 'MEDIUM'} risk."

    if dispatched_emergency_services:
        sentence3 = "Emergency services dispatched."
    elif unroutable or (vendor_id is None and needs_human_review):
        sentence3 = "Escalated to human review — no qualified vendor available."
    elif needs_human_review:
        if vendor_name:
            sentence3 = f"Flagged for review. {vendor_name} dispatched pending approval."
        else:
            sentence3 = "Flagged for human review prior to dispatch."
    elif vendor_name:
        sentence3 = f"Dispatched {vendor_name}."
    elif vendor_id:
        sentence3 = f"Dispatched vendor {vendor_id}."
    else:
        sentence3 = "Routed for dispatch."

    summary = f"{sentence1} {sentence2} {sentence3}"

    if len(summary) > _MAX_CHARS:
        summary = _truncate(summary)

    return summary


def _truncate(text: str) -> str:
    """Truncate at last sentence boundary under the cap."""
    if len(text) <= _MAX_CHARS:
        return text
    sentences = text.split(". ")
    result = ""
    for s in sentences:
        candidate = f"{result}. {s}" if result else s
        if not candidate.endswith("."):
            candidate += "."
        if len(candidate) <= _MAX_CHARS:
            result = candidate.rstrip(".")
        else:
            break
    return (result + ".") if result else text[:_MAX_CHARS]
