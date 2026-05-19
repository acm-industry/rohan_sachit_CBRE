"""Unit tests for `agent.nodes.location`.

Tests location reconciliation logic against the real operational data,
including the profile active/stale gate and the out-of-range-floor
clarification path (issue #19 ACs).

`_NOW` pins the reference time so the staleness check is deterministic
(issue #28) and the suite doesn't rot as wall-clock advances.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data.profiles import Profile  # noqa: E402
from agent.nodes.location import reconcile  # noqa: E402


# ─── Fixtures ──────────────────────────────────────────────────────────

# Pinned reference "now" for deterministic staleness checks.
_NOW = datetime(2025, 3, 5, tzinfo=timezone.utc)

_PROFILE_LEO = Profile(
    caller_name="Leo Green",
    phone_number="+1-310-555-0142",
    email="leo.green@example.com",
    tenant_company="Sherwin Finance Group",
    tenant_id="t_001",
    contact_role="tenant",
    preferred_language="en",
    primary_building_id="bld_001",
    primary_building_name="Westfield Commerce Center",
    primary_address="100 Wilshire Blvd",
    primary_city="Los Angeles",
    primary_floor="Floor 5",
    primary_suite="Suite 503",
    primary_building_type="office",
    last_verified_at="2025-01-15T10:00:00-08:00",  # ~49 days before _NOW → fresh
    active=True,
)

# Same profile, but verified years ago → stale relative to _NOW.
_PROFILE_LEO_STALE = Profile(
    **{**_PROFILE_LEO.__dict__, "last_verified_at": "2022-10-04T10:00:00-08:00"}
)

# Same profile, deactivated by the tenant.
_PROFILE_LEO_INACTIVE = Profile(**{**_PROFILE_LEO.__dict__, "active": False})


# ─── Tests: transcript-stated building ─────────────────────────────────


def test_transcript_building_resolves_via_registry():
    """Transcript says a building name; registry resolves it to canonical."""
    loc = reconcile(
        extracted_building_name="Pacific Ridge Medical Plaza",
        extracted_floor="Floor 7",
        building_confidence=0.95,
        floor_confidence=0.95,
    )
    assert loc.building_name == "Pacific Ridge Medical Plaza"
    assert loc.address == "3200 Pacific Coast Hwy"
    assert loc.city == "Long Beach"
    assert loc.floor == "Floor 7"
    assert loc.source_building == "transcript"
    assert loc.source_floor == "transcript"
    assert loc.building_type == "medical"


def test_transcript_building_substring_match():
    """Transcript says short form; registry resolves via substring."""
    loc = reconcile(
        extracted_building_name="Pacific Ridge",
        extracted_floor="Floor 3",
        building_confidence=0.8,
        floor_confidence=0.9,
    )
    assert loc.building_name == "Pacific Ridge Medical Plaza"
    assert loc.source_building == "transcript"


def test_transcript_building_unknown_uses_raw_name():
    """Transcript says a building not in the registry; use as-is."""
    loc = reconcile(
        extracted_building_name="Mystery Building XYZ",
        extracted_floor="Floor 2",
        building_confidence=0.8,
        floor_confidence=0.9,
    )
    assert loc.building_name == "Mystery Building XYZ"
    assert loc.address is None
    assert loc.source_building == "transcript"


# ─── Tests: profile fallback (active + not stale) ────────────────────


def test_no_extraction_falls_back_to_profile():
    """No building from extraction; a usable profile provides the building."""
    loc = reconcile(
        extracted_building_name=None,
        extracted_floor=None,
        building_confidence=0.0,
        floor_confidence=0.0,
        profile=_PROFILE_LEO,
        now=_NOW,
    )
    assert loc.building_name == "Westfield Commerce Center"
    assert loc.address == "100 Wilshire Blvd"
    assert loc.city == "Los Angeles"
    assert loc.floor == "Floor 5"
    assert loc.source_building == "profile"
    assert loc.source_floor == "profile"


def test_low_confidence_extraction_falls_back_to_profile():
    """Extraction has a building name but low confidence; use profile."""
    loc = reconcile(
        extracted_building_name="Some Guess",
        extracted_floor="Floor 99",
        building_confidence=0.2,
        floor_confidence=0.2,
        profile=_PROFILE_LEO,
        now=_NOW,
    )
    assert loc.building_name == "Westfield Commerce Center"
    assert loc.floor == "Floor 5"
    assert loc.source_building == "profile"
    assert loc.source_floor == "profile"


# ─── Tests: profile active/stale gate (issue #19 AC) ─────────────────


def test_inactive_profile_is_ignored():
    """A deactivated profile is not trusted; precedence falls through to
    the anonymous fallback (None)."""
    loc = reconcile(
        extracted_building_name=None,
        extracted_floor=None,
        profile=_PROFILE_LEO_INACTIVE,
        now=_NOW,
    )
    assert loc.building_name is None
    assert loc.floor is None
    assert loc.source_building == "none"
    assert loc.source_floor == "none"


def test_stale_profile_is_ignored():
    """A profile last verified years ago is stale → not used as a source."""
    loc = reconcile(
        extracted_building_name=None,
        extracted_floor=None,
        profile=_PROFILE_LEO_STALE,
        now=_NOW,
    )
    assert loc.building_name is None
    assert loc.address is None
    assert loc.floor is None
    assert loc.source_building == "none"
    assert loc.source_floor == "none"


def test_stale_profile_does_not_block_transcript():
    """Even with a stale profile, an explicit transcript building still
    resolves normally (transcript is independent of the profile gate)."""
    loc = reconcile(
        extracted_building_name="Pacific Ridge Medical Plaza",
        extracted_floor="Floor 3",
        building_confidence=0.95,
        floor_confidence=0.95,
        profile=_PROFILE_LEO_STALE,
        now=_NOW,
    )
    assert loc.building_name == "Pacific Ridge Medical Plaza"
    assert loc.source_building == "transcript"
    # Address must come from the registry, NOT the stale profile.
    assert loc.address == "3200 Pacific Coast Hwy"


# ─── Tests: transcript overrides profile ─────────────────────────────


def test_transcript_overrides_profile_building():
    """Transcript says a different building than profile; transcript wins."""
    loc = reconcile(
        extracted_building_name="Summit Office Towers",
        extracted_floor="Floor 9",
        building_confidence=0.95,
        floor_confidence=1.0,
        profile=_PROFILE_LEO,
        now=_NOW,
    )
    assert loc.building_name == "Summit Office Towers"
    assert loc.address == "800 Summit Blvd"
    assert loc.city == "San Francisco"
    assert loc.floor == "Floor 9"
    assert loc.source_building == "transcript"
    assert loc.source_floor == "transcript"


def test_transcript_floor_overrides_profile_floor():
    """Same building but transcript says different floor."""
    loc = reconcile(
        extracted_building_name=None,
        extracted_floor="Floor 9",
        building_confidence=0.0,
        floor_confidence=0.95,
        profile=_PROFILE_LEO,
        now=_NOW,
    )
    assert loc.building_name == "Westfield Commerce Center"
    assert loc.floor == "Floor 9"
    assert loc.source_building == "profile"
    assert loc.source_floor == "transcript"


# ─── Tests: floor validation + clarification ─────────────────────────


def test_floor_in_range_returns_ok_no_clarification():
    """Floor within building's floor_count."""
    loc = reconcile(
        extracted_building_name="Westfield Commerce Center",
        extracted_floor="Floor 5",
        building_confidence=0.95,
        floor_confidence=0.95,
    )
    assert loc.floor_check == "ok"
    assert loc.needs_clarification is False


def test_floor_out_of_range_sets_needs_clarification():
    """Floor exceeds building's floor_count (bld_001 has 18 floors):
    the floor is kept verbatim (not silently corrected) and
    needs_clarification is raised."""
    loc = reconcile(
        extracted_building_name="Westfield Commerce Center",
        extracted_floor="Floor 25",
        building_confidence=0.95,
        floor_confidence=0.95,
    )
    assert loc.floor_check == "out_of_range"
    assert loc.needs_clarification is True
    assert loc.floor == "Floor 25"  # kept, NOT corrected


def test_floor_unknown_when_building_lacks_floor_count():
    """Building without floor_count (sparse data) returns unknown — and
    unknown is not a clarification trigger (we can't prove it's wrong)."""
    loc = reconcile(
        extracted_building_name="Central Tower",
        extracted_floor="Floor 10",
        building_confidence=0.95,
        floor_confidence=0.95,
    )
    assert loc.floor_check == "unknown"
    assert loc.needs_clarification is False


# ─── Tests: anonymous caller ────────────────────────────────────────


def test_anonymous_caller_no_profile_no_extraction():
    """No info at all → all None."""
    loc = reconcile()
    assert loc.building_name is None
    assert loc.address is None
    assert loc.floor is None
    assert loc.city is None
    assert loc.source_building == "none"
    assert loc.source_floor == "none"
    assert loc.needs_clarification is False


def test_anonymous_caller_with_transcript_building():
    """Anonymous caller but transcript states building clearly."""
    loc = reconcile(
        extracted_building_name="Torrance Gateway Center",
        extracted_floor="Floor 2",
        building_confidence=1.0,
        floor_confidence=1.0,
    )
    assert loc.building_name == "Torrance Gateway Center"
    assert loc.address == "3500 Carson St"
    assert loc.floor == "Floor 2"
    assert loc.source_building == "transcript"


# ─── Tests: address/city from registry over profile ─────────────────


def test_registry_address_overrides_profile_address():
    """When registry resolves, its address wins even if profile differs."""
    loc = reconcile(
        extracted_building_name="Pacific Ridge Medical Plaza",
        extracted_floor="Floor 3",
        building_confidence=0.9,
        floor_confidence=0.9,
        profile=_PROFILE_LEO,
        now=_NOW,
    )
    assert loc.address == "3200 Pacific Coast Hwy"
    assert loc.city == "Long Beach"


if __name__ == "__main__":
    import inspect

    tests = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__])
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
