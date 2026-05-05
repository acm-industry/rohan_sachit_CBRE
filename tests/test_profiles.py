"""Unit tests for `agent.data.profiles`.

Runs under pytest (`pytest tests/test_profiles.py`) or directly
(`python tests/test_profiles.py`). Uses fixtures from the real
`operational/caller_profiles.json` rather than mocks — the file is small
(250 rows) and the data shape is stable enough that an integration-flavored
unit test catches schema drift cheaply.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python tests/test_profiles.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data import profiles  # noqa: E402


# Two real phone numbers from the dataset — pinned via the data survey on
# this branch (250 profiles, 25 inactive). If the dataset is ever
# regenerated and these specific numbers move, swap them out.
KNOWN_ACTIVE_PHONE = "+1-310-555-0142"      # Leo Green, verified 2024-04-07, active
KNOWN_INACTIVE_PHONE = "+1-562-555-0198"    # Jasmine Clark, verified 2022-03-15, inactive
UNKNOWN_PHONE = "+1-000-000-0000"           # not in dataset


def test_lookup_known_active_returns_profile():
    p = profiles.lookup(KNOWN_ACTIVE_PHONE)
    assert p is not None
    assert p.caller_name == "Leo Green"
    assert p.tenant_company == "Sherwin Finance Group"
    assert p.primary_building_id == "bld_001"
    assert p.primary_address == "100 Wilshire Blvd"
    assert p.primary_floor == "Floor 5"
    assert p.last_verified_at == "2024-04-07T10:00:00-08:00"
    assert p.active is True


def test_lookup_inactive_returns_none_by_default():
    assert profiles.lookup(KNOWN_INACTIVE_PHONE) is None


def test_lookup_inactive_with_include_inactive_returns_profile():
    p = profiles.lookup(KNOWN_INACTIVE_PHONE, include_inactive=True)
    assert p is not None
    assert p.active is False
    assert p.caller_name == "Jasmine Clark"


def test_lookup_unknown_returns_none():
    assert profiles.lookup(UNKNOWN_PHONE) is None


def test_lookup_anonymous_caller_returns_none():
    assert profiles.lookup(None) is None
    assert profiles.lookup("") is None


def test_lookup_handles_whitespace():
    p = profiles.lookup("  " + KNOWN_ACTIVE_PHONE + "  ")
    assert p is not None and p.caller_name == "Leo Green"


def test_is_stale_old_profile_is_stale():
    p = profiles.lookup(KNOWN_INACTIVE_PHONE, include_inactive=True)
    # Verified 2022-03-15; well past 180 days ago against today.
    assert profiles.is_stale(p) is True


def test_is_stale_recent_profile_is_fresh():
    p = profiles.lookup(KNOWN_ACTIVE_PHONE)
    # Pin "now" to a date 30 days after the profile's last_verified_at to
    # decouple this test from the wall clock.
    fixed_now = datetime(2024, 5, 7, 10, 0, 0, tzinfo=timezone.utc)
    assert profiles.is_stale(p, now=fixed_now) is False


def test_is_stale_treats_missing_date_as_stale():
    from dataclasses import replace
    p = profiles.lookup(KNOWN_ACTIVE_PHONE)
    p_no_date = replace(p, last_verified_at=None)
    assert profiles.is_stale(p_no_date) is True


def test_is_stale_threshold_is_tunable():
    p = profiles.lookup(KNOWN_ACTIVE_PHONE)
    # 1-day cutoff against a far-future "now" → stale
    fixed_now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert profiles.is_stale(p, max_age_days=1, now=fixed_now) is True


if __name__ == "__main__":
    # Tiny inline harness so the file runs without pytest installed.
    import inspect

    tests = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__])
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
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
