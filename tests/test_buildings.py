"""Unit tests for `agent.data.buildings`.

Anchored on real data: bld_001 (Westfield Commerce Center, 18 floors, has
basement + rooftop) and bld_007 (Central Tower, missing floor_count) plus
bld_018 (Gateway Industrial Complex, has Mezzanine + loading docks).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data import buildings  # noqa: E402


# --- get_by_id / get_by_name -------------------------------------------------


def test_get_by_id_known():
    b = buildings.get_by_id("bld_001")
    assert b is not None
    assert b.name == "Westfield Commerce Center"
    assert b.address == "100 Wilshire Blvd"
    assert b.city == "Los Angeles"
    assert b.floor_count == 18
    assert b.has_basement is True


def test_get_by_id_unknown_returns_none():
    assert buildings.get_by_id("bld_does_not_exist") is None
    assert buildings.get_by_id(None) is None
    assert buildings.get_by_id("") is None


def test_get_by_name_exact():
    b = buildings.get_by_name("Westfield Commerce Center")
    assert b is not None and b.building_id == "bld_001"


def test_get_by_name_case_insensitive():
    b = buildings.get_by_name("WESTFIELD commerce center")
    assert b is not None and b.building_id == "bld_001"


def test_get_by_name_substring_match():
    # Caller says the short form; catalog has the canonical longer name.
    b = buildings.get_by_name("Westfield")
    assert b is not None and b.building_id == "bld_001"


def test_get_by_name_unknown_returns_none():
    assert buildings.get_by_name("Nonexistent Plaza") is None
    assert buildings.get_by_name(None) is None
    assert buildings.get_by_name("") is None


# --- validate_floor: the three required outcomes ----------------------------


def test_validate_floor_ok_in_range():
    b = buildings.get_by_id("bld_001")  # 18 floors
    assert buildings.validate_floor(b, "Floor 5") == "ok"
    assert buildings.validate_floor(b, "Floor 1") == "ok"
    assert buildings.validate_floor(b, "Floor 18") == "ok"


def test_validate_floor_out_of_range():
    b = buildings.get_by_id("bld_001")  # 18 floors
    assert buildings.validate_floor(b, "Floor 19") == "out_of_range"
    assert buildings.validate_floor(b, "Floor 100") == "out_of_range"
    assert buildings.validate_floor(b, "Floor 0") == "out_of_range"


def test_validate_floor_unknown_when_floor_count_missing():
    b = buildings.get_by_id("bld_007")  # Central Tower — no floor_count
    assert b is not None and b.floor_count is None
    assert buildings.validate_floor(b, "Floor 5") == "unknown"
    assert buildings.validate_floor(b, "Floor 9999") == "unknown"


def test_validate_floor_unknown_when_building_missing():
    assert buildings.validate_floor(None, "Floor 5") == "unknown"


def test_validate_floor_unknown_when_floor_str_missing():
    b = buildings.get_by_id("bld_001")
    assert buildings.validate_floor(b, None) == "unknown"
    assert buildings.validate_floor(b, "") == "unknown"


# --- validate_floor: special floors / basement / rooftop --------------------


def test_validate_floor_ground_floor_is_ok():
    b = buildings.get_by_id("bld_001")
    assert buildings.validate_floor(b, "Ground Floor") == "ok"


def test_validate_floor_basement_ok_when_building_has_one():
    b = buildings.get_by_id("bld_001")  # has_basement=True
    assert buildings.validate_floor(b, "Basement") == "ok"
    assert buildings.validate_floor(b, "B1") == "ok"


def test_validate_floor_rooftop_ok_when_building_has_one():
    b = buildings.get_by_id("bld_001")  # has_rooftop=True
    assert buildings.validate_floor(b, "Rooftop") == "ok"
    assert buildings.validate_floor(b, "Roof") == "ok"


def test_validate_floor_mezzanine_ok_for_special_floor_building():
    b = buildings.get_by_id("bld_018")  # special_floors includes Mezzanine
    assert b is not None and "Mezzanine" in b.special_floors
    assert buildings.validate_floor(b, "Mezzanine") == "ok"
    assert buildings.validate_floor(b, "Loading Dock 1") == "ok"


def test_validate_floor_unparseable_returns_unknown():
    # Caller said something we can't interpret. With floor_count present,
    # we still can't *confirm* it's wrong → unknown (soft-pass).
    b = buildings.get_by_id("bld_001")
    assert buildings.validate_floor(b, "the upstairs bit") == "unknown"


# --- resolve_address_city ---------------------------------------------------


def test_resolve_prefers_registry_over_profile():
    b = buildings.get_by_id("bld_001")
    addr, city = buildings.resolve_address_city(
        building=b,
        profile_address="OUTDATED ADDRESS",
        profile_city="OUTDATED CITY",
    )
    assert addr == "100 Wilshire Blvd"
    assert city == "Los Angeles"


def test_resolve_falls_back_to_profile_when_no_building():
    addr, city = buildings.resolve_address_city(
        building=None,
        profile_address="500 Mission St",
        profile_city="San Francisco",
    )
    assert addr == "500 Mission St"
    assert city == "San Francisco"


def test_resolve_returns_none_pair_when_nothing_known():
    addr, city = buildings.resolve_address_city()
    assert addr is None
    assert city is None


if __name__ == "__main__":
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
