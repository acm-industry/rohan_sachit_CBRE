"""Unit tests for `agent.data.vendors`.

Anchored on real data:
- v_001 Pacific Plumbing Services: plumber, specialties=[pipe_leak,
  restroom_fixture, drainage_backup], 24/7, covers LA + 6 cities.
- v_002 Metro Drain & Pipe: plumber, same specialties, NOT 24/7.
- 32 vendors total, every record has populated coverage_cities and
  building_types_certified — soft-pass paths get hit via synthetic
  Vendor objects.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data import vendors  # noqa: E402
from agent.data.vendors import Vendor  # noqa: E402


# ─── basic loading ─────────────────────────────────────────────────────────


def test_all_vendors_loads_full_catalog():
    assert len(vendors.all_vendors()) == 32


def test_get_by_id_known():
    v = vendors.get_by_id("v_001")
    assert v is not None
    assert v.name == "Pacific Plumbing Services"
    assert v.vendor_type == "plumber"
    assert "pipe_leak" in v.specialties
    assert v.available_24_7 is True


def test_get_by_id_unknown_returns_none():
    assert vendors.get_by_id("v_does_not_exist") is None
    assert vendors.get_by_id(None) is None


# ─── qualify: specialty filter ─────────────────────────────────────────────


def test_qualify_specialty_strict_match_includes_plumbers_for_pipe_leak():
    out = vendors.qualify(subcategory="pipe_leak")
    ids = {v.vendor_id for v in out}
    assert "v_001" in ids and "v_002" in ids
    assert all("pipe_leak" in v.specialties for v in out)


def test_qualify_unknown_subcategory_returns_empty():
    # No vendor has "definitely_not_a_real_thing" in specialties, and the
    # subcategory→vendor-type mapping is empty until issue #20 ships.
    out = vendors.qualify(subcategory="definitely_not_a_real_thing")
    assert out == []


def test_qualify_no_subcategory_returns_all():
    out = vendors.qualify()
    assert len(out) == 32


# ─── qualify: city filter ──────────────────────────────────────────────────


def test_qualify_city_in_coverage_includes_vendor():
    out = vendors.qualify(subcategory="pipe_leak", city="Los Angeles")
    ids = {v.vendor_id for v in out}
    assert "v_001" in ids


def test_qualify_city_not_in_coverage_excludes_vendor():
    # v_001 covers LA, Santa Monica, Burbank, Pasadena, North Hollywood,
    # Torrance, Long Beach. New York is not covered.
    out = vendors.qualify(subcategory="pipe_leak", city="New York")
    ids = {v.vendor_id for v in out}
    assert "v_001" not in ids


def test_qualify_city_soft_passes_when_coverage_empty():
    # Build a synthetic vendor with empty coverage_cities to exercise the
    # soft-pass path (real dataset has no such vendor).
    fake = Vendor(
        vendor_id="v_test",
        name="Universal Vendor",
        vendor_type="plumber",
        specialties=("pipe_leak",),
        coverage_cities=(),  # empty → no city filter applied
        building_types_certified=("office",),
    )
    # Stub the cache so qualify() sees only this vendor.
    vendors._ALL_CACHE = [fake]
    try:
        out = vendors.qualify(subcategory="pipe_leak", city="Anywhere")
        assert len(out) == 1 and out[0].vendor_id == "v_test"
    finally:
        vendors._reset_caches_for_tests()


# ─── qualify: building_type filter ─────────────────────────────────────────


def test_qualify_building_type_in_cert_includes_vendor():
    # v_001 certified for office, retail_office, parking
    out = vendors.qualify(subcategory="pipe_leak", building_type="office")
    ids = {v.vendor_id for v in out}
    assert "v_001" in ids


def test_qualify_building_type_not_in_cert_excludes_vendor():
    # v_001 is NOT certified for medical
    out = vendors.qualify(subcategory="pipe_leak", building_type="medical")
    ids = {v.vendor_id for v in out}
    assert "v_001" not in ids


def test_qualify_building_type_soft_passes_when_cert_empty():
    fake = Vendor(
        vendor_id="v_uncertified",
        name="Universal Vendor",
        vendor_type="plumber",
        specialties=("pipe_leak",),
        coverage_cities=("Los Angeles",),
        building_types_certified=(),  # empty → no cert filter applied
    )
    vendors._ALL_CACHE = [fake]
    try:
        out = vendors.qualify(
            subcategory="pipe_leak", city="Los Angeles", building_type="medical"
        )
        assert len(out) == 1
    finally:
        vendors._reset_caches_for_tests()


# ─── qualify: 24/7 filter ──────────────────────────────────────────────────


def test_qualify_24_7_filter_off_during_business_hours():
    # v_002 is NOT 24/7 but qualifies during business hours
    out = vendors.qualify(subcategory="pipe_leak", is_emergency=False, after_hours=False)
    ids = {v.vendor_id for v in out}
    assert "v_002" in ids


def test_qualify_24_7_filter_off_for_non_emergency_after_hours():
    # Non-emergency at 2am should still include non-24/7 vendors
    out = vendors.qualify(subcategory="pipe_leak", is_emergency=False, after_hours=True)
    ids = {v.vendor_id for v in out}
    assert "v_002" in ids


def test_qualify_24_7_filter_excludes_non_247_on_after_hours_emergency():
    out = vendors.qualify(subcategory="pipe_leak", is_emergency=True, after_hours=True)
    ids = {v.vendor_id for v in out}
    # v_001 is 24/7, qualifies. v_002 is NOT 24/7, must be excluded.
    assert "v_001" in ids
    assert "v_002" not in ids
    assert all(v.available_24_7 for v in out)


def test_qualify_24_7_filter_inactive_for_emergency_business_hours():
    # is_emergency=True but after_hours=False → filter does NOT apply
    out = vendors.qualify(subcategory="pipe_leak", is_emergency=True, after_hours=False)
    ids = {v.vendor_id for v in out}
    assert "v_002" in ids


# ─── qualify: combined filters + escalate path ─────────────────────────────


def test_qualify_combined_filters_returns_intersection():
    out = vendors.qualify(
        subcategory="pipe_leak",
        city="Long Beach",
        building_type="office",
        is_emergency=True,
        after_hours=True,
    )
    # All returned vendors must satisfy every constraint.
    for v in out:
        assert "pipe_leak" in v.specialties
        assert "Long Beach" in v.coverage_cities
        assert "office" in v.building_types_certified
        assert v.available_24_7


def test_qualify_returns_empty_when_nothing_satisfies():
    out = vendors.qualify(
        subcategory="pipe_leak",
        city="Boston",  # no plumber covers Boston in this dataset
    )
    assert out == []


# ─── subcategory→vendor-type mapping (issue #20 hook) ──────────────────────


def test_loose_mapping_empty_by_default():
    # Until issue #20 lands, the mapping is empty.
    assert vendors._subcat_to_vendor_types() == {}


def test_loose_mapping_extends_specialty_filter(monkeypatch=None):
    # Simulate the issue-#20 derived mapping by stubbing the cache.
    vendors._SUBCAT_MAPPING_CACHE = {"weird_new_subcat": ("plumber",)}
    try:
        out = vendors.qualify(subcategory="weird_new_subcat")
        # Every plumber should now qualify even though no one has the
        # specialty string.
        assert len(out) >= 4  # 4 plumbers in the dataset
        assert all(v.vendor_type == "plumber" for v in out)
    finally:
        vendors._reset_caches_for_tests()


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
