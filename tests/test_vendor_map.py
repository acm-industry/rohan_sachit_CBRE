"""Tests for the derived subcategory → vendor_type mapping.

Validates the committed JSON artifact and its integration with
`agent.data.vendors.qualify()`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data import taxonomy  # noqa: E402
from agent.data.vendors import qualify, _reset_caches_for_tests  # noqa: E402

_MAP_PATH = Path(__file__).resolve().parents[1] / "agent" / "data" / "derived" / "subcategory_to_vendor_type.json"


def test_map_file_exists():
    assert _MAP_PATH.exists(), f"Derived map not found at {_MAP_PATH}"


def test_map_covers_all_taxonomy_subcategories():
    """Every subcategory in the canonical taxonomy has a vendor_type mapping."""
    mapping = json.loads(_MAP_PATH.read_text())
    all_subs = set(taxonomy.all_subcategories())
    mapped_subs = set(mapping.keys())
    missing = all_subs - mapped_subs
    assert not missing, f"Subcategories missing from vendor map: {missing}"


def test_map_values_are_nonempty_lists():
    mapping = json.loads(_MAP_PATH.read_text())
    for sub, types in mapping.items():
        assert isinstance(types, list), f"{sub}: expected list, got {type(types)}"
        assert len(types) >= 1, f"{sub}: vendor_type list is empty"


def test_map_vendor_types_exist_in_catalog():
    """Every vendor_type in the map has at least one vendor in the catalog."""
    from agent.data.vendors import all_vendors
    mapping = json.loads(_MAP_PATH.read_text())
    catalog_types = {v.vendor_type for v in all_vendors()}
    for sub, types in mapping.items():
        for vt in types:
            assert vt in catalog_types, (
                f"{sub} maps to vendor_type={vt!r} which has no vendors in the catalog"
            )


def test_qualify_uses_loose_mapping_for_subcategory():
    """qualify() returns vendors via the vendor_type mapping, not just specialties."""
    _reset_caches_for_tests()
    # pipe_leak → plumber vendor_type
    results = qualify(subcategory="pipe_leak")
    assert len(results) > 0
    assert all(v.vendor_type == "plumber" for v in results)


def test_qualify_fire_smoke_returns_fire_life_safety():
    _reset_caches_for_tests()
    results = qualify(subcategory="fire_smoke")
    assert len(results) > 0
    assert all(v.vendor_type == "fire_life_safety" for v in results)


def test_qualify_lighting_returns_facilities():
    _reset_caches_for_tests()
    results = qualify(subcategory="lighting")
    assert len(results) > 0
    assert all(v.vendor_type == "facilities" for v in results)


def test_qualify_loose_path_actually_contributes_a_vendor():
    """Isolate the loose vendor_type path: prove qualify() includes at
    least one vendor that matched ONLY via the derived map (its
    `specialties` does NOT list the subcategory). `refrigerant` → hvac
    has exactly such a vendor in the real catalog, so a regression that
    ignored the derived map would drop it and fail here."""
    _reset_caches_for_tests()
    sub = "refrigerant"
    mapping = json.loads(_MAP_PATH.read_text())
    mapped_types = set(mapping[sub])
    results = qualify(subcategory=sub)
    loose_only = [
        v for v in results
        if sub not in v.specialties and v.vendor_type in mapped_types
    ]
    assert loose_only, (
        "no vendor qualified purely via the derived vendor_type map for "
        f"{sub!r} — the loose path is not being consulted"
    )


def test_known_mappings_spot_check():
    """Spot-check a few well-known mappings."""
    mapping = json.loads(_MAP_PATH.read_text())
    assert mapping["pipe_leak"] == ["plumber"]
    assert mapping["fire_smoke"] == ["fire_life_safety"]
    assert mapping["malfunction"] == ["elevator_service"]
    assert mapping["air_quality"] == ["hvac"]
    assert mapping["waste_odor"] == ["janitorial"]
    assert mapping["active_threat"] == ["security"]
    assert mapping["roof_leak"] == ["roofer"]


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
