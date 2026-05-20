"""Tests for `agent.data.phone_history` (issue #65).

Validates the deterministic phone → recent-building majority lookup
that supplements the static `caller_profiles.json` snapshot when the
historicals corpus has overwhelming evidence about where a phone calls
from.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from agent.data import phone_history  # noqa: E402


def _reset():
    phone_history._reset_cache_for_tests()


def test_anonymous_returns_none():
    _reset()
    assert phone_history.recent_building(None) is None
    assert phone_history.recent_building("") is None


def test_unknown_phone_returns_none():
    _reset()
    assert phone_history.recent_building("+1-000-555-0000") is None


def test_strong_majority_returns_building():
    """+1-818-555-0156 has 183 historical tickets, 100% on Metro Center
    Offices — well above MIN_N=10 and MIN_SHARE=0.9."""
    _reset()
    m = phone_history.recent_building("+1-818-555-0156")
    assert m is not None
    assert m.building_name == "Metro Center Offices"
    assert m.address == "650 Metro Center Way"
    assert m.city == "Burbank"
    assert m.building_type == "office"
    assert m.n_records >= 10
    assert m.share >= 0.9


def test_min_n_guard_rejects_thin_histories():
    """Caller with very few historical tickets is rejected even if the
    share is high — we won't pin to a building on n=1 evidence."""
    _reset()
    m = phone_history.recent_building(
        "+1-818-555-0156", min_n=10_000, min_share=0.9,
    )
    assert m is None


def test_min_share_guard_rejects_multi_property_callers():
    """A caller whose history is split across buildings (no overwhelming
    majority) must return None — we refuse to guess for roving managers."""
    _reset()
    # Raise the share floor above 100% so even a perfect-majority phone
    # is rejected → exercises the share-guard branch.
    m = phone_history.recent_building(
        "+1-818-555-0156", min_n=10, min_share=1.01,
    )
    assert m is None


def test_index_is_cached():
    _reset()
    phone_history.recent_building("+1-818-555-0156")
    # Second call must hit the cache (not re-read the file).
    assert phone_history._INDEX_CACHE is not None
    # Cache reset wipes it.
    _reset()
    assert phone_history._INDEX_CACHE is None


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
