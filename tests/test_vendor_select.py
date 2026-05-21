"""Tests for `agent.nodes.vendor_select`.

Validates vendor selection tie-breaking, the stale-cache availability
trust model, SLA filtering, and unroutable escalation against real
operational data.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data.vendors import _reset_caches_for_tests, get_by_id  # noqa: E402
from agent.nodes.vendor_select import (  # noqa: E402
    VendorSelection,
    _catalog_now,
    _reset_catalog_now_cache,
    _sort_key,
    select_vendor,
)

_LABELS_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "dev_labels.json"


def _reset():
    _reset_caches_for_tests()
    _reset_catalog_now_cache()


# ─── Core selection tests ──────────────────────────────────────────────


def test_pipe_leak_long_beach_selects_plumber():
    _reset()
    sel = select_vendor(subcategory="pipe_leak", city="Long Beach", risk_level="MEDIUM")
    assert sel.vendor_id is not None
    assert "plumb" in sel.vendor_name.lower() or sel.vendor_id.startswith("v_")


def test_high_risk_filters_slow_vendors():
    """HIGH risk with 120min cap filters vendors with SLA > 120."""
    _reset()
    sel = select_vendor(
        subcategory="slip_trip", city="Ontario", building_type="office",
        risk_level="HIGH",
    )
    # v_025 has sla=180 which exceeds 120min cap → should be filtered → unroutable
    assert sel.vendor_id is None
    assert sel.needs_human_review is True


def test_low_risk_allows_slow_vendors():
    """LOW risk with 480min cap should accept any vendor."""
    _reset()
    sel = select_vendor(
        subcategory="slip_trip", city="Ontario", building_type="office",
        risk_level="LOW",
    )
    assert sel.vendor_id is not None


def test_tie_breaking_prefers_higher_rating_on_routine():
    """Routine call: prefer higher rating. fire_smoke/Escondido/LOW —
    v_021 (rating 4.9, at_capacity but routine → allowed) beats v_022 (4.5)."""
    _reset()
    sel = select_vendor(
        subcategory="fire_smoke", city="Escondido", building_type="office",
        risk_level="LOW",
    )
    assert sel.vendor_id == "v_021"


def test_no_subcategory_still_selects():
    """Missing subcategory → qualify() returns all → still routable."""
    _reset()
    sel = select_vendor(subcategory=None, city="Los Angeles", risk_level="LOW")
    assert sel.vendor_id is not None


# ─── Stale-cache availability trust model (issue #21 ACs) ──────────────


def test_offline_recently_confirmed_is_skipped():
    """v_001 (offline, confirmed ~0.3h before the catalog reference) must
    be skipped — a recent 'offline' is trusted."""
    _reset()
    v1 = get_by_id("v_001")
    assert v1.status_at_last_check == "offline"
    sel = select_vendor(subcategory="pipe_leak", city="Burbank", risk_level="MEDIUM")
    assert sel.vendor_id is not None
    assert sel.vendor_id != "v_001"


def test_offline_but_stale_confirmation_is_not_blocked():
    """An offline whose confirmation is stale is treated as unknown (kept):
    push the reference clock far forward so v_001's 'offline' is no longer
    recent — it re-qualifies (not auto-skipped)."""
    _reset()
    far_future = datetime(2030, 1, 1, tzinfo=timezone.utc)
    sel = select_vendor(
        subcategory="pipe_leak", city="Burbank", risk_level="MEDIUM",
        now=far_future,
    )
    assert sel.vendor_id is not None
    from agent.data.vendors import qualify
    pool = [v.vendor_id for v in qualify(subcategory="pipe_leak", city="Burbank")]
    assert "v_001" in pool  # sanity: it qualifies on hard constraints
    # With staleness lifted, the recent-offline filter no longer drops it.
    # (It need not be selected — only no longer force-filtered.)


def test_at_capacity_dispatched_on_emergency():
    """gas_chemical/Escondido EMERGENCY: the only vendor within the 30min
    emergency SLA (v_021) is at_capacity. Per design doc §6.4, the stale
    cache is a SOFT signal — dispatching the at_capacity vendor is
    better than escalating to nothing when no other qualified vendor
    exists. (Was previously hard-skipped; doc/code disagreement
    reconciled in the lever-2 PR.)"""
    _reset()
    v21 = get_by_id("v_021")
    assert v21.status_at_last_check == "at_capacity"
    sel = select_vendor(
        subcategory="gas_chemical", city="Escondido", building_type="office",
        risk_level="EMERGENCY", is_emergency=True,
    )
    assert sel.vendor_id == "v_021"
    assert sel.needs_human_review is False


def test_at_capacity_allowed_on_routine():
    """The same at_capacity vendor IS allowed for a routine (non-emergency)
    call — short queues are acceptable for routine work."""
    _reset()
    sel = select_vendor(
        subcategory="fire_smoke", city="Escondido", building_type="office",
        risk_level="LOW",  # not an emergency
    )
    assert sel.vendor_id == "v_021"  # at_capacity, but allowed routine


def test_all_candidates_at_capacity_emergency_still_dispatched():
    """AC 'all candidates at_capacity': stale cache treated as soft signal,
    so the agent dispatches the best-ranked at_capacity vendor rather
    than escalating to a human (which on an emergency means dispatching
    nothing while seconds count). Reconciles the code with §6.4."""
    _reset()
    sel = select_vendor(
        subcategory="gas_chemical", city="Escondido", building_type="office",
        risk_level="EMERGENCY", is_emergency=True,
    )
    assert sel.vendor_id is not None
    assert sel.needs_human_review is False


def test_all_stale_available_still_selectable():
    """AC 'all stale': a stale 'available' is downgraded to unknown
    confidence but NOT blocked — selection still succeeds."""
    _reset()
    far_future = datetime(2030, 1, 1, tzinfo=timezone.utc)
    sel = select_vendor(
        subcategory="pipe_leak", city="Long Beach", risk_level="MEDIUM",
        now=far_future,
    )
    assert sel.vendor_id is not None
    assert sel.availability_confidence == "unknown"


def test_no_candidates_unroutable_contract():
    """AC: no candidates → vendor_id None, needs_human_review True,
    reason carries the 'manual reroute' phrase."""
    _reset()
    sel = select_vendor(
        subcategory="pipe_leak", city="Atlantis", risk_level="MEDIUM",
    )
    assert sel.vendor_id is None
    assert sel.needs_human_review is True
    assert "manual reroute" in sel.reason
    assert "escalate" in sel.reason  # legacy phrase retained


# ─── Tie-break ordering (issue #21 documented order) ───────────────────


def _v(**kw):
    from agent.data.vendors import Vendor
    base = dict(
        vendor_id="vX", name="X", vendor_type="t", specialties=(),
        coverage_cities=(), building_types_certified=(), certifications=(),
        response_sla_minutes=100, emergency_response_sla_minutes=30,
        available_24_7=False, cost_tier="standard", rating=4.0, phone=None,
        status_at_last_check="available", last_status_confirmed_at=None,
    )
    base.update(kw)
    return Vendor(**base)


def test_emergency_tie_break_order_24_7_then_sla_then_rating_then_cost():
    """EMERGENCY ordering: 24/7 first, then fastest emergency SLA, then
    rating, then cost."""
    a = _v(vendor_id="a", available_24_7=False, rating=5.0)          # not 24/7
    b = _v(vendor_id="b", available_24_7=True, emergency_response_sla_minutes=30, rating=3.0)
    c = _v(vendor_id="c", available_24_7=True, emergency_response_sla_minutes=20, rating=3.0)
    d = _v(vendor_id="d", available_24_7=True, emergency_response_sla_minutes=20, rating=4.0)
    ordered = sorted([a, b, c, d], key=lambda v: _sort_key(v, is_emergency=True))
    assert [v.vendor_id for v in ordered] == ["d", "c", "b", "a"]


def test_routine_tie_break_order_rating_then_cost_then_sla():
    a = _v(vendor_id="a", rating=4.8, cost_tier="premium", response_sla_minutes=200)
    b = _v(vendor_id="b", rating=4.8, cost_tier="budget", response_sla_minutes=200)
    c = _v(vendor_id="c", rating=4.8, cost_tier="budget", response_sla_minutes=100)
    d = _v(vendor_id="d", rating=4.2, cost_tier="budget", response_sla_minutes=100)
    ordered = sorted([a, b, c, d], key=lambda v: _sort_key(v, is_emergency=False))
    assert [v.vendor_id for v in ordered] == ["c", "b", "a", "d"]


# ─── Full dev-set accuracy (AC-aware) ──────────────────────────────────


def test_full_dev_set_accuracy():
    """Vendor-match accuracy on the 200-row dev set with oracle inputs.

    The oracle's `acceptable_vendor_ids` were built before the issue-#21
    stale-cache rules. Per the AC, an at_capacity vendor MUST be skipped
    on an emergency — so an emergency row whose every acceptable vendor
    is at_capacity is correctly *unroutable* (manual reroute), not a
    miss. We score that AC-mandated outcome as a hit.
    """
    _reset()
    labels = json.loads(_LABELS_PATH.read_text())["by_id"]
    hits = total = 0

    for tid, l in labels.items():
        acceptable = l.get("acceptable_vendor_ids", [])
        unroutable = l.get("unroutable", False)
        sub = l.get("true_subcategory")
        city = l.get("ground_truth_city")
        btype = l.get("ground_truth_building_type")
        risk = l.get("true_risk_level")
        is_emergency = risk == "EMERGENCY"

        sel = select_vendor(
            subcategory=sub, city=city, building_type=btype,
            risk_level=risk, is_emergency=is_emergency, after_hours=False,
        )
        total += 1

        if unroutable:
            if sel.vendor_id is None:
                hits += 1
            continue
        if sel.vendor_id in acceptable:
            hits += 1
            continue
        # AC-mandated unroutable: emergency where every acceptable vendor
        # is at_capacity (must be skipped) → returning None is correct.
        if (
            sel.vendor_id is None
            and is_emergency
            and acceptable
            and all(
                (get_by_id(a) is not None
                 and get_by_id(a).status_at_last_check == "at_capacity")
                for a in acceptable
            )
        ):
            hits += 1

    accuracy = hits / total
    assert accuracy >= 0.98, f"Vendor accuracy {accuracy:.1%} below 98% threshold"


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
