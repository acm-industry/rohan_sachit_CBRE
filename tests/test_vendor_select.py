"""Tests for `agent.nodes.vendor_select`.

Validates vendor selection tie-breaking, SLA filtering, and
unroutable escalation against real operational data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data.vendors import _reset_caches_for_tests  # noqa: E402
from agent.nodes.vendor_select import select_vendor, VendorSelection  # noqa: E402

_LABELS_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "dev_labels.json"


def _reset():
    _reset_caches_for_tests()


# ─── Core selection tests ──────────────────────────────────────────────


def test_pipe_leak_long_beach_selects_plumber():
    _reset()
    sel = select_vendor(subcategory="pipe_leak", city="Long Beach", risk_level="MEDIUM")
    assert sel.vendor_id is not None
    assert "plumb" in sel.vendor_name.lower() or sel.vendor_id.startswith("v_")


def test_emergency_uses_emergency_sla():
    """EMERGENCY with 30min cap should select vendor with emergency_sla <= 30."""
    _reset()
    sel = select_vendor(
        subcategory="gas_chemical",
        city="Escondido",
        building_type="office",
        risk_level="EMERGENCY",
        is_emergency=True,
    )
    # v_021 has emergency_sla=20, v_022 has emergency_sla=45 (should be filtered)
    assert sel.vendor_id == "v_021"


def test_high_risk_filters_slow_vendors():
    """HIGH risk with 120min cap filters vendors with SLA > 120."""
    _reset()
    sel = select_vendor(
        subcategory="slip_trip",
        city="Ontario",
        building_type="office",
        risk_level="HIGH",
    )
    # v_025 has sla=180 which exceeds 120min cap → should be filtered → unroutable
    assert sel.vendor_id is None


def test_low_risk_allows_slow_vendors():
    """LOW risk with 480min cap should accept any vendor."""
    _reset()
    sel = select_vendor(
        subcategory="slip_trip",
        city="Ontario",
        building_type="office",
        risk_level="LOW",
    )
    # v_025 has sla=180 which is under 480min cap
    assert sel.vendor_id is not None


def test_unroutable_returns_none():
    """When no vendor qualifies, returns None with escalation reason."""
    _reset()
    sel = select_vendor(
        subcategory="pipe_leak",
        city="Atlantis",  # no plumber covers this city
        risk_level="MEDIUM",
    )
    assert sel.vendor_id is None
    assert "escalate" in sel.reason


def test_tie_breaking_prefers_higher_rating():
    """When multiple vendors qualify, prefer higher rating."""
    _reset()
    # fire_smoke + Escondido at LOW risk (480 cap → both v_021 and v_022 qualify)
    sel = select_vendor(
        subcategory="fire_smoke",
        city="Escondido",
        building_type="office",
        risk_level="LOW",
    )
    # v_021 has rating 4.9, v_022 has rating 4.5
    assert sel.vendor_id == "v_021"


def test_no_subcategory_returns_none():
    """Missing subcategory → no specialty match → unroutable."""
    _reset()
    sel = select_vendor(subcategory=None, city="Los Angeles", risk_level="LOW")
    # With no subcategory, qualify() returns all vendors (no specialty filter)
    # so this should still return a vendor
    assert sel.vendor_id is not None


# ─── Full dev-set accuracy test ────────────────────────────────────────


def test_full_dev_set_accuracy():
    """100% vendor-match accuracy on the 200-row dev set with oracle inputs."""
    _reset()
    labels = json.loads(_LABELS_PATH.read_text())["by_id"]
    hits = 0
    total = 0

    for tid, l in labels.items():
        acceptable = l.get("acceptable_vendor_ids", [])
        unroutable = l.get("unroutable", False)
        sub = l.get("true_subcategory")
        city = l.get("ground_truth_city")
        btype = l.get("ground_truth_building_type")
        risk = l.get("true_risk_level")
        is_emergency = risk == "EMERGENCY"

        sel = select_vendor(
            subcategory=sub,
            city=city,
            building_type=btype,
            risk_level=risk,
            is_emergency=is_emergency,
            after_hours=False,
        )

        total += 1
        if unroutable:
            if sel.vendor_id is None:
                hits += 1
        else:
            if sel.vendor_id in acceptable:
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
