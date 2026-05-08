"""Tests for `agent.data.risk` and `scripts.derive_risk_bands`.

The derivation script is run during the test to produce the table
in-memory, so the unit tests don't depend on the committed JSON file
being current with the historicals.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data import risk, taxonomy  # noqa: E402

# Import the derivation script as a module to test the pure logic.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "derive_risk_bands",
    Path(__file__).resolve().parents[1] / "scripts" / "derive_risk_bands.py",
)
derive_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(derive_mod)


# ─── Constants and helpers ─────────────────────────────────────────────


def test_risk_levels_canonical_order():
    assert risk.RISK_LEVELS == ("LOW", "MEDIUM", "HIGH", "EMERGENCY")
    assert risk.RISK_LEVEL_RANK["LOW"] == 0
    assert risk.RISK_LEVEL_RANK["EMERGENCY"] == 3


def test_compare_risk_returns_signed_difference():
    assert risk.compare_risk("EMERGENCY", "LOW") > 0
    assert risk.compare_risk("LOW", "EMERGENCY") < 0
    assert risk.compare_risk("MEDIUM", "MEDIUM") == 0


def test_is_valid_risk_level():
    assert risk.is_valid_risk_level("LOW") is True
    assert risk.is_valid_risk_level("CRITICAL") is False
    assert risk.is_valid_risk_level("") is False


# ─── Pure derivation logic ─────────────────────────────────────────────


def test_derive_modal_breaks_ties_to_lower_severity():
    """When two risk levels tie in count, pick the LESS severe one —
    don't auto-escalate ambiguous cases."""
    from collections import Counter
    c = Counter({"MEDIUM": 5, "HIGH": 5, "LOW": 1})
    assert derive_mod._modal(c) == "MEDIUM"
    c2 = Counter({"LOW": 3, "MEDIUM": 3})
    assert derive_mod._modal(c2) == "LOW"


def test_derive_modal_handles_empty_counter():
    from collections import Counter
    assert derive_mod._modal(Counter()) == "MEDIUM"


def test_derive_table_shape_per_subcategory():
    records = [
        {"final_subcategory": "pipe_leak", "final_risk_level": "MEDIUM", "intake_risk_level": "MEDIUM"},
        {"final_subcategory": "pipe_leak", "final_risk_level": "MEDIUM", "intake_risk_level": "HIGH"},
        {"final_subcategory": "pipe_leak", "final_risk_level": "LOW",    "intake_risk_level": "LOW"},
        {"final_subcategory": "pipe_leak", "final_risk_level": "HIGH",   "intake_risk_level": "HIGH"},
    ]
    table = derive_mod.derive(records)
    assert "pipe_leak" in table
    e = table["pipe_leak"]
    assert e["n"] == 4
    assert e["modal_risk"] == "MEDIUM"  # 2 MEDIUM, 1 LOW, 1 HIGH
    # Distribution sums to 1 within rounding tolerance.
    assert abs(sum(e["distribution"].values()) - 1.0) < 1e-6
    # 1 of 4 was over-escalated (intake HIGH > final MEDIUM).
    assert e["intake_over_escalation_rate"] == 0.25
    assert e["intake_under_escalation_rate"] == 0.0


def test_derive_skips_records_with_no_subcategory():
    records = [
        {"final_subcategory": None, "final_risk_level": "MEDIUM"},
        {"final_subcategory": "pipe_leak", "final_risk_level": "LOW", "intake_risk_level": "LOW"},
    ]
    table = derive_mod.derive(records)
    assert table.keys() == {"pipe_leak"}
    assert table["pipe_leak"]["n"] == 1


def test_derive_handles_missing_intake_risk_gracefully():
    records = [
        {"final_subcategory": "pipe_leak", "final_risk_level": "MEDIUM"},  # no intake
        {"final_subcategory": "pipe_leak", "final_risk_level": "LOW", "intake_risk_level": "LOW"},
    ]
    table = derive_mod.derive(records)
    e = table["pipe_leak"]
    # Both records counted in distribution; only 1 had a comparable intake.
    assert e["n"] == 2
    assert e["intake_agreement_rate"] == 1.0


# ─── Runtime loader ────────────────────────────────────────────────────


def test_loader_table_covers_full_taxonomy():
    """Every subcategory the classifier can pick must have a base risk."""
    derived = set(risk.all_subcategories_in_table())
    canonical = set(taxonomy.all_subcategories())
    missing = canonical - derived
    assert not missing, (
        f"derived risk table is missing {missing} — re-run "
        "`python scripts/derive_risk_bands.py` after a corpus refresh"
    )


def test_loader_emergency_subcategories():
    """Sanity: known life-safety subcategories should have modal EMERGENCY."""
    for sub in ("active_threat", "fire_smoke", "gas_chemical"):
        assert risk.base_risk_for(sub) == "EMERGENCY", f"{sub} should be EMERGENCY"


def test_loader_routine_subcategories_are_low():
    for sub in ("access_control", "lighting", "carpet_floor", "restroom_supplies"):
        assert risk.base_risk_for(sub) == "LOW", f"{sub} should be LOW"


def test_loader_unknown_subcategory_returns_none():
    assert risk.base_risk_for("definitely_not_real") is None
    assert risk.base_risk_for(None) is None
    assert risk.base_risk_for("") is None


def test_loader_distribution_sums_to_one():
    for sub in taxonomy.all_subcategories():
        d = risk.risk_distribution(sub)
        assert d is not None, f"missing distribution for {sub}"
        assert abs(sum(d.values()) - 1.0) < 1e-4, (
            f"distribution for {sub} doesn't sum to 1: {d}"
        )


def test_loader_known_high_over_escalation_subcategories():
    # The four trap-prone subcategories the classifier work uncovered.
    # Each should have a meaningfully-elevated over-escalation rate.
    for sub in ("air_quality", "waste_odor", "suspicious_person", "roof_leak"):
        rate = risk.intake_over_escalation_rate(sub)
        assert rate > 0.05, (
            f"{sub} over-escalation rate {rate} should be > 5% — historical "
            "intake operators frequently over-stated severity here"
        )


def test_loader_under_escalation_for_unknown_subcategory_is_zero():
    assert risk.intake_under_escalation_rate("definitely_not_real") == 0.0
    assert risk.intake_under_escalation_rate(None) == 0.0


# ─── Inline runner ─────────────────────────────────────────────────────


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
