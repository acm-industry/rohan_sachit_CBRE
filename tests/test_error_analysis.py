"""Tests for `scripts.error_analysis` (issue #25).

Deterministic — reads the committed `eval_runs/dev_baseline.json` +
`evaluation/dev_labels.json`, no LLM/network. Guards the headline
claim: the per-axis breakdown must reconcile with the scoring composite
that produced `eval_runs/README.md` (otherwise the analysis is
measuring the wrong thing).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts import error_analysis as ea  # noqa: E402


def _load():
    return ea.load(ea.DEFAULT_PREDICTIONS, ea.DEFAULT_GT)


def test_eq_ci_semantics():
    assert ea.eq_ci("Floor 7", " floor 7 ") is True
    assert ea.eq_ci("A", "B") is False
    assert ea.eq_ci(None, "x") is False


def test_all_axes_present_and_internally_consistent():
    preds, gt = _load()
    assert len(gt) == 200

    sc = ea.analyze_subcategory(preds, gt)
    assert sc["worst_pairs"] and sc["total_errors"] >= 1

    h = ea.analyze_hitl(preds, gt)
    assert h["tp"] + h["fp"] + h["fn"] + h["tn"] == len(gt)

    ven = ea.analyze_vendor(preds, gt)
    assert sum(ven["failure_buckets"].values()) == ven["total_errors"]


def test_reconciles_with_known_composite_axes():
    """Must agree with the scoring.py composite behind eval_runs/README.md:
    subcat 98%, risk 84%, HITL F1 0.684, fields 88%, vendor 84.5%,
    auto 73/85."""
    preds, gt = _load()
    assert round(ea.analyze_subcategory(preds, gt)["accuracy"], 2) == 0.98
    assert round(ea.analyze_risk(preds, gt)["accuracy"], 2) == 0.84
    assert round(ea.analyze_hitl(preds, gt)["f1"], 3) == 0.684
    assert round(ea.analyze_location(preds, gt)["accuracy"], 2) == 0.88
    assert round(ea.analyze_vendor(preds, gt)["accuracy"], 3) == 0.845
    au = ea.analyze_auto_resolution(preds, gt)
    assert (au["hits"], au["eligible"]) == (73, 85)


def test_vendor_at_capacity_emergency_bucket_is_labelled_ac_mandated():
    """The 13 emergency/at_capacity rows must be called out as the
    issue #21 AC-mandated outcome, not lumped with real bugs."""
    preds, gt = _load()
    buckets = ea.analyze_vendor(preds, gt)["failure_buckets"]
    ac_bucket = [k for k in buckets if "at_capacity" in k and "AC-mandated" in k]
    assert ac_bucket, buckets
    assert buckets[ac_bucket[0]] == 13


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
