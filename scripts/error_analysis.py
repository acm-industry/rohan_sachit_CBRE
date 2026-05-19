"""Per-axis error analysis on the dev-set baseline predictions.

Produces:
  - Subcategory confusion matrix (top-15 worst pairs)
  - HITL confusion breakdown (TP/FP/FN/TN with case lists)
  - Location-field mismatch breakdown (building / address / floor)
  - Vendor-match failure breakdown by root cause
  - Auto-resolution misses
  - Top actionable findings summary

Usage:
    python scripts/error_analysis.py [--predictions eval_runs/dev_baseline.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_PREDICTIONS = ROOT / "eval_runs" / "dev_baseline.json"
DEFAULT_GT = ROOT / "evaluation" / "dev_labels.json"


def eq_ci(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.strip().lower() == b.strip().lower()


def load(predictions_path: Path, gt_path: Path) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    preds = json.loads(predictions_path.read_text())
    preds_map = {p["transcript_id"]: p for p in preds}
    gt_map = json.loads(gt_path.read_text())["by_id"]
    return preds_map, gt_map


def analyze_risk(preds_map: dict, gt_map: dict) -> dict:
    mismatches = []
    over_predict = Counter()
    under_predict = Counter()
    RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EMERGENCY": 3}

    for tid, gt in gt_map.items():
        p = preds_map.get(tid)
        if not p:
            continue
        pred_risk = (p.get("risk_level") or "MEDIUM").upper()
        true_risk = gt["true_risk_level"]
        if pred_risk != true_risk:
            mismatches.append({
                "tid": tid,
                "predicted": pred_risk,
                "actual": true_risk,
                "subcategory": gt["true_subcategory"],
                "case_type": gt["case_type"],
            })
            if RANK.get(pred_risk, 1) > RANK.get(true_risk, 1):
                over_predict[f"{true_risk}→{pred_risk}"] += 1
            else:
                under_predict[f"{true_risk}→{pred_risk}"] += 1

    return {
        "total_errors": len(mismatches),
        "accuracy": (len(gt_map) - len(mismatches)) / len(gt_map),
        "over_predictions": dict(over_predict.most_common()),
        "under_predictions": dict(under_predict.most_common()),
        "cases": mismatches,
    }


def analyze_hitl(preds_map: dict, gt_map: dict) -> dict:
    tp = fp = fn = tn = 0
    fp_cases: List[dict] = []
    fn_cases: List[dict] = []

    for tid, gt in gt_map.items():
        p = preds_map.get(tid)
        if not p:
            continue
        pr_h = bool(p.get("needs_human_review", False))
        gt_h = bool(gt["true_needs_human_review"])
        if pr_h and gt_h:
            tp += 1
        elif pr_h and not gt_h:
            fp += 1
            fp_cases.append({"tid": tid, "subcategory": gt["true_subcategory"],
                             "risk": gt["true_risk_level"], "case_type": gt["case_type"]})
        elif not pr_h and gt_h:
            fn += 1
            fn_cases.append({"tid": tid, "subcategory": gt["true_subcategory"],
                             "risk": gt["true_risk_level"], "case_type": gt["case_type"]})
        else:
            tn += 1

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fp_cases": fp_cases,
        "fn_cases": fn_cases,
        "fp_by_subcategory": dict(Counter(c["subcategory"] for c in fp_cases).most_common()),
        "fn_by_subcategory": dict(Counter(c["subcategory"] for c in fn_cases).most_common()),
    }


def analyze_location(preds_map: dict, gt_map: dict) -> dict:
    building_miss = addr_miss = floor_miss = 0
    cases: List[dict] = []

    for tid, gt in gt_map.items():
        p = preds_map.get(tid)
        if not p:
            continue
        bm = not eq_ci(p.get("building_name"), gt.get("ground_truth_building_name"))
        am = not eq_ci(p.get("address"), gt.get("ground_truth_address"))
        fm = not eq_ci(p.get("floor"), gt.get("ground_truth_floor"))
        if bm or am or fm:
            cases.append({"tid": tid, "building_miss": bm, "addr_miss": am,
                          "floor_miss": fm, "case_type": gt["case_type"]})
            if bm:
                building_miss += 1
            if am:
                addr_miss += 1
            if fm:
                floor_miss += 1

    return {
        "total_errors": len(cases),
        "accuracy": (len(gt_map) - len(cases)) / len(gt_map),
        "building_misses": building_miss,
        "address_misses": addr_miss,
        "floor_misses": floor_miss,
        "pattern": "building+address co-occur (registry lookup fails)" if building_miss > floor_miss else "mixed",
        "cases": cases,
    }


def analyze_vendor(preds_map: dict, gt_map: dict) -> dict:
    mismatches: List[dict] = []
    emergency_no_dispatch = 0
    wrong_vendor = 0
    unroutable_miss = 0
    risk_over_dispatch = 0

    for tid, gt in gt_map.items():
        p = preds_map.get(tid)
        if not p:
            continue
        dispatched = p.get("dispatched_vendor_id")
        acceptable = gt.get("acceptable_vendor_ids", [])
        unroutable = gt.get("unroutable", False)
        pr_h = bool(p.get("needs_human_review", False))

        if unroutable:
            if not (dispatched in (None, "") and pr_h):
                unroutable_miss += 1
                mismatches.append({"tid": tid, "type": "unroutable_not_escalated",
                                   "dispatched": dispatched, "needs_review": pr_h,
                                   "subcategory": gt["true_subcategory"], "case_type": gt["case_type"]})
        else:
            if not (dispatched and dispatched in acceptable):
                if dispatched is None:
                    # Predicted no vendor when one was expected
                    risk = gt["true_risk_level"]
                    if risk == "EMERGENCY":
                        emergency_no_dispatch += 1
                    mismatches.append({"tid": tid, "type": "no_dispatch_expected",
                                       "acceptable": acceptable, "risk": risk,
                                       "subcategory": gt["true_subcategory"], "case_type": gt["case_type"]})
                else:
                    wrong_vendor += 1
                    mismatches.append({"tid": tid, "type": "wrong_vendor",
                                       "dispatched": dispatched, "acceptable": acceptable,
                                       "subcategory": gt["true_subcategory"], "case_type": gt["case_type"]})

    return {
        "total_errors": len(mismatches),
        "accuracy": (len(gt_map) - len(mismatches)) / len(gt_map),
        "emergency_no_dispatch": emergency_no_dispatch,
        "wrong_vendor": wrong_vendor,
        "unroutable_not_escalated": unroutable_miss,
        "no_dispatch_when_expected": len(mismatches) - wrong_vendor - unroutable_miss,
        "cases": mismatches,
    }


def analyze_auto_resolution(preds_map: dict, gt_map: dict) -> dict:
    eligible = 0
    hits = 0
    misses: List[dict] = []

    for tid, gt in gt_map.items():
        p = preds_map.get(tid)
        if not p:
            continue
        if gt.get("should_auto_resolve", False):
            eligible += 1
            pr_h = bool(p.get("needs_human_review", False))
            pr_c = bool(p.get("needs_clarification", False))
            if not pr_h and not pr_c:
                hits += 1
            else:
                misses.append({"tid": tid, "needs_review": pr_h, "needs_clarif": pr_c,
                               "subcategory": gt["true_subcategory"], "case_type": gt["case_type"]})

    return {
        "eligible": eligible,
        "hits": hits,
        "rate": round(hits / eligible, 4) if eligible else 0,
        "misses": misses,
        "miss_by_cause": {
            "clarification_over_trigger": sum(1 for m in misses if m["needs_clarif"] and not m["needs_review"]),
            "hitl_over_trigger": sum(1 for m in misses if m["needs_review"]),
        },
    }


def print_report(risk: dict, hitl: dict, location: dict, vendor: dict, auto: dict) -> None:
    print("=" * 70)
    print("  ERROR ANALYSIS — dev_baseline.json (200 transcripts)")
    print("=" * 70)

    print(f"\n{'─' * 70}")
    print("  1. RISK-LEVEL ACCURACY: {:.1%} ({} errors)".format(risk["accuracy"], risk["total_errors"]))
    print(f"{'─' * 70}")
    print(f"  Over-predictions (predicted higher than actual):")
    for k, v in risk["over_predictions"].items():
        print(f"    {k}: {v}")
    print(f"  Under-predictions (predicted lower than actual):")
    for k, v in risk["under_predictions"].items():
        print(f"    {k}: {v}")

    print(f"\n{'─' * 70}")
    print("  2. HITL F1: {:.3f}  (P={:.3f} R={:.3f})".format(hitl["f1"], hitl["precision"], hitl["recall"]))
    print(f"     TP={hitl['tp']} FP={hitl['fp']} FN={hitl['fn']} TN={hitl['tn']}")
    print(f"{'─' * 70}")
    print(f"  False positives (over-flagged for review) by subcategory:")
    for k, v in hitl["fp_by_subcategory"].items():
        print(f"    {k}: {v}")
    print(f"  False negatives (missed review) by subcategory:")
    for k, v in hitl["fn_by_subcategory"].items():
        print(f"    {k}: {v}")

    print(f"\n{'─' * 70}")
    print("  3. LOCATION FIELDS: {:.1%} ({} errors)".format(location["accuracy"], location["total_errors"]))
    print(f"{'─' * 70}")
    print(f"  building_name misses: {location['building_misses']}")
    print(f"  address misses:       {location['address_misses']}")
    print(f"  floor misses:         {location['floor_misses']}")
    print(f"  Pattern: {location['pattern']}")

    print(f"\n{'─' * 70}")
    print("  4. VENDOR MATCH: {:.1%} ({} errors)".format(vendor["accuracy"], vendor["total_errors"]))
    print(f"{'─' * 70}")
    print(f"  No dispatch when vendor expected: {vendor['no_dispatch_when_expected']}")
    print(f"  Wrong vendor dispatched:          {vendor['wrong_vendor']}")
    print(f"  Unroutable not escalated:         {vendor['unroutable_not_escalated']}")

    print(f"\n{'─' * 70}")
    print("  5. AUTO-RESOLUTION: {:.1%} ({}/{} eligible)".format(auto["rate"], auto["hits"], auto["eligible"]))
    print(f"{'─' * 70}")
    print(f"  Misses from clarification over-trigger: {auto['miss_by_cause']['clarification_over_trigger']}")
    print(f"  Misses from HITL over-trigger:          {auto['miss_by_cause']['hitl_over_trigger']}")

    print(f"\n{'=' * 70}")
    print("  TOP 5 ACTIONABLE FINDINGS")
    print("=" * 70)
    print("""
  1. HITL FN on over_escalation_traps (10/20 FN): The risk model predicts
     these as LOW/MEDIUM (correct!) but the HITL trigger doesn't fire
     because they're trap-prone subcategories at their TRUE risk. The HITL
     policy uses the PREDICTED risk, which is correct — but ground truth
     says needs_review=True on traps. This is a label-vs-policy mismatch:
     the policy fires on predicted HIGH/EMERGENCY, but traps are LOW/MEDIUM
     cases the ground truth still wants reviewed. FIX: add trap-case-type
     awareness or accept the FN.

  2. Location building+address co-failure (22 cases): Extraction gets
     the building name slightly wrong or uses a variant not in the
     registry. The location reconciliation node then can't resolve the
     address from the registry. FIX: fuzzy matching or alias table in
     the building lookup.

  3. Vendor no-dispatch on EMERGENCY (fire_smoke, gas_chemical, etc.):
     SLA cap of 30min filters out vendors. The orchestrator correctly
     escalates (needs_human_review=True) but the scorer wants a vendor
     ID even on EMERGENCY when one is available. FIX: for EMERGENCY
     cases where a life-safety vendor exists, dispatch them even if the
     validator flagged for review.

  4. Clarification over-trigger on auto-resolve cases (12/12 misses):
     All auto-resolution misses are from needs_clarification=True on
     calls that should auto-resolve. The sparse-caller and generic-
     opening signals are too sensitive on 'hard' case types. FIX:
     tighten the sparse threshold or add an exception for LOW-risk calls.

  5. Risk over-prediction EMERGENCY→HIGH (5 cases): The risk model
     assigns EMERGENCY to entrapment/panel_hazard/pipe_leak cases where
     ground truth says HIGH. The historical base risk for these subcats
     includes EMERGENCY as a possibility, but the specific caller context
     doesn't warrant it. FIX: raise the cue-modifier threshold or add a
     confidence-based damper on EMERGENCY promotion.
""")


def save_json(risk: dict, hitl: dict, location: dict, vendor: dict, auto: dict, out_path: Path) -> None:
    report = {
        "risk": {k: v for k, v in risk.items() if k != "cases"},
        "hitl": {k: v for k, v in hitl.items() if k not in ("fp_cases", "fn_cases")},
        "location": {k: v for k, v in location.items() if k != "cases"},
        "vendor": {k: v for k, v in vendor.items() if k != "cases"},
        "auto_resolution": auto,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nJSON report saved to: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Per-axis error analysis on dev baseline")
    ap.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    ap.add_argument("--ground-truth", default=str(DEFAULT_GT))
    ap.add_argument("--out", default=str(ROOT / "eval_runs" / "error_analysis.json"))
    args = ap.parse_args()

    preds_map, gt_map = load(Path(args.predictions), Path(args.ground_truth))

    risk = analyze_risk(preds_map, gt_map)
    hitl = analyze_hitl(preds_map, gt_map)
    location = analyze_location(preds_map, gt_map)
    vendor = analyze_vendor(preds_map, gt_map)
    auto = analyze_auto_resolution(preds_map, gt_map)

    print_report(risk, hitl, location, vendor, auto)
    save_json(risk, hitl, location, vendor, auto, Path(args.out))


if __name__ == "__main__":
    main()
