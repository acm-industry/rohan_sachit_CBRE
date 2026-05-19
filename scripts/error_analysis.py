"""Per-axis error analysis on the dev-set baseline predictions.

Produces (issue #25 acceptance criteria):
  - Subcategory confusion matrix (top-15 worst true→predicted pairs)
  - HITL confusion breakdown (TP/FP/FN/TN with case lists)
  - Location-field mismatch breakdown (building / address / floor)
  - Vendor-match failure breakdown (specialty / city / building-cert /
    SLA / availability miss, + the AC-mandated emergency-at_capacity
    unroutable bucket)
  - Auto-resolution misses
  - Top actionable findings (filed as issues #63–#67)

Correctness criteria mirror `evaluation/scoring.py` exactly, so every
count below reconciles with the composite axes. Pure-stdlib + the repo's
own vendor loader; no LLM, no network — runnable in CI and from
`notebooks/error_analysis.ipynb`.

Usage:
    python scripts/error_analysis.py [--predictions eval_runs/dev_baseline.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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


def analyze_subcategory(preds_map: dict, gt_map: dict, top: int = 15) -> dict:
    """Subcategory confusion matrix — top-N worst (true→predicted) pairs.

    Mirrors scoring.py: a miss is counted when the predicted subcategory
    is not case-insensitively equal to `true_subcategory`; a missing
    prediction is recorded as `<missing>`.
    """
    confusion: Counter = Counter()
    per_true_total: Counter = Counter()
    misses = 0
    for tid, gt in gt_map.items():
        true_sub = gt["true_subcategory"]
        per_true_total[true_sub] += 1
        p = preds_map.get(tid)
        if not p:
            confusion[(true_sub, "<missing>")] += 1
            misses += 1
            continue
        if not eq_ci(p.get("subcategory"), true_sub):
            confusion[(true_sub, p.get("subcategory") or "<missing>")] += 1
            misses += 1

    worst = confusion.most_common(top)
    err_by_true = sorted(
        (
            {
                "true": sub,
                "errors": sum(c for (t, _p), c in confusion.items() if t == sub),
                "n": per_true_total[sub],
            }
            for sub in {t for (t, _p) in confusion}
        ),
        key=lambda r: (-(r["errors"] / r["n"]), -r["errors"]),
    )
    for r in err_by_true:
        r["error_rate"] = round(r["errors"] / r["n"], 3)

    return {
        "total_errors": misses,
        "accuracy": (len(gt_map) - misses) / len(gt_map),
        "worst_pairs": [
            {"true": t, "predicted": p, "count": c} for (t, p), c in worst
        ],
        "error_rate_by_true_subcategory": err_by_true,
    }


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


def _get_vendor():
    """Lazy import so the module stays import-light for unit tests."""
    from agent.data.vendors import get_by_id  # noqa: WPS433
    return get_by_id


def analyze_vendor(preds_map: dict, gt_map: dict) -> dict:
    """Vendor-match failures, bucketed per the issue #25 AC.

    Buckets a routable miss by the FIRST violated constraint in the
    label's `vendor_constraints` (specialty / city / building-cert /
    SLA), else `availability miss`. The 'emergency: all acceptable
    vendors at_capacity' bucket is called out separately because that
    outcome is **AC-mandated by issue #21** (at_capacity vendors are
    skipped on emergencies) — it is a documented policy decision, NOT a
    bug to be 'fixed' by dispatching an at-capacity crew to an emergency.
    """
    get_by_id = _get_vendor()
    mismatches: List[dict] = []
    buckets: Counter = Counter()

    for tid, gt in gt_map.items():
        p = preds_map.get(tid)
        if not p:
            continue
        dispatched = p.get("dispatched_vendor_id")
        acceptable = gt.get("acceptable_vendor_ids", [])
        unroutable = gt.get("unroutable", False)
        pr_h = bool(p.get("needs_human_review", False))
        cons = gt.get("vendor_constraints", {}) or {}

        if unroutable:
            if not (dispatched in (None, "") and pr_h):
                buckets["unroutable not escalated"] += 1
                mismatches.append({"tid": tid, "type": "unroutable not escalated",
                                   "dispatched": dispatched, "needs_review": pr_h,
                                   "subcategory": gt["true_subcategory"],
                                   "case_type": gt["case_type"]})
            continue

        if dispatched and dispatched in acceptable:
            continue  # correct

        if dispatched in (None, ""):
            # Escalated when a vendor was acceptable. Distinguish the
            # AC-mandated emergency/at_capacity unroutable (issue #21)
            # from a genuine over-escalation.
            all_at_cap = bool(acceptable) and all(
                (get_by_id(a) is not None
                 and get_by_id(a).status_at_last_check == "at_capacity")
                for a in acceptable
            )
            if gt["true_risk_level"] == "EMERGENCY" and all_at_cap:
                bucket = "emergency: all acceptable at_capacity (AC-mandated unroutable, #21/#67)"
            else:
                bucket = "escalated though a vendor was acceptable"
            buckets[bucket] += 1
            mismatches.append({"tid": tid, "type": bucket,
                               "acceptable": acceptable,
                               "risk": gt["true_risk_level"],
                               "subcategory": gt["true_subcategory"],
                               "case_type": gt["case_type"]})
            continue

        # A wrong vendor was dispatched — classify by first violated
        # constraint (the AC's specialty / city / cert / SLA buckets).
        v = get_by_id(dispatched)
        if v is None:
            bucket = "dispatched unknown vendor_id"
        elif cons.get("must_match_specialty") and cons["must_match_specialty"] not in v.specialties:
            bucket = "specialty miss"
        elif (cons.get("must_cover_city") and v.coverage_cities
              and cons["must_cover_city"] not in v.coverage_cities):
            bucket = "city miss"
        elif (cons.get("must_certify_building_type") and v.building_types_certified
              and cons["must_certify_building_type"] not in v.building_types_certified):
            bucket = "building-cert miss"
        elif (cons.get("max_response_sla_minutes")
              and (v.response_sla_minutes or 10**9) > cons["max_response_sla_minutes"]):
            bucket = "SLA miss"
        elif v.status_at_last_check in ("offline", "at_capacity"):
            bucket = "availability miss"
        else:
            bucket = "other (acceptable-set mismatch)"
        buckets[bucket] += 1
        mismatches.append({"tid": tid, "type": bucket, "dispatched": dispatched,
                           "acceptable": acceptable,
                           "subcategory": gt["true_subcategory"],
                           "case_type": gt["case_type"]})

    return {
        "total_errors": len(mismatches),
        "accuracy": (len(gt_map) - len(mismatches)) / len(gt_map),
        "failure_buckets": dict(buckets.most_common()),
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


def print_report(subcat: dict, risk: dict, hitl: dict, location: dict,
                  vendor: dict, auto: dict) -> None:
    print("=" * 70)
    print("  ERROR ANALYSIS — dev_baseline.json (200 transcripts)")
    print("=" * 70)

    print(f"\n{'─' * 70}")
    print("  0. SUBCATEGORY CONFUSION: {:.1%} ({} errors)".format(
        subcat["accuracy"], subcat["total_errors"]))
    print(f"{'─' * 70}")
    print("  Worst (true → predicted) pairs:")
    for r in subcat["worst_pairs"]:
        print(f"    {r['true']} → {r['predicted']}: {r['count']}")
    print("  Highest error-rate true subcategories:")
    for r in subcat["error_rate_by_true_subcategory"][:8]:
        print(f"    {r['true']}: {r['errors']}/{r['n']} ({r['error_rate']:.0%})")

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
    for bucket, n in vendor["failure_buckets"].items():
        print(f"    {n:>3}  {bucket}")

    print(f"\n{'─' * 70}")
    print("  5. AUTO-RESOLUTION: {:.1%} ({}/{} eligible)".format(auto["rate"], auto["hits"], auto["eligible"]))
    print(f"{'─' * 70}")
    print(f"  Misses from clarification over-trigger: {auto['miss_by_cause']['clarification_over_trigger']}")
    print(f"  Misses from HITL over-trigger:          {auto['miss_by_cause']['hitl_over_trigger']}")

    print(f"\n{'=' * 70}")
    print("  TOP 5 ACTIONABLE FINDINGS  (filed as follow-up issues for #26)")
    print("=" * 70)
    print("""
  1. [#63 type:bug p1] HITL recall — under-escalation on
     over_escalation_trap + edge (20 FN, FN clustered on trap/edge case
     types). hitl_f1 is 15% of the composite and the lowest axis: the
     single biggest point lever. Tighten the derived HITL policy /
     validator so trap-prone + edge calls pause, without regressing the
     −5 false-911 guard or HITL precision.

  2. [#64 type:feature p2] HITL precision — over-escalation on normal
     (16 FP, mostly 'normal'/'clarification'/'hard'). Co-tune with #63:
     a joint precision/recall target on hitl_f1, not one in isolation.

  3. [#65 type:bug p1] Location — building+address co-fail on 22 rows
     (~17 emitted as None). These are calls with no transcript-stated
     building and no *usable* profile: the issue #19 active/not-stale
     gate correctly drops stale profiles, trading field recall for
     correctness. Recover via a LOW-confidence profile fallback (kept
     out of the authoritative path), not by re-trusting stale profiles.

  4. [#66 type:bug p1] Vendor — ~7 needless escalations (a vendor was
     acceptable) + ~6 city-coverage misses. Correctness bugs in
     vendor_select / qualify, independent of policy.

  5. [#67 type:feature p2] Vendor/emergency — 13 emergencies are
     unroutable because every acceptable vendor is at_capacity. This is
     **AC-mandated by issue #21** (at_capacity vendors are skipped on
     emergencies — dispatching an at-capacity crew to a gas leak is
     exactly what #21 forbids). NOT a bug: it is a documented
     trust-model decision (degrade-dispatch vs. hard-skip). The dev
     oracle's acceptable_vendor_ids predate #21, so these score as
     misses; reconcile in design-doc §6, not by weakening the skip.

  Secondary (deferred to #26): risk_level is 84% overall but weak on
  'edge' and 'clarification' case types — next tier after the HITL and
  field levers above.
""")


def save_json(subcat: dict, risk: dict, hitl: dict, location: dict,
              vendor: dict, auto: dict, out_path: Path) -> None:
    report = {
        "subcategory": subcat,
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

    subcat = analyze_subcategory(preds_map, gt_map)
    risk = analyze_risk(preds_map, gt_map)
    hitl = analyze_hitl(preds_map, gt_map)
    location = analyze_location(preds_map, gt_map)
    vendor = analyze_vendor(preds_map, gt_map)
    auto = analyze_auto_resolution(preds_map, gt_map)

    print_report(subcat, risk, hitl, location, vendor, auto)
    save_json(subcat, risk, hitl, location, vendor, auto, Path(args.out))


if __name__ == "__main__":
    main()
