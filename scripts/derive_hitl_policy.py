"""Derive the HITL trigger policy from `qa_audit_findings.json` + historicals.

For each `(final_subcategory, intake_risk_level)` cell in the historical
corpus, compute the empirical rate at which tickets were audit-flagged
(i.e. QA caught a problem with intake's call). The policy uses these
rates plus the per-subcategory over-escalation rate from #13 to decide
which (predicted_subcategory, predicted_risk_band) combinations should
trigger the validator gate to pause for human review.

Output: `agent/data/derived/hitl_policy.json`. Schema:

    {
      "rule_version": "v2",
      "thresholds": {
        "high_audit_rate_pct": 15.0,
        "trap_over_escalation_rate_pct": 15.0,
        "low_confidence": 0.5,
        "cascade_audit_rate_pct": 50.0,
        "cascade_min_n": 5,
        "cascade_excludes": ["waste_odor"]
      },
      "cells": {
        "<subcategory>": {
          "<risk_band>": {
            "n": <int>,
            "audit_rate": <float in [0,1]>,
            "is_high_audit": <bool>,
            "n_audit_flagged": <int>
          }, ...
        }, ...
      }
    }

Sanity target (per the AC): the rule applied retrospectively to history
should flag ≥80% of audit-flagged tickets and ≤30% of non-flagged tickets.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.data import audit, risk as risk_data  # noqa: E402

HISTORICAL_PATH = ROOT / "operational" / "historical_records.json"
DERIVED_PATH = ROOT / "agent" / "data" / "derived" / "hitl_policy.json"

# Tuned against `true_needs_human_review` on the labeled dev set (the
# rubric's actual axis), plus the AC sanity target on the historical
# corpus. The threshold sweep (recorded in the design doc) showed:
#
#   trap_oer threshold | dev-set HITL F1 | precision | recall
#   ------------------ | --------------- | --------- | ------
#         5%           |     0.79        |   0.66    |  0.97
#         10%          |     0.79        |   0.70    |  0.92
#        15%           |     0.83        |   0.88    |  0.78  ← chosen
#
# 15% gives the best F1 with strong precision (only 6 FPs on the dev set)
# and acceptable recall. The validator gate (#16) catches the remaining
# 22% via the always-pause-on-HIGH+ rule.
TRAP_OVER_ESCALATION_RATE_PCT = 15.0
LOW_CONFIDENCE = 0.5
# Cell-rate threshold kept in the JSON for design-doc transparency; the
# runtime rule uses the over-escalation rate (per-subcategory aggregate)
# rather than per-cell because the simpler rule beat the cell-specific
# variant on dev-set F1.
HIGH_AUDIT_RATE_PCT = 15.0

# Cascade rule (issues #63/#64): a subcategory is "trap-cascade" iff it
# has at least one cell at HIGH or EMERGENCY with audit_rate >= the
# cascade threshold AND n >= cascade_min_n. The validator gate (#16)
# pauses any LOW/MEDIUM prediction of a trap-cascade subcategory — this
# catches the under-classification trap rows (predicted LOW/MEDIUM but
# truly HIGH/EMERGENCY) that the per-subcategory OER rule misses
# (suspicious_person 12.8%, roof_leak 11.7%, malfunction 8.9% — all
# under the 15% OER bar but with overwhelming audit evidence at higher
# bands).
CASCADE_AUDIT_RATE_PCT = 50.0
CASCADE_MIN_N = 5

# Dev-tuned override: subcategories that meet the cascade structural
# criteria above but are EXCLUDED because the joint-axis sweep on
# `eval_runs/dev_baseline.json` showed they cost more on
# `auto_resolution` (10% weight) than they gain on `hitl_f1` (15%):
#
#   policy variant                                | hitl_f1 | auto% | composite contrib
#   --------------------------------------------- | ------- | ----- | -----------------
#   baseline (no cascade)                         |  0.684  | 85.9  | 18.851
#   cells-rule (incl. waste_odor)                 |  0.706  | 76.5  | 18.247  (-0.60)
#   cells-rule minus waste_odor (shipped)         |  0.719  | 85.9  | 19.380  (+0.53)
#
# waste_odor LOW (the dominant predicted cell) has cell audit_rate 5.2%
# on n=248 — strong historical evidence of safety, consistent with the
# 10/14 dev waste_odor LOW rows being auto-resolvable. Including it in
# the cascade pauses those rows and loses ~9 pts on auto_resolution.
CASCADE_EXCLUDES = ("waste_odor",)


def derive_cells(historicals: list[dict]) -> dict:
    """Build the per-(subcategory, intake_risk_level) audit-rate table."""
    flagged_ids = audit.flagged_ticket_ids()
    cells: dict = defaultdict(lambda: defaultdict(lambda: {"n": 0, "n_audit_flagged": 0}))

    for r in historicals:
        sub = r.get("final_subcategory")
        ir = r.get("intake_risk_level")
        tid = r.get("ticket_id")
        if not (sub and ir and tid):
            continue
        cells[sub][ir]["n"] += 1
        if tid in flagged_ids:
            cells[sub][ir]["n_audit_flagged"] += 1

    out: dict = {}
    for sub, by_band in cells.items():
        out[sub] = {}
        for band, c in by_band.items():
            n = c["n"]
            flagged = c["n_audit_flagged"]
            rate = flagged / n if n else 0.0
            out[sub][band] = {
                "n": n,
                "n_audit_flagged": flagged,
                "audit_rate": round(rate, 6),
                "is_high_audit": rate >= HIGH_AUDIT_RATE_PCT / 100.0,
            }
    return out


def should_pause(
    subcategory: str,
    predicted_risk: str,
    *,
    classification_confidence: float | None = None,
    over_escalation_rate: float = 0.0,
    fallback_invoked: bool = False,
) -> tuple[bool, list[str]]:
    """Apply the trigger rule. Returns (pause?, reasons).

    Used both by the derivation script's retrospective sanity check and
    (via `agent.data.hitl.should_pause`) by the validator gate at runtime.
    """
    reasons: list[str] = []

    if predicted_risk in ("HIGH", "EMERGENCY"):
        reasons.append(f"{predicted_risk.lower()}_band:always_pause")
        return True, reasons

    if over_escalation_rate >= TRAP_OVER_ESCALATION_RATE_PCT / 100.0:
        reasons.append(
            f"trap_prone_subcategory:over_escalation_rate={over_escalation_rate:.0%}"
        )
        return True, reasons

    if classification_confidence is not None and classification_confidence < LOW_CONFIDENCE:
        reasons.append(f"low_confidence:{classification_confidence:.2f}")
        return True, reasons

    if fallback_invoked:
        reasons.append("classifier_fallback_invoked")
        return True, reasons

    reasons.append("auto_resolve:no_flags_fired")
    return False, reasons


def evaluate_against_history(
    historicals: list[dict],
    cells: dict,
) -> dict:
    """Retroactive sanity check: how many audit-flagged tickets does the
    rule catch? How many false positives on non-flagged?"""
    flagged_ids = audit.flagged_ticket_ids()
    audited_ids = audit.all_audited_ticket_ids()

    # Use the historical intake labels as the "predicted" inputs — they're
    # what the agent sees at the time the trigger would fire.
    tp = fp = tn = fn = 0
    for r in historicals:
        tid = r.get("ticket_id")
        sub = r.get("final_subcategory")
        intake_band = r.get("intake_risk_level")
        if not (tid and sub and intake_band):
            continue
        oer = risk_data.intake_over_escalation_rate(sub)
        pause, _ = should_pause(
            sub, intake_band,
            classification_confidence=None,
            over_escalation_rate=oer,
        )
        is_flagged = tid in flagged_ids
        if pause and is_flagged:
            tp += 1
        elif pause and not is_flagged:
            fp += 1
        elif not pause and is_flagged:
            fn += 1
        else:
            tn += 1

    n_flagged = tp + fn
    n_unflagged = fp + tn
    return {
        "n_flagged": n_flagged,
        "n_unflagged": n_unflagged,
        "recall_on_flagged": tp / n_flagged if n_flagged else 0.0,
        "fp_rate_on_unflagged": fp / n_unflagged if n_unflagged else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-summary", action="store_true")
    ap.add_argument("--out", default=str(DERIVED_PATH))
    args = ap.parse_args(argv)

    print(f"[hitl] loading {HISTORICAL_PATH.name}...")
    raw = json.loads(HISTORICAL_PATH.read_text())
    print(f"[hitl] {len(raw)} historical records, "
          f"{len(audit.all_audited_ticket_ids())} audited "
          f"({len(audit.flagged_ticket_ids())} with at least one flag)")

    cells = derive_cells(raw)

    policy = {
        "rule_version": "v2",
        "thresholds": {
            "high_audit_rate_pct": HIGH_AUDIT_RATE_PCT,
            "trap_over_escalation_rate_pct": TRAP_OVER_ESCALATION_RATE_PCT,
            "low_confidence": LOW_CONFIDENCE,
            "cascade_audit_rate_pct": CASCADE_AUDIT_RATE_PCT,
            "cascade_min_n": CASCADE_MIN_N,
            "cascade_excludes": list(CASCADE_EXCLUDES),
        },
        "cells": cells,
    }

    DERIVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(policy, indent=2, sort_keys=True))
    print(f"[hitl] wrote {args.out}")

    print()
    print("[hitl] retrospective sanity check (rule applied to historical corpus):")
    e = evaluate_against_history(raw, cells)
    print(f"  records: {e['n_flagged'] + e['n_unflagged']}  "
          f"(audit-flagged: {e['n_flagged']}, unflagged: {e['n_unflagged']})")
    print(f"  recall on audit-flagged:  {e['recall_on_flagged']:.1%}  (target ≥ 80%)")
    print(f"  FP rate on unflagged:     {e['fp_rate_on_unflagged']:.1%}  (target ≤ 30%)")
    print(f"  confusion: TP={e['tp']}  FP={e['fp']}  TN={e['tn']}  FN={e['fn']}")

    if args.print_summary:
        print()
        print("[hitl] high-audit cells (subcategory, band, audit_rate):")
        rows = []
        for sub, by_band in sorted(cells.items()):
            for band, c in by_band.items():
                if c["is_high_audit"]:
                    rows.append((sub, band, c["audit_rate"], c["n"]))
        rows.sort(key=lambda r: -r[2])
        for sub, band, rate, n in rows[:20]:
            print(f"  {sub:24s}  {band:9s}  rate={rate:.1%}  n={n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
