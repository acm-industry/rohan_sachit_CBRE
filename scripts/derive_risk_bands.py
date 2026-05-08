"""Derive the per-subcategory base-risk table from the historical corpus.

Output: `agent/data/derived/base_risk_by_subcategory.json` — committed and
loaded at runtime by `agent.data.risk`.

For each subcategory in `operational/historical_records.json` we compute:
  - n: count of historical tickets
  - distribution: probability over {LOW, MEDIUM, HIGH, EMERGENCY} based on
    `final_risk_level` (the on-site technician's authoritative call;
    `intake_risk_level` is biased toward over-escalation per the QA audit)
  - modal_risk: the most-frequent final_risk_level — the prior the risk
    scorer in #14 starts from before applying caller-cue modifiers
  - intake_over_escalation_rate: fraction where intake > final
  - intake_under_escalation_rate: fraction where intake < final

These last two rates feed issue #15's HITL trigger derivation: subcategories
where intake has historically over-stated severity need the validator gate
to be skeptical of high-risk classifications.

Run:
    python scripts/derive_risk_bands.py
    python scripts/derive_risk_bands.py --print-summary

Re-run whenever `operational/historical_records.json` changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.data.risk import RISK_LEVEL_RANK, RISK_LEVELS  # noqa: E402

HISTORICAL_PATH = ROOT / "operational" / "historical_records.json"
DERIVED_DIR = ROOT / "agent" / "data" / "derived"
OUTPUT_PATH = DERIVED_DIR / "base_risk_by_subcategory.json"


def _modal(counter: Counter) -> str:
    """Return the most-frequent risk level, breaking ties by lower
    severity (don't auto-escalate ties)."""
    if not counter:
        return "MEDIUM"
    most = counter.most_common()
    top_count = most[0][1]
    candidates = [lvl for lvl, n in most if n == top_count]
    candidates.sort(key=lambda lvl: RISK_LEVEL_RANK[lvl])
    return candidates[0]


def derive(records: list[dict]) -> Dict[str, dict]:
    """Build the per-subcategory table from raw historical records."""
    by_subcat: Dict[str, list[dict]] = defaultdict(list)
    for r in records:
        sub = r.get("final_subcategory")
        if sub:
            by_subcat[sub].append(r)

    out: Dict[str, dict] = {}
    for sub, rows in sorted(by_subcat.items()):
        n = len(rows)
        if n == 0:
            continue

        final_counter: Counter = Counter()
        over = under = agree = comparable = 0
        for r in rows:
            fr = r.get("final_risk_level")
            ir = r.get("intake_risk_level")
            if fr in RISK_LEVEL_RANK:
                final_counter[fr] += 1
            if fr in RISK_LEVEL_RANK and ir in RISK_LEVEL_RANK:
                comparable += 1
                if RISK_LEVEL_RANK[ir] > RISK_LEVEL_RANK[fr]:
                    over += 1
                elif RISK_LEVEL_RANK[ir] < RISK_LEVEL_RANK[fr]:
                    under += 1
                else:
                    agree += 1

        # Distribution — explicit zero for missing buckets so downstream
        # consumers get a stable schema regardless of subcategory.
        dist = {lvl: round(final_counter[lvl] / n, 6) for lvl in RISK_LEVELS}

        out[sub] = {
            "n": n,
            "distribution": dist,
            "modal_risk": _modal(final_counter),
            "intake_over_escalation_rate": round(over / comparable, 6) if comparable else 0.0,
            "intake_under_escalation_rate": round(under / comparable, 6) if comparable else 0.0,
            "intake_agreement_rate": round(agree / comparable, 6) if comparable else 0.0,
        }
    return out


def print_summary(table: Dict[str, dict]) -> None:
    print(f"{'subcategory':<22s}  {'n':>5s}  {'modal':<10s}  "
          f"{'P(LOW)':>7s} {'P(MED)':>7s} {'P(HIGH)':>8s} {'P(EMER)':>8s}  "
          f"{'over%':>6s} {'under%':>7s}")
    print("-" * 105)
    for sub, e in sorted(table.items(), key=lambda kv: kv[1]["modal_risk"] + kv[0]):
        d = e["distribution"]
        print(
            f"{sub:<22s}  {e['n']:>5d}  {e['modal_risk']:<10s}  "
            f"{d['LOW']:>7.2f} {d['MEDIUM']:>7.2f} {d['HIGH']:>8.2f} {d['EMERGENCY']:>8.2f}  "
            f"{100*e['intake_over_escalation_rate']:>5.1f}% {100*e['intake_under_escalation_rate']:>6.1f}%"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-summary", action="store_true",
                    help="print the human-readable summary table")
    ap.add_argument("--out", default=str(OUTPUT_PATH),
                    help="output JSON path")
    args = ap.parse_args(argv)

    print(f"[risk] loading {HISTORICAL_PATH.name}...")
    raw = json.loads(HISTORICAL_PATH.read_text())
    print(f"[risk] {len(raw)} historical records")

    table = derive(raw)
    print(f"[risk] derived stats for {len(table)} subcategories")

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, indent=2, sort_keys=True))
    print(f"[risk] wrote {args.out}")

    if args.print_summary:
        print()
        print_summary(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
