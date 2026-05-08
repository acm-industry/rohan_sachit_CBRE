"""Derive the subcategory → vendor_type mapping from historical records.

Each subcategory in the corpus maps to exactly one vendor_type at 100%
frequency (no ambiguity). This script reads the 10K historicals, computes
the mapping, and writes it to `agent/data/derived/subcategory_to_vendor_type.json`.

The output format is `{subcategory: [vendor_type, ...]}` — a list to be
forward-compatible with future subcategories that might need multiple
vendor types. In practice every entry today is a single-element list.

Run:
    python scripts/derive_vendor_map.py
    python scripts/derive_vendor_map.py --print-summary
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

HISTORICAL_PATH = Path(__file__).resolve().parents[1] / "operational" / "historical_records.json"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "agent" / "data" / "derived" / "subcategory_to_vendor_type.json"


def derive(records: List[dict]) -> Dict[str, List[str]]:
    """Compute subcategory → vendor_type(s) from historical assignments.

    Uses `final_subcategory` and `assigned_vendor_type` — the on-site
    authoritative fields, not intake.
    """
    subcat_to_vtypes: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        sub = r.get("final_subcategory")
        vtype = r.get("assigned_vendor_type")
        if sub and vtype:
            subcat_to_vtypes[sub][vtype] += 1

    result: Dict[str, List[str]] = {}
    for sub in sorted(subcat_to_vtypes.keys()):
        counts = subcat_to_vtypes[sub]
        total = sum(counts.values())
        # Include any vendor_type that handles >= 5% of this subcategory's tickets
        qualifying = [vt for vt, c in counts.most_common() if c / total >= 0.05]
        result[sub] = qualifying

    return result


def print_summary(mapping: Dict[str, List[str]], records: List[dict]) -> None:
    subcat_to_vtypes: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        sub = r.get("final_subcategory")
        vtype = r.get("assigned_vendor_type")
        if sub and vtype:
            subcat_to_vtypes[sub][vtype] += 1

    print(f"{'Subcategory':<25} {'Vendor Type(s)':<30} {'N':>5}  {'%':>5}")
    print("-" * 70)
    for sub in sorted(mapping.keys()):
        types = mapping[sub]
        total = sum(subcat_to_vtypes[sub].values())
        primary_count = subcat_to_vtypes[sub][types[0]]
        pct = primary_count / total * 100
        print(f"{sub:<25} {', '.join(types):<30} {total:>5}  {pct:>5.1f}%")
    print(f"\n{len(mapping)} subcategories mapped.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive subcategory → vendor_type mapping")
    ap.add_argument("--print-summary", action="store_true")
    args = ap.parse_args()

    records = json.loads(HISTORICAL_PATH.read_text())
    mapping = derive(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(mapping)} entries → {OUTPUT_PATH}")

    if args.print_summary:
        print()
        print_summary(mapping, records)

    return 0


if __name__ == "__main__":
    sys.exit(main())
