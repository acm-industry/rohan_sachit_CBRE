"""Per-node latency profiler for the orchestrator (issue #27).

Runs `agent.classify:classify` against a small slice of dev transcripts
(default 2 — keeps LLM spend trivial), reads the `latency_ms` breakdown
the orchestrator now stamps into each trainer_log, and prints a per-stage
cost table. Use this when you want to know which node owns the wall-clock
*without* paying for a full 200-row eval.

Usage:
    python scripts/latency_profile.py                # 2 transcripts, dev set
    python scripts/latency_profile.py --n 5          # 5 transcripts
    python scripts/latency_profile.py --out eval_runs/latency_profile.json

This is a developer tool, not part of the eval pipeline — it's safe to
re-run ad hoc and the output is not committed unless you redirect it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from agent.classify import STAGE_NAMES, classify  # noqa: E402

DEFAULT_EVAL = ROOT / "evaluation" / "eval_transcripts_dev.json"


def _profile(rows: List[dict]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        tid = row["transcript_id"]
        t0 = time.perf_counter()
        pred = classify(row["turns"], row.get("caller_phone"))
        wall = round((time.perf_counter() - t0) * 1000.0, 3)
        latency = pred.get("trainer_log", {}).get("ai_prediction", {}).get(
            "latency_ms", {}
        )
        out.append(
            {
                "transcript_id": tid,
                "wall_ms": wall,
                "latency_ms": latency,
                "subcategory": pred.get("subcategory"),
                "risk_level": pred.get("risk_level"),
                "needs_human_review": pred.get("needs_human_review"),
            }
        )
    return out


def _aggregate(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Per-stage min/mean/max across the sampled transcripts."""
    agg: Dict[str, Dict[str, float]] = {}
    for stage in STAGE_NAMES:
        vals = [s["latency_ms"].get(stage, 0.0) for s in samples]
        agg[stage] = {
            "min": round(min(vals), 3),
            "mean": round(statistics.fmean(vals), 3),
            "max": round(max(vals), 3),
        }
    return agg


def _print_table(samples: List[Dict[str, Any]], agg: Dict[str, Dict[str, float]]) -> None:
    print(f"\n=== Per-transcript breakdown (n={len(samples)}) ===\n")
    header = f"{'transcript':<14}" + "".join(f"{s:>10}" for s in STAGE_NAMES)
    print(header)
    print("-" * len(header))
    for s in samples:
        row = f"{s['transcript_id']:<14}"
        for stage in STAGE_NAMES:
            row += f"{s['latency_ms'].get(stage, 0.0):>10.1f}"
        print(row)

    print(f"\n=== Aggregate (ms) ===\n")
    print(f"{'stage':<12}{'min':>10}{'mean':>10}{'max':>10}{'% of total':>14}")
    print("-" * 56)
    total_mean = agg["total"]["mean"] or 1.0
    for stage in STAGE_NAMES:
        a = agg[stage]
        pct = (a["mean"] / total_mean) * 100.0 if stage != "total" else 100.0
        print(f"{stage:<12}{a['min']:>10.1f}{a['mean']:>10.1f}{a['max']:>10.1f}{pct:>13.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", default=str(DEFAULT_EVAL), help="path to eval_transcripts_*.json")
    ap.add_argument("--n", type=int, default=2, help="number of transcripts to sample (default 2)")
    ap.add_argument("--out", default=None, help="optional path to dump the JSON breakdown")
    args = ap.parse_args()

    eval_rows = json.loads(Path(args.eval).read_text())
    rows = eval_rows[: args.n]
    print(f"Profiling {len(rows)} transcript(s) from {args.eval}")

    samples = _profile(rows)
    agg = _aggregate(samples)
    _print_table(samples, agg)

    if args.out:
        Path(args.out).write_text(
            json.dumps({"samples": samples, "aggregate": agg}, indent=2)
        )
        print(f"\nWrote breakdown → {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
