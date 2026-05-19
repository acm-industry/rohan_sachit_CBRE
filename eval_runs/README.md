# Dev-set evaluation baseline (issue #24)

First end-to-end measurement of the full pipeline. This is the baseline
we iterate against; future runs should diff `dev_baseline.json` and append
a new row to the history table below.

## Baseline run

| field | value |
|-------|-------|
| timestamp (UTC) | 2026-05-19T05:15:15Z |
| git SHA | `33fc3bc` (issue-24 integration: fixed nodes + orchestrator wiring) |
| eval set | `evaluation/eval_transcripts_dev.json` (200 labelled transcripts) |
| agent | `agent.classify:classify` |
| chat model | `gpt-4o-mini`, temperature `0.0`, seed `7` (see `agent/config.py`) |
| embedding model | `text-embedding-3-small` |
| predictions | `eval_runs/dev_baseline.json` (200/200, 0 missing, no exceptions) |
| wall time | ~17 min (≈0.2 calls/s, well under the 1-hour AC) |

### Composite

**Final composite: 87.95 / 100**  (before penalty 87.95 — no penalty applied)

### Per-axis

| axis | score | weight |
|------|-------|--------|
| category | 100.00% | 10% |
| subcategory | 98.00% | 15% |
| risk_level | 84.00% | 10% |
| hitl_f1 | 68.42% | 15% |
| clarif_f1 | 75.00% | 5% |
| fields | 88.00% | 10% |
| vendor | 84.50% | 10% |
| auto_resolution | 85.88% | 10% |
| call_summary | 100.00% | 5% |
| trainer_log | 100.00% | 10% |

HITL precision=0.709 recall=0.661 **F1=0.684** ·
Clarification precision=0.675 recall=0.844 **F1=0.750** ·
Auto-resolution: 73 / 85 eligible routine cases.

### Safety audit (the −5 false-911 rule)

**False-911 dispatches on over-escalation traps: 0** (penalty −0.00).

This was the headline blocker on the original PR #56/#52 review: the
validator could autonomously dispatch 911 on a *misclassified* benign
call. The fixed validator now requires `risk==EMERGENCY` **and** a
life-safety subcategory **and** an extracted hard-hazard urgency cue
**and** a non-fallback, confident classification before setting
`dispatched_emergency_services=true`. Manual audit of the 200-row
prediction set confirms **zero** emergency dispatches on the 16
over-escalation-trap rows (and zero across the whole dev set).

### By case type (correct counts / n)

| case_type | n | cat | sub | risk | field | vendor | auto |
|-----------|---|-----|-----|------|-------|--------|------|
| normal | 90 | 90 | 89 | 80 | 71 | 72 | 51 |
| over_escalation_trap | 16 | 16 | 16 | 16 | 16 | 16 | 0 |
| multi_turn_correction | 10 | 10 | 10 | 9 | 10 | 9 | 6 |
| clarification | 24 | 24 | 23 | 20 | 24 | 20 | 0 |
| edge | 16 | 16 | 16 | 5 | 12 | 10 | 2 |
| hard | 36 | 36 | 34 | 30 | 35 | 34 | 14 |
| location_conflict | 8 | 8 | 8 | 8 | 8 | 8 | 0 |

Top confusions (true → predicted, count): `minor_issue→malfunction` 1,
`power_outage→lighting` 1, `refrigerant→no_cooling` 1,
`restroom_fixture→drainage_backup` 1. (Risk on `edge` cases — 5/16 — is
the weakest area and the natural next iteration target; full error
analysis is issue #25.)

## Reproduce

```bash
# 1. Build the RAG index once (~1–3 min, ~$0.02; not committed — see #28).
python -m agent.rag.build_index

# 2. Run the dev eval (real LLM; needs OPENAI_API_KEY in .env).
python evaluation/run_eval.py \
    --agent agent.classify:classify \
    --eval evaluation/eval_transcripts_dev.json \
    --out eval_runs/dev_baseline.json

# 3. Score.
python evaluation/scoring.py \
    --eval evaluation/eval_transcripts_dev.json \
    --ground-truth evaluation/dev_labels.json \
    --predictions eval_runs/dev_baseline.json
```

Output is deterministic modulo the documented LLM-provider noise floor
(temperature 0, fixed seed, pinned model — issue #28).

## History

| date (UTC) | git SHA | composite | false_911 | notes |
|------------|---------|-----------|-----------|-------|
| 2026-05-19T05:15:15Z | `33fc3bc` | 87.95 | 0 | baseline — fixed pipeline (issues #16/#19/#20/#21 + orchestrator) |
| 2026-05-19T (run b)  | `7511166` | 88.30 | 0 | issue #63/#64 — trap-cascade HITL rule, `cascade_excludes=[waste_odor]`. hitl_f1 68.42 → 71.94 (+3.52), auto_resolution 85.88 (unchanged). Predictions: `eval_runs/dev_issue63.json`. |
