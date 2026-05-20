# Latency profile (issue #27) — per-node cost breakdown

First pass at the latency instrumentation acceptance criteria. The
orchestrator (`agent.classify:classify`) now records per-stage wall-clock
into `trainer_log.ai_prediction.latency_ms` for every call. This file
captures a small-sample (n=2) ad-hoc profile run via
`scripts/latency_profile.py` — enough to identify the dominant stage
without paying for a full 200-row dev eval.

## What changed

- New `_time_stage()` context manager in `agent/classify.py` wraps each
  of the 9 pipeline steps. Pre-stamped via `STAGE_NAMES` so the
  breakdown's shape is invariant across runs (a failed stage shows up
  as `0.0`, never silently drops out).
- Retrieval was lifted out of `agent.nodes.classify.classify` into the
  orchestrator so the chroma roundtrip can be timed independently of the
  downstream LLM call. Without this split, "classify" lumps embedding
  cost and structured-output cost together — and on the data below the
  two are an order of magnitude apart.
- `build_ai_prediction()` gained an optional `latency_ms` kwarg that
  flows into the trainer_log unchanged. Old callers (tests, harness
  replays) that don't pass it stay unchanged.
- Fault-isolation paths (extract / classify / risk) now plumb the
  partial `timings` dict into `_safe_fallback()` so a slow-then-failed
  stage still shows up in the breakdown.

## Run conditions

| field | value |
|-------|-------|
| timestamp (UTC) | 2026-05-20 |
| git branch | `issue-27` (off `8ae0116`) |
| chat model | `gpt-4o-mini`, temperature 0.0, seed 7 |
| embedding model | `text-embedding-3-small` |
| n transcripts | 2 (`EVAL-0011`, `EVAL-0019` from dev set) |
| raw breakdown | [`eval_runs/latency_profile.json`](latency_profile.json) |

OpenAI's API was running noticeably hot on this run — one of the two
transcripts (`EVAL-0011`) hit a request-per-day rate cap during extract
and fell through to `_safe_fallback`, which is itself a useful proof
that the instrumentation captures partial costs honestly. A second
run of the same transcript earlier in the day, before the cap, completed
the full pipeline in **6.67s** with the same per-stage shape (LLM nodes
dominant, local nodes negligible) — included below as the reference
"clean" sample.

## Per-stage breakdown

Sampled transcripts (ms):

| transcript | extract | retrieve | classify | location | risk | validate | vendor | clarify | summary | total |
|------------|---------|----------|----------|----------|------|----------|--------|---------|---------|-------|
| EVAL-0019 (full pipeline) | 12245.0 | 1179.0 | 10176.4 | 61.1 | 0.4 | 0.0 | 0.7 | 0.0 | 0.0 | **23663.0** |
| EVAL-0011 (extract-failed → fallback) | 18405.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **18405.4** |
| EVAL-0011 (earlier clean run) | 3349.8 | 727.2 | 2524.4 | 63.5 | 0.3 | 0.3 | 0.6 | 0.0 | 0.0 | **6666.3** |

Aggregate as a fraction of `total` (EVAL-0019, the only sample with a
complete pipeline today):

| stage    | ms      | % of total |
|----------|---------|------------|
| extract  | 12245.0 |     51.7%  |
| classify | 10176.4 |     43.0%  |
| retrieve |  1179.0 |      5.0%  |
| location |    61.1 |      0.3%  |
| vendor   |     0.7 |     <0.01% |
| risk     |     0.4 |     <0.01% |
| validate |     0.0 |     <0.01% |
| clarify  |     0.0 |     <0.01% |
| summary  |     0.0 |     <0.01% |
| **total** | **23663.0** | **100%** |

## Findings

1. **Two LLM nodes own the wall-clock.** `extract` and `classify` —
   the only two stages that issue chat-completion requests — account
   for **94.7% of total latency** on the full-pipeline sample (and a
   similar share on the earlier clean run: 88.2% combined). Everything
   downstream of classification is local Python and totals **<70ms**.
2. **Retrieval is the third-largest cost but a distant third.** Chroma
   roundtrip lands at **0.7–1.2s** in our two clean runs. Worth
   instrumenting separately (it would have been invisible bundled into
   "classify"), but not on the critical path today.
3. **The local nodes are effectively free.** `location`, `risk`,
   `validate`, `vendor`, `clarify`, `summary` together: **<70ms**.
   Any future optimisation effort should not start here.
4. **EVAL-0019 exceeded the 30s timeout window.** At 23.7s it's still
   under the hard cap but well over the 15s P95 target the issue's
   acceptance criteria lay out, and on the wrong side of the timeout if
   either LLM call sees long-tail latency. The clean EVAL-0011 run
   (6.7s) suggests this is a slow-day issue rather than a baseline one,
   but the variability is itself a risk.
5. **`_safe_fallback` is timing-honest on failures.** EVAL-0011's
   extract-failed run still records the 18.4s spent in the failed
   stage, rather than reporting a free 0ms call. Without this, the
   eval-time summary would systematically under-report cost on the
   exact calls that matter most for diagnosing slowness.

## Recommended next moves (not in this PR)

These follow from the data above; pick up under a separate issue
(probably as part of #27's "if P95 > 15s" branch):

- **Shrink the classify prompt.** The full canonical-taxonomy block +
  `k=8` retrieved tickets is sent on every call. The retrieval block
  alone is most of the token budget. Try `k=5` and re-measure — there
  is plenty of headroom on classification accuracy (98% subcategory)
  to spend on speed.
- **Consider a cheaper extract.** `extract` is the per-call cost
  ceiling and runs on every transcript regardless of difficulty. The
  schema is small and well-bounded; an even smaller model on this
  step is worth a head-to-head.
- **Do not touch the local pipeline.** Risk, validator, vendor,
  clarify, summary are sub-millisecond. Any "optimisation" here is
  noise.

## Reproduce

```bash
# Profile 2 dev transcripts. Writes both the table to stdout and a
# JSON breakdown for diffing across runs.
python scripts/latency_profile.py --n 2 --out eval_runs/latency_profile.json

# Profile a different slice, or a single transcript:
python scripts/latency_profile.py --n 1
```

The driver reads `latency_ms` straight off the trainer_log produced by
the orchestrator, so the same numbers are also available in any
`eval_runs/dev_*.json` predictions file produced after this change
lands — making the per-stage cost visible to error analysis (#25) and
the eval-history table in [`README.md`](README.md) without re-running.
