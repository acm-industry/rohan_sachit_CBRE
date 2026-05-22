# UCSB Agentic AI — Final Project

Build a Human-in-the-Loop (HITL) agentic voice bot that automates an international
commercial-real-estate call center. Read **`final-assignment.html`** for the full
brief and **`assignment_brief.md`** for a one-page recap.

## What's in this bundle

```
final_project/
├── final-assignment.html      ← the full assignment (open in browser)
├── assignment_brief.md         ← one-page recap
├── operational/                ← CBRE's live-ops data (treat as production)
│   ├── taxonomy.md             ← intake SOP — labels and risk tiers in prose
│   ├── caller_profiles.json    ← 250 known-caller records
│   ├── buildings.json          ← 52 properties (intentionally sparse)
│   ├── vendors.json            ← 32 dispatch vendors with SLAs
│   └── historical_records.json ← 10,000 past tickets (your live RAG knowledge base)
└── evaluation/                 ← your benchmark
    ├── eval_transcripts_dev.json   ← 200 calls, labeled, for self-eval
    ├── dev_labels.json             ← matching ground truth for dev
    ├── eval_transcripts_test.json  ← 800 calls, no labels (final grading set)
    ├── qa_audit_findings.json      ← post-hoc QA notes for the historical knowledge base
    ├── prediction.py               ← Prediction + TrainerLog dataclasses (your output contract)
    ├── run_eval.py                 ← batch harness (loops over transcripts, calls your agent)
    └── scoring.py                  ← composite scorer; same one we use for final grading
```

## Quick start

```bash
# 1. Set up a Python env (Python ≥ 3.9) and install runtime deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Build the RAG index over historical_records.json (one-time, ~1 min, ~$0.02)
#    — produces a persistent chroma store at agent/rag/chroma_store/.
#    Skip this if a store already exists; pass --force to rebuild.
python -m agent.rag.build_index

# 3. Run the agent on the dev set (with labels — for self-evaluation)
python evaluation/run_eval.py \
    --agent agent.classify:classify \
    --eval evaluation/eval_transcripts_dev.json \
    --out predictions.json

# 4. Score yourself
python evaluation/scoring.py \
    --eval evaluation/eval_transcripts_dev.json \
    --ground-truth evaluation/dev_labels.json \
    --predictions predictions.json

# 5. Final test-set run (used for grading)
python evaluation/run_eval.py \
    --agent agent.classify:classify \
    --eval evaluation/eval_transcripts_test.json \
    --out predictions.json
```

### Agent module path — two equivalent options

Both `--agent` flags below resolve to the same `classify()` function. Either works for grading:

```bash
--agent agent.classify:classify          # native module path
--agent your_submission.agent:classify   # spec-template path (your_submission/agent.py re-exports the same callable)
```

The `your_submission/agent.py` shim exists so the literal example command from the assignment brief runs without modification.

The chroma store is gitignored — it's built on first run from
[`operational/historical_records.json`](operational/historical_records.json)
using the `text-embedding-3-small` model (configurable via `AGENT_EMBEDDING_MODEL`).
Changing the embedding model requires a `--force` rebuild.

The agent exposes a single callable, `agent.classify:classify(turns, caller_phone) -> dict`,
returning the fields documented in [`evaluation/prediction.py`](evaluation/prediction.py).
Each eval row also carries `caller_known_in_profiles: bool` — true when
`caller_phone` matches [`operational/caller_profiles.json`](operational/caller_profiles.json).
The harness forwards only `turns` and `caller_phone`; the agent looks the
profile up itself.

## Environment variables

Required:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Auth for OpenAI chat + embedding models |

Optional (defaults are tuned for determinism):

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_CHAT_MODEL` | `gpt-4.1-mini` | Chat model for extraction / classification / risk |
| `AGENT_CHAT_TEMPERATURE` | `0.0` | Sampling temperature (keep 0 for reproducibility) |
| `AGENT_CHAT_SEED` | `7` | OpenAI seed parameter when supported |
| `AGENT_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for the historical-records RAG index |
| `AGENT_MAX_EXTRACTION_RETRIES` | `2` | Cap on the query-rewrite retry loop |

Either export these in your shell or drop them in a `.env` at the repo root —
the agent loads `.env` automatically via `python-dotenv`. Missing required
vars surface as a `RuntimeError` from `agent.config.get_settings()` naming
the exact variable that's unset.

## Determinism

Per the brief: `classify(turns, phone)` is reproducible for identical input
modulo provider non-determinism. The contract is enforced in three places:

- **One LLM constructor.** Every node builds its chat model through
  `agent.config.build_chat_llm()`, which pins `temperature=0`, the
  `AGENT_CHAT_MODEL` string, and OpenAI's `seed` parameter
  (`AGENT_CHAT_SEED`, default 7). `tests/test_determinism.py` lint-fails
  the build if a bare `ChatOpenAI(...)` constructor shows up anywhere else
  under `agent/`.
- **Sealed file reads.** Source files under `agent/` only read from
  `agent/`, `operational/`, and `evaluation/`. The same test asserts this
  statically against every `Path(...)` constant in the module tree.
- **First-run build, not committed.** The chroma store is gitignored and
  built on first invocation (`python -m agent.rag.build_index`) at a
  one-time cost of roughly 1–3 minutes and ~$0.02 in embedding fees.
  Subsequent runs reuse the on-disk store. Changing
  `AGENT_EMBEDDING_MODEL` requires `--force` to rebuild.

OpenAI's `seed` is best-effort — the API documents occasional drift even
with `temperature=0`. We accept that as the documented provider noise
floor.

## What you submit

1. **Source repo** — your agent code, dependencies, README with run instructions.
2. **`predictions.json`** — output from running your agent over `eval_transcripts_test.json`.
3. **Design document (PDF)** — see the brief for required appendices (derived policy,
   HITL design, RAG strategy, vendor-selection logic, trainer-log spec, scale thought-experiment).

The full evaluation rubric, axis weights, and hard-cost rules are in
`final-assignment.html` → "What We Grade".
