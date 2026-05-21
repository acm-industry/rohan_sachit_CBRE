# CBRE HITL — live-demo backend

Localhost-only FastAPI layer on top of `agent.classify` that drives the
demo-day UI. Not part of the graded submission path — the agent pipeline
(`agent/classify.py::classify`) is byte-identical to `submission-v1`,
and this backend imports a sibling `classify_with_events()` that emits
per-stage events while leaving the grader path untouched.

## Run

```bash
make backend-only        # FastAPI on :8000
make demo                # backend (:8000) + frontend (:3000)
.venv/bin/python -m uvicorn backend.main:app --reload --port 8000   # raw
```

CORS is open to `http://localhost:3000` only.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/transcripts`         | The three canned demo transcripts (id, label, expected outcome) |
| POST | `/api/calls/start`         | `{mode: "canned"\|"voice", transcript_id?, caller_phone?}` → `{call_id}` |
| GET  | `/api/calls/{id}/events`   | SSE stream of per-stage events |
| POST | `/api/calls/{id}/review`   | `{decision: "approve"\|"override", override?}` → `{accepted: true}` |
| POST | `/api/calls/{id}/audio`    | Raw audio bytes (voice mode); transcribes via peer's VoiceSession, then runs the pipeline |
| GET  | `/api/health`              | Liveness probe |

### SSE event contract

```json
{
  "call_id": "string",
  "stage": "extract|retrieve|classify|location|risk|validate|vendor|clarify|summary|trainer_log",
  "status": "started|complete|failed|gate_open|gate_resumed",
  "timing_ms": 1234.5,
  "payload": { /* stage-specific */ }
}
```

The validator gate emits `validate.gate_open` with the AI prediction,
validator reasons, classifier reasoning, retrieved tickets, and the
full transcript. The frontend opens its review modal, then POSTs to
`/api/calls/{id}/review`; the pipeline emits `validate.gate_resumed`
and continues with any override applied to category / subcategory /
risk_level / dispatched_vendor_id / dispatched_emergency_services /
needs_human_review.

## Canned demos

`backend/demo/canned.py` ships three transcripts hand-picked from
`evaluation/eval_transcripts_dev.json`:

- **EVAL-0976** — routine AC failure, auto-resolves with no review.
- **EVAL-0788** — "smoke alarm — just burnt toast", over-escalation trap;
  validator should auto-resolve as `JANITORIAL/waste_odor`, no 911.
- **EVAL-0076** — visible flames in an electrical room, true EMERGENCY,
  911 dispatched.

## Voice integration

`backend/integrations/voice.py` imports peer's `VoiceSession` (issue #32);
if the package isn't installed it falls back to a stub that returns a
canned transcript so the rest of the pipeline still runs end-to-end on
the demo laptop. `VOICE_AVAILABLE` exposes which path is active.

## Env vars

Documented in the root `.env.example`:

- `OPENAI_API_KEY` — required (tier-1).
- `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY` — required only when peer's
  voice package is installed; the stub ignores them.
- `LOG_LEVEL` — optional; defaults to `INFO`.

## Tests

```bash
make test-backend
```

Three suites:

- `tests/test_backend_routes.py` — every endpoint with the real
  pipeline mocked.
- `tests/test_classify_with_events.py` — runs `classify_with_events()`
  against patched nodes; asserts every stage fires in order and that
  the validator pause/override flow works.
- `tests/test_classify_unchanged.py` — guards the grader path
  (`classify()`) with both a source-hash pin and a mocked-output
  snapshot. Fails loudly if anyone edits the function body.
