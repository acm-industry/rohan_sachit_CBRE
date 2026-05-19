"""Schema conformance test for `agent.classify:classify`.

This is the canary that fires the moment any downstream node breaks the
output contract — silent schema drift is the most common way to zero out
the rubric (missing fields → wrong on the corresponding axis).

Design intent (from issue #2):

- Round-trip the agent's output through `evaluation.prediction.Prediction`
  + `TrainerLog`. If a key is missing, mistyped, or out of the canonical
  enum, instantiation raises and this test fails loudly.
- Run in < 5s without hitting any LLM. The reference classify on main is
  currently a stub (issue #1) so direct calls are deterministic and free.
  Once the LLM-backed orchestrator (#56 / #24) lands, this test will need
  to monkey-patch the chat client — at that point the conformance check
  here becomes the gating CI test for the full pipeline.
- At least 3 fixture transcripts (per AC). We use real dev transcripts
  spanning anonymous + known callers, varying turn counts.

The test is structurally independent of any specific node implementation
— it asserts only what `evaluation.prediction.Prediction` enforces +
the rubric's downstream expectations:

  - all required fields present and typed correctly
  - `risk_level` ∈ {LOW, MEDIUM, HIGH, EMERGENCY}
  - `dispatched_emergency_services` is a `bool`, not a truthy non-bool
  - `call_summary` is a non-empty string at least 30 chars (scoring.py's
    bar for the call-summary axis)
  - `trainer_log` has all four required sub-keys (`full_transcript`,
    `ai_prediction`, `human_override`, `final_decision`)
  - `(category, subcategory)` either match the canonical taxonomy or
    fall back to the stub "OTHER/other" placeholder
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from agent.classify import classify  # noqa: E402
from prediction import Prediction, TrainerLog  # noqa: E402


RISK_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH", "EMERGENCY"})
REQUIRED_TOP_KEYS = (
    "category",
    "subcategory",
    "risk_level",
    "needs_human_review",
    "needs_clarification",
    "building_name",
    "address",
    "floor",
    "dispatched_vendor_id",
    "dispatched_emergency_services",
    "call_summary",
    "trainer_log",
)
REQUIRED_TRAINER_LOG_KEYS = (
    "full_transcript",
    "ai_prediction",
    "human_override",
    "final_decision",
)


def _load_dev_fixtures(transcript_ids: List[str]) -> List[Dict[str, Any]]:
    rows = json.loads((ROOT / "evaluation" / "eval_transcripts_dev.json").read_text())
    by_id = {r["transcript_id"]: r for r in rows}
    out = []
    for tid in transcript_ids:
        if tid not in by_id:
            raise RuntimeError(f"fixture {tid!r} not in eval_transcripts_dev.json")
        out.append(by_id[tid])
    return out


# Real dev rows — picked to span the input space:
#  EVAL-0004: known caller (profile), short dialogue
#  EVAL-0006: anonymous caller, mid-length, location stated explicitly
#  EVAL-0011: anonymous caller, mid-call correction (caller flips topic)
_FIXTURES = ("EVAL-0004", "EVAL-0006", "EVAL-0011")


# ─── Per-fixture schema round-trip ──────────────────────────────────────


def test_classify_output_round_trips_through_prediction_dataclass():
    """The headline check: `Prediction(**classify(...))` instantiates
    cleanly on every fixture. Catches any missing/mistyped field added
    by a downstream node."""
    for row in _load_dev_fixtures(_FIXTURES):
        out = classify(row["turns"], row.get("caller_phone"))
        # Pull trainer_log out, instantiate it first, then the parent.
        tl_raw = out["trainer_log"]
        tl = TrainerLog(**tl_raw)
        Prediction(
            transcript_id=row["transcript_id"],
            category=out["category"],
            subcategory=out["subcategory"],
            risk_level=out["risk_level"],
            needs_human_review=out["needs_human_review"],
            needs_clarification=out["needs_clarification"],
            building_name=out.get("building_name"),
            address=out.get("address"),
            floor=out.get("floor"),
            dispatched_vendor_id=out.get("dispatched_vendor_id"),
            dispatched_emergency_services=out["dispatched_emergency_services"],
            call_summary=out["call_summary"],
            trainer_log=tl,
        )


def test_classify_output_has_every_required_top_level_key():
    for row in _load_dev_fixtures(_FIXTURES):
        out = classify(row["turns"], row.get("caller_phone"))
        missing = [k for k in REQUIRED_TOP_KEYS if k not in out]
        assert not missing, f"{row['transcript_id']} missing {missing}"


def test_trainer_log_has_every_required_subkey():
    for row in _load_dev_fixtures(_FIXTURES):
        out = classify(row["turns"], row.get("caller_phone"))
        tl = out["trainer_log"]
        assert isinstance(tl, dict), f"trainer_log not a dict on {row['transcript_id']}"
        missing = [k for k in REQUIRED_TRAINER_LOG_KEYS if k not in tl]
        assert not missing, f"{row['transcript_id']} trainer_log missing {missing}"


# ─── Field-level type + value checks ────────────────────────────────────


def test_risk_level_is_in_canonical_enum():
    for row in _load_dev_fixtures(_FIXTURES):
        out = classify(row["turns"], row.get("caller_phone"))
        assert out["risk_level"] in RISK_LEVELS, (
            f"{row['transcript_id']}: risk_level={out['risk_level']!r} "
            f"not in {sorted(RISK_LEVELS)}"
        )


def test_boolean_fields_are_actually_booleans():
    """Catches the classic 'truthy non-bool' bug — Pydantic would coerce
    `1`, `"true"`, or `None` to bool in some contexts but the scorer
    uses `bool(...)` literally."""
    for row in _load_dev_fixtures(_FIXTURES):
        out = classify(row["turns"], row.get("caller_phone"))
        for k in ("needs_human_review", "needs_clarification", "dispatched_emergency_services"):
            assert isinstance(out[k], bool), (
                f"{row['transcript_id']}: {k}={out[k]!r} is "
                f"{type(out[k]).__name__}, expected bool"
            )


def test_call_summary_is_nonempty_string_at_least_30_chars():
    """scoring.py rewards `call_summary` only when length >= 30 chars."""
    for row in _load_dev_fixtures(_FIXTURES):
        out = classify(row["turns"], row.get("caller_phone"))
        cs = out["call_summary"]
        assert isinstance(cs, str), f"{row['transcript_id']}: call_summary not a string"
        assert len(cs.strip()) >= 30, (
            f"{row['transcript_id']}: call_summary too short "
            f"({len(cs.strip())} chars; scoring axis requires >= 30)"
        )


def test_optional_string_fields_are_str_or_none():
    """`building_name`, `address`, `floor`, `dispatched_vendor_id` are
    `Optional[str]` per `Prediction`. They must be a string or None —
    not `0`, empty dict, or other 'absent' sentinels that would
    silently fail equality checks in scoring.py."""
    for row in _load_dev_fixtures(_FIXTURES):
        out = classify(row["turns"], row.get("caller_phone"))
        for k in ("building_name", "address", "floor", "dispatched_vendor_id"):
            v = out.get(k)
            assert v is None or isinstance(v, str), (
                f"{row['transcript_id']}: {k}={v!r} is {type(v).__name__}, "
                "expected Optional[str]"
            )


def test_full_transcript_in_trainer_log_is_nonempty():
    """The trainer log's `full_transcript` is the supervised signal for
    any future fine-tune. It must contain the actual dialogue, not an
    empty string or placeholder."""
    for row in _load_dev_fixtures(_FIXTURES):
        out = classify(row["turns"], row.get("caller_phone"))
        ft = out["trainer_log"]["full_transcript"]
        assert isinstance(ft, str) and len(ft.strip()) > 0, (
            f"{row['transcript_id']}: trainer_log.full_transcript empty"
        )


def test_ai_prediction_and_final_decision_are_dicts():
    """Scoring counts the trainer-log axis as passing only when the four
    sub-keys are present AND `final_decision` is a dict (per scoring.py's
    `isinstance(tl, dict) and 'final_decision' in tl` check)."""
    for row in _load_dev_fixtures(_FIXTURES):
        out = classify(row["turns"], row.get("caller_phone"))
        tl = out["trainer_log"]
        assert isinstance(tl["ai_prediction"], dict)
        assert isinstance(tl["final_decision"], dict)


def test_human_override_is_none_or_dict():
    """When no override happened, `human_override` must be `None`
    (not missing). When one happened, it should be a dict diff."""
    for row in _load_dev_fixtures(_FIXTURES):
        out = classify(row["turns"], row.get("caller_phone"))
        ho = out["trainer_log"]["human_override"]
        assert ho is None or isinstance(ho, dict), (
            f"{row['transcript_id']}: human_override={ho!r} "
            f"is {type(ho).__name__}, expected None or dict"
        )


# ─── Inputs the harness will actually pass ─────────────────────────────


def test_anonymous_caller_input_handled():
    """`run_eval.py` passes `caller_phone=None` for some rows. Must not
    crash and must still return a valid Prediction."""
    out = classify(
        [{"speaker": "caller", "text": "Toilet clogged on Floor 4."}],
        None,
    )
    Prediction(
        transcript_id="synthetic",
        category=out["category"],
        subcategory=out["subcategory"],
        risk_level=out["risk_level"],
        needs_human_review=out["needs_human_review"],
        needs_clarification=out["needs_clarification"],
        building_name=out.get("building_name"),
        address=out.get("address"),
        floor=out.get("floor"),
        dispatched_vendor_id=out.get("dispatched_vendor_id"),
        dispatched_emergency_services=out["dispatched_emergency_services"],
        call_summary=out["call_summary"],
        trainer_log=TrainerLog(**out["trainer_log"]),
    )


def test_empty_turns_does_not_crash():
    """Defensive: even a zero-turn input should return a valid (if low-
    quality) Prediction rather than crashing. The harness's per-call
    timeout wraps individual failures, but a hard crash on empty input
    would still poison the run."""
    out = classify([], None)
    assert "category" in out
    assert "trainer_log" in out
    # Round-trip works
    TrainerLog(**out["trainer_log"])


def test_minimal_single_caller_turn_handled():
    out = classify([{"speaker": "caller", "text": "x"}], "+1-310-555-0142")
    Prediction(
        transcript_id="synthetic-minimal",
        category=out["category"],
        subcategory=out["subcategory"],
        risk_level=out["risk_level"],
        needs_human_review=out["needs_human_review"],
        needs_clarification=out["needs_clarification"],
        building_name=out.get("building_name"),
        address=out.get("address"),
        floor=out.get("floor"),
        dispatched_vendor_id=out.get("dispatched_vendor_id"),
        dispatched_emergency_services=out["dispatched_emergency_services"],
        call_summary=out["call_summary"],
        trainer_log=TrainerLog(**out["trainer_log"]),
    )


# ─── Performance budget (per AC: < 5s) ──────────────────────────────────


def test_full_contract_check_completes_under_5s():
    """The whole contract suite must run fast enough to land in CI on
    every push. Per the AC: < 5s without hitting any LLM."""
    t0 = time.time()
    rows = _load_dev_fixtures(_FIXTURES)
    for row in rows:
        out = classify(row["turns"], row.get("caller_phone"))
        TrainerLog(**out["trainer_log"])
    elapsed = time.time() - t0
    assert elapsed < 5.0, (
        f"contract test budget blown: {elapsed:.2f}s on {len(rows)} fixtures "
        "(target < 5s) — a downstream node is doing something slow on import "
        "or in classify()"
    )


# ─── Inline runner ──────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect

    tests = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__])
             if n.startswith("test_") and callable(f)]
    failed = 0
    t0 = time.time()
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed in {time.time() - t0:.2f}s")
    sys.exit(1 if failed else 0)
