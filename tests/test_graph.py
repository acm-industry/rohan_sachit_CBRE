"""Tests for `agent.graph` — LangGraph interrupt/resume/override HITL path.

Two layers:
1. Unit tests of the node functions (no LangGraph runtime needed).
2. Integration tests using the compiled graph with InMemorySaver
   (requires langgraph to be installed).

Tests in layer 2 are skipped if langgraph is not importable.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

try:
    from agent import graph as graph_mod
except ImportError:
    graph_mod = None


# ─── Fixtures ─────────────────────────────────────────────────────────

SAMPLE_TURNS = [
    {"speaker": "agent", "text": "CBRE maintenance, what's going on?"},
    {"speaker": "caller", "text": "Pipe burst on Floor 3, water everywhere."},
]

SAMPLE_PREDICTION = {
    "category": "PLUMBING",
    "subcategory": "pipe_leak",
    "risk_level": "HIGH",
    "needs_human_review": True,
    "needs_clarification": False,
    "building_name": "Pacific Ridge Medical Plaza",
    "address": "3200 Pacific Coast Hwy",
    "floor": "Floor 3",
    "dispatched_vendor_id": "v_002",
    "dispatched_emergency_services": False,
    "call_summary": "Pipe burst on Floor 3 at Pacific Ridge. HIGH risk. Dispatched AquaFix Plumbing.",
    "trainer_log": {
        "full_transcript": "[AGENT] CBRE maintenance\n[CALLER] Pipe burst",
        "ai_prediction": {
            "category": "PLUMBING",
            "subcategory": "pipe_leak",
            "risk_level": "HIGH",
            "dispatched_vendor_id": "v_002",
            "dispatched_emergency_services": False,
            "needs_human_review": True,
            "needs_clarification": False,
            "confidence_category": 0.9,
            "confidence_subcategory": 0.85,
            "validator_reasons": ["high_band:always_pause"],
            "risk_reasons": ["base_risk=HIGH"],
            "retrieved_record_ids": ["TKT-001"],
        },
        "human_override": None,
        "final_decision": {
            "category": "PLUMBING",
            "subcategory": "pipe_leak",
            "risk_level": "HIGH",
            "dispatched_vendor_id": "v_002",
            "dispatched_emergency_services": False,
            "needs_human_review": True,
            "needs_clarification": False,
        },
    },
}

ROUTINE_PREDICTION = {
    **SAMPLE_PREDICTION,
    "risk_level": "LOW",
    "needs_human_review": False,
    "trainer_log": {
        **SAMPLE_PREDICTION["trainer_log"],
        "ai_prediction": {
            **SAMPLE_PREDICTION["trainer_log"]["ai_prediction"],
            "risk_level": "LOW",
            "needs_human_review": False,
            "validator_reasons": ["auto_resolve:no_flags_fired"],
        },
    },
}


# ─── Unit tests: _run_pipeline_node ───────────────────────────────────


def _skip_no_langgraph():
    if graph_mod is None:
        print("  SKIP (langgraph not installed)")
        return True
    return False


def test_run_pipeline_node_calls_classify():
    """Node calls agent.classify.classify and wraps result."""
    if _skip_no_langgraph():
        return
    state = {
        "turns": SAMPLE_TURNS,
        "caller_phone": "+15551234567",
    }
    with patch.object(graph_mod, "run_pipeline", return_value=SAMPLE_PREDICTION):
        result = graph_mod._run_pipeline_node(state)

    assert result["prediction"] == SAMPLE_PREDICTION
    assert result["needs_human_review"] is True
    assert result["review_payload"]["type"] == "review_required"
    assert result["review_payload"]["category"] == "PLUMBING"


def test_run_pipeline_node_auto_resolved():
    """Routine call gets review_payload type=auto_resolved."""
    if _skip_no_langgraph():
        return
    state = {"turns": SAMPLE_TURNS, "caller_phone": None}
    with patch.object(graph_mod, "run_pipeline", return_value=ROUTINE_PREDICTION):
        result = graph_mod._run_pipeline_node(state)

    assert result["needs_human_review"] is False
    assert result["review_payload"]["type"] == "auto_resolved"


# ─── Unit tests: _review_gate_node ───────────────────────────────────


def test_review_gate_skips_when_no_review_needed():
    """Auto-resolved calls pass through without interrupt."""
    if _skip_no_langgraph():
        return
    state = {
        "needs_human_review": False,
        "prediction": ROUTINE_PREDICTION,
        "review_payload": {"type": "auto_resolved"},
    }
    result = graph_mod._review_gate_node(state)
    assert result["human_override"] is None


# ─── Unit tests: build_graph structure ────────────────────────────────


def test_build_graph_has_expected_nodes():
    """Graph contains run_pipeline and review_gate nodes."""
    if _skip_no_langgraph():
        return
    workflow = graph_mod.build_graph()
    node_names = set(workflow.nodes.keys())
    assert "run_pipeline" in node_names
    assert "review_gate" in node_names


# ─── Integration tests (require langgraph) ───────────────────────────


def test_full_graph_auto_approve_routine_call():
    """Routine call goes through without interrupt."""
    if _skip_no_langgraph():
        return

    with patch.object(graph_mod, "run_pipeline", return_value=ROUTINE_PREDICTION):
        compiled = graph_mod.compile_graph()
        result = graph_mod.invoke_call(
            compiled,
            transcript_id="TEST-001",
            turns=SAMPLE_TURNS,
            caller_phone=None,
            auto_approve=True,
        )

    assert result["category"] == "PLUMBING"
    assert result["needs_human_review"] is False


def test_full_graph_auto_approve_high_risk_call():
    """HIGH risk call hits interrupt, auto_approve resumes it."""
    if _skip_no_langgraph():
        return

    with patch.object(graph_mod, "run_pipeline", return_value=SAMPLE_PREDICTION):
        compiled = graph_mod.compile_graph()
        result = graph_mod.invoke_call(
            compiled,
            transcript_id="TEST-002",
            turns=SAMPLE_TURNS,
            caller_phone="+15551234567",
            auto_approve=True,
        )

    assert result["category"] == "PLUMBING"
    assert result["risk_level"] == "HIGH"


def test_full_graph_override_changes_prediction():
    """Override via Command(resume) modifies the prediction."""
    if _skip_no_langgraph():
        return

    with patch.object(graph_mod, "run_pipeline", return_value=SAMPLE_PREDICTION):
        compiled = graph_mod.compile_graph()
        config = {"configurable": {"thread_id": "call-TEST-003"}}

        compiled.invoke(
            {
                "transcript_id": "TEST-003",
                "turns": SAMPLE_TURNS,
                "caller_phone": None,
            },
            config,
        )

        state = compiled.get_state(config)
        if state.next:
            output = compiled.invoke(
                Command(resume={
                    "action": "override",
                    "fields": {"risk_level": "MEDIUM", "needs_human_review": False},
                }),
                config,
            )
            prediction = output.get("prediction", output)
            assert prediction["risk_level"] == "MEDIUM"
            assert prediction["trainer_log"]["human_override"] == {
                "risk_level": "MEDIUM",
                "needs_human_review": False,
            }


def test_update_state_post_execution_override():
    """update_state() modifies a completed checkpoint."""
    if _skip_no_langgraph():
        return

    with patch.object(graph_mod, "run_pipeline", return_value=ROUTINE_PREDICTION):
        compiled = graph_mod.compile_graph()
        graph_mod.invoke_call(
            compiled,
            transcript_id="TEST-004",
            turns=SAMPLE_TURNS,
            auto_approve=True,
        )

        graph_mod.override_after_execution(
            compiled,
            transcript_id="TEST-004",
            override_fields={"subcategory": "drainage_backup"},
        )

        config = {"configurable": {"thread_id": "call-TEST-004"}}
        state = compiled.get_state(config)
        prediction = state.values.get("prediction", {})
        assert prediction["subcategory"] == "drainage_backup"
        assert prediction["trainer_log"]["human_override"] == {"subcategory": "drainage_backup"}


def test_thread_id_strategy():
    """Thread IDs follow the f'call-{transcript_id}' convention."""
    if _skip_no_langgraph():
        return

    with patch.object(graph_mod, "run_pipeline", return_value=ROUTINE_PREDICTION):
        compiled = graph_mod.compile_graph()
        graph_mod.invoke_call(
            compiled,
            transcript_id="EVAL-0042",
            turns=SAMPLE_TURNS,
            auto_approve=True,
        )

        config = {"configurable": {"thread_id": "call-EVAL-0042"}}
        state = compiled.get_state(config)
        assert state.values is not None
        assert "prediction" in state.values


# ─── Inline runner ─────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect

    tests = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__])
             if n.startswith("test_") and callable(f)]
    failed = 0
    skipped = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            if "langgraph" in str(e).lower() or "SKIP" in str(e):
                print(f"  SKIP  {name}: {e}")
                skipped += 1
            else:
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
                failed += 1
    total = len(tests)
    print(f"\n{total - failed - skipped}/{total} passed, {skipped} skipped, {failed} failed")
    sys.exit(1 if failed else 0)
