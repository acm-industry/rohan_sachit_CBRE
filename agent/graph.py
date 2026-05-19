"""LangGraph pipeline with interrupt/resume/override HITL semantics.

Wraps the sequential orchestrator (`agent.classify`) into a LangGraph
StateGraph that supports:

1. **Interrupt** — when the validator gate fires `needs_human_review=True`,
   the graph pauses via `interrupt()` and surfaces a review payload.
2. **Resume (approve)** — `Command(resume={"action": "approve"})` continues
   with the AI's prediction unchanged.
3. **Resume (override)** — `Command(resume={"action": "override", "fields": {...}})`
   replaces specified fields and records the diff in trainer_log.human_override.
4. **Post-execution override** — `update_state()` on a completed checkpoint
   modifies the final decision after the fact (supervisor review).

Thread-ID strategy: `f"call-{transcript_id}"` — one thread per call,
enabling replay and audit trail via the checkpointer.

During eval (no live human), the harness auto-approves every interrupt
so scoring completes without blocking. This is documented behavior per
the issue #17 spec.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.classify import classify as run_pipeline
from agent.nodes.trainer_log import assemble_trainer_log, build_ai_prediction


class CallState(TypedDict, total=False):
    """Graph state shape — carries the full prediction through the pipeline."""
    transcript_id: str
    turns: List[dict]
    caller_phone: Optional[str]
    prediction: Dict[str, Any]
    needs_human_review: bool
    review_payload: Dict[str, Any]
    human_override: Optional[Dict[str, Any]]


def _run_pipeline_node(state: CallState) -> dict:
    """Execute the full classification pipeline (all 9 nodes).

    This is the load-bearing node — it runs extract, classify, location,
    risk, validate, vendor, clarify, summary, and trainer_log assembly.
    """
    turns = state["turns"]
    caller_phone = state.get("caller_phone")

    prediction = run_pipeline(turns, caller_phone)

    needs_review = bool(prediction.get("needs_human_review", False))

    return {
        "prediction": prediction,
        "needs_human_review": needs_review,
        "review_payload": {
            "type": "review_required" if needs_review else "auto_resolved",
            "category": prediction.get("category"),
            "subcategory": prediction.get("subcategory"),
            "risk_level": prediction.get("risk_level"),
            "dispatched_vendor_id": prediction.get("dispatched_vendor_id"),
            "dispatched_emergency_services": prediction.get("dispatched_emergency_services"),
            "call_summary": prediction.get("call_summary"),
            "validator_reasons": (
                prediction.get("trainer_log", {})
                .get("ai_prediction", {})
                .get("validator_reasons", [])
            ),
        },
    }


def _review_gate_node(state: CallState) -> dict:
    """Conditionally interrupt for human review.

    If the validator flagged `needs_human_review=True`, this node calls
    `interrupt()` which pauses the graph. The human (or eval harness)
    resumes with either:
      - {"action": "approve"}
      - {"action": "override", "fields": {"risk_level": "LOW", ...}}
    """
    if not state.get("needs_human_review"):
        return {"human_override": None}

    decision = interrupt(state["review_payload"])

    action = decision.get("action", "approve") if isinstance(decision, dict) else "approve"

    if action == "override" and isinstance(decision, dict):
        override_fields = decision.get("fields", {})
        prediction = state["prediction"]
        updated_prediction = {**prediction, **override_fields}

        ai_pred = prediction.get("trainer_log", {}).get("ai_prediction", {})
        updated_prediction["trainer_log"] = assemble_trainer_log(
            full_transcript=prediction.get("trainer_log", {}).get("full_transcript", ""),
            ai_prediction=ai_pred,
            human_override=override_fields,
            final_decision={**ai_pred, **override_fields},
        )

        return {
            "prediction": updated_prediction,
            "human_override": override_fields,
        }

    return {"human_override": None}


def build_graph() -> StateGraph:
    """Construct the StateGraph (uncompiled). Call .compile(checkpointer=...) to use."""
    workflow = StateGraph(CallState)

    workflow.add_node("run_pipeline", _run_pipeline_node)
    workflow.add_node("review_gate", _review_gate_node)

    workflow.add_edge(START, "run_pipeline")
    workflow.add_edge("run_pipeline", "review_gate")
    workflow.add_edge("review_gate", END)

    return workflow


def compile_graph(*, checkpointer=None):
    """Compile the graph with a checkpointer. Defaults to InMemorySaver."""
    if checkpointer is None:
        checkpointer = InMemorySaver()
    workflow = build_graph()
    return workflow.compile(checkpointer=checkpointer)


def invoke_call(
    graph,
    *,
    transcript_id: str,
    turns: List[dict],
    caller_phone: Optional[str] = None,
    auto_approve: bool = True,
) -> Dict[str, Any]:
    """Run a single call through the graph with interrupt handling.

    Args:
        graph: compiled StateGraph.
        transcript_id: unique ID for thread/checkpoint.
        turns: dialogue turn list.
        caller_phone: caller phone number (optional).
        auto_approve: if True (eval mode), automatically approves any
            interrupt without human intervention.

    Returns:
        The final prediction dict.
    """
    config = {"configurable": {"thread_id": f"call-{transcript_id}"}}

    initial_state = {
        "transcript_id": transcript_id,
        "turns": turns,
        "caller_phone": caller_phone,
    }

    output = graph.invoke(initial_state, config)

    if auto_approve:
        state = graph.get_state(config)
        if state.next:
            output = graph.invoke(
                Command(resume={"action": "approve"}), config
            )

    return output.get("prediction", output)


def override_after_execution(
    graph,
    *,
    transcript_id: str,
    override_fields: Dict[str, Any],
) -> None:
    """Post-execution override via update_state().

    Modifies the checkpoint for a completed call — simulates a supervisor
    correcting the AI's decision during post-shift review.
    """
    config = {"configurable": {"thread_id": f"call-{transcript_id}"}}

    state = graph.get_state(config)
    if not state.values:
        raise ValueError(f"No checkpoint found for thread call-{transcript_id}")

    prediction = state.values.get("prediction", {})
    ai_pred = prediction.get("trainer_log", {}).get("ai_prediction", {})

    updated_prediction = {**prediction, **override_fields}
    updated_prediction["trainer_log"] = assemble_trainer_log(
        full_transcript=prediction.get("trainer_log", {}).get("full_transcript", ""),
        ai_prediction=ai_pred,
        human_override=override_fields,
        final_decision={**ai_pred, **override_fields},
    )

    graph.update_state(
        config,
        {"prediction": updated_prediction, "human_override": override_fields},
    )
