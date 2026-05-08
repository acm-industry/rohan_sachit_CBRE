"""Integration tests for the agent pipeline (agent.classify:classify).

Tests the orchestrator's fallback path and schema conformance.
Full end-to-end testing requires LLM dependencies (pydantic, langchain);
these tests validate the safe-fallback path and output schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SAMPLE_TURNS = [
    {"speaker": "agent", "text": "CBRE maintenance, this is Dana. What's going on?"},
    {"speaker": "caller", "text": "The breakroom sink on Floor 9 won't drain."},
]

REQUIRED_KEYS = {
    "category", "subcategory", "risk_level", "needs_human_review",
    "needs_clarification", "building_name", "address", "floor",
    "dispatched_vendor_id", "dispatched_emergency_services",
    "call_summary", "trainer_log",
}

TRAINER_LOG_REQUIRED_KEYS = {"full_transcript", "ai_prediction", "human_override", "final_decision"}


def _import_safe_fallback():
    """Import just the _safe_fallback and _flatten without triggering LLM imports."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "agent_classify",
        str(Path(__file__).resolve().parents[1] / "agent" / "classify.py"),
    )
    # We can't load the module normally because it imports pydantic at top level.
    # Instead, read the file and extract the helper functions we need to test.
    source = (Path(__file__).resolve().parents[1] / "agent" / "classify.py").read_text()

    # Execute only the _flatten and _safe_fallback functions in isolation
    ns = {"__name__": "agent.classify", "logging": __import__("logging")}
    # Extract the two functions we need (they have no LLM deps)
    lines = source.split("\n")
    in_func = False
    func_lines = []
    target_funcs = {"def _flatten", "def _safe_fallback"}
    current_indent = 0

    for line in lines:
        if any(line.startswith(t) for t in target_funcs):
            in_func = True
            func_lines.append(line)
            current_indent = len(line) - len(line.lstrip())
        elif in_func:
            if line.strip() == "" or line.startswith(" ") or line.startswith("\t"):
                func_lines.append(line)
            else:
                in_func = False
                func_lines.append("")

    func_code = "\n".join(func_lines)
    exec(compile(func_code, "<test>", "exec"), ns)
    return ns["_safe_fallback"], ns["_flatten"]


_safe_fallback, _flatten = _import_safe_fallback()


def test_prediction_schema_has_all_keys():
    """Verify output keys match what scoring.py expects."""
    result = _safe_fallback(SAMPLE_TURNS, "test error")
    assert REQUIRED_KEYS.issubset(result.keys()), f"Missing keys: {REQUIRED_KEYS - result.keys()}"


def test_trainer_log_has_required_keys():
    """trainer_log must have full_transcript, ai_prediction, human_override, final_decision."""
    result = _safe_fallback(SAMPLE_TURNS, "test error")
    tl = result["trainer_log"]
    assert isinstance(tl, dict)
    assert TRAINER_LOG_REQUIRED_KEYS.issubset(tl.keys()), f"Missing: {TRAINER_LOG_REQUIRED_KEYS - tl.keys()}"


def test_safe_fallback_no_false_911():
    """Fallback path must never dispatch emergency services."""
    result = _safe_fallback(SAMPLE_TURNS, "any error")
    assert result["dispatched_emergency_services"] is False
    assert result["needs_human_review"] is True


def test_safe_fallback_has_call_summary():
    """Call summary must be >= 30 chars to pass scoring."""
    result = _safe_fallback(SAMPLE_TURNS, "test")
    assert len(result["call_summary"]) >= 30


def test_full_transcript_in_trainer_log():
    """Trainer log must contain the full transcript text."""
    result = _safe_fallback(SAMPLE_TURNS, "test")
    tl = result["trainer_log"]
    assert "breakroom sink" in tl["full_transcript"]
    assert "[CALLER]" in tl["full_transcript"]


def test_flatten_preserves_speaker_tags():
    """_flatten produces [AGENT]/[CALLER] tagged lines."""
    text = _flatten(SAMPLE_TURNS)
    assert "[AGENT]" in text
    assert "[CALLER]" in text
    assert "breakroom sink" in text


def test_safe_fallback_final_decision_matches_base():
    """When no override, final_decision should mirror the base prediction."""
    result = _safe_fallback(SAMPLE_TURNS, "err")
    tl = result["trainer_log"]
    assert tl["human_override"] is None
    assert tl["final_decision"]["needs_human_review"] is True
    assert tl["final_decision"]["dispatched_emergency_services"] is False


if __name__ == "__main__":
    import inspect

    tests = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__])
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
