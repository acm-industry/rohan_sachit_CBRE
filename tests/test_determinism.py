"""Determinism and sealed-packaging checks (issue #28).

The brief's contract:
  - `classify()` is reproducible on the same input (modulo provider noise).
  - All LLM calls go through one constructor — temperature 0, explicit
    model, OpenAI `seed` set.
  - No file reads outside `operational/`, `evaluation/`, or `agent/`.

These tests are static / stubbed so they run in CI without network. The
real-LLM smoke test for end-to-end reproducibility lives in
`test_classify.py::test_e2e_*` (which only runs when `OPENAI_API_KEY` is set).
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import classify as classify_module  # noqa: E402
from agent import config  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = REPO_ROOT / "agent"


# ─── Config contract ───────────────────────────────────────────────────


def test_default_settings_pin_determinism_knobs():
    """Defaults must keep the pipeline reproducible without env tweaks."""
    s = config.get_settings(validate=False)
    assert s.chat_temperature == 0.0, (
        f"temperature default must be 0.0 for determinism; got {s.chat_temperature}"
    )
    assert s.chat_seed is not None, "chat_seed default must be set (OpenAI seed param)"
    assert isinstance(s.chat_model, str) and s.chat_model, (
        "chat_model must be an explicit string (no None / empty default)"
    )
    assert isinstance(s.embedding_model, str) and s.embedding_model


def test_build_chat_llm_passes_determinism_kwargs():
    """The central LLM builder forwards temperature, model, and seed.

    Patches `langchain_openai.ChatOpenAI` to a recording stub so we can
    assert on the exact constructor kwargs without needing a real API key.
    """
    import langchain_openai

    captured: dict = {}

    class _RecordingChatOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    real = langchain_openai.ChatOpenAI
    langchain_openai.ChatOpenAI = _RecordingChatOpenAI
    try:
        config.build_chat_llm(config.get_settings(validate=False))
    finally:
        langchain_openai.ChatOpenAI = real

    assert captured.get("temperature") == 0.0
    assert isinstance(captured.get("model"), str) and captured["model"]
    assert "seed" in captured and captured["seed"] is not None


# ─── classify() reproducibility ────────────────────────────────────────


def test_classify_is_deterministic_for_identical_input():
    """Twice on the same input → equal output dict.

    The orchestrator's only stochastic surface is the LLM provider —
    pinned via `temperature=0` + explicit `seed` in `agent.config`,
    but the provider's "deterministic mode" is best-effort, not a
    guarantee. To make this test verify *pipeline* determinism (not
    provider stability), we install the same LLM + RAG stubs used by
    `test_contract.py`, then assert two consecutive calls produce
    identical output. Real-LLM end-to-end reproducibility is covered
    separately by `test_classify.py` (API-key-gated).
    """
    import agent.config
    import agent.nodes.classify as _classify_node
    from agent.nodes.classify import (
        CategoryEnum, Classification, SubcategoryEnum,
    )
    from agent.nodes.extract import Extraction, FieldConfidence

    def _canned(schema):
        if schema is Extraction:
            return Extraction(
                problem_summary="determinism-test stub",
                building_name=None, floor=None, suite=None,
                urgency_cues=[], caller_role="tenant", language="en",
                confidence=FieldConfidence(
                    problem_summary=0.5, building_name=0.0,
                    floor=0.0, suite=0.0, caller_role=0.0,
                ),
            )
        if schema is Classification:
            return Classification(
                category=CategoryEnum.ELEVATOR,
                subcategory=SubcategoryEnum.minor_issue,
                confidence_category=0.5, confidence_subcategory=0.5,
                reasoning="determinism-test stub",
                is_fallback=False, retrieved_record_ids=[],
            )
        raise AssertionError(f"unexpected schema: {schema!r}")

    class _StubChat:
        def with_structured_output(self, schema):
            class _Inv:
                def invoke(_self, _prompt): return _canned(schema)
            return _Inv()

    orig_build = agent.config.build_chat_llm
    orig_retrieve = _classify_node.retrieve
    agent.config.build_chat_llm = lambda settings=None: _StubChat()
    _classify_node.retrieve = lambda *_a, **_kw: []
    try:
        turns = [
            {"speaker": "agent", "text": "Hello, how can I help?"},
            {"speaker": "caller", "text": "There's water on the floor of suite 408."},
        ]
        phone = "+15551234567"
        a = classify_module.classify(turns, phone)
        b = classify_module.classify(turns, phone)
        assert a == b, "classify() must return identical output for identical input"
    finally:
        agent.config.build_chat_llm = orig_build
        _classify_node.retrieve = orig_retrieve


# ─── No `ChatOpenAI(` constructors outside agent/config.py ─────────────


def test_no_chatopenai_constructor_outside_config():
    """All LLM call sites must funnel through `config.build_chat_llm`.

    A bare `ChatOpenAI(...)` constructor elsewhere bypasses the central
    temperature/seed/model contract and silently breaks determinism. This
    catches the regression at lint time rather than at grading time.
    """
    pattern = re.compile(r"\bChatOpenAI\s*\(")
    violations: list[str] = []
    for py in AGENT_DIR.rglob("*.py"):
        if py.resolve() == (AGENT_DIR / "config.py").resolve():
            continue
        text = py.read_text()
        for ln, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                violations.append(f"{py.relative_to(REPO_ROOT)}:{ln}: {line.strip()}")
    assert not violations, (
        "Found `ChatOpenAI(` constructors outside agent/config.py — "
        "route them through `config.build_chat_llm()` instead:\n  "
        + "\n  ".join(violations)
    )


# ─── File-path policy: agent/ only reads from agent/, operational/, evaluation/ ─


# Top-level directories any source file under `agent/` is allowed to read from.
_ALLOWED_TOP_LEVEL_DIRS = {"agent", "operational", "evaluation"}


def _resolve_path_arg(node: ast.AST) -> Path | None:
    """Best-effort static eval of a `Path(...)` chain rooted at __file__.

    Recognises the patterns the codebase actually uses:
      Path(__file__).resolve().parent / "x" / "y.json"
      Path(__file__).resolve().parents[2] / "operational" / "z.json"
    Returns the resolved Path (relative to this test's view of the file)
    or None if the expression is too dynamic to evaluate statically.
    """
    # Walk binary-op chains of `/` joins down to the leftmost call expr.
    parts: list[str] = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
            parts.insert(0, node.right.value)
        else:
            return None
        node = node.left

    base = _resolve_pathlib_base(node)
    if base is None:
        return None
    return base.joinpath(*parts)


def _resolve_pathlib_base(node: ast.AST) -> Path | None:
    """Resolve `Path(__file__).resolve().parent[s][i]` for the file under analysis.

    `node` is expected to be the call/attribute/subscript chain ending in
    a Path-typed expression. Returns the absolute Path or None when the
    expression isn't one of the supported shapes.
    """
    # parents[i] subscript
    if isinstance(node, ast.Subscript):
        inner = _resolve_pathlib_base(node.value)
        if inner is None:
            return None
        # Must be a parents subscript: previous resolver returns the file path
        # itself for `Path(__file__).resolve()` and applies `.parent` walks via
        # the attribute branch; for `parents[i]` we walk i+1 ancestors from
        # the file path.
        if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int):
            return None
        # We can't distinguish `.parents[i]` from `.parent` here without
        # the attribute context; the attribute branch handles `.parents`
        # by deferring to this function with the subscript node.
        return inner.parents[node.slice.value]
    if isinstance(node, ast.Attribute):
        inner = _resolve_pathlib_base(node.value)
        if inner is None:
            return None
        if node.attr == "resolve":
            return inner
        if node.attr == "parent":
            return inner.parent
        if node.attr == "parents":
            # The `.parents` attribute itself isn't a path — only `.parents[i]`
            # is, and that lookup goes through Subscript above. Return inner so
            # the Subscript handler can do the arithmetic.
            return inner
        return None
    if isinstance(node, ast.Call):
        # `Path(__file__)` or `Path(__file__).resolve()`
        if isinstance(node.func, ast.Attribute) and node.func.attr == "resolve":
            return _resolve_pathlib_base(node.func.value)
        if isinstance(node.func, ast.Name) and node.func.id == "Path":
            if len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == "__file__":
                # The path of the source file we're currently analysing is
                # injected by the caller; return a sentinel here that gets
                # rebound by the caller via a closure.
                return _CURRENT_FILE_PATH
    return None


_CURRENT_FILE_PATH: Path | None = None


def test_no_file_reads_outside_allowed_dirs():
    """Every `Path(__file__)...` constant in agent/ must point inside agent/, operational/, or evaluation/."""
    global _CURRENT_FILE_PATH
    violations: list[str] = []
    for py in AGENT_DIR.rglob("*.py"):
        _CURRENT_FILE_PATH = py
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # Only inspect module-level / class-level constant assignments
            # — these are the data-file paths the codebase declares once.
            if not isinstance(node, ast.Assign):
                continue
            resolved = _resolve_path_arg(node.value)
            if resolved is None:
                continue
            try:
                rel = resolved.resolve().relative_to(REPO_ROOT.resolve())
            except (ValueError, OSError):
                violations.append(
                    f"{py.relative_to(REPO_ROOT)}: path escapes repo root → {resolved}"
                )
                continue
            top = rel.parts[0] if rel.parts else ""
            if top not in _ALLOWED_TOP_LEVEL_DIRS:
                violations.append(
                    f"{py.relative_to(REPO_ROOT)}: reads from disallowed dir '{top}/' → {rel}"
                )
    _CURRENT_FILE_PATH = None
    assert not violations, (
        "Source files under agent/ may only read from "
        f"{sorted(_ALLOWED_TOP_LEVEL_DIRS)}; offending paths:\n  "
        + "\n  ".join(violations)
    )


# ─── Inline runner ─────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect
    import traceback

    tests = [
        (n, f) for n, f in inspect.getmembers(sys.modules[__name__])
        if n.startswith("test_") and callable(f)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
            failed += 1
    total = len(tests)
    print(f"\n{total - failed}/{total} passed, {failed} failed")
    sys.exit(1 if failed else 0)
