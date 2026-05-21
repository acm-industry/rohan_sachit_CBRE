"""Spec-compliant entrypoint shim.

The final-assignment brief shows the example grader command as:

    python evaluation/run_eval.py \
        --agent your_submission.agent:classify \
        --eval evaluation/eval_transcripts_test.json

This module exists solely so that literal command works. The real
orchestrator is at `agent/classify.py`; this is a one-line re-export.

Either of these `--agent` flags is equivalent and produces identical
predictions:

    --agent your_submission.agent:classify   # the spec's literal template
    --agent agent.classify:classify          # the native module path
"""
from agent.classify import classify

__all__ = ["classify"]
