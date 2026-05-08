"""CBRE HITL call-intake agent.

Single public entrypoint:

    from agent.classify import classify
    classify(turns: list[dict], caller_phone: str | None) -> dict

The returned dict matches `evaluation/prediction.py::Prediction`. Downstream
issues replace the stub in `agent/classify.py` with the real LangGraph
pipeline (extract → retrieve → classify → validator → log).
"""
__version__ = "0.1.0"
