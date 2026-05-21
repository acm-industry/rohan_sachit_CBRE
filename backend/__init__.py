"""Demo-only FastAPI backend.

This package is a thin, localhost-only layer over `agent.classify` that
the live demo frontend talks to. It is NOT part of the graded submission
path — `agent.classify:classify` is untouched, and this backend imports
the events-aware sibling `classify_with_events()` to drive per-stage
animation and HITL pause/resume in the UI.
"""
