"""Graph nodes for the LangGraph pipeline.

Each module here is one node: a pure function from state to state-update.
The graph wiring lives in agent/graph.py (lands in issue #17). For now
nodes are also runnable standalone via their public entrypoints —
`extract.extract(turns, caller_phone) -> Extraction` etc. — which keeps
unit tests simple and lets the issue-1 stub call into them piecemeal as
the pipeline fills out.
"""
