"""RAG layer over `operational/historical_records.json`.

Layout:
- `build_index.py` — CLI to embed + persist the 10K-record corpus.
- `retriever.py` — (issue #8) retrieval API on top of the persisted store.

Constants here are the single source of truth for builder + retriever:
they must agree on the on-disk store path, the chroma collection name,
and the embedding model. Changing the embedding model requires a full
rebuild.
"""
from __future__ import annotations

from pathlib import Path

# Where the persisted chroma store lives. Gitignored — built on first run
# (or on demand via `python -m agent.rag.build_index --force`).
STORE_DIR: Path = Path(__file__).resolve().parent / "chroma_store"

# Single chroma collection per build; keep in lockstep with the retriever.
COLLECTION_NAME: str = "historical_records"

# Schema of every chroma metadata row. Listed here so the builder, the
# retriever, and any analysis script all agree. `was_audit_flagged` is a
# placeholder: it lands as `False` in this issue and gets joined against
# `evaluation/qa_audit_findings.json` in issue #9.
METADATA_KEYS: tuple[str, ...] = (
    "ticket_id",
    "intake_category",
    "intake_subcategory",
    "intake_risk_level",
    "final_category",
    "final_subcategory",
    "final_risk_level",
    "assigned_vendor_id",
    "building_type",
    "city",
    "was_audit_flagged",
)
