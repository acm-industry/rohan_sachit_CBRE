"""Build the RAG vector index over `operational/historical_records.json`.

Run:
    python -m agent.rag.build_index               # build if no store yet
    python -m agent.rag.build_index --force       # rebuild from scratch
    python -m agent.rag.build_index --limit 100   # subset (smoke test)

What this builds
----------------
A persisted chromadb collection at `agent/rag/chroma_store/`. Each ticket
becomes one document. The document text is a structured concatenation of
the verbatim caller transcript and the intake / dispatch / resolution
notes — caller transcript first because it's the richest signal per the
brief. Metadata covers the join keys downstream nodes need: intake +
final category / subcategory / risk_level, the assigned vendor, building
type, city, and a `was_audit_flagged` placeholder that issue #9 fills.

Embedding model defaults to `text-embedding-3-small` via `agent.config`.
Changing the model requires a rebuild — the on-disk store has no concept
of which embedder produced its vectors.

Cost / time
-----------
Average ticket text is ~400 chars (~100 tokens). The full 10K corpus is
~1M tokens; at text-embedding-3-small pricing that's ≈ $0.02. Wall time
on a typical laptop laptop: 1–3 minutes (well under the AC's 10 min cap).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

# chromadb's posthog telemetry is incompatible with Python 3.14 and emits
# noisy "capture() takes 1 positional argument" warnings. Opt out before
# any chromadb code path runs and silence the leftover error logger that
# fires even when telemetry is disabled.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import logging  # noqa: E402
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from langchain_core.documents import Document  # noqa: E402

from agent import config  # noqa: E402
from agent.data import audit  # noqa: E402
from agent.rag import COLLECTION_NAME, METADATA_KEYS, STORE_DIR  # noqa: E402


HISTORICAL_PATH: Path = (
    Path(__file__).resolve().parents[2] / "operational" / "historical_records.json"
)

# Hard cap on per-document text length. text-embedding-3-small accepts
# ~8K tokens (~32K chars). Real-world tickets average ~400 chars and
# top out at ~540, so this is a safety belt rather than a routine path.
MAX_DOC_CHARS: int = 24_000

# Embedding batch size sent to OpenAI. Chroma + langchain handle batching,
# but we set this explicitly for predictable throughput at 10K scale.
EMBEDDING_BATCH_SIZE: int = 256


# ─── Pure builders (no I/O, no network — easily unit-testable) ─────────


def _format_document_text(record: dict) -> str:
    """Produce one embeddable text block per ticket.

    Order matters for retrieval quality: the caller's verbatim quote
    leads, then operator shorthand, then dispatch + resolution. Empty
    sections are dropped so they don't dilute the embedding.
    """
    parts: List[str] = []
    if record.get("call_recording_transcript"):
        parts.append(f"CALLER TRANSCRIPT:\n{record['call_recording_transcript']}")
    if record.get("intake_notes"):
        parts.append(f"INTAKE NOTES: {record['intake_notes']}")
    if record.get("dispatch_notes"):
        parts.append(f"DISPATCH NOTES: {record['dispatch_notes']}")
    if record.get("resolution_notes"):
        parts.append(f"RESOLUTION NOTES: {record['resolution_notes']}")
    text = "\n\n".join(parts)
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS]
    return text


def _format_metadata(record: dict) -> dict:
    """Extract the metadata fields downstream nodes filter and rank by.

    Joins `evaluation/qa_audit_findings.json` (issue #9) to stamp the
    audit booleans into each row. Records without a finding default to
    every audit flag = False; records with a finding but no specific
    flags set (auditor reviewed and confirmed intake) likewise stay False
    on `was_audit_flagged` — that key tracks "real problem the auditor
    caught", not "this ticket was reviewed".

    Chroma metadata values must be primitives (str / int / float / bool);
    `None`s are dropped because some chromadb versions choke on them in
    where-clauses.
    """
    finding = audit.lookup(record.get("ticket_id"))
    md: dict[str, Any] = {
        "ticket_id": record.get("ticket_id"),
        "intake_category": record.get("intake_category"),
        "intake_subcategory": record.get("intake_subcategory"),
        "intake_risk_level": record.get("intake_risk_level"),
        "final_category": record.get("final_category"),
        "final_subcategory": record.get("final_subcategory"),
        "final_risk_level": record.get("final_risk_level"),
        "assigned_vendor_id": record.get("assigned_vendor_id"),
        "building_type": record.get("building_type"),
        "city": record.get("city"),
        "was_audit_flagged": bool(finding and finding.has_any_flag),
        "audit_over_escalated": bool(finding and finding.was_over_escalated),
        "audit_reclassified": bool(finding and finding.was_reclassified),
        "audit_floor_wrong": bool(finding and finding.floor_was_wrong_at_intake),
    }
    return {k: v for k, v in md.items() if v is not None}


def _to_documents(records: Iterable[dict]) -> List[Document]:
    out: List[Document] = []
    for r in records:
        text = _format_document_text(r)
        if not text.strip():
            continue
        out.append(Document(page_content=text, metadata=_format_metadata(r)))
    return out


def _load_records(limit: Optional[int] = None) -> List[dict]:
    raw = json.loads(HISTORICAL_PATH.read_text())
    if limit is not None:
        raw = raw[:limit]
    return raw


# ─── Build orchestration ───────────────────────────────────────────────


def build(
    *,
    force: bool = False,
    limit: Optional[int] = None,
    verbose: bool = True,
    embeddings: Any = None,
    store_dir: Optional[Path] = None,
) -> int:
    """Build (or refresh) the chroma store. Returns the document count.

    Returns -1 when an existing store is left intact (no `--force`).

    `embeddings` and `store_dir` are injection points for tests — pass a
    `FakeEmbeddings` instance and a `tmp_path` to validate end-to-end
    without hitting OpenAI.
    """
    from langchain_chroma import Chroma  # heavy imports kept off module load

    target_dir = Path(store_dir) if store_dir else STORE_DIR

    if target_dir.exists() and any(target_dir.iterdir()):
        if not force:
            if verbose:
                print(
                    f"[rag] store exists at {target_dir} — skipping "
                    "(use --force to rebuild)"
                )
            return -1
        if verbose:
            print(f"[rag] --force: removing existing store at {target_dir}")
        shutil.rmtree(target_dir)
        # chromadb keeps an in-process SharedSystemClient cache keyed on
        # persist path; without clearing it, a subsequent open of the same
        # path inside the same Python process tries to write through stale
        # SQLite handles and trips "attempt to write a readonly database".
        try:
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass

    if embeddings is None:
        from langchain_openai import OpenAIEmbeddings

        settings = config.get_settings()
        if verbose:
            print(f"[rag] embedding model: {settings.embedding_model}")
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            chunk_size=EMBEDDING_BATCH_SIZE,
        )
    elif verbose:
        print("[rag] using injected embeddings (test mode)")

    if verbose:
        print(f"[rag] loading historicals from {HISTORICAL_PATH.name}...")
    records = _load_records(limit=limit)
    docs = _to_documents(records)
    if verbose:
        print(f"[rag] {len(docs)} documents to embed (from {len(records)} records)")

    target_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(target_dir),
        collection_name=COLLECTION_NAME,
    )
    elapsed = time.time() - t0

    if verbose:
        rate = len(docs) / elapsed if elapsed > 0 else 0
        print(
            f"[rag] embedded + persisted {len(docs)} docs in {elapsed:.1f}s "
            f"({rate:.0f} docs/s)"
        )
        print(f"[rag] store: {target_dir}")
    return len(docs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the RAG vector index over operational/historical_records.json"
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if a store already exists at agent/rag/chroma_store/",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap on records (smoke tests / debugging)",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress logging",
    )
    args = ap.parse_args(argv)
    n = build(force=args.force, limit=args.limit, verbose=not args.quiet)
    return 0 if n >= 0 else 0  # both paths are clean exits


if __name__ == "__main__":
    sys.exit(main())


# Re-export schema for tests / external scripts.
__all__ = (
    "build",
    "main",
    "MAX_DOC_CHARS",
    "METADATA_KEYS",
    "HISTORICAL_PATH",
)
