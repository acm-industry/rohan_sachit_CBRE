"""Retrieval API on top of the persisted RAG index.

Public surface is `retrieve(query, k, filters)` returning a list of
`RetrievedRecord`s. Callers (the classifier in #11, the conflict-resolver
in #9, etc.) get a typed, embedding-agnostic façade over the chroma store
that issue #7 builds — they don't have to know about `Chroma`,
`OpenAIEmbeddings`, or chromadb's `where` syntax.

Filters are pre-applied at the index level (chromadb's metadata filter,
not a post-hoc Python filter on top of vector results) so a `k=5` request
returns 5 *qualifying* hits rather than 5 hits filtered down to fewer.

Usage:

    from agent.rag.retriever import retrieve

    # Simple: no filter, default k=5
    hits = retrieve("water leaking from ceiling")

    # With a single-field metadata filter (chroma equality shorthand):
    hits = retrieve("door card reader broken", filters={"city": "Long Beach"})

    # With a multi-field filter (must use chroma's $and operator):
    hits = retrieve(
        "no heat",
        k=8,
        filters={"$and": [{"building_type": "office"}, {"city": "Los Angeles"}]},
    )
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

# Same telemetry-noise suppression as build_index.py — chromadb's posthog
# hook breaks under Python 3.14 and we opt out before any chroma import.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from agent import config  # noqa: E402
from agent.rag import COLLECTION_NAME, STORE_DIR  # noqa: E402


DEFAULT_K = 5
# The set of metadata keys callers can safely pass through `filters` —
# matches the schema written by the builder. Anything outside this list
# would silently match nothing in the chroma store. A unit test asserts
# this stays in lockstep with `agent.rag.METADATA_KEYS`.
FILTERABLE_KEYS = (
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
    "audit_over_escalated",
    "audit_reclassified",
    "audit_floor_wrong",
)


@dataclass(frozen=True)
class RetrievedRecord:
    """One hit returned by `retrieve()`.

    `text` is the full embeddable document body the builder produced — the
    structured concatenation of caller transcript + intake / dispatch /
    resolution notes. `metadata` holds every field listed in
    `agent.rag.METADATA_KEYS` that was non-None for this ticket.
    `distance` is chroma's L2 distance between the query embedding and
    this document's embedding (lower = more similar; 0.0 = identical).
    None when the underlying call didn't surface a score.

    The `corrected_*` properties are the AC's "relabel" path for
    audit-flagged records: when the QA team reclassified or de-escalated
    a ticket, classifier prompts should learn from the corrected label,
    not the flawed intake label.
    """
    text: str
    metadata: Mapping[str, Any]
    distance: Optional[float] = None

    # ─── Raw metadata accessors ────────────────────────────────────────

    @property
    def ticket_id(self) -> Optional[str]:
        return self.metadata.get("ticket_id")

    @property
    def intake_category(self) -> Optional[str]:
        return self.metadata.get("intake_category")

    @property
    def intake_subcategory(self) -> Optional[str]:
        return self.metadata.get("intake_subcategory")

    @property
    def intake_risk_level(self) -> Optional[str]:
        return self.metadata.get("intake_risk_level")

    @property
    def final_category(self) -> Optional[str]:
        return self.metadata.get("final_category")

    @property
    def final_subcategory(self) -> Optional[str]:
        return self.metadata.get("final_subcategory")

    @property
    def final_risk_level(self) -> Optional[str]:
        return self.metadata.get("final_risk_level")

    @property
    def assigned_vendor_id(self) -> Optional[str]:
        return self.metadata.get("assigned_vendor_id")

    @property
    def is_audit_flagged(self) -> bool:
        return bool(self.metadata.get("was_audit_flagged"))

    @property
    def was_reclassified(self) -> bool:
        return bool(self.metadata.get("audit_reclassified"))

    @property
    def was_over_escalated(self) -> bool:
        return bool(self.metadata.get("audit_over_escalated"))

    # ─── Corrected accessors — the audit-aware view ────────────────────

    @property
    def corrected_category(self) -> Optional[str]:
        """`final_category` when the auditor reclassified, else `intake_category`."""
        if self.was_reclassified and self.metadata.get("final_category"):
            return self.metadata.get("final_category")
        return self.metadata.get("intake_category")

    @property
    def corrected_subcategory(self) -> Optional[str]:
        """`final_subcategory` when the auditor reclassified, else `intake_subcategory`.

        This is the *primary* label the classifier should learn from on
        retrieved records — using `intake_subcategory` directly perpetuates
        the original miss on flagged tickets (e.g. "roof_leak" intake that
        was actually a `structural` life-safety call).
        """
        if self.was_reclassified and self.metadata.get("final_subcategory"):
            return self.metadata.get("final_subcategory")
        return self.metadata.get("intake_subcategory")

    @property
    def corrected_risk_level(self) -> Optional[str]:
        """`final_risk_level` when the auditor flagged over-escalation, else intake."""
        if self.was_over_escalated and self.metadata.get("final_risk_level"):
            return self.metadata.get("final_risk_level")
        return self.metadata.get("intake_risk_level")


_STORE: Any = None


def _get_store(*, embeddings: Any = None, store_dir: Any = None) -> Any:
    """Return the persisted Chroma store, building the embedder lazily.

    `embeddings` and `store_dir` are injection points for tests — the
    production path constructs `OpenAIEmbeddings` from `agent.config` and
    points at `STORE_DIR`. Callers should usually let those defaults stand.
    """
    global _STORE
    target_dir = store_dir if store_dir is not None else STORE_DIR

    if _STORE is not None and embeddings is None and store_dir is None:
        return _STORE

    from pathlib import Path
    target_path = Path(str(target_dir))
    if not target_path.exists() or not any(target_path.iterdir()):
        raise RuntimeError(
            f"RAG store not found at {target_path}. Run "
            "`python -m agent.rag.build_index` to populate it."
        )

    from langchain_chroma import Chroma

    if embeddings is None:
        from langchain_openai import OpenAIEmbeddings

        s = config.get_settings()
        embeddings = OpenAIEmbeddings(model=s.embedding_model)

    store = Chroma(
        persist_directory=str(target_path),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Cache only the production-path store; test invocations with explicit
    # injection params get a fresh instance every call.
    if embeddings is None and store_dir is None:
        _STORE = store
    return store


def _normalize_filters(filters: Optional[Mapping[str, Any]]) -> Optional[dict]:
    """Pass-through chroma `where` clauses unchanged; light validation.

    Chroma accepts equality shorthand (`{"city": "LA"}`) and operator
    forms (`{"city": {"$eq": "LA"}}`, `{"$and": [...]}`). We accept both.
    Single-field shorthand against a non-known key emits a debug log so
    typos surface during development without rejecting the call.
    """
    if not filters:
        return None
    if "$and" in filters or "$or" in filters or "$not" in filters:
        return dict(filters)
    for k in filters:
        if k not in FILTERABLE_KEYS and not k.startswith("$"):
            logging.getLogger(__name__).debug(
                "filter on metadata key %r is outside FILTERABLE_KEYS; "
                "results may be empty",
                k,
            )
    return dict(filters)


def retrieve(
    query: str,
    *,
    k: int = DEFAULT_K,
    filters: Optional[Mapping[str, Any]] = None,
    embeddings: Any = None,
    store_dir: Any = None,
) -> List[RetrievedRecord]:
    """Vector-search the historical-records corpus.

    Args:
        query: free-text query string. Embedded by the same model that
            built the index (text-embedding-3-small by default).
        k: max number of hits to return (default 5; classification may
            want 8–10 for richer disambiguation).
        filters: optional metadata pre-filter applied at index level.
            Pass a chromadb `where`-style dict; see module docstring for
            shape examples.
        embeddings, store_dir: test-only injection points. Production
            callers should omit both.

    Returns:
        Up to `k` `RetrievedRecord`s ordered by ascending distance
        (most similar first). Empty list if no docs match.

    Raises:
        ValueError: on empty query or non-positive k.
        RuntimeError: when the chroma store hasn't been built.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    store = _get_store(embeddings=embeddings, store_dir=store_dir)
    where = _normalize_filters(filters)

    pairs = store.similarity_search_with_score(query=query, k=k, filter=where)
    return [
        RetrievedRecord(
            text=doc.page_content,
            metadata=dict(doc.metadata or {}),
            distance=float(score) if score is not None else None,
        )
        for doc, score in pairs
    ]


def _reset_caches_for_tests() -> None:
    """Drop the cached Chroma client so tests can swap stores mid-process."""
    global _STORE
    _STORE = None


def detect_label_conflict(
    records: List["RetrievedRecord"],
    *,
    use_corrected: bool = True,
) -> Optional[dict]:
    """Surface conflicting subcategory labels across a retrieved set.

    The brief: "When a retrieved set contains conflicting labels for
    similar calls, the agent surfaces the conflict (test/log) rather
    than silently picking the most frequent." This is the surfacer.

    Returns None when records agree (or fewer than 2 meaningful labels
    exist). When they disagree, returns a dict the classifier or the
    clarification node can log / act on:

        {
          "conflict": True,
          "labels": {"pipe_leak": 3, "drainage_backup": 2},
          "consensus": "pipe_leak",   # the modal label
          "consensus_share": 0.6,     # 3/5
          "n_records": 5,
          "use_corrected": True,
        }

    Set `use_corrected=False` to inspect the raw intake labels (useful
    for diagnostics — "what did intake think?" — but not what the
    classifier should consume).
    """
    from collections import Counter

    if not records:
        return None

    if use_corrected:
        labels = [r.corrected_subcategory for r in records]
    else:
        labels = [r.intake_subcategory for r in records]
    labels = [l for l in labels if l]

    if len(labels) < 2:
        return None

    counts = Counter(labels)
    if len(counts) <= 1:
        return None  # all agree

    consensus, top_count = counts.most_common(1)[0]
    return {
        "conflict": True,
        "labels": dict(counts),
        "consensus": consensus,
        "consensus_share": top_count / len(labels),
        "n_records": len(labels),
        "use_corrected": use_corrected,
    }
