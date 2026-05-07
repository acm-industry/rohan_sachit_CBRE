"""Tests for `agent.rag.retriever`.

Strategy: build a tiny chroma store at `tmp_path` with `_FakeEmbeddings`
and the first ~50 historical records, then exercise `retrieve()` against
it. FakeEmbeddings means semantic quality of results is meaningless — we
assert structural properties (k bound, filter pass-through, schema of the
returned `RetrievedRecord`s, error handling). Real semantic retrieval is
validated end-to-end in the issue-#7 acceptance run.

The "3 representative queries" smoke test from the AC is implemented as
three filter scenarios (no filter / single-field / compound), since the
queries themselves can't usefully differ when the embedder is a hash
function.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import logging  # noqa: E402
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.rag import COLLECTION_NAME  # noqa: E402
from agent.rag.build_index import build  # noqa: E402
from agent.rag.retriever import (  # noqa: E402
    DEFAULT_K,
    FILTERABLE_KEYS,
    RetrievedRecord,
    _normalize_filters,
    _reset_caches_for_tests,
    retrieve,
)


# ─── Deterministic embedder identical to the one used in test_rag_build ──


class _FakeEmbeddings:
    def __init__(self, dim: int = 64):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        out: List[float] = []
        i = 0
        while len(out) < self.dim:
            out.append((h[i % 32] / 127.5) - 1.0)
            i += 1
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


# ─── Pure helpers ──────────────────────────────────────────────────────────


def test_normalize_filters_passes_through_simple_equality():
    assert _normalize_filters({"city": "Los Angeles"}) == {"city": "Los Angeles"}


def test_normalize_filters_passes_through_operator_forms():
    f = {"$and": [{"city": "LA"}, {"building_type": "office"}]}
    assert _normalize_filters(f) == f


def test_normalize_filters_returns_none_for_empty_input():
    assert _normalize_filters(None) is None
    assert _normalize_filters({}) is None


def test_filterable_keys_match_metadata_schema():
    # Drift guard: if METADATA_KEYS in agent/rag/__init__.py grows or
    # shrinks, FILTERABLE_KEYS should track. was_audit_flagged is the
    # only "currently False everywhere" key but it's still filterable.
    from agent.rag import METADATA_KEYS
    assert set(FILTERABLE_KEYS) == set(METADATA_KEYS)


# ─── Validation ────────────────────────────────────────────────────────────


def test_retrieve_rejects_empty_query(tmp_path):
    target = tmp_path / "store"
    build(force=True, limit=5, verbose=False,
          embeddings=_FakeEmbeddings(), store_dir=target)
    _reset_caches_for_tests()
    try:
        retrieve("", embeddings=_FakeEmbeddings(), store_dir=target)
        raise AssertionError("expected ValueError on empty query")
    except ValueError as e:
        assert "non-empty" in str(e)


def test_retrieve_rejects_non_positive_k(tmp_path):
    target = tmp_path / "store"
    build(force=True, limit=5, verbose=False,
          embeddings=_FakeEmbeddings(), store_dir=target)
    _reset_caches_for_tests()
    try:
        retrieve("anything", k=0, embeddings=_FakeEmbeddings(), store_dir=target)
        raise AssertionError("expected ValueError on k=0")
    except ValueError as e:
        assert "k must be positive" in str(e)


def test_retrieve_raises_when_store_not_built(tmp_path):
    target = tmp_path / "nonexistent_store"
    _reset_caches_for_tests()
    try:
        retrieve("anything", embeddings=_FakeEmbeddings(), store_dir=target)
        raise AssertionError("expected RuntimeError on missing store")
    except RuntimeError as e:
        assert "build_index" in str(e)


# ─── Core retrieval — the three representative scenarios ───────────────────


def _setup_store(tmp_path):
    target = tmp_path / "store"
    build(force=True, limit=50, verbose=False,
          embeddings=_FakeEmbeddings(), store_dir=target)
    _reset_caches_for_tests()
    return target


def test_retrieve_no_filter_returns_k_records(tmp_path):
    target = _setup_store(tmp_path)
    hits = retrieve("water leak", k=5, embeddings=_FakeEmbeddings(), store_dir=target)
    assert len(hits) == 5
    for h in hits:
        assert isinstance(h, RetrievedRecord)
        assert h.text  # non-empty
        assert h.metadata
        assert h.distance is not None


def test_retrieve_default_k_is_DEFAULT_K(tmp_path):
    target = _setup_store(tmp_path)
    hits = retrieve("anything", embeddings=_FakeEmbeddings(), store_dir=target)
    assert len(hits) == DEFAULT_K


def test_retrieve_with_city_filter_only_returns_matching(tmp_path):
    target = _setup_store(tmp_path)
    # Pick a city that's actually present in the first 50 records.
    import json
    raw = json.loads(open("operational/historical_records.json").read())[:50]
    cities_present = {r["city"] for r in raw if r.get("city")}
    target_city = next(iter(cities_present))

    hits = retrieve(
        "any query",
        k=10,
        filters={"city": target_city},
        embeddings=_FakeEmbeddings(),
        store_dir=target,
    )
    # Every hit must satisfy the filter.
    assert len(hits) > 0
    for h in hits:
        assert h.metadata.get("city") == target_city


def test_retrieve_with_compound_filter(tmp_path):
    target = _setup_store(tmp_path)
    # Build a compound filter we know matches at least one record.
    import json, collections
    raw = json.loads(open("operational/historical_records.json").read())[:50]
    pairs = collections.Counter((r.get("building_type"), r.get("city")) for r in raw)
    (bt, ct), n = pairs.most_common(1)[0]
    assert n > 0

    hits = retrieve(
        "any query",
        k=10,
        filters={"$and": [{"building_type": bt}, {"city": ct}]},
        embeddings=_FakeEmbeddings(),
        store_dir=target,
    )
    for h in hits:
        assert h.metadata.get("building_type") == bt
        assert h.metadata.get("city") == ct


def test_retrieve_returns_empty_when_filter_matches_nothing(tmp_path):
    target = _setup_store(tmp_path)
    hits = retrieve(
        "any query",
        filters={"city": "Atlantis"},
        embeddings=_FakeEmbeddings(),
        store_dir=target,
    )
    assert hits == []


def test_retrieve_results_ordered_by_ascending_distance(tmp_path):
    target = _setup_store(tmp_path)
    hits = retrieve(
        "any query",
        k=10,
        embeddings=_FakeEmbeddings(),
        store_dir=target,
    )
    distances = [h.distance for h in hits]
    assert distances == sorted(distances), (
        f"results should be sorted nearest-first; got {distances}"
    )


def test_retrieved_record_convenience_accessors(tmp_path):
    target = _setup_store(tmp_path)
    hits = retrieve("any query", k=1,
                    embeddings=_FakeEmbeddings(), store_dir=target)
    h = hits[0]
    # Convenience properties should mirror the underlying metadata dict.
    assert h.ticket_id == h.metadata.get("ticket_id")
    assert h.intake_category == h.metadata.get("intake_category")
    assert h.intake_subcategory == h.metadata.get("intake_subcategory")
    assert h.final_subcategory == h.metadata.get("final_subcategory")
    assert h.assigned_vendor_id == h.metadata.get("assigned_vendor_id")


# ─── Inline runner ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect, tempfile

    tests = [
        (n, f) for n, f in inspect.getmembers(sys.modules[__name__])
        if n.startswith("test_") and callable(f)
    ]
    failed = 0
    for name, fn in tests:
        sig = inspect.signature(fn)
        kwargs = {}
        if "tmp_path" in sig.parameters:
            kwargs["tmp_path"] = Path(tempfile.mkdtemp(prefix=f"{name}_"))
        try:
            fn(**kwargs)
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
