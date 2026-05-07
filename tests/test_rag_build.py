"""Unit + integration tests for `agent.rag.build_index`.

Two layers:

1. **Pure-function tests** — exercise the chunker / metadata projector
   against synthetic records. No I/O, no embeddings, no API.

2. **End-to-end build with FakeEmbeddings** — runs the real `build()`
   against a `tmp_path` store and a deterministic embedding stub. Proves
   the chroma persistence wiring works without spending a cent on
   OpenAI calls. The full real build is validated manually as part of
   the issue-#7 acceptance run (see PR notes).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

# Suppress chromadb telemetry warnings under Python 3.14 (see build_index.py).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import logging  # noqa: E402
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.rag import COLLECTION_NAME, METADATA_KEYS  # noqa: E402
from agent.rag.build_index import (  # noqa: E402
    MAX_DOC_CHARS,
    _format_document_text,
    _format_metadata,
    _to_documents,
    build,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


def _make_record(**overrides) -> dict:
    base = {
        "ticket_id": "tkt_test_001",
        "call_recording_transcript": "Caller says the breakroom sink is leaking.",
        "intake_notes": "kitchen sink leak floor 9",
        "dispatch_notes": "vendor en route, ETA 30m",
        "resolution_notes": "p-trap replaced, cleared blockage",
        "intake_category": "PLUMBING",
        "intake_subcategory": "pipe_leak",
        "intake_risk_level": "MEDIUM",
        "final_category": "PLUMBING",
        "final_subcategory": "pipe_leak",
        "final_risk_level": "MEDIUM",
        "assigned_vendor_id": "v_001",
        "building_type": "office",
        "city": "Los Angeles",
    }
    base.update(overrides)
    return base


# ─── _format_document_text ─────────────────────────────────────────────


def test_document_text_leads_with_caller_transcript():
    text = _format_document_text(_make_record())
    assert text.startswith("CALLER TRANSCRIPT:")
    # Order: caller → intake → dispatch → resolution
    pos_caller = text.find("CALLER TRANSCRIPT:")
    pos_intake = text.find("INTAKE NOTES:")
    pos_dispatch = text.find("DISPATCH NOTES:")
    pos_resolution = text.find("RESOLUTION NOTES:")
    assert pos_caller < pos_intake < pos_dispatch < pos_resolution


def test_document_text_drops_empty_sections():
    r = _make_record(dispatch_notes="", resolution_notes=None)
    text = _format_document_text(r)
    assert "DISPATCH NOTES:" not in text
    assert "RESOLUTION NOTES:" not in text
    # Caller + intake still present
    assert "CALLER TRANSCRIPT:" in text and "INTAKE NOTES:" in text


def test_document_text_truncates_oversized_records():
    # Build a synthetic giant transcript well past MAX_DOC_CHARS.
    huge = "x" * (MAX_DOC_CHARS + 5_000)
    text = _format_document_text(_make_record(call_recording_transcript=huge))
    assert len(text) == MAX_DOC_CHARS


def test_document_text_empty_when_all_sections_blank():
    r = _make_record(
        call_recording_transcript="",
        intake_notes="",
        dispatch_notes="",
        resolution_notes="",
    )
    assert _format_document_text(r) == ""


# ─── _format_metadata ──────────────────────────────────────────────────


def test_metadata_carries_every_required_key():
    md = _format_metadata(_make_record())
    for key in METADATA_KEYS:
        assert key in md, f"missing required metadata key {key!r}"


def test_metadata_audit_fields_default_false_for_unaudited_record():
    # The synthetic ticket_id "tkt_test_001" isn't in qa_audit_findings.json,
    # so the audit lookup returns None and every audit boolean stays False.
    md = _format_metadata(_make_record())
    assert md["was_audit_flagged"] is False
    assert md["audit_over_escalated"] is False
    assert md["audit_reclassified"] is False
    assert md["audit_floor_wrong"] is False


def test_metadata_audit_fields_populated_for_known_flagged_ticket():
    # TKT-2023-02756 was reclassified (intake roof_leak → final structural).
    md = _format_metadata(_make_record(ticket_id="TKT-2023-02756"))
    assert md["was_audit_flagged"] is True
    assert md["audit_reclassified"] is True
    # over-escalated and floor-wrong flags depend on the actual finding —
    # we don't assert their exact values, just that they're proper booleans.
    assert isinstance(md["audit_over_escalated"], bool)
    assert isinstance(md["audit_floor_wrong"], bool)


def test_metadata_drops_none_values_to_avoid_chroma_type_errors():
    r = _make_record(assigned_vendor_id=None, city=None)
    md = _format_metadata(r)
    assert "assigned_vendor_id" not in md
    assert "city" not in md
    # Required keys that are present remain.
    assert md["ticket_id"] == "tkt_test_001"


def test_metadata_values_are_chroma_compatible_primitives():
    md = _format_metadata(_make_record())
    for k, v in md.items():
        assert isinstance(v, (str, int, float, bool)), (
            f"metadata value for {k!r} is {type(v).__name__}, "
            "chroma only accepts primitives"
        )


# ─── _to_documents ─────────────────────────────────────────────────────


def test_to_documents_skips_records_with_no_text():
    records = [
        _make_record(ticket_id="tkt_001"),
        _make_record(
            ticket_id="tkt_002",
            call_recording_transcript="",
            intake_notes="",
            dispatch_notes="",
            resolution_notes="",
        ),
        _make_record(ticket_id="tkt_003"),
    ]
    docs = _to_documents(records)
    assert len(docs) == 2
    assert {d.metadata["ticket_id"] for d in docs} == {"tkt_001", "tkt_003"}


def test_to_documents_preserves_record_order():
    records = [_make_record(ticket_id=f"tkt_{i:03d}") for i in range(5)]
    docs = _to_documents(records)
    assert [d.metadata["ticket_id"] for d in docs] == [
        "tkt_000", "tkt_001", "tkt_002", "tkt_003", "tkt_004"
    ]


# ─── End-to-end build with FakeEmbeddings ──────────────────────────────


class _FakeEmbeddings:
    """Deterministic, zero-cost embedder for E2E tests.

    Returns a fixed-length vector derived from a stable hash of the text
    so semantic similarity is meaningless but identical inputs produce
    identical vectors (good enough for shape / persistence assertions).
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Stretch the 32-byte hash to `dim` floats in [-1, 1].
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


def test_e2e_build_creates_persistent_store_with_fake_embeddings(tmp_path):
    """Build over the first 25 records, persist to tmp_path, reopen, sanity check."""
    target = tmp_path / "chroma_store"
    n = build(
        force=True,
        limit=25,
        verbose=False,
        embeddings=_FakeEmbeddings(),
        store_dir=target,
    )
    assert n == 25
    assert target.exists() and any(target.iterdir())

    # Reopen the persisted store and confirm the doc count + metadata schema.
    from langchain_chroma import Chroma

    store = Chroma(
        persist_directory=str(target),
        collection_name=COLLECTION_NAME,
        embedding_function=_FakeEmbeddings(),
    )
    persisted = store.get()
    assert len(persisted["ids"]) == 25
    # Pick a metadata row at random and assert every required key is there.
    sample_md = persisted["metadatas"][0]
    for key in METADATA_KEYS:
        # was_audit_flagged is bool False — present even when filtered.
        # Other keys may be dropped if None in source, but our test fixtures
        # never have None for the canonical fields, so they should all stick.
        assert key in sample_md, f"missing {key} in persisted metadata"


def test_e2e_build_skips_when_store_exists_without_force(tmp_path):
    target = tmp_path / "chroma_store"
    # First build populates.
    build(force=True, limit=5, verbose=False,
          embeddings=_FakeEmbeddings(), store_dir=target)
    assert target.exists()

    # Second build without --force should no-op and return -1 (the negative
    # return value is the stable contract — log message is incidental).
    n = build(force=False, limit=5, verbose=False,
              embeddings=_FakeEmbeddings(), store_dir=target)
    assert n == -1


def test_e2e_force_rebuild_replaces_store(tmp_path):
    target = tmp_path / "chroma_store"
    # First build with 5 records.
    build(force=True, limit=5, verbose=False,
          embeddings=_FakeEmbeddings(), store_dir=target)
    # Force rebuild with 10 records.
    n = build(force=True, limit=10, verbose=False,
              embeddings=_FakeEmbeddings(), store_dir=target)
    assert n == 10

    from langchain_chroma import Chroma

    store = Chroma(
        persist_directory=str(target),
        collection_name=COLLECTION_NAME,
        embedding_function=_FakeEmbeddings(),
    )
    assert len(store.get()["ids"]) == 10  # not 15 — old store was wiped


# ─── Inline runner so the file works with or without pytest ────────────


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
            # Each tmp_path is a fresh directory so tests stay isolated.
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
