"""Loader for `evaluation/qa_audit_findings.json`.

Audit findings are post-hoc QA reviews over ~10% of the historical CMMS
corpus. Each entry references one ticket_id and may carry up to three
flags:

- `was_over_escalated`: intake_risk_level was too high vs final_risk_level.
- `was_reclassified`: intake_subcategory was wrong; final_subcategory differs.
- `floor_was_wrong_at_intake`: intake captured the wrong floor; the
  audited `actual_floor` is the correct one.

A finding may exist with *no* flags set — that means the auditor reviewed
the ticket and confirmed the intake decision. `has_any_flag` is the
canonical signal for "this ticket had a real problem the QA team caught."

This module is consumed by:
- `agent/rag/build_index.py` — to stamp audit metadata into chroma at
  index-build time, so the classifier can prefer corrected labels over
  flawed intake labels.
- `agent/rag/retriever.py` — `RetrievedRecord.corrected_*` properties
  derive from these flags.

Empirically (988 findings):
- 188 / 988 over-escalated
- 279 / 988 reclassified
- 391 / 988 floor wrong at intake
-  20 / 988 with multiple flags
- 150 / 988 audited and found OK (no flags set)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

_AUDIT_PATH: Path = (
    Path(__file__).resolve().parents[2] / "evaluation" / "qa_audit_findings.json"
)


@dataclass(frozen=True)
class AuditFinding:
    ticket_id: str
    audit_id: str
    was_over_escalated: bool
    was_reclassified: bool
    floor_was_wrong_at_intake: bool
    over_escalation_reason: Optional[str] = None
    reclassification_note: Optional[str] = None
    actual_floor: Optional[str] = None
    notes: Optional[str] = None
    audited_by: Optional[str] = None
    audited_at: Optional[str] = None

    @property
    def has_any_flag(self) -> bool:
        """True iff at least one of the three flags is set.

        An audit entry can exist with no flags — that's "auditor reviewed
        this ticket and confirmed the intake decision." Those tickets are
        NOT what the classifier should treat as suspect.
        """
        return (
            self.was_over_escalated
            or self.was_reclassified
            or self.floor_was_wrong_at_intake
        )


_INDEX_CACHE: Optional[Dict[str, AuditFinding]] = None


def _build_index() -> Dict[str, AuditFinding]:
    raw = json.loads(_AUDIT_PATH.read_text())
    out: Dict[str, AuditFinding] = {}
    for r in raw:
        tid = r.get("ticket_id")
        if not tid:
            continue
        out[tid] = AuditFinding(
            ticket_id=tid,
            audit_id=r.get("audit_id", ""),
            was_over_escalated=bool(r.get("was_over_escalated")),
            was_reclassified=bool(r.get("was_reclassified")),
            floor_was_wrong_at_intake=bool(r.get("floor_was_wrong_at_intake")),
            over_escalation_reason=r.get("over_escalation_reason"),
            reclassification_note=r.get("reclassification_note"),
            actual_floor=r.get("actual_floor"),
            notes=r.get("notes"),
            audited_by=r.get("audited_by"),
            audited_at=r.get("audited_at"),
        )
    return out


def _index() -> Dict[str, AuditFinding]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = _build_index()
    return _INDEX_CACHE


def lookup(ticket_id: Optional[str]) -> Optional[AuditFinding]:
    """Return the audit finding for `ticket_id`, or None if not audited."""
    if not ticket_id:
        return None
    return _index().get(ticket_id)


def flagged_ticket_ids() -> Set[str]:
    """Set of ticket_ids with at least one audit flag set.

    Excludes audited-and-clean tickets (~150 of 988 in the current data).
    """
    return {tid for tid, f in _index().items() if f.has_any_flag}


def all_audited_ticket_ids() -> Set[str]:
    """Set of every ticket_id with an audit entry, flagged or not."""
    return set(_index().keys())


def _reset_cache_for_tests() -> None:
    global _INDEX_CACHE
    _INDEX_CACHE = None
