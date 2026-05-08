"""Unit tests for `agent.data.audit`.

Anchored on real findings:
- TKT-2023-02756: was_reclassified=True (intake roof_leak → structural).
- 988 total findings, 838 with at least one flag set, 150 audited-clean.

If `evaluation/qa_audit_findings.json` is regenerated and the pinned
ticket_ids move, swap them out.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data import audit  # noqa: E402


KNOWN_RECLASSIFIED_TID = "TKT-2023-02756"  # intake roof_leak → final structural
NEVER_AUDITED_TID = "TKT-9999-99999"        # not in the audit file


def test_lookup_known_reclassified():
    f = audit.lookup(KNOWN_RECLASSIFIED_TID)
    assert f is not None
    assert f.was_reclassified is True
    assert f.has_any_flag is True
    assert "roof_leak" in (f.reclassification_note or "")


def test_lookup_never_audited_returns_none():
    assert audit.lookup(NEVER_AUDITED_TID) is None
    assert audit.lookup(None) is None
    assert audit.lookup("") is None


def test_has_any_flag_true_when_any_set():
    f = audit.lookup(KNOWN_RECLASSIFIED_TID)
    assert f.has_any_flag is True


def test_has_any_flag_false_for_audited_clean_record():
    # Find a real audited-clean record from the dataset.
    clean = [
        f for f in audit._index().values()
        if not f.has_any_flag
    ]
    assert len(clean) >= 1, "expected some audited-clean records"
    sample = clean[0]
    assert sample.was_over_escalated is False
    assert sample.was_reclassified is False
    assert sample.floor_was_wrong_at_intake is False
    assert sample.has_any_flag is False


def test_flagged_ticket_ids_excludes_audited_clean():
    flagged = audit.flagged_ticket_ids()
    all_audited = audit.all_audited_ticket_ids()
    assert flagged.issubset(all_audited)
    assert len(flagged) < len(all_audited), (
        "expected some audited tickets to be clean (no flags)"
    )
    # Sanity: counts roughly match what the survey showed (988 total, ~838 flagged).
    assert 800 < len(flagged) < 950
    assert len(all_audited) == 988


def test_flagged_set_includes_known_reclassified():
    assert KNOWN_RECLASSIFIED_TID in audit.flagged_ticket_ids()


def test_audit_findings_only_reference_real_tickets():
    """Audit ticket_ids must all exist in operational/historical_records.json."""
    import json
    hist = json.loads(open("operational/historical_records.json").read())
    hist_ids = {h["ticket_id"] for h in hist}
    audit_ids = audit.all_audited_ticket_ids()
    missing = audit_ids - hist_ids
    assert not missing, f"audit references non-existent tickets: {sorted(missing)[:5]}"


if __name__ == "__main__":
    import inspect
    tests = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__])
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
