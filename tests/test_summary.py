"""Tests for `agent.nodes.summary`.

Validates summary generation style, length, and content.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.nodes.summary import generate_summary, _MAX_CHARS  # noqa: E402


# ─── Style and content tests ─────────────────────────────────────────


def test_basic_summary_structure():
    """Summary includes problem, location, risk, and action."""
    s = generate_summary(
        subcategory="pipe_leak",
        risk_level="MEDIUM",
        building_name="Pacific Ridge Medical Plaza",
        floor="Floor 3",
        vendor_name="AquaFix Plumbing",
        vendor_id="v_002",
    )
    assert "Pipe leak" in s
    assert "Pacific Ridge" in s
    assert "Floor 3" in s
    assert "MEDIUM risk" in s
    assert "AquaFix Plumbing" in s


def test_emergency_dispatch_summary():
    """Emergency services dispatched note."""
    s = generate_summary(
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        building_name="One Commerce Center",
        floor="Floor 7",
        dispatched_emergency_services=True,
    )
    assert "Fire/smoke" in s
    assert "Emergency services dispatched" in s
    assert "EMERGENCY risk" in s


def test_unroutable_escalation_summary():
    """Unroutable case mentions escalation."""
    s = generate_summary(
        subcategory="pipe_leak",
        risk_level="HIGH",
        building_name="Metro Center Offices",
        city="Burbank",
        needs_human_review=True,
        unroutable=True,
    )
    assert "Escalated" in s or "escalat" in s.lower()


def test_flagged_for_review_with_vendor():
    """Human review flagged but vendor pre-selected."""
    s = generate_summary(
        subcategory="unauthorized_access",
        risk_level="HIGH",
        building_name="Metro Center Offices",
        floor="Floor 8",
        vendor_name="SecureGuard Services",
        vendor_id="v_026",
        needs_human_review=True,
    )
    assert "review" in s.lower()
    assert "SecureGuard" in s


def test_no_first_person():
    """Summary must not contain first-person language."""
    s = generate_summary(
        subcategory="door_mechanical",
        risk_level="LOW",
        building_name="Bayshore Commons",
        floor="Floor 2",
        vendor_name="DoorTech",
        vendor_id="v_010",
    )
    assert " I " not in s
    assert " we " not in s.lower()
    assert "determined" not in s.lower()


# ─── Length constraint ────────────────────────────────────────────────


def test_length_under_cap():
    """Summary must not exceed the 280-char cap."""
    s = generate_summary(
        subcategory="pipe_leak",
        risk_level="MEDIUM",
        building_name="The Very Long Building Name That Takes Up Many Characters International Plaza",
        floor="Floor 27",
        city="San Francisco",
        vendor_name="AquaFix Professional Plumbing and Water Restoration Services LLC",
        vendor_id="v_002",
    )
    assert len(s) <= _MAX_CHARS


def test_minimum_length_for_scoring():
    """Summary must be at least 30 chars to pass scoring.py."""
    s = generate_summary(
        subcategory="minor_issue",
        risk_level="LOW",
    )
    assert len(s) >= 30


# ─── Fixture summaries for manual review ─────────────────────────────


def test_fixture_summaries_printable():
    """Print 5 fixture summaries for manual style review."""
    fixtures = [
        dict(subcategory="pipe_leak", risk_level="MEDIUM",
             building_name="Pacific Ridge Medical Plaza", floor="Floor 3",
             vendor_name="AquaFix Plumbing", vendor_id="v_002"),
        dict(subcategory="fire_smoke", risk_level="EMERGENCY",
             building_name="One Commerce Center", floor="Floor 7",
             city="Irvine", dispatched_emergency_services=True),
        dict(subcategory="unauthorized_access", risk_level="HIGH",
             building_name="Metro Center Offices", floor="Floor 8",
             vendor_name="SecureGuard", vendor_id="v_026",
             needs_human_review=True),
        dict(subcategory="door_mechanical", risk_level="LOW",
             building_name="Bayshore Commons", floor="Floor 2",
             vendor_name="DoorTech Repairs", vendor_id="v_010"),
        dict(subcategory="gas_chemical", risk_level="EMERGENCY",
             building_name="Harbor Industrial Complex",
             city="Long Beach", needs_human_review=True,
             unroutable=True),
    ]
    for i, f in enumerate(fixtures, 1):
        s = generate_summary(**f)
        assert len(s) >= 30, f"Fixture {i} too short"
        assert len(s) <= _MAX_CHARS, f"Fixture {i} exceeds cap"
        # Print for manual review
        print(f"  [{i}] ({len(s)} chars) {s}")


if __name__ == "__main__":
    import inspect

    tests = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__])
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in sorted(tests):
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
