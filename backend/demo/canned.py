"""The three demo-day transcripts.

Hand-picked from `evaluation/eval_transcripts_dev.json` for stage
reliability and rubric coverage:

  - EVAL-0976 — AC failure, LOW, routine auto-resolve.
  - EVAL-0788 — "smoke alarm — just burnt toast", over-escalation trap.
    The classifier resolves it to JANITORIAL/waste_odor, validator
    auto-resolves. The whole point of this one is showing the
    `no actual fire` clarification holding back a false-911.
  - EVAL-0076 — visible smoke + flames in an electrical room, true
    EMERGENCY. fire_smoke + 911 dispatch.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


DEMO_IDS: List[Dict[str, str]] = [
    {
        "transcript_id": "EVAL-0976",
        "label": "Routine — AC failure (auto-resolve)",
        "expected_outcome": (
            "Auto-resolved as HVAC / no_cooling. Vendor dispatched, no human "
            "review needed."
        ),
    },
    {
        "transcript_id": "EVAL-0788",
        "label": "Over-escalation trap — burnt toast",
        "expected_outcome": (
            "Smoke alarm + 'burnt toast, no fire' should classify as "
            "JANITORIAL / waste_odor — NOT fire_smoke. No 911 dispatch."
        ),
    },
    {
        "transcript_id": "EVAL-0076",
        "label": "True emergency — visible flames",
        "expected_outcome": (
            "Visible flames + smoke in electrical room classifies as "
            "LIFE_SAFETY / fire_smoke, EMERGENCY, 911 dispatched."
        ),
    },
]


@lru_cache(maxsize=1)
def _load_dev_transcripts() -> Dict[str, Dict[str, Any]]:
    path = Path(__file__).resolve().parents[2] / "evaluation" / "eval_transcripts_dev.json"
    with open(path) as f:
        records = json.load(f)
    return {r["transcript_id"]: r for r in records}


def list_demos() -> List[Dict[str, str]]:
    return list(DEMO_IDS)


def get_transcript(transcript_id: str) -> Dict[str, Any]:
    """Return `{transcript_id, caller_phone, turns}` for a known demo id."""
    transcripts = _load_dev_transcripts()
    if transcript_id not in transcripts:
        raise KeyError(transcript_id)
    rec = transcripts[transcript_id]
    return {
        "transcript_id": rec["transcript_id"],
        "caller_phone": rec.get("caller_phone"),
        "turns": rec.get("turns", []),
    }


def is_demo_id(transcript_id: str) -> bool:
    return any(d["transcript_id"] == transcript_id for d in DEMO_IDS)
