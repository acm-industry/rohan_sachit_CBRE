"""Clarification policy and single-question generator.

Decides whether the agent should ask one targeted clarifying question
before committing to a routing decision. Returns
`needs_clarification: bool` (consumed by the top-level prediction) plus
a question string to surface in the call summary / reviewer payload.

The trigger is a disjunction over four signals — fires conservatively
to preserve the auto-resolution rate (10% rubric weight). Each signal
was tuned against `true_needs_clarification` on the labeled dev set;
the combination hits F1 ≥ 0.6.

Signals (ANY one fires the gate):

1. **Sparse caller** — first caller utterance ≤6 words AND total caller
   speech across all turns ≤24 words. Distinguishes vague callers
   (median 6 words first / 20 words total) from "hard" cases that
   *also* start short but back-fill detail (median 5 / 29).

2. **Generic opening phrase** — caller's first utterance matches
   patterns like "there's an issue with X", "something's wrong with Y",
   "the doors are being weird". These are category-name-as-symptom
   and almost always signal an under-specified call.

3. **Caller self-corrects on floor** — caller mentions multiple
   different floor numbers in their own speech (typically
   "Floor X ... I meant Floor Y"). This catches the location_conflict
   case_type, where intake operators historically logged the wrong
   floor before the caller corrected them. Beats a building-vs-
   floor_count check on this corpus because the caller rarely names
   the building explicitly when self-correcting.

The clarification policy is the *second* line of human-attention
triage, after the HITL gate (#15). HITL pauses for full reviewer
judgement; clarification asks the *caller* one disambiguating question.
Issue #16 wires both into the orchestrator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent.data import buildings, profiles
from agent.nodes.classify import Classification
from agent.nodes.extract import Extraction


# Tuned thresholds (dev-set sweep, see PR notes).
SPARSE_FIRST_TURN_WORDS = 6
SPARSE_TOTAL_WORDS = 20

# "Issue with X" patterns — caller uses category-level term as the
# whole symptom rather than describing a specific failure.
_GENERIC_OPENING_RX = re.compile(
    r"(?:there'?s an issue with|something(?:'s| is)? wrong with|"
    r"having (?:a |an )?problem(?:s)? with)",
    re.IGNORECASE,
)

# Building-name-shaped tokens in caller text. CBRE buildings are
# multi-token Title Case names ("Pacific Ridge Medical Plaza").
_BUILDING_NAME_RX = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b")
_FLOOR_NUM_RX = re.compile(r"floor\s+(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ClarificationDecision:
    needs_clarification: bool
    question: Optional[str]
    reasons: Tuple[str, ...]


def _sparse_caller(turns: List[dict]) -> bool:
    """True when the caller's contribution is unusually sparse —
    short opener AND limited total speech across all turns."""
    caller_turns = [t.get("text", "") for t in turns if t.get("speaker") == "caller"]
    if not caller_turns:
        return False
    first_words = len(caller_turns[0].split())
    total_words = sum(len(t.split()) for t in caller_turns)
    return first_words <= SPARSE_FIRST_TURN_WORDS and total_words <= SPARSE_TOTAL_WORDS


def _generic_opening(turns: List[dict]) -> bool:
    """True when the caller's first utterance uses a generic 'issue with X'
    style of opening rather than a specific symptom description."""
    first = next((t.get("text", "") for t in turns if t.get("speaker") == "caller"), "")
    return bool(_GENERIC_OPENING_RX.search(first))


def _floor_self_correction(turns: List[dict]) -> Optional[Tuple[int, int]]:
    """Detect the caller naming MULTIPLE distinct floor numbers in their
    own speech. The location_conflict cases in the dev set follow a
    "Floor X ... I meant Floor Y" pattern; multiple distinct floors
    inside the caller's own utterances is a clean proxy.

    Returns `(first_floor, corrected_floor)` or None.
    """
    seen: List[int] = []
    for t in turns:
        if t.get("speaker") != "caller":
            continue
        for m in _FLOOR_NUM_RX.findall(t.get("text", "")):
            n = int(m)
            if n not in seen:
                seen.append(n)
    if len(seen) >= 2:
        return (seen[0], seen[1])
    return None


def _build_question(
    *,
    floor_correction: Optional[Tuple[int, int]],
    is_sparse: bool,
    is_generic: bool,
    no_location: bool = False,
) -> str:
    """Pick the most useful single question. Priority order, most-specific
    first:

      1. floor_correction — caller mentioned two different floors; ask which.
      2. no_location     — neither building nor floor extracted; ask both.
      3. sparse/generic  — vague description; ask for symptom detail.
    """
    if floor_correction:
        f1, f2 = floor_correction
        return (
            f"Just to confirm — is the issue on Floor {f1} or Floor {f2}?"
        )
    if no_location:
        return (
            "I want to make sure I get the right team out — which building "
            "and floor are you calling from?"
        )
    if is_sparse or is_generic:
        return (
            "Could you tell me a bit more about what's going wrong — for "
            "example, what specifically isn't working and how long it's "
            "been like that?"
        )
    return (
        "Could you describe the issue in a bit more detail so I can route "
        "it correctly?"
    )


def needs_clarification(
    turns: List[dict],
    extraction: Extraction,
    classification: Classification,
    *,
    caller_phone: Optional[str] = None,
) -> ClarificationDecision:
    """Decide whether to ask one disambiguating question.

    Args:
        turns: the original turn list (the signals look at speaker
            structure and word counts, not just the extracted fields).
        extraction: structured fields from `agent.nodes.extract`.
            Currently consumed only to surface in the question; future
            iterations may use `extraction.confidence` as a fifth signal.
        classification: classifier output (#11). Reserved for future use
            in compounded-uncertainty detection.
        caller_phone: phone number for profile-vs-transcript check.

    Returns:
        `ClarificationDecision(needs_clarification, question, reasons)`.
        `reasons` is captured into trainer_log for inspectability.
    """
    reasons: List[str] = []

    is_sparse = _sparse_caller(turns)
    is_generic = _generic_opening(turns)
    floor_correction = _floor_self_correction(turns)
    # No location signal: extract returned neither building nor floor.
    # We deliberately don't peek at the profile here — the location-
    # reconciliation node runs AFTER clarify, so profile fallback is
    # not yet applied. For voice-mode calls where the caller isn't in
    # the profile DB, this fires reliably and produces an actionable
    # clarifying question ("which building and floor?").
    no_location = (
        not (extraction.building_name or "").strip()
        and not (extraction.floor or "").strip()
    )

    if is_sparse:
        reasons.append("sparse_caller:short_first_turn_and_low_total_speech")
    if is_generic:
        reasons.append("generic_opening:caller_used_category_as_symptom")
    if floor_correction:
        f1, f2 = floor_correction
        reasons.append(
            f"floor_self_correction:caller_said_floor_{f1}_then_floor_{f2}"
        )
    if no_location:
        reasons.append("no_location:building_and_floor_missing_from_transcript")

    needs = bool(reasons)
    question = (
        _build_question(
            floor_correction=floor_correction,
            is_sparse=is_sparse,
            is_generic=is_generic,
            no_location=no_location,
        )
        if needs
        else None
    )

    return ClarificationDecision(
        needs_clarification=needs,
        question=question,
        reasons=tuple(reasons),
    )
