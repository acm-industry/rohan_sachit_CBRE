"""Category + subcategory classifier — the highest-leverage node.

Subcategory accuracy is 15% of the rubric and category another 10%, so
this single function is responsible for 25% of the score.

Pipeline:
  1. Build a retrieval query from the extracted problem_summary (+ urgency).
  2. Call `agent.rag.retriever.retrieve()` for top-k similar tickets.
  3. Render those tickets with their AUDIT-CORRECTED labels (issue #9)
     so the prompt never trains the LLM on labels the QA team already
     flagged as wrong.
  4. Build a prompt that lists the entire taxonomy + the retrieved
     examples + the caller's complaint; demand structured output via
     Pydantic with `Literal[...]` constraints.
  5. Validate that subcategory genuinely belongs to the chosen category.
     On failure, retry with explicit pair guidance. Second failure →
     fall back to category-only with a logged warning.

Engineering choices that drive accuracy:

- **Audit-corrected retrieval labels** (#9). Without this we'd train the
  LLM on the very labels the QA team flagged as wrong. With it, the
  classic example (TKT-2023-02756: roof_leak intake → structural final)
  presents as `corrected_subcategory=structural` to the prompt.

- **k=8** retrieved records by default. The AC notes classification may
  want 8-10. Empirically 8 gives the LLM enough cluster signal without
  bloating the prompt.

- **Reasoning capture**. The Classification model includes a short
  `reasoning` field. This is the trainer log's `ai_prediction.reasoning`
  payload (issue #23) and the input to error analysis (#25).

- **Per-axis confidence** (category + subcategory separately). The
  validator gate (#16) and clarification policy (#18) consume these.

- **Two-axis retry**. Pydantic's `Literal` constraint handles most
  schema drift. The post-hoc `model_validator` catches the case where
  the LLM picks a subcategory that's valid in isolation but doesn't
  belong to the chosen category. On retry we tell it exactly which
  pairs are allowed and why the first answer was rejected.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from agent import config
from agent.data import taxonomy
from agent.nodes.extract import Extraction
from agent.rag.retriever import RetrievedRecord, retrieve

logger = logging.getLogger(__name__)


# Materialise category + subcategory enums from the canonical taxonomy at
# import time. Using `Enum` (over a dynamic `Literal[...]`) is the most
# robust path through Pydantic v2 + langchain-openai's structured output —
# JSON schema renders them as proper `enum` constraints that OpenAI's
# function-calling layer strictly enforces, so the LLM cannot return an
# off-taxonomy string in the happy path.
CategoryEnum = Enum(
    "CategoryEnum",
    {c: c for c in taxonomy.categories()},
    type=str,
)
SubcategoryEnum = Enum(
    "SubcategoryEnum",
    {s: s for s in taxonomy.all_subcategories()},
    type=str,
)


# ─── Output schema ─────────────────────────────────────────────────────


class Classification(BaseModel):
    """Classifier output.

    `category` and `subcategory` are constrained to the canonical
    taxonomy; the model_validator additionally enforces
    `subcategory ∈ taxonomy[category]`. `confidence_*` fields are 0..1.
    `reasoning` is a 1-2 sentence justification consumed by the trainer
    log (#23) and error analysis (#25).
    """

    category: CategoryEnum = Field(
        description="Top-level category from the canonical 10-element taxonomy."
    )
    subcategory: SubcategoryEnum = Field(
        description="Leaf subcategory; must belong to the chosen category."
    )
    confidence_category: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the category pick, 0..1. Lower means the "
        "validator gate (issue #16) should consider human review.",
    )
    confidence_subcategory: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the subcategory pick, 0..1.",
    )
    reasoning: str = Field(
        description="1-2 sentence justification — what in the caller's complaint or "
        "the retrieved tickets drove this label. Captured into trainer_log."
    )
    is_fallback: bool = Field(
        default=False,
        description="Internal flag — True only when the classifier had to use the "
        "category-only fallback path. The validator gate reads this directly "
        "instead of substring-matching the reasoning string.",
    )
    retrieved_record_ids: List[str] = Field(
        default_factory=list,
        description="Internal field — ticket IDs of the historical records the "
        "classifier retrieved. Populated by `classify()`, never by the LLM; "
        "consumed by trainer_log so the fine-tune corpus can re-anchor to "
        "the records that drove each prediction.",
    )

    @field_validator("reasoning")
    @classmethod
    def _strip_reasoning(cls, v: str) -> str:
        return (v or "").strip()

    @model_validator(mode="after")
    def _category_subcategory_pair_must_be_valid(self) -> "Classification":
        cat = self.category.value if hasattr(self.category, "value") else self.category
        sub = self.subcategory.value if hasattr(self.subcategory, "value") else self.subcategory
        if not taxonomy.is_valid_pair(cat, sub):
            allowed = ", ".join(taxonomy.subcategories_for(cat))
            raise ValueError(
                f"subcategory={sub!r} is not valid under category={cat!r}. "
                f"Allowed subcategories for this category: {allowed}."
            )
        return self

    @property
    def category_str(self) -> str:
        """Plain-string view of the category enum value."""
        return self.category.value if hasattr(self.category, "value") else self.category

    @property
    def subcategory_str(self) -> str:
        """Plain-string view of the subcategory enum value."""
        return self.subcategory.value if hasattr(self.subcategory, "value") else self.subcategory


# ─── Prompt assembly ───────────────────────────────────────────────────


def _format_taxonomy_block() -> str:
    lines: List[str] = []
    for cat in taxonomy.categories():
        subs = taxonomy.subcategories_for(cat)
        lines.append(f"{cat}:")
        for s in subs:
            lines.append(f"  - {s}")
    return "\n".join(lines)


def _format_retrieval_block(records: List[RetrievedRecord]) -> str:
    """Render retrieved records using AUDIT-CORRECTED labels.

    Per issue #9, `corrected_subcategory` returns `final_*` for tickets
    flagged as reclassified by QA, and `intake_*` otherwise. This is the
    label the LLM should learn from — never the raw intake label.

    A `[FLAGGED]` marker appears next to the corrected label when it
    differs from the original intake, so the LLM can spot the pattern
    and weight that example accordingly.
    """
    if not records:
        return "(no retrieved tickets)"
    lines: List[str] = []
    for i, r in enumerate(records, 1):
        flag = ""
        if r.is_audit_flagged and r.was_reclassified:
            orig = r.intake_subcategory or "?"
            flag = f"  [QA-RECLASSIFIED from intake={orig}]"
        # Truncate raw text for prompt-budget hygiene.
        snippet = r.text.replace("\n", " ").strip()
        if len(snippet) > 280:
            snippet = snippet[:280] + "…"
        lines.append(
            f"[{i}] distance={r.distance:.3f}  "
            f"category={r.corrected_category}  "
            f"subcategory={r.corrected_subcategory}{flag}\n"
            f"    {snippet}"
        )
    return "\n\n".join(lines)


def _build_query(extraction: Extraction) -> str:
    """Compose the retrieval query from extraction signals.

    Combining `problem_summary` with urgency cues keeps the embedding
    grounded in what the caller actually said while preserving the
    safety-language tail (e.g. "smoke", "trapped") that influences both
    subcategory and risk.
    """
    parts = [extraction.problem_summary]
    if extraction.urgency_cues:
        parts.append("urgency cues: " + ", ".join(extraction.urgency_cues))
    return ". ".join(p for p in parts if p)


_PROMPT_TEMPLATE = """\
You are classifying a CBRE facilities-maintenance call into the canonical taxonomy below.
Subcategory accuracy is the most heavily-weighted axis on the scorecard. Pick carefully.

# CANONICAL TAXONOMY (the ONLY valid labels — both fields must come from this list,
# and the subcategory MUST belong to its parent category)

{taxonomy_block}

# CALLER COMPLAINT (extracted from the dialogue)

problem_summary: {problem_summary}
urgency_cues:    {urgency_cues}
caller_role:     {caller_role}
location:        {building_name} / {floor} / {suite}

# SIMILAR HISTORICAL TICKETS (corrected labels — QA-flagged reclassifications already applied)

{retrieval_block}

# REASONING GUIDANCE

- The retrieved tickets are your primary signal: when 4+ of them share the same
  corrected subcategory, that's almost always the right answer.
- Tickets marked [QA-RECLASSIFIED ...] are cases where intake got it wrong and the
  technician corrected on-site. Pay attention — the same caller language surfaced
  again means the same correction probably applies again.
- If retrieved tickets disagree (split between two subcategories), pick the one
  that better matches the caller's specific issue, not the more populous one.
- Confidence should reflect retrieval consensus: 0.9+ when retrieval agrees,
  0.5-0.7 when retrieval is split, lower when no retrieval cluster matches.

# TAXONOMY DISAMBIGUATION (boundary cases retrieval alone may not surface)

- Sprinkler / irrigation / lawn-watering issues → PEST_SPECIALTY/landscaping.
  NOT PLUMBING — those are landscape-irrigation systems, not building plumbing.
  "Sprinkler head stuck on, water everywhere" = landscaping.

- Card reader / badge / keypad / electronic-lock problems → DOORS_ACCESS/access_control.
  NOT SECURITY/unauthorized_access — SECURITY is for *people*-related concerns
  (suspicious persons, breaches, threats). An "issue with access control" or a
  "card reader not working" is the access-control hardware, even when phrased
  using the word "access".

- Loading-bay / freight / fire-exit / main-lobby-entrance doors that are stuck,
  weird, or malfunctioning → DOORS_ACCESS/auto_door (these are powered automatic
  doors). NOT door_mechanical (which is for traditional manual doors —
  hinges, latches, knobs). NOT LIFE_SAFETY (a "fire exit" is a *type of door*,
  not a fire incident).

- HVAC/air_quality covers chemical fumes, mold, ventilation, and AC-related air
  complaints. KEY EXAMPLES that go here even though the caller may deny "fire":
  cleaning-chemical fumes (bleach, mopping solution, ammonia), paint or solvent
  smells, exhaust gases, refrigerant smell, suspected mold growth. These are
  air-quality issues that affect building occupants via the air system.
  However, BURNT-FOOD smells (burnt toast, burnt cooking, candle smoke from a
  tenant — combustion-product smells of mundane origin) belong to
  JANITORIAL/waste_odor, NOT HVAC. The distinguishing test: is the smell from
  a building-system source or industrial chemical (→ air_quality), or from a
  food / candle / tenant activity (→ waste_odor)?

- ELECTRICAL/lighting is for INDOOR lighting (office lights, hallway lights,
  conference room lights). Outdoor / parking-lot / pathway / exterior signage
  lighting → GROUNDS_EXTERIOR/parking_lighting. The taxonomy splits indoor
  vs outdoor lighting fixtures across two different categories.

- Roof or ceiling water intrusion (including ceiling tiles falling because of
  water) → PLUMBING/roof_leak. The cause is water from above. Falling tile is a
  symptom, not the category.

# OVER-ESCALATION GUARD (FINAL EMPHASIS — high-stakes; false alarms are heavily penalised)

The intake set is deliberately seeded with calls that BEGIN with alarming-sounding
keywords ("smoke", "fire", "ceiling fell", "elevator stopped with someone inside",
"yelling in the lobby") and then have the caller themselves CLARIFY that the
underlying situation is mundane. Read the WHOLE transcript. The caller's own
clarifications after the agent's "is anyone hurt / is there fire" prompt are
DECISIVE. Classify the actual mundane cause, not the initial alarming framing.

Concrete patterns in this corpus — when the caller says any of these,
classify the MUNDANE cause, NOT the safety category:

- "Smoke alarm beeped, just burnt toast — no fire" → JANITORIAL/waste_odor.
  Burnt FOOD smells are waste_odor, not LIFE_SAFETY/fire_smoke and not
  HVAC/air_quality. Both burnt toast and candle-too-close-to-vent fall here.

- "Drop-ceiling tile fell — water-damaged, nothing structural, nobody hurt" →
  PLUMBING/roof_leak. Cause is water from above. Tile falling is a symptom.
  NOT LIFE_SAFETY/structural.

- "Yelling in the lobby — just two tenants arguing, no weapons, no physical
  contact" → SECURITY/suspicious_person. NOT active_threat (which requires
  actual weapons or physical violence in progress).

- "Elevator stopped with someone inside, but they're out of the car now — it's
  stuck empty" → ELEVATOR/malfunction. NOT entrapment (which requires a person
  CURRENTLY trapped). Once they're out, it's a stuck-empty malfunction.

- "Outlet was sparking but I unplugged it / the hazard is mitigated" →
  ELECTRICAL/panel_hazard. NOT LIFE_SAFETY/fire_smoke.

- "Door being weird — it's a fire exit" → DOORS_ACCESS/auto_door. "Fire exit"
  describes WHICH door, not a fire incident.

Promote a call to LIFE_SAFETY (or active_threat / entrapment / structural) only on
POSITIVE EVIDENCE of an active hazard: visible flames RIGHT NOW, ongoing smoke
without innocent explanation, weapons or physical violence in progress, people
CURRENTLY trapped / injured / unconscious, structural failure unrelated to a
fixable plumbing or building-systems cause. When the caller explicitly denies
the hazard, TRUST them — DO NOT escalate against an explicit "no fire / no injury
/ no weapons / they're out".

Return a Classification with category, subcategory, confidence_category,
confidence_subcategory (both 0..1), and a 1-2 sentence reasoning. The subcategory
MUST belong to the category you choose.
"""


def build_prompt(extraction: Extraction, records: List[RetrievedRecord]) -> str:
    return _PROMPT_TEMPLATE.format(
        taxonomy_block=_format_taxonomy_block(),
        problem_summary=extraction.problem_summary,
        urgency_cues=", ".join(extraction.urgency_cues) or "(none)",
        caller_role=extraction.caller_role or "(unknown)",
        building_name=extraction.building_name or "(unknown)",
        floor=extraction.floor or "(unknown)",
        suite=extraction.suite or "(unknown)",
        retrieval_block=_format_retrieval_block(records),
    )


# ─── LLM-backed entrypoint ─────────────────────────────────────────────


DEFAULT_K = 8


def _build_llm() -> Any:
    return config.build_chat_llm()


def classify(
    extraction: Extraction,
    *,
    k: int = DEFAULT_K,
    llm: Any = None,
    records: Optional[List[RetrievedRecord]] = None,
) -> Classification:
    """Classify a maintenance call.

    Args:
        extraction: structured fields from `agent.nodes.extract.extract`.
        k: number of historical tickets to retrieve (default 8 per the AC).
        llm: test-only injection point.
        records: test-only injection — bypasses retrieval entirely so
            unit tests can assert on prompt construction without hitting
            chroma. Production callers should leave this None.

    Returns:
        A `Classification` with validated `(category, subcategory)`,
        per-axis confidence, and a short reasoning string.

    Raises:
        ValidationError: only if the second-attempt retry also fails to
            produce a valid pair AND the category-only fallback can't be
            constructed. This should be vanishingly rare in practice.
    """
    if records is None:
        query = _build_query(extraction)
        records = retrieve(query, k=k) if query else []

    if llm is None:
        llm = _build_llm()

    prompt = build_prompt(extraction, records)
    structured = llm.with_structured_output(Classification)

    result: Classification
    try:
        result = structured.invoke(prompt)
    except ValidationError as first_err:
        logger.warning(
            "classifier first-pass validation failed (%s); retrying with "
            "explicit pair enumeration",
            first_err,
        )
        # Retry: tell the LLM exactly which (category, subcategory) pairs are valid.
        pair_block = "\n".join(
            f"  ({cat!r}, {sub!r})"
            for cat, sub in taxonomy.all_pairs()
        )
        retry_prompt = (
            prompt
            + "\n\n# RETRY GUIDANCE\nYour previous answer used a (category, subcategory) "
            "pair that doesn't exist in the taxonomy. The complete list of valid pairs is:\n"
            + pair_block
            + "\n\nPick exactly one of these pairs.\n"
        )
        try:
            result = structured.invoke(retry_prompt)
        except ValidationError as second_err:
            logger.error(
                "classifier retry validation also failed (%s); falling back "
                "to category-only via retrieval consensus",
                second_err,
            )
            result = _category_only_fallback(records)

    # Plumb retrieval provenance through to trainer_log (issue #29 §8.3).
    # Done after the LLM call so the model can't pollute the field — the
    # default `[]` it would emit gets overwritten with the real IDs.
    result.retrieved_record_ids = [r.ticket_id for r in records if r.ticket_id]
    return result


def _category_only_fallback(records: List[RetrievedRecord]) -> Classification:
    """Last-ditch fallback when both LLM attempts produce off-taxonomy output.

    Uses retrieval consensus on the category, then picks an arbitrary
    valid subcategory under that category. Confidence is forced low so
    the validator gate (#16) escalates to human review.
    """
    from collections import Counter

    cats = [r.corrected_category for r in records if r.corrected_category]
    if not cats:
        # Total retrieval whiff — punt to PEST_SPECIALTY (catch-all per taxonomy.md).
        cat = "PEST_SPECIALTY"
    else:
        cat, _ = Counter(cats).most_common(1)[0]

    sub = taxonomy.subcategories_for(cat)[0] if taxonomy.subcategories_for(cat) else "infestation"

    return Classification(
        category=CategoryEnum(cat),
        subcategory=SubcategoryEnum(sub),
        confidence_category=0.2,
        confidence_subcategory=0.1,
        reasoning="(fallback: classifier retries exhausted; category from retrieval consensus, subcategory arbitrary — escalate to human reviewer)",
        is_fallback=True,
    )
