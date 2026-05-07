"""First node of the agent graph.

Converts the raw `turns` list (multi-turn caller dialogue) into a
structured `Extraction` — problem summary, location, urgency cues,
caller role, language, plus per-field confidence used by the
clarification policy (#18) and risk scorer (#14).

When `caller_phone` matches a known profile (#4), the profile's
last-known fields are merged in **before** the LLM extraction pass as a
starting prior. The prompt instructs the model that the **transcript
always wins on conflict** — profiles are stale by design (see brief),
so a caller saying "Floor 9" supersedes a profile saying "Floor 12".
The model returns the transcript-anchored answer with high confidence
and the profile-derived field with moderate confidence.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator

from agent import config
from agent.data import profiles
from agent.data.profiles import Profile


# ─── Output schema ─────────────────────────────────────────────────────


class FieldConfidence(BaseModel):
    """Per-field 0..1 confidence used by the clarification policy.

    Scoring guide (in the prompt — repeated here as the canonical scale):
      1.0       — explicitly stated in the transcript with no ambiguity
      0.7-0.9   — clearly inferred from transcript context
      0.4-0.6   — weak inference, partial info, or profile-derived
      0.0-0.3   — unknown / fell back to a default we don't trust
    """
    problem_summary: float = Field(ge=0.0, le=1.0)
    building_name: float = Field(ge=0.0, le=1.0)
    floor: float = Field(ge=0.0, le=1.0)
    suite: float = Field(ge=0.0, le=1.0)
    caller_role: float = Field(ge=0.0, le=1.0)


class Extraction(BaseModel):
    """Structured extraction from one caller dialogue."""

    problem_summary: str = Field(
        description="1-2 sentence factual summary of what the caller is reporting. "
        "Operator-style: terse, no emotional language."
    )
    building_name: Optional[str] = Field(
        default=None,
        description="Name of the building where the issue is. Transcript wins over profile.",
    )
    floor: Optional[str] = Field(
        default=None,
        description='Floor designation, e.g. "Floor 9", "Ground Floor", "Basement". '
        "Transcript wins over profile.",
    )
    suite: Optional[str] = Field(
        default=None,
        description='Suite number, e.g. "Suite 408". Transcript wins over profile.',
    )
    urgency_cues: List[str] = Field(
        default_factory=list,
        description="Verbatim phrases from the transcript that signal urgency or "
        'safety risk. Examples: "flooding", "smoke", "fire", "trapped", "gas leak", '
        '"unconscious", "no heat overnight", "getting worse", "people trapped". '
        "Empty list when no such phrases appear.",
    )
    caller_role: Optional[str] = Field(
        default=None,
        description='Role of the caller, e.g. "tenant", "office_manager", '
        '"security_contact", "vendor", "anonymous". Inferred from transcript or profile.',
    )
    language: Optional[str] = Field(
        default=None,
        description='ISO 639-1 language code of the dialogue, e.g. "en", "es", "zh". '
        'Default to "en" when the dialogue is clearly English.',
    )
    confidence: FieldConfidence

    @field_validator("urgency_cues", mode="before")
    @classmethod
    def _strip_empty_cues(cls, v: Any) -> List[str]:
        if not v:
            return []
        return [s for s in v if isinstance(s, str) and s.strip()]


# ─── Pure helpers (no LLM, no I/O) ─────────────────────────────────────


def _flatten_turns(turns: List[dict]) -> str:
    """Turn list → single text block. Preserves speaker tags."""
    return "\n".join(
        f"[{t.get('speaker', '?').upper()}] {t.get('text', '')}".rstrip()
        for t in turns
    )


def _format_profile_block(profile: Optional[Profile]) -> str:
    """Render a profile as a pre-extraction prior.

    Returns a fixed sentinel string when no profile is available so the
    prompt is identical-shape across known and anonymous callers.
    """
    if profile is None:
        return "(no caller profile — anonymous caller or unknown phone number)"
    return json.dumps(
        {
            "caller_name": profile.caller_name,
            "tenant_company": profile.tenant_company,
            "contact_role": profile.contact_role,
            "primary_building_name": profile.primary_building_name,
            "primary_floor": profile.primary_floor,
            "primary_suite": profile.primary_suite,
            "preferred_language": profile.preferred_language,
        },
        indent=2,
    )


_EXTRACTION_PROMPT_TEMPLATE = """\
You are extracting structured fields from a CBRE call-center maintenance dialogue.
Operate in operator-shorthand: terse, factual, no embellishment.

KNOWN CALLER DEFAULTS (a stale prior; the TRANSCRIPT ALWAYS WINS on conflict):
{profile_block}

DIALOGUE:
{transcript}

EXTRACTION RULES:

1. problem_summary — 1-2 sentence operator-style summary of what the CALLER is reporting.
   Ignore the agent's own routing or escalation chatter. No emotional language.

2. building_name / floor / suite — the location of the ISSUE.
   - If the transcript states a location, USE THAT, even when it disagrees with the profile.
   - If the transcript is silent on a particular field, fall back to the profile default.
   - Confidence is HIGH (>=0.9) for transcript-stated; MODERATE (0.4-0.6) for profile-derived.

3. urgency_cues — verbatim phrases from the transcript that signal urgency or safety risk.
   Look for exactly the kinds of phrases below; capture them as they appear. Empty list when none.
     safety:    "flooding", "smoke", "fire", "trapped", "gas leak", "unconscious", "sparking"
     escalating: "no heat overnight", "getting worse", "can't wait", "people trapped"

4. caller_role — tenant, office_manager, security_contact, billing_contact, vendor, anonymous, etc.
   Prefer the profile's contact_role when one is provided; otherwise infer from the transcript.

5. language — ISO 639-1 code. Default to "en" unless the dialogue is clearly in another language.

CONFIDENCE GUIDE (per field):
   1.0       explicitly stated in the transcript
   0.7-0.9   clearly inferred from transcript context
   0.4-0.6   weak inference, partial info, or profile-derived fallback
   0.0-0.3   unknown / no evidence

DO NOT hallucinate. Use null/None for any field that genuinely can't be determined.
"""


def build_prompt(turns: List[dict], profile: Optional[Profile]) -> str:
    """Public for tests; otherwise an implementation detail of `extract`."""
    return _EXTRACTION_PROMPT_TEMPLATE.format(
        profile_block=_format_profile_block(profile),
        transcript=_flatten_turns(turns),
    )


# ─── LLM-backed entrypoint ─────────────────────────────────────────────


def extract(
    turns: List[dict],
    caller_phone: Optional[str] = None,
    *,
    llm: Any = None,
) -> Extraction:
    """Extract structured fields from a caller dialogue.

    Args:
        turns: list of `{speaker, text}` dicts (the harness's input shape).
        caller_phone: phone number to look up in `caller_profiles.json`.
            None for anonymous callers; unknown numbers also yield no profile.
        llm: test-only injection. Production callers omit this; the
            function constructs `ChatOpenAI` from `agent.config` settings.

    Returns:
        An `Extraction` populated by the LLM. Pydantic validation
        guarantees the schema; field values still need a sanity check
        from downstream nodes (location reconciliation in #19,
        clarification trigger in #18).
    """
    profile = profiles.lookup(caller_phone)
    prompt = build_prompt(turns, profile)

    if llm is None:
        from langchain_openai import ChatOpenAI

        s = config.get_settings()
        kwargs = {"model": s.chat_model, "temperature": s.chat_temperature}
        if s.chat_seed is not None:
            kwargs["seed"] = s.chat_seed
        llm = ChatOpenAI(**kwargs)

    structured = llm.with_structured_output(Extraction)
    return structured.invoke(prompt)
