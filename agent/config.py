"""Central runtime configuration for the agent.

All tunable knobs (model name, embedding model, temperature, retry caps)
live here. Each is overridable via an environment variable; defaults are
picked to favor determinism (`temperature=0`, fixed `seed`) per the
submission contract.

Secrets are validated lazily — importing this module is side-effect-free
beyond a `.env` load. The first call to `get_settings()` (or an explicit
`Settings.validate_or_raise()`) raises a `RuntimeError` naming the exact
missing environment variable. Pass `validate=False` to bypass when wiring up
LLM-free code paths (e.g. data-loader unit tests, the stub in agent/classify).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


REQUIRED_ENV_VARS: Tuple[str, ...] = ("OPENAI_API_KEY",)


@dataclass(frozen=True)
class Settings:
    chat_model: str
    chat_temperature: float
    chat_seed: Optional[int]
    embedding_model: str
    max_extraction_retries: int
    openai_api_key: Optional[str]

    def validate_or_raise(self) -> None:
        missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): "
                f"{', '.join(missing)}. Set them in your shell or in a .env "
                "file at the repo root."
            )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_int_opt(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def get_settings(*, validate: bool = True) -> Settings:
    """Build a `Settings` from the current environment.

    Set `validate=False` only for code paths that genuinely don't need the
    LLM (data loaders, schema tests, the issue-#1 stub).
    """
    settings = Settings(
        chat_model=os.environ.get("AGENT_CHAT_MODEL", "gpt-4o-mini"),
        chat_temperature=_env_float("AGENT_CHAT_TEMPERATURE", 0.0),
        chat_seed=_env_int_opt("AGENT_CHAT_SEED", 7),
        embedding_model=os.environ.get(
            "AGENT_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        max_extraction_retries=_env_int_opt("AGENT_MAX_EXTRACTION_RETRIES", 2),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
    if validate:
        settings.validate_or_raise()
    return settings
