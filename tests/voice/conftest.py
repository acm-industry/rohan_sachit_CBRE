"""Shared fixtures for voice tests."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_audio_chunk() -> bytes:
    """1 second of silence as 16-bit PCM, 16 kHz mono."""
    num_samples = 16000
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


@pytest.fixture
def short_audio_chunk() -> bytes:
    """100ms of silence — enough to feed to transcribe_chunk."""
    num_samples = 1600
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))
