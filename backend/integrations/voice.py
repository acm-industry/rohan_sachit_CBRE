"""Voice (Deepgram STT + ElevenLabs TTS) integration façade.

Peer is building `VoiceSession` in issue #32 (separate package). Until
their PR merges we cannot import it; this module exposes a `VoiceSession`
symbol via a try/except so the backend never hard-fails at import time.
The `VOICE_AVAILABLE` flag tells the routes whether the real
implementation is wired in. The stub keeps `/api/calls/start` callable
in `voice` mode for local dev — it accepts audio bytes and returns a
canned transcript, so the rest of the pipeline can be exercised end-to-end
on the demo laptop even before the voice PR lands.

The peer's intended public surface (from issue-32 PR description):

    class VoiceSession:
        def __init__(self, *, deepgram_key, elevenlabs_key): ...
        async def transcribe(audio_bytes) -> list[dict]   # turns
        async def speak(text: str) -> bytes               # tts audio

We re-export `VoiceSession` only — anything more specific would couple
us to a signature that may still change before issue-32 merges.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

VOICE_AVAILABLE = False
VoiceSession: Any = None

try:  # pragma: no cover — depends on peer's package being installed
    from voice import VoiceSession as _RealVoiceSession  # type: ignore
    VoiceSession = _RealVoiceSession
    VOICE_AVAILABLE = True
    logger.info("voice integration: peer's VoiceSession loaded")
except Exception as e:  # noqa: BLE001
    logger.info("voice integration: peer package not available (%s) — using stub", e)

    class _StubVoiceSession:
        """Minimal stand-in until peer's VoiceSession lands.

        Returns a fixed two-turn transcript regardless of audio input.
        Lets the demo pipeline run end-to-end on a laptop with no
        Deepgram/ElevenLabs keys.
        """

        def __init__(self, *, deepgram_key: Optional[str] = None,
                     elevenlabs_key: Optional[str] = None) -> None:
            self.deepgram_key = deepgram_key
            self.elevenlabs_key = elevenlabs_key

        async def transcribe(self, audio_bytes: bytes) -> List[dict]:
            return [
                {"speaker": "agent", "text": "CBRE maintenance — what's going on?"},
                {
                    "speaker": "caller",
                    "text": (
                        "[stub transcript — peer's VoiceSession not installed] "
                        "Live audio mode is not wired up on this build."
                    ),
                },
            ]

        async def speak(self, text: str) -> bytes:
            return b""

    VoiceSession = _StubVoiceSession
