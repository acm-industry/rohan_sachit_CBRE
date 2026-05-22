"""Live API smoke tests — only run when VOICE_LIVE_TEST=1 is set.

These hit real Deepgram/ElevenLabs APIs and require valid keys in the env.
Run with: VOICE_LIVE_TEST=1 pytest tests/voice/test_live.py -v
"""

from __future__ import annotations

import asyncio
import os
import struct

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VOICE_LIVE_TEST") != "1",
    reason="Live API tests skipped — set VOICE_LIVE_TEST=1 to run",
)

if os.environ.get("VOICE_LIVE_TEST") == "1":
    pytest.importorskip(
        "deepgram", reason="Deepgram SDK is required for live voice tests"
    )
    pytest.importorskip(
        "elevenlabs", reason="ElevenLabs SDK is required for live voice tests"
    )

from voice.session import VoiceSession


def _silence_pcm(duration_s: float = 1.0) -> bytes:
    num_samples = int(16000 * duration_s)
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


@pytest.fixture
def live_session() -> VoiceSession:
    dg_key = os.environ.get("DEEPGRAM_API_KEY", "")
    eleven_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not dg_key or not eleven_key:
        pytest.skip("Missing DEEPGRAM_API_KEY or ELEVENLABS_API_KEY")
    return VoiceSession(deepgram_api_key=dg_key, eleven_api_key=eleven_key)


class TestLiveSTT:
    def test_connect_and_send_silence(
        self, live_session: VoiceSession
    ) -> None:
        async def run() -> None:
            await live_session.connect()
            try:
                result = await live_session.transcribe_chunk(_silence_pcm(0.5))
                # Silence may produce None or empty transcript — both are valid
                if result is not None:
                    assert "text" in result
                    assert "is_final" in result
            finally:
                await live_session.close()

        asyncio.run(run())


class TestLiveTTS:
    def test_speak_returns_audio(self, live_session: VoiceSession) -> None:
        async def run() -> None:
            await live_session.connect()
            try:
                audio = await live_session.speak("Hello, this is a test.")
                assert len(audio) > 0
                # MP3 files start with ID3 tag or MPEG sync word
                assert audio[:3] == b"ID3" or audio[:2] == b"\xff\xfb"
            finally:
                await live_session.close()

        asyncio.run(run())
