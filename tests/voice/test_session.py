"""Unit tests for VoiceSession — mocked, no live API calls."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice.session import VoiceSession


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(
        deepgram_api_key="test-dg-key",
        eleven_api_key="test-eleven-key",
    )


class TestInit:
    def test_initial_state(self, session: VoiceSession) -> None:
        assert session._connected is False
        assert session._dg_client is None
        assert session._eleven_client is None

    def test_custom_voice_id(self) -> None:
        s = VoiceSession("k1", "k2", voice_id="custom-voice")
        assert s._voice_id == "custom-voice"


class TestTranscribeChunk:
    @pytest.mark.asyncio
    async def test_raises_when_not_connected(
        self, session: VoiceSession, short_audio_chunk: bytes
    ) -> None:
        with pytest.raises(RuntimeError, match="not connected"):
            await session.transcribe_chunk(short_audio_chunk)

    @pytest.mark.asyncio
    async def test_returns_none_when_no_transcript(
        self, session: VoiceSession, short_audio_chunk: bytes
    ) -> None:
        session._connected = True
        session._dg_socket = AsyncMock()

        result = await session.transcribe_chunk(short_audio_chunk)
        assert result is None
        session._dg_socket.send_media.assert_awaited_once_with(short_audio_chunk)

    @pytest.mark.asyncio
    async def test_returns_transcript_event(
        self, session: VoiceSession, short_audio_chunk: bytes
    ) -> None:
        session._connected = True
        session._dg_socket = AsyncMock()
        session._transcript_queue.put_nowait({"text": "hello", "is_final": False})

        result = await session.transcribe_chunk(short_audio_chunk)
        assert result == {"text": "hello", "is_final": False}


class TestEndUtterance:
    @pytest.mark.asyncio
    async def test_raises_when_not_connected(self, session: VoiceSession) -> None:
        with pytest.raises(RuntimeError, match="not connected"):
            await session.end_utterance()

    @pytest.mark.asyncio
    async def test_returns_accumulated_transcript(
        self, session: VoiceSession
    ) -> None:
        session._connected = True
        session._dg_socket = AsyncMock()
        session._final_transcript_parts = ["hello", "world"]

        transcript = await session.end_utterance()
        assert transcript == "hello world"
        assert session._final_transcript_parts == []


class TestSpeak:
    @pytest.mark.asyncio
    async def test_raises_when_not_connected(self, session: VoiceSession) -> None:
        with pytest.raises(RuntimeError, match="not connected"):
            await session.speak("hi")

    @pytest.mark.asyncio
    async def test_returns_audio_bytes(self, session: VoiceSession) -> None:
        async def fake_stream():
            yield b"\x00\x01"
            yield b"\x02\x03"

        mock_client = AsyncMock()
        mock_client.text_to_speech.convert = AsyncMock(return_value=fake_stream())
        session._eleven_client = mock_client

        audio = await session.speak("hello")
        assert audio == b"\x00\x01\x02\x03"
        mock_client.text_to_speech.convert.assert_awaited_once()


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_sets_up_clients(self, session: VoiceSession) -> None:
        mock_socket = AsyncMock()
        mock_socket.on = MagicMock()
        mock_socket.start_listening = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_socket)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("voice.session.AsyncDeepgramClient") as MockDG, \
             patch("voice.session.AsyncElevenLabs") as MockEleven:
            mock_dg_instance = MagicMock()
            mock_dg_instance.listen.v1.connect.return_value = mock_ctx
            MockDG.return_value = mock_dg_instance

            await session.connect()

            assert session._connected is True
            MockDG.assert_called_once_with(api_key="test-dg-key")
            MockEleven.assert_called_once_with(api_key="test-eleven-key")


class TestClose:
    @pytest.mark.asyncio
    async def test_close_finishes_connection(self, session: VoiceSession) -> None:
        session._connected = True
        session._dg_socket = AsyncMock()
        session._dg_ctx = AsyncMock()
        session._dg_ctx.__aexit__ = AsyncMock(return_value=False)

        task = asyncio.create_task(asyncio.sleep(10))
        session._listener_task = task

        await session.close()
        assert session._connected is False
        session._dg_socket.send_close_stream.assert_awaited_once()
