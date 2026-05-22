"""Unit tests for VoiceSession — mocked, no live API calls."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
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
        assert session._dg_connection is None
        assert session._eleven_client is None

    def test_custom_voice_id(self) -> None:
        s = VoiceSession("k1", "k2", voice_id="custom-voice")
        assert s._voice_id == "custom-voice"


class TestTranscribeChunk:
    def test_raises_when_not_connected(
        self, session: VoiceSession, short_audio_chunk: bytes
    ) -> None:
        async def run() -> None:
            with pytest.raises(RuntimeError, match="not connected"):
                await session.transcribe_chunk(short_audio_chunk)

        asyncio.run(run())

    def test_returns_none_when_no_transcript(
        self, session: VoiceSession, short_audio_chunk: bytes
    ) -> None:
        async def run() -> None:
            session._connected = True
            session._dg_connection = AsyncMock()

            result = await session.transcribe_chunk(short_audio_chunk)
            assert result is None
            session._dg_connection.send.assert_awaited_once_with(short_audio_chunk)

        asyncio.run(run())

    def test_returns_transcript_event(
        self, session: VoiceSession, short_audio_chunk: bytes
    ) -> None:
        async def run() -> None:
            session._connected = True
            session._dg_connection = AsyncMock()
            session._transcript_queue.put_nowait(
                {"text": "hello", "is_final": False}
            )

            result = await session.transcribe_chunk(short_audio_chunk)
            assert result == {"text": "hello", "is_final": False}

        asyncio.run(run())


class TestEndUtterance:
    def test_raises_when_not_connected(self, session: VoiceSession) -> None:
        async def run() -> None:
            with pytest.raises(RuntimeError, match="not connected"):
                await session.end_utterance()

        asyncio.run(run())

    def test_returns_accumulated_transcript(self, session: VoiceSession) -> None:
        async def run() -> None:
            session._connected = True
            session._dg_connection = AsyncMock()
            session._final_transcript_parts = ["hello", "world"]

            with patch("voice.session.asyncio.sleep", new=AsyncMock()):
                transcript = await session.end_utterance()

            assert transcript == "hello world"
            assert session._final_transcript_parts == []
            session._dg_connection.finalize.assert_awaited_once()

        asyncio.run(run())


class TestSpeak:
    def test_raises_when_tts_sdk_missing(self, session: VoiceSession) -> None:
        async def run() -> None:
            with patch("voice.session.AsyncElevenLabs", None):
                with pytest.raises(RuntimeError, match="ElevenLabs SDK"):
                    await session.speak("hi")

        asyncio.run(run())

    def test_returns_audio_bytes(self, session: VoiceSession) -> None:
        async def run() -> None:
            async def fake_stream():
                yield b"\x00\x01"
                yield b"\x02\x03"

            mock_client = MagicMock()
            mock_client.text_to_speech.convert = MagicMock(return_value=fake_stream())
            session._eleven_client = mock_client

            with patch("voice.session.AsyncElevenLabs", MagicMock()):
                audio = await session.speak("hello")

            assert audio == b"\x00\x01\x02\x03"
            mock_client.text_to_speech.convert.assert_called_once()

        asyncio.run(run())


class TestConnect:
    def test_connect_sets_up_clients(self, session: VoiceSession) -> None:
        async def run() -> None:
            mock_connection = MagicMock()
            mock_connection.on = MagicMock()
            mock_connection.start = AsyncMock(return_value=True)
            mock_connection.finish = AsyncMock()
            mock_connection.keep_alive = AsyncMock()

            mock_dg_instance = MagicMock()
            mock_dg_instance.listen.asyncwebsocket.v.return_value = mock_connection

            events = SimpleNamespace(
                Open="open",
                Error="error",
                Transcript="transcript",
            )

            def on_event(event: str, handler) -> None:
                if event == events.Open:
                    session._ready_event.set()

            mock_connection.on.side_effect = on_event

            with patch("voice.session.DeepgramClient") as MockDG, \
                 patch("voice.session.AsyncElevenLabs") as MockEleven, \
                 patch("voice.session.LiveTranscriptionEvents", events), \
                 patch("voice.session.LiveOptions") as MockOptions:
                MockDG.return_value = mock_dg_instance

                await session.connect()

                assert session._connected is True
                MockDG.assert_called_once_with(api_key="test-dg-key")
                MockEleven.assert_called_once_with(api_key="test-eleven-key")
                mock_dg_instance.listen.asyncwebsocket.v.assert_called_once_with("1")
                assert mock_connection.on.call_count == 3
                mock_connection.start.assert_awaited_once_with(MockOptions.return_value)

            await session.close()

        asyncio.run(run())

    def test_connect_raises_when_voice_sdks_missing(
        self, session: VoiceSession
    ) -> None:
        async def run() -> None:
            with patch("voice.session.DeepgramClient", None):
                with pytest.raises(RuntimeError, match="Deepgram SDK"):
                    await session.connect()

        asyncio.run(run())


class TestClose:
    def test_close_finishes_connection(self, session: VoiceSession) -> None:
        async def run() -> None:
            session._connected = True
            session._dg_connection = AsyncMock()

            task = asyncio.create_task(asyncio.sleep(10))
            session._keepalive_task = task

            await session.close()
            assert session._connected is False
            assert session._keepalive_task is None
            session._dg_connection.finish.assert_awaited_once()

        asyncio.run(run())
