"""VoiceSession — streaming STT via Deepgram, TTS via ElevenLabs."""

from __future__ import annotations

import asyncio
from typing import Any

from deepgram import AsyncDeepgramClient
from deepgram.listen.v1.socket_client import (
    AsyncV1SocketClient,
    EventType,
    ListenV1Results,
)
from elevenlabs import AsyncElevenLabs


_DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # ElevenLabs "George"


class VoiceSession:
    """Wraps Deepgram real-time STT and ElevenLabs TTS into a single session.

    Usage from backend:

        session = VoiceSession(deepgram_key, eleven_key)
        await session.connect()
        result = await session.transcribe_chunk(audio_bytes)
        transcript = await session.end_utterance()
        audio = await session.speak("Hello, how can I help?")
        await session.close()
    """

    def __init__(
        self,
        deepgram_api_key: str,
        eleven_api_key: str,
        voice_id: str = _DEFAULT_VOICE_ID,
    ) -> None:
        self._dg_key = deepgram_api_key
        self._eleven_key = eleven_api_key
        self._voice_id = voice_id

        self._dg_client: AsyncDeepgramClient | None = None
        self._dg_socket: AsyncV1SocketClient | None = None
        self._dg_ctx: Any = None
        self._eleven_client: AsyncElevenLabs | None = None
        self._listener_task: asyncio.Task | None = None

        self._transcript_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._final_transcript_parts: list[str] = []
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Deepgram live-transcription websocket."""
        self._dg_client = AsyncDeepgramClient(api_key=self._dg_key)

        self._dg_ctx = self._dg_client.listen.v1.connect(
            model="nova-2",
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            utterance_end_ms="1500",
            vad_events=True,
            punctuate=True,
            language="en",
        )
        self._dg_socket = await self._dg_ctx.__aenter__()
        self._dg_socket.on(EventType.MESSAGE, self._on_message)

        self._listener_task = asyncio.create_task(self._dg_socket.start_listening())

        self._eleven_client = AsyncElevenLabs(api_key=self._eleven_key)
        self._connected = True

    async def close(self) -> None:
        """Tear down the Deepgram websocket."""
        if self._dg_socket:
            await self._dg_socket.send_close_stream()
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._dg_ctx:
            await self._dg_ctx.__aexit__(None, None, None)
        self._connected = False

    # ------------------------------------------------------------------
    # STT — streaming
    # ------------------------------------------------------------------

    async def transcribe_chunk(self, audio_chunk: bytes) -> dict | None:
        """Send an audio chunk to Deepgram; return a transcript event if available.

        Returns:
            {'text': str, 'is_final': bool} when Deepgram emits a result,
            None if nothing new yet.
        """
        if not self._connected:
            raise RuntimeError("VoiceSession is not connected — call connect() first")

        await self._dg_socket.send_media(audio_chunk)

        try:
            return self._transcript_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def end_utterance(self) -> str:
        """Flush the Deepgram buffer and return the full utterance transcript."""
        if not self._connected:
            raise RuntimeError("VoiceSession is not connected — call connect() first")

        await self._dg_socket.send_finalize()
        # Brief wait for final results to arrive
        await asyncio.sleep(0.3)

        # Drain remaining events
        while not self._transcript_queue.empty():
            event = self._transcript_queue.get_nowait()
            if event["is_final"]:
                self._final_transcript_parts.append(event["text"])

        transcript = " ".join(self._final_transcript_parts).strip()
        self._final_transcript_parts.clear()
        return transcript

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> bytes:
        """Synthesize speech from text via ElevenLabs. Returns mp3 bytes."""
        if self._eleven_client is None:
            raise RuntimeError("VoiceSession is not connected — call connect() first")

        audio_stream = self._eleven_client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
        )

        chunks: list[bytes] = []
        async for chunk in audio_stream:
            chunks.append(chunk)
        return b"".join(chunks)

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    async def _on_message(self, message: Any) -> None:
        """Deepgram message event handler."""
        if not isinstance(message, ListenV1Results):
            return

        alternatives = message.channel.alternatives
        if not alternatives:
            return

        sentence = alternatives[0].transcript
        if not sentence:
            return

        is_final = bool(message.is_final)
        if is_final:
            self._final_transcript_parts.append(sentence)

        self._transcript_queue.put_nowait({"text": sentence, "is_final": is_final})
