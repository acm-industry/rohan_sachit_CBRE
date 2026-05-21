"""VoiceSession — streaming STT via Deepgram, TTS via ElevenLabs.

Written against deepgram-sdk v4.x and elevenlabs-sdk v1.x. Pure I/O —
no agent imports, no HTTP server. Designed to be imported and called
from a FastAPI backend (or any async caller) on a per-utterance basis.

Usage (single utterance — recommended for live demo):

    session = VoiceSession(deepgram_key, eleven_key)
    await session.connect()
    try:
        # Stream audio chunks (PCM linear16, 16kHz, mono) in real-ish time;
        # pace them at <= 2x real-time or Deepgram may drop frames.
        for chunk in mic_chunks:
            await session.transcribe_chunk(chunk)
        # 1-2s of trailing silence helps Deepgram VAD detect end-of-utterance.
        for _ in range(15):
            await session.transcribe_chunk(b"\\x00" * 3200)
        transcript = await session.end_utterance()
        audio_mp3 = await session.speak("I've logged your request.")
    finally:
        await session.close()

Notes on the persistent-connection pattern (multiple utterances on one
session): Deepgram closes idle websockets after ~10s. A keepalive loop
pings every 5s while connected. Some patch combinations of finalize()
also seem to put the connection into a drain state — for safety, a
per-utterance lifecycle (open → send → end → close, fresh session per
turn) is the most reliable pattern and is what the demo + backend use.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
from elevenlabs import AsyncElevenLabs


_DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # ElevenLabs "George"
_OPEN_EVENT_TIMEOUT_S = 3.0
_KEEPALIVE_INTERVAL_S = 5.0
_FINALIZE_GRACE_S = 0.5


class VoiceSession:
    """Wraps Deepgram real-time STT and ElevenLabs TTS into a single session."""

    def __init__(
        self,
        deepgram_api_key: str | None = None,
        eleven_api_key: str | None = None,
        *,
        deepgram_key: str | None = None,
        elevenlabs_key: str | None = None,
        voice_id: str = _DEFAULT_VOICE_ID,
    ) -> None:
        # Accept both naming conventions: the backend uses
        # `deepgram_key`/`elevenlabs_key`, this module's original signature
        # used `deepgram_api_key`/`eleven_api_key`. Either works.
        self._dg_key = deepgram_api_key or deepgram_key
        self._eleven_key = eleven_api_key or elevenlabs_key
        if not self._dg_key:
            raise ValueError("Deepgram API key is required (deepgram_api_key= or deepgram_key=)")
        if not self._eleven_key:
            raise ValueError("ElevenLabs API key is required (eleven_api_key= or elevenlabs_key=)")
        self._voice_id = voice_id

        self._dg_client: DeepgramClient | None = None
        self._dg_connection: Any = None  # AsyncListenWebSocketClient
        self._eleven_client: AsyncElevenLabs | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._ready_event: asyncio.Event = asyncio.Event()

        self._transcript_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._final_transcript_parts: list[str] = []
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Deepgram live-transcription websocket."""
        self._dg_client = DeepgramClient(api_key=self._dg_key)
        self._dg_connection = self._dg_client.listen.asyncwebsocket.v("1")
        self._ready_event = asyncio.Event()

        self._dg_connection.on(LiveTranscriptionEvents.Open, self._on_open)
        self._dg_connection.on(LiveTranscriptionEvents.Error, self._on_error)
        self._dg_connection.on(
            LiveTranscriptionEvents.Transcript, self._on_transcript
        )

        options = LiveOptions(
            model="nova-2",
            language="en",
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            utterance_end_ms="1500",
            vad_events=True,
            punctuate=True,
        )

        started = await self._dg_connection.start(options)
        if not started:
            raise RuntimeError("Failed to open Deepgram websocket")

        # Wait for the websocket Open event before declaring connected, so
        # callers don't race the handshake. Fall through after timeout —
        # some patch versions of the SDK don't emit Open reliably but the
        # connection still works.
        try:
            await asyncio.wait_for(
                self._ready_event.wait(), timeout=_OPEN_EVENT_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            pass

        self._eleven_client = AsyncElevenLabs(api_key=self._eleven_key)
        self._connected = True
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def close(self) -> None:
        """Tear down the Deepgram websocket."""
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._keepalive_task = None

        if self._dg_connection is not None:
            try:
                await self._dg_connection.finish()
            except Exception:
                pass
        self._connected = False

    async def _keepalive_loop(self) -> None:
        """Ping Deepgram periodically to prevent idle-timeout (dpgr.am/net0001)."""
        try:
            while self._connected:
                await asyncio.sleep(_KEEPALIVE_INTERVAL_S)
                if self._connected and self._dg_connection is not None:
                    try:
                        await self._dg_connection.keep_alive()
                    except Exception:
                        # Connection may be torn down mid-ping; exit cleanly.
                        break
        except asyncio.CancelledError:
            pass

    async def _on_open(self, *args: Any, **kwargs: Any) -> None:
        """Deepgram Open event handler — signals websocket is ready."""
        self._ready_event.set()

    async def _on_error(self, *args: Any, **kwargs: Any) -> None:
        """Deepgram Error event handler — surface for ops visibility."""
        err = (
            kwargs.get("error")
            or (args[1] if len(args) > 1 else args[0] if args else None)
        )
        print(f"[VoiceSession] Deepgram error: {err!r}", file=sys.stderr)

    # ------------------------------------------------------------------
    # STT — streaming
    # ------------------------------------------------------------------

    async def transcribe_chunk(self, audio_chunk: bytes) -> dict | None:
        """Send an audio chunk to Deepgram; return a transcript event if available.

        Audio must be PCM linear16, 16000 Hz, mono. Chunks should be paced
        at <= 2x real-time (e.g. 50 ms between 100 ms chunks); burst-sending
        the full utterance at once will cause Deepgram to silently drop
        frames and return an empty transcript.

        Returns:
            {'text': str, 'is_final': bool} when Deepgram emits a result,
            None if nothing new yet.
        """
        if not self._connected:
            raise RuntimeError(
                "VoiceSession is not connected — call connect() first"
            )

        await self._dg_connection.send(audio_chunk)

        try:
            return self._transcript_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def transcribe(self, audio_bytes: bytes) -> list[dict]:
        """One-shot transcription convenience — handles the full lifecycle.

        Opens a fresh Deepgram session, streams the audio with pacing,
        flushes, closes, and returns turns in the
        `[{"speaker", "text"}, ...]` shape the backend's voice route
        expects. Use this when you have a complete utterance in hand and
        want a single transcript back (typical for the FastAPI route that
        receives a full audio blob via POST).

        Audio must be PCM linear16, 16000 Hz, mono.
        """
        # Auto-manage the session lifecycle.
        await self.connect()
        try:
            chunk_size = 3200  # 100 ms at 16 kHz int16 mono
            # Pace audio at 2x real-time (50 ms between 100 ms chunks); Deepgram
            # drops frames silently if we burst-send.
            for i in range(0, len(audio_bytes), chunk_size):
                await self.transcribe_chunk(audio_bytes[i : i + chunk_size])
                await asyncio.sleep(0.05)
            # Trailing silence so Deepgram VAD detects end-of-speech.
            silence = b"\x00" * chunk_size
            for _ in range(15):  # 1.5 s of silence
                await self.transcribe_chunk(silence)
                await asyncio.sleep(0.05)
            transcript = await self.end_utterance()
        finally:
            await self.close()

        # Return turns in the dialogue shape the agent pipeline + frontend
        # expect. We synthesize a single agent greeting + the caller's
        # transcribed speech as one caller turn.
        caller_text = transcript or "[no speech detected]"
        return [
            {"speaker": "agent", "text": "CBRE maintenance, what's going on?"},
            {"speaker": "caller", "text": caller_text},
        ]

    async def end_utterance(self) -> str:
        """Flush the Deepgram buffer and return the full utterance transcript.

        The transcript is assembled directly from `_final_transcript_parts`,
        which the `_on_transcript` handler populates as finals arrive on the
        websocket. The `_transcript_queue` exists for callers that want to
        observe live (interim) transcript events via `transcribe_chunk()`'s
        return value — it is NOT drained here, since that would double-count
        every final.
        """
        if not self._connected:
            raise RuntimeError(
                "VoiceSession is not connected — call connect() first"
            )

        # Ask Deepgram to flush buffered finals.
        try:
            await self._dg_connection.finalize()
        except Exception:
            # Some SDK builds don't expose finalize(); fall through and rely
            # on the natural utterance-end VAD trigger (utterance_end_ms).
            pass

        # Brief wait for finals to arrive on the websocket and be appended
        # by the handler.
        await asyncio.sleep(_FINALIZE_GRACE_S)

        transcript = " ".join(self._final_transcript_parts).strip()
        self._final_transcript_parts.clear()
        return transcript

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> bytes:
        """Synthesize speech from text via ElevenLabs. Returns mp3 bytes.

        ElevenLabs is HTTP-based (no persistent websocket like Deepgram),
        so this works standalone — you can construct a VoiceSession purely
        for TTS without ever calling connect(). The ElevenLabs client is
        created lazily on first speak().
        """
        if self._eleven_client is None:
            self._eleven_client = AsyncElevenLabs(api_key=self._eleven_key)

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

    async def _on_transcript(self, *args: Any, **kwargs: Any) -> None:
        """Deepgram Transcript event handler.

        The v4 SDK calls handlers as `handler(socket, result, **kwargs)`
        when registered as bound methods. To be robust across patch
        releases we accept *args/**kwargs and locate the result object
        by duck-typing.
        """
        result = kwargs.get("result")
        if result is None:
            for arg in args:
                if hasattr(arg, "channel"):
                    result = arg
                    break
        if result is None:
            return

        try:
            alts = result.channel.alternatives
            sentence = alts[0].transcript if alts else ""
        except (AttributeError, IndexError):
            return

        if not sentence:
            return

        is_final = bool(getattr(result, "is_final", False))
        if is_final:
            self._final_transcript_parts.append(sentence)

        self._transcript_queue.put_nowait(
            {"text": sentence, "is_final": is_final}
        )
