"""Interactive voice-agent demo — mic in → Deepgram STT → ElevenLabs TTS → speakers.

Press Enter to start recording, speak, press Enter again to stop.
The agent transcribes what you said and speaks back an echo.
Ctrl-C to quit.

Requires DEEPGRAM_API_KEY and ELEVENLABS_API_KEY in .env (or env).
Uses the project's VoiceSession from voice/session.py — same module the
backend would call into. This is a thin harness around it so you can
feel the round-trip latency on your laptop without standing up the
backend + frontend.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from voice import VoiceSession  # noqa: E402

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_MS = 100
CHUNK_BYTES = SAMPLE_RATE * 2 * CHANNELS * CHUNK_MS // 1000  # 16-bit mono


def record_until_enter() -> bytes:
    """Record from default mic until user presses Enter. Returns raw PCM16."""
    chunks: list[np.ndarray] = []
    stop = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n  (mic warning: {status})", file=sys.stderr)
        if not stop.is_set():
            chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
    ):
        print("→ Recording. Press Enter again to stop... ", end="", flush=True)
        input()
        stop.set()

    if not chunks:
        return b""
    audio = np.concatenate(chunks)
    dur = len(audio) / SAMPLE_RATE
    print(f"  ({dur:.1f}s captured)")
    return audio.tobytes()


def play_mp3(mp3_bytes: bytes) -> None:
    """Play mp3 audio via macOS's afplay (built-in)."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes)
        path = f.name
    try:
        subprocess.run(["afplay", path], check=True)
    finally:
        os.unlink(path)


async def main() -> None:
    dg = os.environ.get("DEEPGRAM_API_KEY")
    el = os.environ.get("ELEVENLABS_API_KEY")
    if not dg or not el:
        print(
            "ERROR: DEEPGRAM_API_KEY and ELEVENLABS_API_KEY must be set "
            "(check .env at repo root)",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("Voice demo  ·  mic → Deepgram → ElevenLabs → speakers")
    print("Loop: Enter to record, Enter again to stop. Ctrl-C to exit.")
    print("=" * 60)

    # Per-utterance connection lifecycle — sidesteps Deepgram's ~10s idle
    # timeout while the user is between utterances. Each turn opens a fresh
    # websocket, sends the audio, gets the transcript, and closes. The TTS
    # client (ElevenLabs is HTTP-based, not websocket) is reused across
    # turns for slightly faster speak() calls.

    try:
        while True:
            print("\n→ Press Enter when ready to speak... ", end="", flush=True)
            input()

            audio = record_until_enter()
            if not audio:
                print("  (no audio — try again)")
                continue

            # Save captured audio to disk so you can verify what the mic
            # actually got. Play back with: afplay /tmp/voice_demo_last.wav
            import wave
            with wave.open("/tmp/voice_demo_last.wav", "wb") as w:
                w.setnchannels(CHANNELS)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(audio)
            # Compute audio level for diagnostic
            samples = np.frombuffer(audio, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            peak = int(np.max(np.abs(samples)))
            print(f"  [debug] audio: rms={rms:.0f}, peak={peak} (silence ~ rms<50, speech ~ rms>500)")
            print(f"  [debug] wav saved to /tmp/voice_demo_last.wav — play with: afplay /tmp/voice_demo_last.wav")

            print("  [opening Deepgram + ElevenLabs session...]")
            session = VoiceSession(dg, el)
            await session.connect()

            try:
                print("  → sending to Deepgram (paced at half-real-time)...")
                for i in range(0, len(audio), CHUNK_BYTES):
                    await session.transcribe_chunk(audio[i : i + CHUNK_BYTES])
                    await asyncio.sleep(0.05)  # 50ms = half-real-time pacing
                # Brief tail of silence so Deepgram VAD sees end-of-speech.
                silence = b"\x00" * CHUNK_BYTES
                for _ in range(15):  # 1.5s of silence
                    await session.transcribe_chunk(silence)
                    await asyncio.sleep(0.05)

                transcript = await session.end_utterance()
                print(f"  📝 transcript: {transcript!r}")

                if not transcript.strip():
                    print("  (empty transcript — speak louder or check mic)")
                    continue

                response = f"I heard you say: {transcript}"
                print(f"  🗣  TTS: {response!r}")
                mp3 = await session.speak(response)
                print(f"  ▶  playing {len(mp3) // 1024} KB of mp3 audio...")
                play_mp3(mp3)
            finally:
                await session.close()

    except KeyboardInterrupt:
        print("\n\n[exiting cleanly]")


if __name__ == "__main__":
    asyncio.run(main())
