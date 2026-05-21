"""Call lifecycle endpoints.

  POST /api/calls/start                    — spin up a call worker
  GET  /api/calls/{call_id}/events         — SSE stream
  POST /api/calls/{call_id}/review         — resume the validator gate
  POST /api/calls/{call_id}/audio          — push live audio (voice mode)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.classify import classify_with_events
from backend.demo.canned import get_transcript, is_demo_id
from backend.events.bus import CallBus, CallRegistry, is_sentinel
from backend.events.emitter import make_emitter
from backend.integrations.voice import VOICE_AVAILABLE, VoiceSession


logger = logging.getLogger(__name__)

router = APIRouter()


class StartCallBody(BaseModel):
    mode: str  # "canned" | "voice"
    transcript_id: Optional[str] = None
    caller_phone: Optional[str] = None


class ReviewBody(BaseModel):
    decision: str  # "approve" | "override"
    override: Optional[Dict[str, Any]] = None


def _registry(request: Request) -> CallRegistry:
    reg: CallRegistry = request.app.state.registry
    return reg


async def _run_pipeline(
    bus: CallBus,
    turns: List[dict],
    caller_phone: Optional[str],
    *,
    speak_response: bool = False,
) -> None:
    """Run the agent pipeline as a background task and close the bus.

    When `speak_response=True` (voice-mode calls), the pipeline's summary
    is captured and synthesized to speech via ElevenLabs; an additional
    `voice_response` SSE event is emitted with base64-encoded mp3 audio
    so the frontend can play it back to the caller.
    """
    captured_summary: Dict[str, Optional[str]] = {"value": None}
    base_emitter = make_emitter(bus)

    async def emitter(event: Dict[str, Any]) -> None:
        await base_emitter(event)
        if (
            speak_response
            and event.get("stage") == "summary"
            and event.get("status") == "complete"
        ):
            payload = event.get("payload") or {}
            captured_summary["value"] = payload.get("call_summary")

    try:
        await classify_with_events(
            turns,
            caller_phone,
            emitter=emitter,
            pause_at_validator=True,
            wait_for_resume=bus.wait_for_resume,
        )

        if speak_response and captured_summary["value"] and VOICE_AVAILABLE:
            await _speak_summary(bus, captured_summary["value"])
    except Exception as e:  # noqa: BLE001
        logger.exception("pipeline failed for %s: %s", bus.call_id, e)
        await bus.publish(
            {
                "call_id": bus.call_id,
                "stage": "pipeline",
                "status": "failed",
                "timing_ms": 0,
                "payload": {"error": str(e)},
            }
        )
    finally:
        await bus.close()


async def _speak_summary(bus: CallBus, summary: str) -> None:
    """Synthesize the call_summary to MP3, emit audio + agent transcript turn."""
    speakable = _make_speakable(summary)
    # Show the agent's spoken response in the ticker first.
    await bus.publish(
        {
            "call_id": bus.call_id,
            "stage": "transcript",
            "status": "complete",
            "timing_ms": 0,
            "payload": {"speaker": "agent", "text": speakable},
        }
    )
    await _speak_text(bus, speakable)


_AGENT_GREETING = "CBRE maintenance, what's going on?"


async def _speak_greeting(bus: CallBus) -> None:
    """Emit the agent greeting transcript + TTS audio on the bus.

    Called from /api/calls/start in voice mode. The caller hears this
    when their SSE subscription opens (typically ~50ms after start_call
    returns), well before they begin speaking.
    """
    # Brief pause so the frontend has time to subscribe to the SSE stream
    # before we publish — events on the queue are still delivered if they
    # land before the GET, but pacing them after subscription keeps the
    # audio playback synchronized with the visible transcript line.
    await asyncio.sleep(0.25)
    await bus.publish(
        {
            "call_id": bus.call_id,
            "stage": "transcript",
            "status": "complete",
            "timing_ms": 0,
            "payload": {"speaker": "agent", "text": _AGENT_GREETING},
        }
    )
    await _speak_text(bus, _AGENT_GREETING)


async def _speak_text(bus: CallBus, text: str) -> None:
    """Synthesize `text` to MP3 via ElevenLabs and publish a voice_response.

    Use for any agent-spoken line (greeting, summary, mid-call clarifying
    question, etc.). Safe to fire-and-forget as a background task; errors
    are logged but never raise.
    """
    import base64

    try:
        session = VoiceSession(
            deepgram_key=os.environ.get("DEEPGRAM_API_KEY"),
            elevenlabs_key=os.environ.get("ELEVENLABS_API_KEY"),
        )
        # speak() is standalone — no connect() needed (TTS is HTTP, not WS).
        mp3 = await session.speak(text)
        await bus.publish(
            {
                "call_id": bus.call_id,
                "stage": "voice_response",
                "status": "complete",
                "timing_ms": 0,
                "payload": {
                    "audio_base64": base64.b64encode(mp3).decode("ascii"),
                    "audio_mime": "audio/mpeg",
                    "text": text,
                },
            }
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("voice synthesis failed for %r: %s", text[:50], e)


def _make_speakable(summary: str) -> str:
    """Rewrite a call_summary so ElevenLabs reads it naturally.

    The agent's summary is operator-style shorthand: 'Pipe leak at Building 3,
    Floor 7. MEDIUM risk. Dispatched Metro Drain & Pipe.' ElevenLabs handles
    most of this fine but mispronounces all-caps risk bands and digit
    abbreviations. Apply a few targeted substitutions.
    """
    import re

    s = summary
    # Risk bands: read as words, lowercase
    s = re.sub(r"\bEMERGENCY\b", "emergency", s)
    s = re.sub(r"\bHIGH\b", "high", s)
    s = re.sub(r"\bMEDIUM\b", "medium", s)
    s = re.sub(r"\bLOW\b", "low", s)
    # Digit floor / building abbreviations → spelled-out forms TTS reads cleanly
    s = re.sub(r"\bFloor (\d+)\b", r"floor number \1", s)
    s = re.sub(r"\bBuilding (\d+)\b", r"building number \1", s)
    return s


@router.post("/api/calls/start")
async def start_call(body: StartCallBody, request: Request) -> Dict[str, str]:
    reg = _registry(request)
    call_id = uuid.uuid4().hex

    if body.mode == "canned":
        if not body.transcript_id or not is_demo_id(body.transcript_id):
            raise HTTPException(400, "unknown transcript_id for canned mode")
        rec = get_transcript(body.transcript_id)
        turns = rec["turns"]
        caller_phone = rec["caller_phone"]
        bus = await reg.create(call_id)
        asyncio.create_task(_run_pipeline(bus, turns, caller_phone))
        return {"call_id": call_id}

    if body.mode == "voice":
        # Voice mode: create the bus now; pipeline kicks off in /audio
        # once the caller posts the audio blob. We stash the phone so
        # the audio endpoint can pick it up. Also fire the agent's
        # greeting on the bus as a background task — the frontend opens
        # its SSE subscription immediately after this returns, so the
        # caller hears "CBRE maintenance, what's going on?" when they
        # connect, *before* they start speaking (like a real call line).
        bus = await reg.create(call_id)
        bus.resume_payload = None  # unused for voice; kept for symmetry
        request.app.state.voice_phone[call_id] = body.caller_phone

        if VOICE_AVAILABLE:
            asyncio.create_task(_speak_greeting(bus))

        return {"call_id": call_id}

    raise HTTPException(400, f"unknown mode: {body.mode}")


async def _sse_stream(bus: CallBus):
    """Convert bus events to SSE-formatted lines.

    Events with `__sse_type__` are emitted as named SSE events (the value
    becomes the SSE `event:` line); everything else is emitted as a default
    unnamed event for `EventSource.onmessage`. This lets the backend
    publish transcript chunks under the named `transcript` channel that
    the frontend's `onTranscript` handler subscribes to.
    """
    yield ": connected\n\n"
    while True:
        event = await bus.queue.get()
        if is_sentinel(event):
            yield "event: done\ndata: {}\n\n"
            return
        sse_type = event.pop("__sse_type__", None) if isinstance(event, dict) else None
        payload = json.dumps(event, default=str)
        if sse_type:
            yield f"event: {sse_type}\ndata: {payload}\n\n"
        else:
            yield f"data: {payload}\n\n"


@router.get("/api/calls/{call_id}/events")
async def stream_events(call_id: str, request: Request) -> StreamingResponse:
    bus = _registry(request).get(call_id)
    if bus is None:
        raise HTTPException(404, f"unknown call_id: {call_id}")
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        _sse_stream(bus),
        media_type="text/event-stream",
        headers=headers,
    )


@router.post("/api/calls/{call_id}/review")
async def submit_review(call_id: str, body: ReviewBody, request: Request) -> Dict[str, bool]:
    bus = _registry(request).get(call_id)
    if bus is None:
        raise HTTPException(404, f"unknown call_id: {call_id}")
    if body.decision not in {"approve", "override"}:
        raise HTTPException(400, "decision must be 'approve' or 'override'")
    bus.submit_review(body.decision, body.override)
    return {"accepted": True}


@router.post("/api/calls/{call_id}/audio")
async def push_audio(call_id: str, request: Request):
    bus = _registry(request).get(call_id)
    if bus is None:
        raise HTTPException(404, f"unknown call_id: {call_id}")

    audio_bytes = await request.body()

    session = VoiceSession(
        deepgram_key=os.environ.get("DEEPGRAM_API_KEY"),
        elevenlabs_key=os.environ.get("ELEVENLABS_API_KEY"),
    )
    try:
        turns = await session.transcribe(audio_bytes)
    except Exception as e:  # noqa: BLE001
        logger.exception("voice transcription failed: %s", e)
        raise HTTPException(500, f"transcription failed: {e}")

    caller_phone = request.app.state.voice_phone.pop(call_id, None)

    # Emit caller turn(s) as transcript stage events. We skip the leading
    # agent greeting turn here because /api/calls/start already published
    # it (with TTS) when the SSE subscription opened — re-emitting would
    # duplicate the line in the ticker.
    for turn in turns:
        if turn.get("speaker") == "agent" and turn.get("text") == _AGENT_GREETING:
            continue
        await bus.publish(
            {
                "call_id": call_id,
                "stage": "transcript",
                "status": "complete",
                "timing_ms": 0,
                "payload": {
                    "speaker": turn.get("speaker", "agent"),
                    "text": turn.get("text", ""),
                },
            }
        )

    await bus.publish(
        {
            "call_id": call_id,
            "stage": "voice",
            "status": "complete",
            "timing_ms": 0,
            "payload": {"turns": turns, "voice_available": VOICE_AVAILABLE},
        }
    )

    # speak_response=True wires the agent's TTS reply at end of pipeline.
    asyncio.create_task(
        _run_pipeline(bus, turns, caller_phone, speak_response=True)
    )
    return {"ok": True, "voice_available": VOICE_AVAILABLE}
