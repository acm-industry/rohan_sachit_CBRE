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
) -> None:
    """Run the agent pipeline as a background task and close the bus."""
    try:
        await classify_with_events(
            turns,
            caller_phone,
            emitter=make_emitter(bus),
            pause_at_validator=True,
            wait_for_resume=bus.wait_for_resume,
        )
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
        # the audio endpoint can pick it up.
        bus = await reg.create(call_id)
        bus.resume_payload = None  # unused for voice; kept for symmetry
        request.app.state.voice_phone[call_id] = body.caller_phone
        return {"call_id": call_id}

    raise HTTPException(400, f"unknown mode: {body.mode}")


async def _sse_stream(bus: CallBus):
    """Convert bus events to SSE-formatted lines."""
    # Initial comment line keeps the connection open while clients attach.
    yield ": connected\n\n"
    while True:
        event = await bus.queue.get()
        if is_sentinel(event):
            yield "event: done\ndata: {}\n\n"
            return
        yield f"data: {json.dumps(event, default=str)}\n\n"


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

    # Emit a transcript event up front so the UI can render the turns.
    await bus.publish(
        {
            "call_id": call_id,
            "stage": "voice",
            "status": "complete",
            "timing_ms": 0,
            "payload": {"turns": turns, "voice_available": VOICE_AVAILABLE},
        }
    )

    asyncio.create_task(_run_pipeline(bus, turns, caller_phone))
    return {"ok": True, "voice_available": VOICE_AVAILABLE}
