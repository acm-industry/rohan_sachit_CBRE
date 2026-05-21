"""End-to-end-ish tests for the FastAPI backend routes.

We patch `agent.classify.classify_with_events` (already imported into
`backend.routes.calls`) so we never touch OpenAI or chromadb. The mock
implementation drives the emitter manually, exercising both the SSE
stream and the HITL review handshake.

Uses the project's existing `asyncio.run()` test style (no pytest-asyncio).
"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import httpx
from httpx import ASGITransport

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.main import create_app  # noqa: E402


def _make_fake_pipeline(events_to_emit: List[Dict[str, Any]],
                       pauses: bool = False):
    async def fake(turns, caller_phone, emitter=None, *, pause_at_validator=False,
                   wait_for_resume=None):
        for ev in events_to_emit:
            await emitter(ev)
            if pauses and ev.get("status") == "gate_open":
                decision = await wait_for_resume()
                await emitter({
                    "stage": "validate",
                    "status": "gate_resumed",
                    "timing_ms": 0,
                    "payload": decision,
                })
        return {"category": "PLUMBING", "subcategory": "drainage_backup",
                "risk_level": "MEDIUM", "needs_human_review": False,
                "needs_clarification": False, "building_name": None,
                "address": None, "floor": None,
                "dispatched_vendor_id": "v_002",
                "dispatched_emergency_services": False,
                "call_summary": "ok", "trainer_log": {}}
    return fake


def _parse_sse_blocks(buffer: str) -> tuple[list[dict], str, bool]:
    """Parse complete SSE blocks out of `buffer`. Returns (events, remaining, done)."""
    events: list[dict] = []
    done = False
    while "\n\n" in buffer:
        block, _, buffer = buffer.partition("\n\n")
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        if block.startswith("event: done"):
            done = True
            continue
        data_line = next(
            (ln for ln in block.splitlines() if ln.startswith("data:")),
            None,
        )
        if data_line:
            try:
                events.append(json.loads(data_line[len("data:"):].strip()))
            except json.JSONDecodeError:
                events.append({"raw": data_line})
    return events, buffer, done


async def _read_sse_until_done(resp: httpx.Response, *,
                               on_event=None, max_events: int = 200) -> list[dict]:
    collected: list[dict] = []
    buffer = ""
    async for chunk in resp.aiter_text():
        buffer += chunk
        events, buffer, done = _parse_sse_blocks(buffer)
        for ev in events:
            collected.append(ev)
            if on_event is not None:
                await on_event(ev)
            if len(collected) >= max_events:
                return collected
        if done:
            break
    return collected


class TestBackendRoutes(unittest.TestCase):

    # ─── /api/transcripts ────────────────────────────────────────────

    def test_list_transcripts(self):
        async def go():
            app = create_app()
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                async with httpx.AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as c2:
                    r = await c2.get("/api/transcripts")
            return r

        r = asyncio.run(go())
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body), 3)
        ids = {row["transcript_id"] for row in body}
        self.assertEqual(ids, {"EVAL-0976", "EVAL-0788", "EVAL-0076"})

    def test_health(self):
        async def go():
            app = create_app()
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                return await ac.get("/api/health")
        r = asyncio.run(go())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ok": True})

    # ─── /api/calls/start ────────────────────────────────────────────

    def test_start_rejects_bad_mode(self):
        async def go():
            app = create_app()
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                return await ac.post("/api/calls/start", json={"mode": "lol"})
        r = asyncio.run(go())
        self.assertEqual(r.status_code, 400)

    def test_start_rejects_unknown_transcript(self):
        async def go():
            app = create_app()
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                return await ac.post(
                    "/api/calls/start",
                    json={"mode": "canned", "transcript_id": "EVAL-9999"},
                )
        r = asyncio.run(go())
        self.assertEqual(r.status_code, 400)

    def test_review_on_unknown_call_404(self):
        async def go():
            app = create_app()
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                return await ac.post(
                    "/api/calls/nope/review", json={"decision": "approve"}
                )
        r = asyncio.run(go())
        self.assertEqual(r.status_code, 404)

    # ─── canned call: streams events ─────────────────────────────────

    def test_canned_call_streams_events(self):
        fake_events = [
            {"stage": "extract", "status": "started", "timing_ms": 0, "payload": {}},
            {"stage": "extract", "status": "complete", "timing_ms": 1.2,
             "payload": {"problem_summary": "sink"}},
            {"stage": "classify", "status": "complete", "timing_ms": 5,
             "payload": {"category": "PLUMBING"}},
        ]

        async def go():
            app = create_app()
            with patch(
                "backend.routes.calls.classify_with_events",
                _make_fake_pipeline(fake_events),
            ):
                async with httpx.AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as ac:
                    start = await ac.post(
                        "/api/calls/start",
                        json={"mode": "canned", "transcript_id": "EVAL-0976"},
                    )
                    assert start.status_code == 200
                    call_id = start.json()["call_id"]

                    async with ac.stream(
                        "GET", f"/api/calls/{call_id}/events", timeout=10
                    ) as resp:
                        assert resp.status_code == 200
                        events = await _read_sse_until_done(resp)
                    return call_id, events

        call_id, events = asyncio.run(go())
        stages = [(e.get("stage"), e.get("status")) for e in events]
        self.assertIn(("extract", "started"), stages)
        self.assertIn(("classify", "complete"), stages)
        # Every event tagged with the right call_id
        self.assertTrue(all(e.get("call_id") == call_id for e in events))

    # ─── HITL gate: /review signals the bus ──────────────────────────
    #
    # The full pause→review→resume contract is covered end-to-end in
    # `tests/test_classify_with_events.py` against the real pipeline.
    # Here we just verify the route plumbing: a POST to /review with a
    # known call_id flips the bus's resume_event with the right payload.

    def test_review_endpoint_signals_bus(self):
        async def go():
            app = create_app()
            reg = app.state.registry
            bus = await reg.create("test-call-1")
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                r = await ac.post(
                    "/api/calls/test-call-1/review",
                    json={"decision": "override",
                          "override": {"risk_level": "LOW"}},
                )
            return r, bus

        r, bus = asyncio.run(go())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"accepted": True})
        self.assertTrue(bus.resume_event.is_set())
        self.assertEqual(bus.resume_payload["decision"], "override")
        self.assertEqual(bus.resume_payload["override"], {"risk_level": "LOW"})

    def test_review_rejects_bad_decision(self):
        async def go():
            app = create_app()
            await app.state.registry.create("test-call-2")
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                return await ac.post(
                    "/api/calls/test-call-2/review",
                    json={"decision": "maybe"},
                )

        r = asyncio.run(go())
        self.assertEqual(r.status_code, 400)

    # ─── voice mode (stubbed) ────────────────────────────────────────

    def test_voice_audio_uses_stub_transcript(self):
        fake_events = [
            {"stage": "extract", "status": "complete", "timing_ms": 1, "payload": {}},
        ]

        async def go():
            app = create_app()
            with patch(
                "backend.routes.calls.classify_with_events",
                _make_fake_pipeline(fake_events),
            ):
                async with httpx.AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as ac:
                    start = await ac.post(
                        "/api/calls/start",
                        json={"mode": "voice", "caller_phone": "+15551112222"},
                    )
                    assert start.status_code == 200
                    call_id = start.json()["call_id"]

                    r = await ac.post(
                        f"/api/calls/{call_id}/audio",
                        content=b"\x00\x01\x02\x03",
                    )
                    return r

        r = asyncio.run(go())
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])


if __name__ == "__main__":
    unittest.main()
