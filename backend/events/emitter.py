"""Bridges `classify_with_events`'s `emitter` callback to a `CallBus`.

Wraps the per-stage event dict in the public SSE contract:
    {call_id, stage, status, timing_ms, payload}
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from backend.events.bus import CallBus


def make_emitter(bus: CallBus):
    """Return an async emitter for `classify_with_events()`.

    The pipeline calls it with `{stage, status, timing_ms, payload}`;
    we tag the `call_id` and push it onto the bus. We then `await
    asyncio.sleep(0)` to yield the event loop, which lets the SSE
    consumer task wake up, drain the new event off the queue, and yield
    it as an HTTP chunk to the browser — *before* the next stage of the
    pipeline begins its work. Without this, fast stages (or the gap
    between a stage's "started" event and its LLM await) can pile their
    events onto the queue inside one event-loop tick, and the browser
    only sees them as a burst at the end.
    """

    async def _emit(event: Dict[str, Any]) -> None:
        wrapped = {"call_id": bus.call_id, **event}
        await bus.publish(wrapped)
        await asyncio.sleep(0)

    return _emit
