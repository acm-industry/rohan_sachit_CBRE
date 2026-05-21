"""Bridges `classify_with_events`'s `emitter` callback to a `CallBus`.

Wraps the per-stage event dict in the public SSE contract:
    {call_id, stage, status, timing_ms, payload}
"""
from __future__ import annotations

from typing import Any, Dict

from backend.events.bus import CallBus


def make_emitter(bus: CallBus):
    """Return an async emitter for `classify_with_events()`.

    The pipeline calls it with `{stage, status, timing_ms, payload}`;
    we tag the `call_id` and push it onto the bus.
    """

    async def _emit(event: Dict[str, Any]) -> None:
        wrapped = {"call_id": bus.call_id, **event}
        await bus.publish(wrapped)

    return _emit
