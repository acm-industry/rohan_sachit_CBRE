"""Per-call asyncio event bus.

One `CallBus` per `call_id`. The pipeline pushes events into it from a
worker task; the SSE endpoint consumes them. The bus also carries the
HITL resume signal (the validator gate blocks on `wait_for_resume()`,
the `/review` endpoint flips it).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


_SENTINEL_DONE: Dict[str, Any] = {"__done__": True}


@dataclass
class CallBus:
    call_id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    resume_event: asyncio.Event = field(default_factory=asyncio.Event)
    resume_payload: Optional[Dict[str, Any]] = None
    done: bool = False

    async def publish(self, event: Dict[str, Any]) -> None:
        await self.queue.put(event)

    async def close(self) -> None:
        if self.done:
            return
        self.done = True
        await self.queue.put(_SENTINEL_DONE)

    def submit_review(self, decision: str, override: Optional[Dict[str, Any]] = None) -> None:
        self.resume_payload = {"decision": decision, "override": override or {}}
        self.resume_event.set()

    async def wait_for_resume(self) -> Dict[str, Any]:
        await self.resume_event.wait()
        return self.resume_payload or {"decision": "approve", "override": {}}


class CallRegistry:
    """In-process map of {call_id: CallBus}.

    Localhost-only demo — no Redis, no persistence. Lives on the FastAPI
    app state so tests can swap it out.
    """

    def __init__(self) -> None:
        self._buses: Dict[str, CallBus] = {}
        self._lock = asyncio.Lock()

    async def create(self, call_id: str) -> CallBus:
        async with self._lock:
            bus = CallBus(call_id=call_id)
            self._buses[call_id] = bus
            return bus

    def get(self, call_id: str) -> Optional[CallBus]:
        return self._buses.get(call_id)

    async def drop(self, call_id: str) -> None:
        async with self._lock:
            self._buses.pop(call_id, None)


def is_sentinel(event: Dict[str, Any]) -> bool:
    return event is _SENTINEL_DONE or event.get("__done__") is True
