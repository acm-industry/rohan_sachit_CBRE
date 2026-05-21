"""FastAPI app entrypoint for the live demo backend.

Run with:
    uvicorn backend.main:app --reload --port 8000
or:
    make backend-only

Localhost-only — CORS opens to http://localhost:3000 (Next.js dev) only.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.events.bus import CallRegistry
from backend.routes import calls, transcripts


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def create_app() -> FastAPI:
    app = FastAPI(title="CBRE HITL demo backend")

    # Initialise per-process state eagerly. We don't use lifespan because
    # ASGITransport (used in tests) doesn't run it, and the registry has
    # nothing to tear down — it's a plain in-memory map.
    app.state.registry = CallRegistry()
    app.state.voice_phone = {}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(transcripts.router)
    app.include_router(calls.router)

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    return app


app = create_app()
