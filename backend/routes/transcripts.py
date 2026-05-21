"""GET /api/transcripts — list the three canned demo transcripts."""
from __future__ import annotations

from fastapi import APIRouter

from backend.demo.canned import list_demos


router = APIRouter()


@router.get("/api/transcripts")
async def get_transcripts() -> list[dict]:
    return list_demos()
