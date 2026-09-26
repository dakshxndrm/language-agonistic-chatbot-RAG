"""BHASHA FastAPI app."""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from livekit import api
from pydantic import BaseModel

from server.config import get_settings

app = FastAPI(title="BHASHA", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "profile": s.bhasha_profile,
        "providers": s.resolved,
    }


class TokenResponse(BaseModel):
    token: str
    url: str
    room: str


@app.post("/livekit/token")
async def livekit_token(room: str | None = None) -> TokenResponse:
    """Mints a join token for the web client (web/) to connect with. One
    room per conversation — the client generates its own room name if it
    doesn't pass one, so concurrent sessions never collide."""
    s = get_settings()
    room_name = room or f"bhasha-{uuid.uuid4().hex[:12]}"
    identity = f"user-{uuid.uuid4().hex[:8]}"

    token = (
        api.AccessToken(s.livekit_api_key, s.livekit_api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    return TokenResponse(token=token, url=s.livekit_url, room=room_name)
