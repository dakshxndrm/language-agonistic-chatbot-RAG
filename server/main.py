"""BHASHA FastAPI app."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from livekit import api
from pydantic import BaseModel

from server.config import get_settings

logger = logging.getLogger("bhasha")
_active_agent_tasks: dict[str, asyncio.Task] = {}


async def _run_agent_pipeline(room_name: str, token: str) -> None:
    try:
        from server.pipeline import run_pipeline

        await run_pipeline(room_name, token)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("Pipeline task in room %s failed: %s", room_name, e)
    finally:
        _active_agent_tasks.pop(room_name, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Enforces immediate validation of profile and fail-loud air-gap guarantees (C1, C5)
    get_settings()
    yield
    # Clean up active pipeline tasks on shutdown
    for task in list(_active_agent_tasks.values()):
        if not task.done():
            task.cancel()


app = FastAPI(title="BHASHA", version="0.1.0", lifespan=lifespan)


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
async def livekit_token(request: Request, room: str | None = None) -> TokenResponse:
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

    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if forwarded_proto == "https" and forwarded_host:
        lk_url = f"wss://{forwarded_host}"
    else:
        lk_url = s.livekit_url

    # Automatically dispatch the Pipecat voice agent into the room
    if not os.getenv("PYTEST_CURRENT_TEST"):
        if room_name in _active_agent_tasks:
            old_task = _active_agent_tasks[room_name]
            if not old_task.done():
                old_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(old_task), timeout=1.5)
                except (TimeoutError, asyncio.CancelledError, Exception):
                    pass
        agent_identity = f"agent-{uuid.uuid4().hex[:8]}"
        agent_token = (
            api.AccessToken(s.livekit_api_key, s.livekit_api_secret)
            .with_identity(agent_identity)
            .with_name("BHASHA Agent")
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )
        _active_agent_tasks[room_name] = asyncio.create_task(
            _run_agent_pipeline(room_name, agent_token)
        )

    return TokenResponse(token=token, url=lk_url, room=room_name)


class QueryRequest(BaseModel):
    question: str
    question_type: str = "factoid"
    lang: str | None = None


@app.post("/query")
async def text_query(req: QueryRequest) -> dict:
    """HTTP text-query fallback (Finding 16) for environments without WebRTC mic access."""
    from server.lang.contract import build_contract, generate_with_contract
    from server.pipeline import detect_question_type
    from server.providers.base import LangContract
    from server.providers.factory import get_llm
    from server.rag.orchestrate import orchestrate
    from server.rag.prompt import build_context, build_messages
    from server.visual_stage import extract_visual_payload

    if req.lang:
        script = "Deva" if req.lang == "hi" else "Latn"
        contract = LangContract(lang=req.lang, script=script)
    else:
        contract = await build_contract(req.question)

    q_type = req.question_type if req.question_type != "factoid" else detect_question_type(req.question)
    result = await orchestrate(req.question, question_type=q_type)
    messages = build_messages(req.question)
    context = build_context(result)
    llm = get_llm()
    answer, fidelity_passed, repairs = await generate_with_contract(
        llm, messages, contract, context
    )
    visual = extract_visual_payload(req.question, answer, result.hits, result.graph_paths)
    return {
        "question": req.question,
        "lang": contract.lang,
        "script": contract.script,
        "answer": answer,
        "citations": [c.article for c in result.hits if c.article],
        "fidelity_passed": fidelity_passed,
        "repairs": repairs,
        "visual_stage": visual.to_dict(),
    }


@app.post("/query/stream")
async def stream_query(req: QueryRequest) -> StreamingResponse:
    """Server-Sent Events streaming query endpoint for real-time text chat (sub-600ms TTFT)."""
    from server.lang.contract import build_contract
    from server.pipeline import detect_question_type
    from server.providers.base import LangContract
    from server.providers.factory import get_llm
    from server.rag.orchestrate import orchestrate
    from server.rag.prompt import build_context, build_messages
    from server.visual_stage import extract_visual_payload

    if req.lang:
        script = "Deva" if req.lang == "hi" else "Latn"
        contract = LangContract(lang=req.lang, script=script)
    else:
        contract = await build_contract(req.question)

    q_type = req.question_type if req.question_type != "factoid" else detect_question_type(req.question)
    result = await orchestrate(req.question, question_type=q_type)
    messages = build_messages(req.question)
    context = build_context(result)
    llm = get_llm()

    async def event_generator():
        # 1. Start event with detected contract metadata
        yield f"data: {json.dumps({'event': 'start', 'lang': contract.lang, 'script': contract.script})}\n\n"

        # 2. Token-by-token stream
        tokens: list[str] = []
        try:
            async for tok in llm.stream(messages, contract, context):
                tokens.append(tok)
                yield f"data: {json.dumps({'event': 'token', 'token': tok})}\n\n"
        except Exception as e:
            logger.error("Error streaming tokens: %s", e)
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
            return

        full_answer = "".join(tokens)
        visual = extract_visual_payload(req.question, full_answer, result.hits, result.graph_paths)

        # 3. Final completion event with citations and visual stage
        yield f"data: {json.dumps({'event': 'done', 'answer': full_answer, 'citations': [c.article for c in result.hits if c.article], 'visual_stage': visual.to_dict()})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Mount built React frontend if dist exists
_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")
