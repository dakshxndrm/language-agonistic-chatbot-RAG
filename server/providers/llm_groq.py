"""Groq-hosted LLM. Groq's chat endpoint is plain OpenAI-compatible REST, and
httpx is already pinned (§3), so a second SDK for one HTTP call buys nothing.

`stream()` is the conversational path (contract + RAG context -> cited answer,
M2) and must be genuinely cancellable: it's a plain async generator over SSE
lines, so `asyncio.CancelledError` raised while awaiting the next chunk closes
the httpx stream via the `async with` block, same as any other cancelled task.

`complete()` is a plain non-streaming call for one-shot extraction (M0.5's
build_graph.py) — it does not go through the contract/context machinery
because entity extraction has nothing to do with reply language or citations.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

import httpx

from server.config import get_settings
from server.providers.base import LangContract, LLMProvider

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_ATTEMPTS = 6
MAX_WAIT_SECONDS = 60  # a multi-minute hang mid-conversation is worse than a clear error
_DURATION_RE = re.compile(r"(?:(\d+)m)?([\d.]+)s")
_TRY_AGAIN_RE = re.compile(r"try again in ([\d.]+m?[\d.]*s)", re.IGNORECASE)


def _parse_duration(value: str) -> float:
    m = _DURATION_RE.match(value.strip())
    if not m:
        return 5.0
    minutes, seconds = m.groups()
    return (int(minutes) * 60 if minutes else 0) + float(seconds)


def _wait_seconds(resp: httpx.Response, body_text: str) -> float:
    """The x-ratelimit-reset-tokens header only reflects the per-minute
    window. A 429 can also come from a tokens-PER-DAY cap (confirmed live:
    "Rate limit reached ... on tokens per day (TPD) ... try again in
    7m30.144s"), where that header is irrelevant — the error message's own
    "try again in Xs" is the authoritative wait, so prefer it when present."""
    m = _TRY_AGAIN_RE.search(body_text)
    if m:
        return _parse_duration(m.group(1))
    return _parse_duration(resp.headers.get("x-ratelimit-reset-tokens", "5s"))


class GroqLLM(LLMProvider):
    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        s = get_settings()
        self._api_key = api_key or s.groq_api_key
        self._model = model or s.llm_model

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _system_prompt(self, contract: LangContract, context: list[dict]) -> str:
        lines = [
            f"Reply only in language '{contract.lang}', script '{contract.script}', "
            f"register '{contract.register}'. Never switch script mid-reply.",
        ]
        if context:
            lines.append("Cite only the numbered sources below by their chunk_id. "
                         "If the sources don't answer the question, say you don't know (C8).")
            for c in context:
                lines.append(f"[{c['chunk_id']}] {c['text']}")
        return "\n".join(lines)

    async def stream(self, messages: list[dict], contract: LangContract,
                     context: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": self._system_prompt(contract, context)},
                        *messages],
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                async with client.stream("POST", GROQ_URL, json=payload,
                                         headers=self._headers()) as resp:
                    if resp.status_code == 429:
                        body_text = (await resp.aread()).decode("utf-8", "replace")
                        wait = _wait_seconds(resp, body_text)
                        if attempt == MAX_ATTEMPTS or wait > MAX_WAIT_SECONDS:
                            resp.raise_for_status()
                        await asyncio.sleep(wait + 0.5)
                        continue  # nothing yielded yet — safe to retry the whole request
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: "):]
                        if data == "[DONE]":
                            break
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta
                    return

    async def complete(self, messages: list[dict], *, json_mode: bool = False) -> str:
        payload = {"model": self._model, "messages": messages, "stream": False}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                resp = await client.post(GROQ_URL, json=payload, headers=self._headers())
                if resp.status_code != 429:
                    if resp.status_code >= 400:
                        raise httpx.HTTPStatusError(
                            f"{resp.status_code} calling Groq: {resp.text[:500]}",
                            request=resp.request, response=resp,
                        )
                    return resp.json()["choices"][0]["message"]["content"]
                wait = _wait_seconds(resp, resp.text)
                if attempt == MAX_ATTEMPTS or wait > MAX_WAIT_SECONDS:
                    resp.raise_for_status()
                await asyncio.sleep(wait + 0.5)  # small margin past the window edge

    async def health(self) -> bool:
        try:
            reply = await self.complete([{"role": "user", "content": "reply with just: ok"}])
            return bool(reply)
        except Exception:
            return False
