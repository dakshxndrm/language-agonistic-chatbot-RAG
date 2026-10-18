"""Local Ollama LLM (onprem/hybrid fallback for LLM_PROVIDER=ollama, C5).

UNVERIFIED against a live Ollama instance: this dev machine has no GPU and no
Ollama install (§2a) — confirmed by a live connection attempt to
localhost:11434, not assumed. Implemented against Ollama's documented
/api/chat NDJSON streaming contract, matching llm_groq.py's structure, but it
has not been run end-to-end. Verify on GPU-equipped hardware before relying on
it (same caveat as Milestone 6).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from server.config import get_settings
from server.providers.base import LangContract, LLMProvider

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        s = get_settings()
        self._base_url = base_url or s.ollama_base_url
        self._model = model or DEFAULT_MODEL

    def _system_prompt(self, contract: LangContract, context: list[dict]) -> str:
        lines = [
            f"Reply only in language '{contract.lang}', script '{contract.script}', "
            f"register '{contract.register}'. Never switch script mid-reply.",
        ]
        if context:
            lines.append("Cite only the numbered sources below by their chunk_id. "
                         "If the sources don't answer the question, say you don't know.")
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
        url = f"{self._base_url}/api/chat"
        async with httpx.AsyncClient(timeout=60.0) as client, \
                client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                delta = chunk.get("message", {}).get("content")
                if delta:
                    yield delta
                if chunk.get("done"):
                    break

    async def complete(self, messages: list[dict], *, json_mode: bool = False) -> str:
        payload = {"model": self._model, "messages": messages, "stream": False}
        if json_mode:
            payload["format"] = "json"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
