"""Groq-hosted whisper-large-v3-turbo — the English/code-mixed leg of the
dual-path ASR router (§3, §4). Confirmed live: this endpoint has its own
request-count rate limit, entirely separate from the chat-completions
token-per-minute pool llm_groq.py shares with build_graph.py's extraction.
"""

from __future__ import annotations

import time

import httpx

from server.config import get_settings
from server.lang.detect import majority_script
from server.providers._pcm import pcm_to_wav_bytes
from server.providers.base import ASRProvider, Transcript

TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "whisper-large-v3-turbo"


class GroqASR(ASRProvider):
    name = "groq-whisper"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().groq_api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def transcribe(self, pcm: bytes, sample_rate: int,
                         hint_lang: str | None = None) -> Transcript:
        wav = pcm_to_wav_bytes(pcm, sample_rate)
        files = {"file": ("audio.wav", wav, "audio/wav")}
        data = {"model": MODEL}
        if hint_lang:
            data["language"] = hint_lang.split("-")[0]  # whisper wants bare "en", not "en-US"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        start = time.monotonic()
        client = self._get_client()
        resp = await client.post(TRANSCRIBE_URL, files=files, data=data, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        latency_ms = (time.monotonic() - start) * 1000

        text = body.get("text", "").strip()
        return Transcript(
            text=text,
            lang=hint_lang.split("-")[0] if hint_lang else "en",
            script=majority_script(text) if text else "Latn",
            confidence=1.0,   # whisper's verbose_json gives per-segment logprobs, not used here
            is_final=True,
            latency_ms=latency_ms,
            asr_engine_used=self.name,
        )

    async def health(self) -> bool:
        try:
            silence = b"\x00\x00" * 1600
            t = await self.transcribe(silence, 16000)
            return t is not None
        except Exception:
            return False
