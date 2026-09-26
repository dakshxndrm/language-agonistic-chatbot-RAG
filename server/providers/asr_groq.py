"""Groq-hosted whisper-large-v3-turbo — the English/code-mixed leg of the
dual-path ASR router (§3, §4). Confirmed live: this endpoint has its own
request-count rate limit, entirely separate from the chat-completions
token-per-minute pool llm_groq.py shares with build_graph.py's extraction.
"""

from __future__ import annotations

import time

import httpx

from server.config import get_settings
from server.providers._pcm import pcm_to_wav_bytes
from server.providers.base import ASRProvider, Transcript

TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "whisper-large-v3-turbo"


class GroqASR(ASRProvider):
    name = "groq-whisper"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().groq_api_key

    async def transcribe(self, pcm: bytes, sample_rate: int,
                         hint_lang: str | None = None) -> Transcript:
        wav = pcm_to_wav_bytes(pcm, sample_rate)
        files = {"file": ("audio.wav", wav, "audio/wav")}
        data = {"model": MODEL}
        if hint_lang:
            data["language"] = hint_lang.split("-")[0]  # whisper wants bare "en", not "en-US"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(TRANSCRIBE_URL, files=files, data=data, headers=headers)
            resp.raise_for_status()
            body = resp.json()
        latency_ms = (time.monotonic() - start) * 1000

        return Transcript(
            text=body.get("text", "").strip(),
            lang=hint_lang.split("-")[0] if hint_lang else "en",
            script="Latn",
            confidence=1.0,   # whisper's verbose_json gives per-segment logprobs, not used here
            is_final=True,
            latency_ms=latency_ms,
        )

    async def health(self) -> bool:
        try:
            silence = b"\x00\x00" * 1600
            t = await self.transcribe(silence, 16000)
            return t is not None
        except Exception:
            return False
