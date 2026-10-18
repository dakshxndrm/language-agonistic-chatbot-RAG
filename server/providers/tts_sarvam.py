"""Sarvam Bulbul TTS (§3, C3 — never Piper, archived Oct 2025).

Sarvam's REST endpoint returns one base64 WAV payload per call, not an
incremental stream, so `synthesize()` yields it as a single PCM chunk. It's
still genuinely cancellable: cancelling the awaiting task tears down the
httpx request like any other, same guarantee as llm_groq.py's stream().
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator

import httpx

from server.config import get_settings
from server.providers._pcm import wav_bytes_to_pcm
from server.providers.base import TTSProvider

TTS_URL = "https://api.sarvam.ai/text-to-speech"
MODEL = "bulbul:v3"
SAMPLE_RATE = 22050  # verified live from a real Bulbul response's WAV header

# Sarvam wants BCP-47-with-region codes; our contract uses bare language
# families (§4's LangContract.lang). "hi-en" (Hinglish) maps to hi-IN — Bulbul
# is advertised as code-mixed-capable for that voice, not a separate locale.
LANG_MAP = {
    "en": "en-IN", "hi": "hi-IN", "hi-en": "hi-IN", "mr": "mr-IN",
    "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
    "gu": "gu-IN", "ml": "ml-IN", "pa": "pa-IN", "or": "or-IN",
}


class SarvamTTS(TTSProvider):
    name = "sarvam-bulbul"
    sample_rate = SAMPLE_RATE

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().sarvam_api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def synthesize(self, text: str, lang: str,
                         voice: str | None = None) -> AsyncIterator[bytes]:
        payload = {
            "inputs": [text],
            "target_language_code": LANG_MAP.get(lang, "en-IN"),
            "model": MODEL,
        }
        if voice:
            payload["speaker"] = voice
        headers = {"api-subscription-key": self._api_key}
        client = self._get_client()
        resp = await client.post(TTS_URL, json=payload, headers=headers)
        resp.raise_for_status()
        for b64_audio in resp.json()["audios"]:
            # ABC contract is raw PCM (base.py) — Bulbul returns a WAV
            # container, so unwrap it rather than leaking the format.
            pcm, _sr, _ch = wav_bytes_to_pcm(base64.b64decode(b64_audio))
            yield pcm

    async def health(self) -> bool:
        try:
            async for chunk in self.synthesize("ok", "en"):
                return len(chunk) > 0
            return False
        except Exception:
            return False
