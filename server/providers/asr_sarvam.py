"""Sarvam Saarika ASR — the Indic-speech leg of the dual-path router (§3, §4).
"""

from __future__ import annotations

import time

import httpx

from server.config import get_settings
from server.providers._pcm import pcm_to_wav_bytes
from server.providers.base import ASRProvider, Transcript

ASR_URL = "https://api.sarvam.ai/speech-to-text"
MODEL = "saaras:v3"  # saarika:v2 was deprecated server-side; verified live

# Saarika transcribes into each language's native script — this is what that
# language is written in, not a detection result (lang/detect.py handles that
# for arbitrary text downstream).
NATIVE_SCRIPT = {
    "hi": "Deva", "mr": "Deva", "ne": "Deva", "sa": "Deva",
    "bn": "Beng", "as": "Beng",
    "ta": "Taml", "te": "Telu", "kn": "Knda", "ml": "Mlym",
    "gu": "Gujr", "pa": "Guru", "or": "Orya", "en": "Latn",
}


class SarvamASR(ASRProvider):
    name = "sarvam-saarika"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().sarvam_api_key

    async def transcribe(self, pcm: bytes, sample_rate: int,
                         hint_lang: str | None = None) -> Transcript:
        wav = pcm_to_wav_bytes(pcm, sample_rate)
        files = {"file": ("audio.wav", wav, "audio/wav")}
        data = {"model": MODEL}
        if hint_lang:
            data["language_code"] = hint_lang
        headers = {"api-subscription-key": self._api_key}

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(ASR_URL, files=files, data=data, headers=headers)
            resp.raise_for_status()
            body = resp.json()
        latency_ms = (time.monotonic() - start) * 1000

        lang = (body.get("language_code") or hint_lang or "hi").split("-")[0]
        return Transcript(
            text=body.get("transcript", ""),
            lang=lang,
            script=NATIVE_SCRIPT.get(lang, "Deva"),
            confidence=1.0,   # Saarika's API doesn't return a per-utterance confidence
            is_final=True,
            latency_ms=latency_ms,
        )

    async def health(self) -> bool:
        try:
            silence = b"\x00\x00" * 1600  # 100ms of silence at 16kHz mono
            t = await self.transcribe(silence, 16000)
            return t is not None
        except Exception:
            return False
