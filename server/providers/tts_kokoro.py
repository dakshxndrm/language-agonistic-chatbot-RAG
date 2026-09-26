"""Kokoro-82M local TTS (§3, C3 — Piper is banned, archived Oct 2025).
Onprem/hybrid fallback for TTS_PROVIDER=kokoro. Verified live on this
CPU-only dev machine: real audio generated for English.

Kokoro's own G2P coverage among our target languages is only English + Hindi
(Hindi via an espeak-ng fallback) — that's a ceiling of the model itself, not
a gap in this code, so other Indic languages degrade to English phonemization
with a logged warning rather than silently mis-pronouncing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from server.providers.base import TTSProvider

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
LANG_CODE = {"en": "a", "hi": "h", "hi-en": "h"}
DEFAULT_VOICE = {"a": "af_heart", "h": "hf_alpha"}


class KokoroTTS(TTSProvider):
    name = "kokoro"
    sample_rate = SAMPLE_RATE

    def __init__(self):
        self._pipelines: dict[str, object] = {}

    def _get_pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            from kokoro import KPipeline

            self._pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return self._pipelines[lang_code]

    def _synthesize_sync(self, lang_code: str, text: str, voice: str) -> list[bytes]:
        # torch inference on CPU is blocking — must run inside asyncio.to_thread
        # (C6), same reasoning as embed_local.py's model calls.
        import numpy as np

        pipeline = self._get_pipeline(lang_code)
        chunks = []
        for result in pipeline(text, voice=voice):
            if result.audio is None:
                continue
            pcm16 = (result.audio.numpy().clip(-1.0, 1.0) * 32767).astype(np.int16)
            chunks.append(pcm16.tobytes())
        return chunks

    async def synthesize(self, text: str, lang: str,
                         voice: str | None = None) -> AsyncIterator[bytes]:
        lang_code = LANG_CODE.get(lang)
        if lang_code is None:
            logger.warning("kokoro has no G2P for lang=%r — falling back to English", lang)
            lang_code = "a"
        chosen_voice = voice or DEFAULT_VOICE[lang_code]
        chunks = await asyncio.to_thread(self._synthesize_sync, lang_code, text, chosen_voice)
        for chunk in chunks:
            yield chunk

    async def health(self) -> bool:
        try:
            async for chunk in self.synthesize("ok", "en"):
                return len(chunk) > 0
            return False
        except Exception:
            return False
