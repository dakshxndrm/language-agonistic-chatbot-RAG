"""faster-whisper, CPU int8 — the offline fallback for ASR_EN_PROVIDER=local /
ASR_INDIC_PROVIDER=local (onprem, C5). §2a: this dev machine has no GPU, so
the pinned whisper-large-v3-turbo is too slow to iterate against here — model
size is configurable specifically so a small/base model can be used for
correctness checks without turning every dev loop into a benchmark.
"""

from __future__ import annotations

import asyncio
import os
import time

from server.providers.base import ASRProvider, Transcript

DEFAULT_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")


class LocalASR(ASRProvider):
    name = "faster-whisper"

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE, device: str = "cpu",
                compute_type: str = "int8"):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def load(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._model_size, device=self._device,
                                       compute_type=self._compute_type)

    async def _ensure_loaded(self) -> None:
        if self._model is None:
            await asyncio.to_thread(self.load)

    def _transcribe_sync(self, audio, language: str | None) -> tuple[str, object]:
        # faster-whisper's transcribe() returns a LAZY generator of segments —
        # decoding happens on iteration, so the join must run in this same
        # thread call, not after it returns to the event loop (C6).
        segments, info = self._model.transcribe(audio, language=language)
        text = " ".join(seg.text.strip() for seg in segments)
        return text, info

    async def transcribe(self, pcm: bytes, sample_rate: int,
                         hint_lang: str | None = None) -> Transcript:
        import numpy as np

        await self._ensure_loaded()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        start = time.monotonic()
        language = hint_lang.split("-")[0] if hint_lang else None
        text, info = await asyncio.to_thread(self._transcribe_sync, audio, language)
        latency_ms = (time.monotonic() - start) * 1000

        return Transcript(
            text=text,
            lang=info.language,
            script="Latn" if info.language == "en" else "Deva",
            confidence=float(info.language_probability),
            is_final=True,
            latency_ms=latency_ms,
            asr_engine_used="faster-whisper",
        )

    async def health(self) -> bool:
        try:
            silence = b"\x00\x00" * 1600
            t = await self.transcribe(silence, 16000)
            return t is not None
        except Exception:
            return False
