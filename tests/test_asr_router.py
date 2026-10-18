"""Dual-path routing (§4 feature #1) is the one thing that must never guess —
these pin the routing table down as a pure function, no audio/network needed.
"""

from __future__ import annotations

import asyncio

from server.providers.asr_router import choose_route, route_transcribe
from server.providers.base import ASRProvider, Transcript


def test_devanagari_hindi_routes_indic():
    assert choose_route("hi") == "indic"


def test_english_routes_en():
    assert choose_route("en") == "en"


def test_hinglish_code_mixed_routes_en_not_indic():
    # "hi-en" is code-mixed and Latin-script — the English/code-mixed engine
    # is the one built to handle it, not the Indic-tuned one.
    assert choose_route("hi-en") == "en"


def test_no_hint_defaults_to_en():
    assert choose_route(None) == "en"


def test_other_indic_languages_route_indic():
    for lang in ("mr", "bn", "ta", "te", "kn", "ml", "gu", "pa"):
        assert choose_route(lang) == "indic"


class MockASR(ASRProvider):
    def __init__(self, name: str, should_fail: bool = False, delay: float = 0.0):
        self.name = name
        self.should_fail = should_fail
        self.delay = delay

    async def transcribe(
        self, pcm: bytes, sample_rate: int, hint_lang: str | None = None
    ) -> Transcript:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            raise RuntimeError(f"{self.name} failed")
        return Transcript(
            text="mock transcript",
            lang=hint_lang or "en",
            script="Latn",
            confidence=0.99,
            is_final=True,
            latency_ms=10.0,
            asr_engine_used=self.name,
        )

    async def health(self) -> bool:
        return True


async def test_route_transcribe_indic_fallback_on_error(monkeypatch):
    sarvam_mock = MockASR("sarvam-mock", should_fail=True)
    whisper_mock = MockASR("whisper-mock", should_fail=False)

    monkeypatch.setattr("server.providers.asr_router.get_asr_indic", lambda: sarvam_mock)
    monkeypatch.setattr("server.providers.asr_router.get_asr_en", lambda: whisper_mock)

    res = await route_transcribe(b"\x00" * 320, 16000, hint_lang="hi")
    assert res.text == "mock transcript"
    assert res.asr_engine_used == "whisper-mock"


async def test_route_transcribe_indic_fallback_on_timeout(monkeypatch):
    sarvam_mock = MockASR("sarvam-mock", should_fail=False, delay=0.1)
    whisper_mock = MockASR("whisper-mock", should_fail=False, delay=0.0)

    monkeypatch.setattr("server.providers.asr_router.SARVAM_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("server.providers.asr_router.get_asr_indic", lambda: sarvam_mock)
    monkeypatch.setattr("server.providers.asr_router.get_asr_en", lambda: whisper_mock)

    res = await route_transcribe(b"\x00" * 320, 16000, hint_lang="hi")
    assert res.text == "mock transcript"
    assert res.asr_engine_used == "whisper-mock"
