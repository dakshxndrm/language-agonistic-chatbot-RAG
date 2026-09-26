"""Dual-path ASR router (§4's non-generic feature #1): Indic speech and
English/code-mixed speech go to different engines, chosen once per utterance
from a hint (VAD/session state provides this — e.g. the language the user
selected, or the previous turn's detected language), not from decoding audio
twice.
"""

from __future__ import annotations

import logging

from server.providers.base import ASRProvider, Transcript
from server.providers.factory import get_asr_en, get_asr_indic

logger = logging.getLogger(__name__)

# Bare language families that route to the Indic-tuned engine. "en" and
# code-mixed "<lang>-en" both go to the English/code-mixed engine — that
# engine is the one built to handle code-switching, not the Indic one.
INDIC_LANGS = {"hi", "mr", "bn", "ta", "te", "kn", "ml", "gu", "pa", "or", "as", "ne", "sa"}


def choose_route(hint_lang: str | None) -> str:
    """Returns "indic" or "en" — pure function so routing decisions are
    unit-testable without any audio or network I/O."""
    if not hint_lang:
        return "en"
    base = hint_lang.split("-")[0]
    return "indic" if base in INDIC_LANGS and not hint_lang.endswith("-en") else "en"


async def route_transcribe(pcm: bytes, sample_rate: int,
                           hint_lang: str | None = None) -> Transcript:
    route = choose_route(hint_lang)
    provider: ASRProvider = get_asr_indic() if route == "indic" else get_asr_en()
    logger.info("asr_router: hint_lang=%r -> route=%s provider=%s",
               hint_lang, route, provider.name)
    return await provider.transcribe(pcm, sample_rate, hint_lang=hint_lang)
