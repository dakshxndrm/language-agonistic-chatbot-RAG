"""Language + script detection.

fastText lid.176 gives a coarse language family guess (e.g. "hi", "en", "mr").
It is NOT trusted to tell "hi" apart from "hi-en" (Hinglish): romanized Hindi
gets misclassified by lid.176 constantly, because it was trained mostly on
native-script text. Instead, the same fastText guess flips between "hi" and
"hi-en" purely on which Unicode block the text is written in — script is the
ONLY signal for that distinction, by design.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_PATH = Path("data/models/lid.176.ftz")
LOW_CONFIDENCE = 0.4

# Common Hindi/Hinglish function words that survive romanization intact —
# these are the highest-frequency tell for "this Latin-script text is Hindi"
# when fastText's own confidence collapses (see detect() below).
HINGLISH_MARKERS = frozenset({
    "hai", "nahi", "nahin", "kya", "kaise", "kyun", "kyu", "tha", "thi", "the",
    "ka", "ki", "ke", "mein", "hoon", "hun", "aur", "bhi", "kar", "karo",
    "kiya", "gaya", "gayi", "koi", "kuch", "yeh", "woh", "aap", "tum", "hum",
    "mera", "tera", "uska", "iska", "se", "ko", "par", "ne",
})
_WORD_RE = re.compile(r"[a-zA-Z]+")


def _looks_hinglish(text: str) -> bool:
    words = _WORD_RE.findall(text.lower())
    return any(w in HINGLISH_MARKERS for w in words)

# Unicode block ranges for the scripts this project actually sees. "Common"
# codepoints (digits, punctuation, whitespace) never count toward any script,
# so a script histogram isn't diluted by numbers or "." on either side.
SCRIPT_RANGES: list[tuple[str, int, int]] = [
    ("Deva", 0x0900, 0x097F),   # Devanagari: Hindi, Marathi, Sanskrit
    ("Beng", 0x0980, 0x09FF),   # Bengali
    ("Guru", 0x0A00, 0x0A7F),   # Gurmukhi: Punjabi
    ("Gujr", 0x0A80, 0x0AFF),   # Gujarati
    ("Orya", 0x0B00, 0x0B7F),   # Odia
    ("Taml", 0x0B80, 0x0BFF),   # Tamil
    ("Telu", 0x0C00, 0x0C7F),   # Telugu
    ("Knda", 0x0C80, 0x0CFF),   # Kannada
    ("Mlym", 0x0D00, 0x0D7F),   # Malayalam
    ("Latn", 0x0041, 0x005A),   # Latin uppercase
    ("Latn", 0x0061, 0x007A),   # Latin lowercase
]


@dataclass(frozen=True)
class Detection:
    lang: str       # BCP-47-ish: "en", "hi", "hi-en", "mr", ...
    script: str     # ISO 15924: "Latn", "Deva", "Beng", ...
    confidence: float


def script_histogram(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if unicodedata.category(ch)[0] in ("N", "P", "Z", "C"):  # digits/punct/space/control
            continue
        for script, lo, hi in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[script] = counts.get(script, 0) + 1
                break
    return counts


def majority_script(text: str) -> str:
    hist = script_histogram(text)
    if not hist:
        return "Latn"  # no lettered chars at all — default rather than guess
    return max(hist, key=hist.get)


class LanguageDetector:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        self._model_path = Path(model_path)
        self._model = None

    def load(self) -> None:
        if self._model is None:
            import fasttext

            self._model = fasttext.load_model(str(self._model_path))

    async def _ensure_loaded(self) -> None:
        if self._model is None:
            await asyncio.to_thread(self.load)

    async def detect(self, text: str) -> Detection:
        text = text.strip()
        if not text:
            return Detection(lang="en", script="Latn", confidence=0.0)

        await self._ensure_loaded()
        script = majority_script(text)

        # fastText chokes on embedded newlines; a query/utterance is one line anyway.
        # Calling the underlying pybind method directly, not FastText.predict():
        # that wrapper does `np.array(probs, copy=False)`, which raises under
        # numpy>=2.0 (this is a real bug in fasttext-wheel 0.9.2, not a guess).
        line = text.replace("\n", " ") + "\n"
        predictions = self._model.f.predict(line, 1, 0.0, "strict")
        if not predictions:
            return Detection(lang="en", script=script, confidence=0.0)
        prob, label = predictions[0]
        base_lang = label.removeprefix("__label__")
        confidence = float(prob)

        if script != "Latn":
            return Detection(lang=base_lang, script=script, confidence=confidence)

        if base_lang == "en" and confidence >= LOW_CONFIDENCE:
            return Detection(lang="en", script="Latn", confidence=confidence)

        # lid.176 is unreliable on romanized Hindi — verified live: "Ashoka ne
        # kaunsa yudh lada tha?" gets guessed as "la" (Latin, the dead language)
        # at confidence 0.09. A near-zero-confidence guess is the tell, so fall
        # back to a marker-word check for that case specifically.
        # ponytail: word-list heuristic, ceiling is anything outside this list;
        # upgrade to a dedicated Hinglish classifier if this proves insufficient.
        if confidence < LOW_CONFIDENCE and _looks_hinglish(text):
            return Detection(lang="hi-en", script="Latn", confidence=confidence)

        lang = "en" if base_lang == "en" else f"{base_lang}-en"
        return Detection(lang=lang, script="Latn", confidence=confidence)

        if script != "Latn":
            return Detection(lang=base_lang, script=script, confidence=confidence)

        lang = "en" if base_lang == "en" else f"{base_lang}-en"
        return Detection(lang=lang, script="Latn", confidence=confidence)


_detector: LanguageDetector | None = None


def get_detector() -> LanguageDetector:
    global _detector
    if _detector is None:
        _detector = LanguageDetector()
    return _detector
