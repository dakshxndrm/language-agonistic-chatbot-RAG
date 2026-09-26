"""hi vs hi-en must be decided by script alone (CLAUDE.md M2 spec) — these
tests pin that down at both the pure-function level (no model needed) and
against the real fastText model, since that's exactly where lid.176's
unreliability on romanized Hindi was caught live (see detect.py's docstring).
"""

from server.lang.detect import Detection, _looks_hinglish, get_detector, majority_script


def test_majority_script_devanagari():
    assert majority_script("अशोक ने कौन सा युद्ध लड़ा था?") == "Deva"


def test_majority_script_latin():
    assert majority_script("Which war made Ashoka give up violence?") == "Latn"


def test_majority_script_ignores_digits_and_punctuation():
    assert majority_script("1757, Plassey!") == "Latn"


def test_majority_script_empty_text_defaults_latin():
    assert majority_script("42!") == "Latn"


def test_hinglish_markers_detected():
    assert _looks_hinglish("Ashoka ne kaunsa yudh lada tha?")
    assert not _looks_hinglish("Which war made Ashoka give up violence?")


async def test_devanagari_text_detected_as_hi():
    det = await get_detector().detect("अशोक ने कौन सा युद्ध लड़ा था?")
    assert det == Detection(lang="hi", script="Deva", confidence=det.confidence)
    assert det.confidence > 0.5


async def test_english_text_detected_as_en():
    det = await get_detector().detect("Which war made Ashoka give up violence?")
    assert det.lang == "en"
    assert det.script == "Latn"


async def test_romanized_hindi_falls_back_to_hi_en():
    # This is the exact case that exposed lid.176's low-confidence miss.
    det = await get_detector().detect("Ashoka ne kaunsa yudh lada tha?")
    assert det.lang == "hi-en"
    assert det.script == "Latn"
