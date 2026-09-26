"""Dual-path routing (§4 feature #1) is the one thing that must never guess —
these pin the routing table down as a pure function, no audio/network needed.
"""

from server.providers.asr_router import choose_route


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
