"""Profile resolution is the one thing in Milestone 0 that can silently leak data."""

import pytest

from server.config import ConfigError, Settings


def make(**kw) -> Settings:
    # _env_file=None so a developer's real .env can never change a test outcome.
    return Settings(_env_file=None, **kw)


def test_hybrid_keeps_declared_providers():
    s = make(bhasha_profile="hybrid", llm_provider="groq", tts_provider="sarvam")
    assert s.resolved["llm"] == "groq"
    assert s.resolved["tts"] == "sarvam"


def test_onprem_overrides_every_cloud_provider():
    s = make(
        bhasha_profile="onprem",
        asr_en_provider="groq",
        asr_indic_provider="sarvam",
        llm_provider="groq",
        tts_provider="sarvam",
    )
    assert s.resolved == {
        "asr_en": "local",
        "asr_indic": "local",
        "llm": "ollama",
        "tts": "kokoro",
        "embed": "local",
        "rerank": "local",
        "graph": "neo4j_local",
    }


@pytest.mark.parametrize(
    "key", ["groq_api_key", "sarvam_api_key", "cohere_api_key", "openai_api_key"]
)
def test_onprem_fails_loudly_on_any_cloud_key(key):
    with pytest.raises(ConfigError, match=key.upper()):
        make(bhasha_profile="onprem", **{key: "sk-not-a-real-key"})


def test_cloud_keys_are_fine_off_onprem():
    s = make(bhasha_profile="cloud", groq_api_key="sk-x")
    assert s.resolved["llm"] == "groq"
