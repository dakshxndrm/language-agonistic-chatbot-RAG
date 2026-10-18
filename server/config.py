"""Settings + profile resolution.

The deployment story (C1) is "swap providers with an env var". That only works if
the swap is resolved exactly once, here, at startup — never with an `if provider ==
"groq"` sprinkled through business logic. Everything downstream reads
`settings.resolved` and nothing else.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Profile = Literal["cloud", "hybrid", "onprem"]

# What each cloud-backed provider degrades to when the profile forbids the network.
LOCAL_EQUIVALENT = {
    "asr_en_provider": "local",
    "asr_indic_provider": "local",
    "llm_provider": "ollama",
    "tts_provider": "kokoro",
    "embed_provider": "local",
    "rerank_provider": "local",
    "graph_provider": "neo4j_local",
}

CLOUD_KEYS = ("groq_api_key", "sarvam_api_key", "cohere_api_key", "openai_api_key")


class ConfigError(RuntimeError):
    """Raised at import/startup time. Deliberately fatal — see C5."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    bhasha_profile: Profile = "hybrid"

    asr_en_provider: Literal["groq", "local"] = "groq"
    asr_indic_provider: Literal["sarvam", "local"] = "sarvam"
    llm_provider: Literal["groq", "ollama", "vllm"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"  # High-capacity open model with ultra-fast inference on Groq LPU
    ollama_base_url: str = "http://localhost:11434"
    tts_provider: Literal["sarvam", "kokoro"] = "sarvam"
    embed_provider: Literal["cohere", "local"] = "cohere"
    rerank_provider: Literal["cohere", "local"] = "cohere"

    groq_api_key: str = ""
    sarvam_api_key: str = ""
    cohere_api_key: str = ""
    openai_api_key: str = ""

    database_url: str = "postgresql+asyncpg://bhasha:bhasha@localhost:5432/bhasha"
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    vad_silence_ms: int = 600
    retrieve_top_k: int = 20
    rerank_top_k: int = 5
    fidelity_script_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

    graph_provider: Literal["neo4j_local", "neo4j_aura", "none"] = "none"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    graph_max_hops: int = 2
    graph_trigger_types: str = "multi_hop,comparative"  # comma-separated question types (C12)

    @property
    def graph_trigger_types_list(self) -> list[str]:
        return [t.strip() for t in self.graph_trigger_types.split(",") if t.strip()]

    @model_validator(mode="after")
    def _resolve_profile(self) -> Settings:
        if self.bhasha_profile != "onprem":
            return self

        # Fail loud rather than quietly ignoring a key the operator believes is
        # in use: a leaked key in an "air-gapped" deployment is the worst outcome.
        leaked = [k.upper() for k in CLOUD_KEYS if getattr(self, k)]
        if leaked:
            raise ConfigError(
                f"BHASHA_PROFILE=onprem forbids cloud credentials, but these are set: "
                f"{', '.join(leaked)}. Unset them or switch profile."
            )

        for field, local in LOCAL_EQUIVALENT.items():
            object.__setattr__(self, field, local)
        return self

    @property
    def resolved(self) -> dict[str, str]:
        """The provider actually in force for each pipeline stage."""
        return {
            "asr_en": self.asr_en_provider,
            "asr_indic": self.asr_indic_provider,
            "llm": self.llm_provider,
            "tts": self.tts_provider,
            "embed": self.embed_provider,
            "rerank": self.rerank_provider,
            "graph": self.graph_provider,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
