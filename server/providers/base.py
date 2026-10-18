"""The provider interface. Every ASR / LLM / TTS / embedding call in BHASHA goes
through one of these ABCs (C1) so a deployment can swap cloud for local with an
env var and nothing downstream notices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Transcript:
    text: str
    lang: str            # BCP-47-ish: "en", "hi", "hi-en", "mr"
    script: str          # ISO 15924: "Latn", "Deva", "Beng", "Taml"
    confidence: float
    is_final: bool
    latency_ms: float
    asr_engine_used: str = "whisper"


@dataclass(frozen=True)
class LangContract:
    lang: str
    script: str
    register: str = "neutral"   # "formal" | "neutral" | "casual"


class ASRProvider(ABC):
    name: str

    @abstractmethod
    async def transcribe(self, pcm: bytes, sample_rate: int,
                         hint_lang: str | None = None) -> Transcript: ...

    @abstractmethod
    async def health(self) -> bool: ...


class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def stream(self, messages: list[dict], contract: LangContract,
                     context: list[dict]) -> AsyncIterator[str]:
        """Yields token deltas. MUST be cancellable via asyncio.CancelledError."""

    @abstractmethod
    async def health(self) -> bool: ...


class TTSProvider(ABC):
    name: str

    sample_rate: int  # PCM frames yielded by synthesize() are at this rate

    @abstractmethod
    async def synthesize(self, text: str, lang: str,
                         voice: str | None = None) -> AsyncIterator[bytes]:
        """Yields raw 16-bit mono PCM chunks. MUST be cancellable mid-stream."""

    @abstractmethod
    async def health(self) -> bool: ...


class EmbeddingProvider(ABC):
    name: str

    @abstractmethod
    async def embed(self, texts: list[str],
                    is_query: bool = False) -> list[list[float]]: ...


class RerankProvider(ABC):
    name: str

    @abstractmethod
    async def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance scores, higher is better, same order as `passages`."""


@dataclass(frozen=True)
class GraphHit:
    # ordered nodes+edges, e.g.
    # [{"entity": "Shah Jahan"}, {"rel": "FATHER_OF"}, {"entity": "Aurangzeb"}]
    path: list[dict]
    source_chunk_ids: list[str]   # which corpus chunks this triple was extracted from
    confidence: float


class GraphProvider(ABC):
    name: str

    @abstractmethod
    async def query_entity(self, entity_name: str, max_hops: int = 2) -> list[GraphHit]: ...

    @abstractmethod
    async def health(self) -> bool:
        """MUST return False gracefully (no exception) if Neo4j is unreachable — see C11."""
