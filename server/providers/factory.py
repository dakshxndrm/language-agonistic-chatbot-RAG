"""Resolves EMBED_PROVIDER / RERANK_PROVIDER to a concrete provider exactly once
(C1) — retrieve.py and index_corpus.py call get_embedder()/get_reranker() and
never know or care whether that's Cohere or the local bge- models.
"""

from __future__ import annotations

from server.config import get_settings
from server.providers.base import (
    ASRProvider,
    EmbeddingProvider,
    GraphProvider,
    LLMProvider,
    RerankProvider,
    TTSProvider,
)

_embedder: EmbeddingProvider | None = None
_reranker: RerankProvider | None = None
_graph: GraphProvider | None = None
_llm: LLMProvider | None = None
_asr_en: ASRProvider | None = None
_asr_indic: ASRProvider | None = None
_tts: TTSProvider | None = None


def get_embedder() -> EmbeddingProvider:
    global _embedder
    if _embedder is None:
        if get_settings().embed_provider == "cohere":
            from server.providers.embed_cohere import CohereEmbedder

            _embedder = CohereEmbedder()
        else:
            from server.providers.embed_local import LocalEmbedder

            _embedder = LocalEmbedder()
    return _embedder


def get_reranker() -> RerankProvider:
    global _reranker
    if _reranker is None:
        if get_settings().rerank_provider == "cohere":
            from server.providers.rerank_cohere import CohereReranker

            _reranker = CohereReranker()
        else:
            from server.providers.rerank_local import LocalReranker

            _reranker = LocalReranker()
    return _reranker


def get_graph_provider() -> GraphProvider:
    global _graph
    if _graph is None:
        if get_settings().graph_provider == "none":
            from server.providers.graph_neo4j import NullGraphProvider

            _graph = NullGraphProvider()
        else:
            from server.providers.graph_neo4j import Neo4jGraphProvider

            _graph = Neo4jGraphProvider()
    return _graph


def get_llm() -> LLMProvider:
    global _llm
    if _llm is None:
        provider = get_settings().llm_provider
        if provider == "groq":
            from server.providers.llm_groq import GroqLLM

            _llm = GroqLLM()
        elif provider == "ollama":
            from server.providers.llm_ollama import OllamaLLM

            _llm = OllamaLLM()
        else:
            raise NotImplementedError(
                f"LLM_PROVIDER={provider!r} has no concrete provider yet (C9) — "
                "only groq and ollama are implemented."
            )
    return _llm


def get_asr_en() -> ASRProvider:
    global _asr_en
    if _asr_en is None:
        if get_settings().asr_en_provider == "groq":
            from server.providers.asr_groq import GroqASR

            _asr_en = GroqASR()
        else:
            from server.providers.asr_local import LocalASR

            _asr_en = LocalASR()
    return _asr_en


def get_asr_indic() -> ASRProvider:
    global _asr_indic
    if _asr_indic is None:
        if get_settings().asr_indic_provider == "sarvam":
            from server.providers.asr_sarvam import SarvamASR

            _asr_indic = SarvamASR()
        else:
            from server.providers.asr_local import LocalASR

            _asr_indic = LocalASR()
    return _asr_indic


def get_tts() -> TTSProvider:
    global _tts
    if _tts is None:
        if get_settings().tts_provider == "sarvam":
            from server.providers.tts_sarvam import SarvamTTS

            _tts = SarvamTTS()
        else:
            from server.providers.tts_kokoro import KokoroTTS

            _tts = KokoroTTS()
    return _tts
