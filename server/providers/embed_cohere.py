"""Cohere embed-multilingual-v3.0 — the default embedder (§3, §5). Free tier,
1024-d, and pretrained for cross-lingual retrieval, so it needs no local GPU/CPU
budget on a 16GB dev laptop. Falls back to bge-m3 (embed_local.py) only under
BHASHA_PROFILE=onprem, where C5 forbids outbound calls.
"""

from __future__ import annotations

import asyncio

import cohere
from cohere.errors import TooManyRequestsError

from server.config import get_settings
from server.providers.base import EmbeddingProvider

MODEL_NAME = "embed-multilingual-v3.0"
DIM = 1024
MAX_ATTEMPTS = 5


MAX_QUERY_CACHE = 256
_query_cache: dict[str, list[float]] = {}


class CohereEmbedder(EmbeddingProvider):
    name = "cohere-embed-multilingual-v3.0"

    def __init__(self, api_key: str | None = None):
        # pydantic-settings parses .env into Settings only — it never exports into
        # os.environ, so the key has to come from get_settings(), not os.getenv().
        self._api_key = api_key or get_settings().cohere_api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = cohere.AsyncClientV2(api_key=self._api_key)
        return self._client

    async def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        if is_query and len(texts) == 1 and texts[0] in _query_cache:
            return [_query_cache[texts[0]]]
        client = self._ensure_client()
        # The free/trial key caps at 100k tokens/min, easy to blow through on a
        # 2000+ chunk corpus. Retry with backoff rather than making every caller
        # (index_corpus.py, retrieve.py) know about Cohere-specific rate limits.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = await client.embed(
                    texts=texts,
                    model=MODEL_NAME,
                    input_type="search_query" if is_query else "search_document",
                    embedding_types=["float"],
                )
                embeddings = resp.embeddings.float_
                if is_query and len(texts) == 1 and embeddings:
                    if len(_query_cache) >= MAX_QUERY_CACHE:
                        _query_cache.pop(next(iter(_query_cache)))
                    _query_cache[texts[0]] = embeddings[0]
                return embeddings
            except TooManyRequestsError:
                if attempt == MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def health(self) -> bool:
        try:
            v = await self.embed(["ok"])
            return len(v) == 1 and len(v[0]) == DIM
        except Exception:
            return False
