"""Cohere rerank-multilingual-v3.0 — the default reranker (§3, §5). Free tier,
cross-lingual, no local cross-encoder to load on a 16GB dev laptop. Falls back
to bge-reranker-v2-m3 (rerank_local.py) only under BHASHA_PROFILE=onprem.
"""

from __future__ import annotations

import asyncio

import cohere
from cohere.errors import TooManyRequestsError

from server.config import get_settings
from server.providers.base import RerankProvider

# ponytail: trial Cohere keys cap /v2/rerank at a hard 10 calls/minute (stated in
# the 429 body, unlike embed's token-based limit) — retry on a fixed ~7s interval
# rather than exponential backoff, since the ceiling resets every minute regardless
# of how long we wait. A paid key removes the need for this entirely.
MODEL_NAME = "rerank-multilingual-v3.0"
MAX_ATTEMPTS = 12
RETRY_SECONDS = 7


class CohereReranker(RerankProvider):
    name = "cohere-rerank-multilingual-v3.0"

    def __init__(self, api_key: str | None = None):
        # pydantic-settings parses .env into Settings only — it never exports into
        # os.environ, so the key has to come from get_settings(), not os.getenv().
        self._api_key = api_key or get_settings().cohere_api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = cohere.AsyncClientV2(api_key=self._api_key)
        return self._client

    async def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance scores, higher is better. Cohere returns results sorted by
        relevance with the original index attached, so we un-sort them back into
        `passages` order to match the LocalReranker contract."""
        if not passages:
            return []
        client = self._ensure_client()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = await client.rerank(query=query, documents=passages, model=MODEL_NAME)
                break
            except TooManyRequestsError:
                if attempt == MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(RETRY_SECONDS)
        scores = [0.0] * len(passages)
        for r in resp.results:
            scores[r.index] = r.relevance_score
        return scores
