"""BAAI/bge-reranker-v2-m3 — a cross-encoder, so it sees the query and the chunk
together and can catch the near-misses that bi-encoder cosine similarity ranks
too highly. Used when RERANK_PROVIDER=local (onprem profile, or a developer
without a Cohere key).
"""

from __future__ import annotations

import asyncio
import os

from server.providers.base import RerankProvider

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class LocalReranker(RerankProvider):
    name = "bge-reranker-v2-m3"

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self._model_name = model_name
        self._device = device or os.getenv("EMBED_DEVICE", "cpu")
        self._model = None
        self._lock = asyncio.Lock()

    def load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device, max_length=512)

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is None:
                await asyncio.to_thread(self.load)

    async def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance logits, higher is better. Not normalised — only the order matters."""
        if not passages:
            return []
        await self._ensure_loaded()
        pairs = [(query, p) for p in passages]
        scores = await asyncio.to_thread(self._model.predict, pairs, batch_size=8)
        return [float(s) for s in scores]
