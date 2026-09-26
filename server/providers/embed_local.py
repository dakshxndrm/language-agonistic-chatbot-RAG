"""BAAI/bge-m3 embeddings, always local (EMBED_PROVIDER=local).

bge-m3 is the reason cross-lingual retrieval works at all: a Hindi or Tamil
question and an English Wikipedia chunk land in the same 1024-d space, so we
never have to translate the query.
"""

from __future__ import annotations

import asyncio
import os

from server.providers.base import EmbeddingProvider

MODEL_NAME = "BAAI/bge-m3"
DIM = 1024


class LocalEmbedder(EmbeddingProvider):
    """BAAI/bge-m3, run on-device. Used when EMBED_PROVIDER=local (onprem profile,
    or a developer without a Cohere key) — one process-wide instance, since loading
    the model costs ~10s and ~2.5GB RAM."""

    name = "bge-m3"

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self._model_name = model_name
        self._device = device or os.getenv("EMBED_DEVICE", "cpu")
        self._model = None
        self._lock = asyncio.Lock()

    def load(self) -> None:
        """Blocking. Call from startup or via asyncio.to_thread."""
        if self._model is None:
            # Imported here, not at module scope, so `import server.providers`
            # stays cheap for anything that doesn't actually embed.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
            # bge-m3 defaults to an 8192-token window. Our chunks are <=512 tokens
            # (build_corpus.py), and leaving the window wide made a 16GB CPU box
            # swap itself to death mid-index. Cap it.
            self._model.max_seq_length = 512

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._lock:          # two concurrent first-calls must not both load
            if self._model is None:
                await asyncio.to_thread(self.load)

    async def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        await self._ensure_loaded()
        # bge-m3 is trained without an instruction prefix, so queries and passages
        # are encoded identically — is_query exists for interface compatibility.
        vectors = await asyncio.to_thread(
            self._model.encode,
            texts,
            normalize_embeddings=True,   # lets pgvector cosine == dot product
            batch_size=4,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    async def health(self) -> bool:
        try:
            v = await self.embed(["ok"])
            return len(v) == 1 and len(v[0]) == DIM
        except Exception:
            return False
