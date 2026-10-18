"""Semantic Question Caching with pgvector (Phase 5).

Frequent historical and evaluation questions (e.g. "Who founded the Maurya Dynasty?",
"1857 ka vidroh kyu shuru hua?") achieve cosine similarity >= 0.96 against pre-verified
answers. Bypassing RAG + LLM drops Time-To-First-Audio (TTFA) from ~700ms to <120ms.

Constraint C11/C12 adherence: If Postgres or pgvector is offline, this degrades
silently to a cache miss, never breaking the conversation pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from server.providers.factory import get_embedder
from server.rag.db import get_pool

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.96


@dataclass(frozen=True)
class CacheHit:
    id: str
    query_text: str
    response_text: str
    audio_cache_path: str | None
    metadata: dict[str, Any]
    similarity: float


_MATCH_SQL = """
SELECT id, query_text, response_text, audio_cache_path, metadata,
       1 - (query_embedding <=> $1) AS similarity
FROM query_cache
WHERE 1 - (query_embedding <=> $1) >= $2
ORDER BY similarity DESC
LIMIT 1;
"""

_INSERT_SQL = """
INSERT INTO query_cache (query_text, query_embedding, response_text, audio_cache_path, metadata)
VALUES ($1, $2, $3, $4, $5)
RETURNING id;
"""


async def check_semantic_cache(
    query: str, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> CacheHit | None:
    """Checks pgvector query_cache for semantic matches >= threshold."""
    try:
        vec = (await get_embedder().embed([query], is_query=True))[0]
        pool = await get_pool()
        row = await pool.fetchrow(_MATCH_SQL, vec, similarity_threshold)
        if not row:
            return None

        meta_raw = row["metadata"]
        meta_dict = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw or {})
        hit = CacheHit(
            id=str(row["id"]),
            query_text=row["query_text"],
            response_text=row["response_text"],
            audio_cache_path=row["audio_cache_path"],
            metadata=meta_dict,
            similarity=float(row["similarity"]),
        )
        logger.info(
            "Semantic cache HIT (similarity=%.4f) for query: %r",
            hit.similarity,
            query,
        )
        return hit
    except Exception as exc:
        logger.debug("Semantic cache check bypassed: %s", exc)
        return None


async def store_semantic_cache(
    query: str,
    response_text: str,
    audio_cache_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """Stores query embedding and response in pgvector query_cache."""
    try:
        vec = (await get_embedder().embed([query], is_query=True))[0]
        pool = await get_pool()
        cache_id = await pool.fetchval(
            _INSERT_SQL,
            query,
            vec,
            response_text,
            audio_cache_path,
            json.dumps(metadata or {}),
        )
        logger.info("Stored query in semantic cache: id=%s", cache_id)
        return str(cache_id)
    except Exception as exc:
        logger.debug("Failed to store in semantic cache: %s", exc)
        return None
