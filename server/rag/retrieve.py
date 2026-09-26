"""Hybrid retrieval: dense + sparse, fused with RRF, then cross-encoder reranked.

Why hybrid. Dense (bge-m3) is what makes a Hindi question find an English chunk,
but it is fuzzy about rare proper nouns and dates. Postgres full-text is exact on
"Plassey" or "1757" and useless on Devanagari. They fail in different places, so
fusing them beats either alone — that claim is what scripts/eval_retrieval.py
exists to test.
"""

from __future__ import annotations

from dataclasses import dataclass

from server.config import get_settings
from server.providers.factory import get_embedder, get_reranker
from server.rag.db import get_pool

RRF_K = 60   # standard damping constant; keeps rank-1 from dominating the fusion


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    article: str
    section: str
    text: str
    score: float


_DENSE_SQL = """
SELECT chunk_id, article, section, text
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1
LIMIT $2
"""

# ts_rank_cd over a weighted tsvector, not textbook BM25 — Postgres ships this,
# and pulling in pg_search just for Okapi weighting is not worth a new service.
_SPARSE_SQL = """
SELECT chunk_id, article, section, text, ts_rank_cd(tsv, q) AS rank
FROM chunks, websearch_to_tsquery('english', $1) q
WHERE tsv @@ q
ORDER BY rank DESC
LIMIT $2
"""


async def _dense(query: str, k: int) -> list[dict]:
    vec = (await get_embedder().embed([query], is_query=True))[0]
    pool = await get_pool()
    rows = await pool.fetch(_DENSE_SQL, vec, k)
    return [dict(r) for r in rows]


async def _sparse(query: str, k: int) -> list[dict]:
    pool = await get_pool()
    # websearch_to_tsquery never raises on junk input, unlike to_tsquery.
    rows = await pool.fetch(_SPARSE_SQL, query, k)
    return [dict(r) for r in rows]


def rrf(rankings: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion. Uses rank position only, so the dense cosine score
    and the ts_rank_cd score never have to be made commensurable."""
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for pos, row in enumerate(ranking, start=1):
            entry = fused.setdefault(row["chunk_id"], {**row, "score": 0.0})
            entry["score"] += 1.0 / (k + pos)
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)


async def retrieve(query: str, top_k: int | None = None, rerank_top_k: int | None = None,
                   mode: str = "hybrid") -> list[Hit]:
    """mode: "hybrid" | "dense" | "sparse" | "hybrid_norerank" — the non-default
    modes exist so Milestone 5 can ablate them."""
    s = get_settings()
    top_k = top_k or s.retrieve_top_k
    rerank_top_k = rerank_top_k or s.rerank_top_k

    if mode == "dense":
        candidates = await _dense(query, top_k)
    elif mode == "sparse":
        candidates = await _sparse(query, top_k)
    else:
        candidates = rrf([await _dense(query, top_k), await _sparse(query, top_k)])

    if not candidates:
        return []

    if mode in ("dense", "sparse", "hybrid_norerank"):
        picked = candidates[:rerank_top_k]
        return [Hit(c["chunk_id"], c["article"], c["section"], c["text"],
                    float(c.get("score", c.get("rank", 0.0)))) for c in picked]

    scores = await get_reranker().score(query, [c["text"] for c in candidates])
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)
    return [Hit(c["chunk_id"], c["article"], c["section"], c["text"], float(sc))
            for c, sc in ranked[:rerank_top_k]]


async def chunks_by_ids(chunk_ids: list[str]) -> list[Hit]:
    """Fetches specific chunks by id, e.g. the ones cited on a graph path —
    orchestrate.py uses this to pull real chunk text for graph-sourced hits."""
    if not chunk_ids:
        return []
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT chunk_id, article, section, text FROM chunks WHERE chunk_id = ANY($1)",
        chunk_ids,
    )
    return [Hit(r["chunk_id"], r["article"], r["section"], r["text"], 0.0) for r in rows]
