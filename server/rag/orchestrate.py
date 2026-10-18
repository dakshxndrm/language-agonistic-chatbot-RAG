"""orchestrate.py — the single entry point rag/prompt.py (M2) calls.

Vector retrieval always runs: it's the part that must produce a valid answer
on its own (C11). Graph augmentation only runs for multi_hop/comparative
questions (C12) and, when it finds anything, contributes two things: the
actual chunk text behind each graph path (via chunks_by_ids, so the LLM has
real prose to cite) and the raw entity-relation paths themselves (so a prompt
can show the reasoning chain, e.g. "Shah Jahan -FATHER_OF-> Aurangzeb").
"""

from __future__ import annotations

from dataclasses import dataclass

import asyncio

from server.providers.base import GraphHit
from server.rag.graph_retrieve import graph_retrieve
from server.rag.retrieve import Hit, chunks_by_ids, retrieve


@dataclass(frozen=True)
class OrchestrationResult:
    hits: list[Hit]              # vector hits, plus any graph-cited chunks not already present
    graph_paths: list[GraphHit]  # raw entity-relation chains, for citing the multi-hop reasoning
    vector_only_hits: list[Hit]  # kept separate so eval can report the vector-only baseline


async def orchestrate(question: str, question_type: str = "factoid",
                      top_k: int | None = None, rerank_top_k: int | None = None,
                      mode: str = "hybrid") -> OrchestrationResult:
    vector_hits, graph_paths = await asyncio.gather(
        retrieve(question, top_k=top_k, rerank_top_k=rerank_top_k, mode=mode),
        graph_retrieve(question, question_type),
    )

    seen = {h.chunk_id for h in vector_hits}
    graph_chunk_ids = [cid for gh in graph_paths for cid in gh.source_chunk_ids if cid not in seen]
    graph_hits = await chunks_by_ids(list(dict.fromkeys(graph_chunk_ids))[:5])  # dedupe, keep order, cap at 5

    return OrchestrationResult(
        hits=[*vector_hits, *graph_hits],
        graph_paths=graph_paths,
        vector_only_hits=vector_hits,
    )
