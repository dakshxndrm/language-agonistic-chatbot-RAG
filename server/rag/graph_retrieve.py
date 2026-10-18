"""Multi-hop/comparative-only graph augmentation (C12), merged into retrieval
by orchestrate.py.

Entity extraction from the question goes through the LLM, not a capitalisation
regex: the golden set's multi_hop/comparative questions span en, hi-deva,
hinglish-latn and hi-deva-mixed, but the graph's node names are the English
Wikipedia titles build_graph.py extracted. Bridging "गांधी" -> "Gandhi" is
cross-lingual entity linking, which needs the same multilingual model the rest
of this pipeline already uses — a regex heuristic only works on English.
"""

from __future__ import annotations

import json
import logging

from server.config import get_settings
from server.providers.base import GraphHit, LLMProvider
from server.providers.factory import get_graph_provider, get_llm

logger = logging.getLogger(__name__)

ENTITY_EXTRACT_PROMPT = """Identify the key named entities (people, dynasties,
places, battles, events) this question is about, using their standard ENGLISH
Wikipedia article names, even if the question is in Hindi, Hinglish, or mixed
script. Output strict JSON: {"entities": [str, ...]}. Return at most 3.
"""


async def extract_entities(question: str, llm: LLMProvider | None = None) -> list[str]:
    llm = llm or get_llm()
    if hasattr(llm, "complete"):
        reply = await llm.complete(
            [{"role": "system", "content": ENTITY_EXTRACT_PROMPT},
             {"role": "user", "content": question}],
            json_mode=True,
        )
    else:
        reply = "".join([c async for c in llm.stream([
            {"role": "system", "content": ENTITY_EXTRACT_PROMPT},
            {"role": "user", "content": question}
        ], None, [])])
    try:
        entities = json.loads(reply).get("entities", [])
    except json.JSONDecodeError:
        return []
    return [e.strip() for e in entities if isinstance(e, str) and e.strip()][:3]


def _path_key(hit: GraphHit) -> tuple:
    return tuple(tuple(sorted(step.items())) for step in hit.path)


async def graph_retrieve(question: str, question_type: str) -> list[GraphHit]:
    """[] immediately for any type outside GRAPH_TRIGGER_TYPES (C12) — never
    queries the graph "just in case". [] (never raises) if the graph is
    unreachable (C11) — the caller still gets a valid vector-only answer."""
    s = get_settings()
    if question_type not in s.graph_trigger_types_list:
        return []

    graph = get_graph_provider()
    if not await graph.health():
        logger.warning("graph unreachable — skipping graph augmentation (C11)")
        return []

    entities = await extract_entities(question)
    if not entities:
        return []

    hits: list[GraphHit] = []
    seen: set[tuple] = set()
    for entity in entities:
        for hit in await graph.query_entity(entity, max_hops=s.graph_max_hops):
            key = _path_key(hit)
            if key not in seen:
                seen.add(key)
                hits.append(hit)
    hits.sort(key=lambda h: getattr(h, "confidence", 1.0) or 1.0, reverse=True)
    return hits[:10]

