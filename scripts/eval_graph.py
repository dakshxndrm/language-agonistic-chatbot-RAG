#!/usr/bin/env python3
"""eval_graph.py — does graph augmentation actually help multi_hop/comparative
questions? (the ablation C12 exists to make meaningful).

For each multi_hop/comparative question in the golden set: extract entities,
walk the graph, and check whether the chunk_ids on the returned paths belong
to the question's gold articles. This is the graph-only half of the
vector-vs-vector+graph split eval_retrieval.py reports after M1.5's fusion.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.rag.graph_retrieve import extract_entities, graph_retrieve  # noqa: E402


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if r.get("gold_articles")]


def chunk_to_article_map(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fh:
        return {json.loads(line)["chunk_id"]: json.loads(line)["article"]
                for line in fh if line.strip()}


async def main() -> int:
    golden = load_golden(Path("data/golden_set.jsonl"))
    chunk_article = chunk_to_article_map(Path("data/corpus.jsonl"))

    triggered = [q for q in golden if q["type"] in ("multi_hop", "comparative")]
    print(f"[info] {len(triggered)} multi_hop/comparative questions "
         f"(of {len(golden)} total with gold articles)\n")

    covered = 0
    any_hit = 0
    for q in triggered:
        entities = await extract_entities(q["question"])
        hits = await graph_retrieve(q["question"], q["type"])
        found_articles = {chunk_article.get(cid) for h in hits for cid in h.source_chunk_ids}
        found_articles.discard(None)
        gold = set(q["gold_articles"])
        hit_gold = bool(found_articles & gold)
        any_hit += bool(hits)
        covered += hit_gold
        print(f"{q['id']:5} [{q['type']:11}] entities={entities} "
             f"paths={len(hits)} found={sorted(found_articles)} gold={sorted(gold)} "
             f"{'HIT' if hit_gold else 'miss'}")

    n = len(triggered)
    print(f"\n[summary] {any_hit}/{n} questions got >=1 graph path; "
         f"{covered}/{n} questions had a graph path landing on a gold article")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
