#!/usr/bin/env python3
"""eval_retrieval.py — context recall@K on data/golden_set.jsonl, split by flavour.

Recall@K here = "did the top-K retrieved chunks include at least one chunk from
every gold article?". Article-level, because the golden set labels articles, not
chunk ids. A question whose gold_articles list is empty (out-of-scope traps) is
skipped — there is nothing to recall.

The per-flavour split is the point: if `en` scores well and `hi_deva` does not,
the cross-lingual claim is dead, and we would rather find that out here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.rag.db import close_pool  # noqa: E402
from server.rag.orchestrate import orchestrate  # noqa: E402


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if r.get("gold_articles")]


def chunk_to_article_map(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fh:
        return {json.loads(line)["chunk_id"]: json.loads(line)["article"]
                for line in fh if line.strip()}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("data/golden_set.jsonl"))
    ap.add_argument("-k", type=int, default=5, help="recall@K")
    ap.add_argument("--mode", default="hybrid",
                    choices=["hybrid", "hybrid_norerank", "dense", "sparse"])
    ap.add_argument("--verbose", action="store_true", help="print every miss")
    args = ap.parse_args()

    questions = load_golden(args.golden)
    chunk_article = chunk_to_article_map(Path("data/corpus.jsonl"))
    print(f"[info] {len(questions)} questions with gold articles | mode={args.mode} | K={args.k}\n")

    per_flavour_vec: dict[str, list[float]] = defaultdict(list)
    per_flavour_combined: dict[str, list[float]] = defaultdict(list)
    graph_eligible_vec: list[float] = []
    graph_eligible_combined: list[float] = []
    misses: list[tuple[str, str, list[str], list[str]]] = []

    for q in questions:
        result = await orchestrate(q["question"], question_type=q["type"],
                                   rerank_top_k=args.k, mode=args.mode)
        gold = q["gold_articles"]

        found_vec = {h.article for h in result.vector_only_hits}
        recall_vec = sum(1 for g in gold if g in found_vec) / len(gold)
        per_flavour_vec[q["flavour"]].append(recall_vec)

        found_combined = found_vec | {chunk_article.get(cid) for gh in result.graph_paths
                                      for cid in gh.source_chunk_ids}
        found_combined.discard(None)
        recall_combined = sum(1 for g in gold if g in found_combined) / len(gold)
        per_flavour_combined[q["flavour"]].append(recall_combined)

        if q["type"] in ("multi_hop", "comparative"):
            graph_eligible_vec.append(recall_vec)
            graph_eligible_combined.append(recall_combined)

        if recall_vec < 1.0:
            misses.append((q["id"], q["question"], gold, sorted(found_vec)))

    width = max(len(f) for f in per_flavour_vec)
    print(f"| {'flavour':<{width}} | n  | vector@{args.k} | +graph@{args.k} |")
    print(f"|{'-' * (width + 2)}|----|-----------|-----------|")
    for flavour in sorted(per_flavour_vec):
        v, c = per_flavour_vec[flavour], per_flavour_combined[flavour]
        print(f"| {flavour:<{width}} | {len(v):<2} | {sum(v) / len(v):>9.3f} | "
             f"{sum(c) / len(c):>9.3f} |")
    allv = [v for vals in per_flavour_vec.values() for v in vals]
    allc = [v for vals in per_flavour_combined.values() for v in vals]
    print(f"| {'ALL':<{width}} | {len(allv):<2} | {sum(allv) / len(allv):>9.3f} | "
         f"{sum(allc) / len(allc):>9.3f} |")

    if graph_eligible_vec:
        n = len(graph_eligible_vec)
        print(f"\n[graph ablation] multi_hop/comparative only (n={n}): "
             f"vector-only={sum(graph_eligible_vec) / n:.3f} vs "
             f"vector+graph={sum(graph_eligible_combined) / n:.3f}")

    print(f"\n[info] {len(misses)} question(s) below full recall (vector-only)")
    if args.verbose:
        for qid, question, gold, found in misses:
            print(f"  {qid} {question[:60]!r}\n      gold={gold}\n      got ={found}")

    await close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
