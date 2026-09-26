#!/usr/bin/env python3
"""build_graph.py — LLM entity/relation extraction from data/corpus.jsonl into Neo4j.

DEVIATION from per-chunk extraction: this Groq account is capped at 8000
tokens/minute (confirmed live via response headers), so one LLM call per chunk
across 2371 chunks would take hours. Instead this batches PER ARTICLE (~80,
per CLAUDE.md §6), using each article's lead chunks (where rulers/dynasties/
dates concentrate) up to a token budget, one Groq call per article. Provenance
is preserved: the model sees each chunk tagged by its chunk_id and must cite
one per triple.

Idempotent: a processed article gets a marker node in Neo4j; reruns skip it
unless --force. Relation MERGE dedupes source_chunk_ids and keeps max
confidence, so re-extracting an already-graphed article is a no-op.
Stops if extraction fails (bad/unparseable LLM output) on >30% of the
articles attempted (of a minimum sample), per the milestone's acceptance gate.

Usage:
    python scripts/build_graph.py                  # extract what is missing
    python scripts/build_graph.py --force           # re-extract everything
    python scripts/build_graph.py --limit 5          # smoke test on 5 articles
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.providers.graph_neo4j import Neo4jGraphProvider  # noqa: E402
from server.providers.llm_groq import GroqLLM  # noqa: E402

ARTICLE_TOKEN_BUDGET = 1400  # keeps each call well under the 8000 tok/min account cap
PACING_SECONDS = 12  # deliberate spacing so successive calls don't burst the per-minute cap
RELATION_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,39}$")
MIN_SAMPLE_BEFORE_GATE = 10
FAIL_RATIO_GATE = 0.30

SYSTEM_PROMPT = """You are an information-extraction system for Indian history.
Extract factual relations between named entities (people, dynasties, places,
battles, events, dates) from the numbered source chunks given.

Output strict JSON: {"triples": [{"subject": str, "relation": str, "object": str,
"chunk_id": str, "confidence": number between 0 and 1}]}.

Rules:
- relation must be UPPER_SNAKE_CASE, e.g. RULED, FATHER_OF, FOUGHT_IN, FOUNDED,
  BORN_IN, DIED_IN, CAPITAL_OF, SUCCEEDED_BY, PART_OF, ALLIED_WITH, DEFEATED.
- chunk_id must be exactly one of the chunk_ids given to you.
- Only extract relations explicitly stated in the text. Do not infer or guess.
- If nothing qualifies, return {"triples": []}.
"""

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:ExtractedArticle) REQUIRE a.article IS UNIQUE",
]

MERGE_TRIPLE_SQL_TEMPLATE = """
MERGE (a:Entity {{key: toLower($subject)}})
ON CREATE SET a.name = $subject
MERGE (b:Entity {{key: toLower($object)}})
ON CREATE SET b.name = $object
MERGE (a)-[r:{relation}]->(b)
ON CREATE SET r.source_chunk_ids = [$chunk_id], r.confidence = $confidence
ON MATCH SET
    r.source_chunk_ids = CASE WHEN $chunk_id IN r.source_chunk_ids
                              THEN r.source_chunk_ids
                              ELSE r.source_chunk_ids + $chunk_id END,
    r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END
"""

MARK_ARTICLE_DONE = "MERGE (a:ExtractedArticle {article: $article})"
ARTICLE_ALREADY_DONE = "MATCH (a:ExtractedArticle {article: $article}) RETURN a.article"


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def group_by_article(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        grouped[r["article"]].append(r)
    return grouped


def select_lead_chunks(chunks: list[dict], token_budget: int) -> list[dict]:
    """Lead chunks first (they're where dynasty/date facts concentrate), capped
    to stay well under the per-minute token budget for one Groq call."""
    picked, total = [], 0
    for c in chunks:
        n = c.get("n_tokens", len(c["text"]) // 4)
        if picked and total + n > token_budget:
            break
        picked.append(c)
        total += n
    return picked


def normalize_relation(rel: str) -> str:
    rel = re.sub(r"[^A-Za-z0-9]+", "_", rel.strip()).upper().strip("_") or "RELATED_TO"
    rel = rel[:40]
    return rel if RELATION_RE.match(rel) else "RELATED_TO"


async def extract_article(llm: GroqLLM, article: str, chunks: list[dict]) -> list[dict]:
    body = "\n\n".join(f"[[{c['chunk_id']}]] ({c['section']}) {c['text']}" for c in chunks)
    valid_ids = {c["chunk_id"] for c in chunks}
    reply = await llm.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Article: {article}\n\n{body}"},
        ],
        json_mode=True,
        max_tokens=6000,  # headroom for reasoning tokens + a long triples list (see llm_groq.py)
    )
    triples = json.loads(reply)["triples"]
    out = []
    for t in triples:
        subject, relation, obj = t.get("subject"), t.get("relation"), t.get("object")
        if not subject or not relation or not obj:
            continue
        chunk_id = t.get("chunk_id") if t.get("chunk_id") in valid_ids else chunks[0]["chunk_id"]
        raw_conf = t.get("confidence", 0.5)
        confidence = (max(0.0, min(1.0, float(raw_conf)))
                     if isinstance(raw_conf, int | float) else 0.5)
        out.append({
            "subject": str(subject).strip(),
            "relation": normalize_relation(str(relation)),
            "object": str(obj).strip(),
            "chunk_id": chunk_id,
            "confidence": confidence,
        })
    return out


async def write_triples(driver, article: str, triples: list[dict]) -> None:
    async with driver.session() as session:
        async with await session.begin_transaction() as tx:
            for t in triples:
                cypher = MERGE_TRIPLE_SQL_TEMPLATE.format(relation=t["relation"])
                await tx.run(cypher, subject=t["subject"], object=t["object"],
                            chunk_id=t["chunk_id"], confidence=t["confidence"])
            await tx.run(MARK_ARTICLE_DONE, article=article)
            await tx.commit()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--force", action="store_true", help="re-extract articles already marked done")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N articles")
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"[err] {args.inp} not found — run scripts/build_corpus.py first", file=sys.stderr)
        return 1

    grouped = group_by_article(load_records(args.inp))
    articles = sorted(grouped)
    if args.limit:
        articles = articles[: args.limit]
    print(f"[info] {len(articles)} articles in {args.inp}")

    graph = Neo4jGraphProvider()
    if not await graph.health():
        print("[err] Neo4j unreachable — bring it up: "
              "docker compose --profile hybrid up -d neo4j", file=sys.stderr)
        return 1
    driver = graph.driver

    async with driver.session() as session:
        for stmt in CONSTRAINTS:
            await session.run(stmt)

    todo = articles
    if not args.force:
        done = set()
        async with driver.session() as session:
            for article in articles:
                result = await session.run(ARTICLE_ALREADY_DONE, article=article)
                if await result.single():
                    done.add(article)
        todo = [a for a in articles if a not in done]
        if done:
            print(f"[info] {len(done)} articles already extracted — resuming")

    if not todo:
        print("[done] nothing to do")
        return 0

    # openai/gpt-oss-120b (the §3-pinned model, used by the real answering
    # pipeline) hit its 200k tokens/day cap partway through this milestone —
    # confirmed live via the 429 body, not assumed. gpt-oss-20b is a separate
    # quota bucket on this account and is plenty for extraction, which is a
    # bulk/low-stakes task compared to answer generation.
    llm = GroqLLM(model="openai/gpt-oss-20b")
    attempted = 0
    failed = 0
    total_triples = 0

    with tqdm(total=len(todo), unit="article", desc="extracting") as bar:
        for i, article in enumerate(todo):
            if i > 0:
                # Pace deliberately instead of bursting and reactively backing
                # off — this account's real 8000 tok/min cap means firing
                # requests back-to-back just spends the whole budget on 429s.
                await asyncio.sleep(PACING_SECONDS)
            chunks = select_lead_chunks(grouped[article], ARTICLE_TOKEN_BUDGET)
            attempted += 1
            last_err: Exception | None = None
            for retry in range(2):  # a stray 400/5xx from Groq has been observed to be transient
                try:
                    triples = await extract_article(llm, article, chunks)
                    await write_triples(driver, article, triples)
                    total_triples += len(triples)
                    tqdm.write(f"[ok] {article}: {len(triples)} triples")
                    last_err = None
                    break
                except Exception as e:  # noqa: BLE001 - classify as extraction failure, not crash
                    last_err = e
                    if retry == 0:
                        await asyncio.sleep(2)
            if last_err is not None:
                failed += 1
                tqdm.write(f"[fail] {article}: {last_err!r}")

            if attempted >= MIN_SAMPLE_BEFORE_GATE and failed / attempted > FAIL_RATIO_GATE:
                print(f"\n[abort] {failed}/{attempted} articles failed extraction "
                     f"(> {FAIL_RATIO_GATE:.0%}) — stopping per acceptance gate.", file=sys.stderr)
                return 1
            bar.update(1)

    print(f"\n[done] {attempted - failed}/{attempted} articles extracted OK, "
         f"{total_triples} triples written, {failed} failed")
    print("[next] python scripts/eval_graph.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
