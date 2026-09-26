#!/usr/bin/env python3
"""index_corpus.py — embed data/corpus.jsonl with bge-m3 and upsert into `chunks`.

Idempotent: chunk_id is the primary key and every write is an upsert, so re-running
updates rows instead of duplicating them.
Resumable: chunks that already carry an embedding are skipped, so a crash halfway
through costs you only the batch that was in flight. Use --force to re-embed all.

Usage:
    python scripts/index_corpus.py                  # embed what is missing
    python scripts/index_corpus.py --force          # re-embed everything
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import asyncpg
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.providers.embed_local import DIM  # noqa: E402
from server.providers.factory import get_embedder  # noqa: E402
from server.rag.db import close_pool, get_pool  # noqa: E402

UPSERT_SQL = """
INSERT INTO chunks (chunk_id, article, url, section, text, n_tokens, lang,
                    revision_id, fetched_at, embedding)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (chunk_id) DO UPDATE SET
    article     = EXCLUDED.article,
    url         = EXCLUDED.url,
    section     = EXCLUDED.section,
    text        = EXCLUDED.text,
    n_tokens    = EXCLUDED.n_tokens,
    lang        = EXCLUDED.lang,
    revision_id = EXCLUDED.revision_id,
    fetched_at  = EXCLUDED.fetched_at,
    embedding   = EXCLUDED.embedding
"""


async def write_batch(pool, rows: list[tuple], attempts: int = 3) -> None:
    """One transaction per batch: a crash leaves whole batches committed, never a
    half-written row, which is what makes the resume clean.

    Retries because an hour-long CPU embed job on a loaded laptop will occasionally
    time out acquiring a connection, and losing 40 minutes of work to that is silly.
    """
    for attempt in range(1, attempts + 1):
        try:
            async with pool.acquire() as conn, conn.transaction():
                await conn.executemany(UPSERT_SQL, rows)
            return
        except (asyncpg.PostgresConnectionError, TimeoutError, OSError) as e:
            if attempt == attempts:
                raise
            tqdm.write(f"[warn] db write failed ({e!r}); retry {attempt}/{attempts - 1}")
            await asyncio.sleep(2 * attempt)


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="re-embed rows that already have one")
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"[err] {args.inp} not found — run scripts/build_corpus.py first", file=sys.stderr)
        return 1

    records = load_records(args.inp)
    print(f"[info] {len(records)} chunks in {args.inp}")

    pool = await get_pool()
    done: set[str] = set()
    if not args.force:
        rows = await pool.fetch("SELECT chunk_id FROM chunks WHERE embedding IS NOT NULL")
        done = {r["chunk_id"] for r in rows}
        if done:
            print(f"[info] {len(done)} already embedded — resuming")

    todo = [r for r in records if r["chunk_id"] not in done]
    if not todo:
        print("[done] nothing to do")
        await close_pool()
        return 0

    embedder = get_embedder()
    if hasattr(embedder, "load"):  # local models only — Cohere has no weights to load
        print(f"[info] loading {embedder.name} (first run downloads ~2.3GB)")
        await asyncio.to_thread(embedder.load)
    else:
        print(f"[info] using {embedder.name}")

    written = 0
    with tqdm(total=len(todo), unit="chunk", desc="embedding") as bar:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            vectors = await embedder.embed([r["text"] for r in batch])
            assert len(vectors[0]) == DIM, f"expected {DIM}-d vectors, got {len(vectors[0])}"

            rows = [
                (r["chunk_id"], r["article"], r["url"], r["section"], r["text"],
                 r["n_tokens"], r.get("lang", "en"), r.get("revision_id"),
                 datetime.fromisoformat(r["fetched_at"]), v)
                for r, v in zip(batch, vectors, strict=True)
            ]
            await write_batch(pool, rows)
            written += len(batch)
            bar.update(len(batch))

    total = await pool.fetchval("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
    await close_pool()
    print(f"\n[done] wrote {written} chunks; {total} embedded rows in the table")
    print("[next] python scripts/eval_retrieval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
