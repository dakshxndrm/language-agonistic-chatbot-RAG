#!/usr/bin/env python3
"""
build_corpus.py — Build the Indian History knowledge corpus for BHASHA.

Fetches the curated Wikipedia article list, strips boilerplate, splits each
article into section-aware chunks, and writes a JSONL file ready to embed.

Why section-aware chunking: Wikipedia history articles are long and strictly
hierarchical. Chunking on raw character counts splits mid-paragraph and
destroys the section heading, which is the single most useful piece of
retrieval metadata we have ("Ashoka > Kalinga War > Aftermath").

Usage:
    pip install requests tqdm tiktoken
    python scripts/build_corpus.py \
        --articles data/corpus_articles.txt \
        --out data/corpus.jsonl \
        --chunk-tokens 512 --overlap-tokens 64

Output schema (one JSON object per line):
    {
      "chunk_id":   "ashoka::0007",
      "article":    "Ashoka",
      "url":        "https://en.wikipedia.org/wiki/Ashoka",
      "section":    "Kalinga War > Aftermath",
      "text":       "...",
      "n_tokens":   498,
      "lang":       "en",
      "revision_id": 1234567890,
      "fetched_at": "2026-09-12T00:00:00Z"
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import requests

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "BHASHA-corpus-builder/1.0 (final-year-project; contact: 2023pietcsdaksh041@poornima.org)"

# Sections that carry no factual content worth retrieving.
SKIP_SECTIONS = {
    "see also", "references", "further reading", "external links",
    "notes", "bibliography", "citations", "sources", "gallery",
    "footnotes", "explanatory notes", "works cited",
}


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #
def get_token_counter():
    """Prefer tiktoken; fall back to a whitespace approximation."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s)), enc
    except Exception:  # pragma: no cover
        print("[warn] tiktoken unavailable — using ~0.75 words/token estimate", file=sys.stderr)
        return lambda s: int(len(s.split()) / 0.75), None


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def fetch_article(title: str, session: requests.Session) -> dict | None:
    """Fetch plaintext extract + revision id for one article."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|revisions",
        "rvprop": "ids",
        "explaintext": 1,
        "exsectionformat": "wiki",   # keeps == Heading == markers
        "redirects": 1,
        "titles": title,
        "formatversion": 2,
    }
    for attempt in range(4):
        try:
            r = session.get(API, params=params, timeout=30)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing"):
                print(f"[miss] {title}", file=sys.stderr)
                return None
            page = pages[0]
            return {
                "article": page["title"],
                "text": page.get("extract", ""),
                "revision_id": page.get("revisions", [{}])[0].get("revid"),
                "url": "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_"),
            }
        except Exception as e:
            wait = 2 ** attempt
            print(f"[retry {attempt + 1}/4 in {wait}s] {title}: {e}", file=sys.stderr)
            time.sleep(wait)
    return None


# --------------------------------------------------------------------------- #
# Cleaning + section splitting
# --------------------------------------------------------------------------- #
HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)


def split_sections(text: str) -> Iterator[tuple[str, str]]:
    """Yield (breadcrumb, body) pairs. Breadcrumb = 'Parent > Child'."""
    matches = list(HEADING_RE.finditer(text))
    stack: list[str] = []

    # Lead paragraph, before the first heading.
    lead_end = matches[0].start() if matches else len(text)
    lead = text[:lead_end].strip()
    if lead:
        yield ("Introduction", lead)

    for i, m in enumerate(matches):
        level = len(m.group(1)) - 1          # '==' -> level 1
        title = m.group(2).strip()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():body_end].strip()

        stack = stack[: level - 1]
        stack.append(title)

        if title.lower() in SKIP_SECTIONS:
            continue
        if not body or len(body) < 120:       # stub sections add noise
            continue
        yield (" > ".join(stack), body)


def clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_section(body: str, count_tokens, max_tokens: int, overlap: int) -> list[str]:
    """Greedy paragraph packing with sentence-level overlap."""
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # A single oversized paragraph: split on sentence boundaries.
        if para_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf, buf_tokens = [], 0
            for s in sentences:
                st = count_tokens(s)
                if buf_tokens + st > max_tokens and buf:
                    chunks.append(" ".join(buf))
                    tail = buf[-2:] if overlap else []
                    buf = tail + [s]
                    buf_tokens = sum(count_tokens(x) for x in buf)
                else:
                    buf.append(s)
                    buf_tokens += st
            if buf:
                chunks.append(" ".join(buf))
            continue

        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            # Overlap: carry the trailing sentences of the last paragraph.
            tail_sents = re.split(r"(?<=[.!?])\s+", current[-1])
            carry, carry_tokens = [], 0
            for s in reversed(tail_sents):
                st = count_tokens(s)
                if carry_tokens + st > overlap:
                    break
                carry.insert(0, s)
                carry_tokens += st
            current = ([" ".join(carry)] if carry else []) + [para]
            current_tokens = carry_tokens + para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if count_tokens(c) >= 40]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_titles(path: Path) -> list[str]:
    titles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            titles.append(line)
    return titles


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the BHASHA Indian History corpus.")
    ap.add_argument("--articles", type=Path, default=Path("data/corpus_articles.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--chunk-tokens", type=int, default=512)
    ap.add_argument("--overlap-tokens", type=int, default=64)
    ap.add_argument("--sleep", type=float, default=0.35, help="Politeness delay between requests")
    args = ap.parse_args()

    count_tokens, _ = get_token_counter()
    titles = load_titles(args.articles)
    print(f"[info] {len(titles)} articles queued")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    args.out.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    total_chunks = 0
    ok_articles = 0

    with args.out.open("w", encoding="utf-8") as fh:
        for i, title in enumerate(titles, 1):
            art = fetch_article(title, session)
            time.sleep(args.sleep)
            if not art or not art["text"]:
                continue
            ok_articles += 1

            slug = re.sub(r"[^a-z0-9]+", "_", art["article"].lower()).strip("_")
            idx = 0
            for section, body in split_sections(art["text"]):
                for piece in chunk_section(clean(body), count_tokens,
                                           args.chunk_tokens, args.overlap_tokens):
                    record = {
                        "chunk_id": f"{slug}::{idx:04d}",
                        "article": art["article"],
                        "url": art["url"],
                        "section": section,
                        "text": piece,
                        "n_tokens": count_tokens(piece),
                        "lang": "en",
                        "revision_id": art["revision_id"],
                        "fetched_at": now,
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    idx += 1
                    total_chunks += 1
            print(f"[{i:3d}/{len(titles)}] {art['article']:<42} {idx:4d} chunks")

    print(f"\n[done] {ok_articles}/{len(titles)} articles -> {total_chunks} chunks -> {args.out}")
    print("[next] python scripts/index_corpus.py --in data/corpus.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
