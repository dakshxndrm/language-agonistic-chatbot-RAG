# BUILD_PROMPT.md — How to drive the BHASHA build

> **How to use this file.**
> 1. Put `CLAUDE.md` at the repo root. Claude Code reads it automatically every session.
> 2. Paste **Prompt 0** to start.
> 3. After each milestone verifies, paste the next prompt. **One milestone per session** — start a fresh session between them so context stays clean.
> 4. If Claude drifts, paste the **Course-Correction** block at the bottom.

---

## 🚀 Prompt 0 — Kickoff

```
You are building BHASHA, a self-hostable real-time speech-to-speech voice agent
that answers questions about Indian history from a cited Wikipedia corpus, and
replies in the same language AND script the user spoke.

Read CLAUDE.md first. It contains hard constraints, the pinned stack, the provider
interface, and the env contract. Everything in it is binding.

Also read these three files, which already exist and are tested. Do not rewrite them:
  - scripts/build_corpus.py
  - data/corpus_articles.txt
  - data/golden_set.jsonl

Your job in THIS session is Milestone 0 only. Do not start Milestone 1.

MILESTONE 0 — Foundation
Deliverables:
  1. docker-compose.yml with three profiles (cloud, hybrid, onprem) containing:
     postgres:16 with pgvector, livekit-server (self-hosted), and a caddy sidecar
     terminating TLS with a self-signed cert on :8443. Only the services each
     profile needs.
  2. .env.example with every variable from CLAUDE.md §5, dummy values.
  3. server/config.py using pydantic-settings. It must resolve BHASHA_PROFILE into
     concrete provider choices, and FAIL LOUDLY at startup if profile=onprem while
     any cloud API key is set.
  4. server/rag/schema.sql — tables for: chunks (with pgvector column + HNSW index +
     a tsvector column for BM25), conversations, turns. Include the columns emitted
     by build_corpus.py — read it to get them exactly right.
  5. server/main.py — minimal FastAPI app with GET /health that reports the active
     profile and the resolved provider for each stage.
  6. pyproject.toml with the pinned deps, ruff config, pytest config.
  7. tests/test_config.py proving the onprem override and the fail-loud behaviour.

ACCEPTANCE — tell me these exact commands and what I should see:
  docker compose --profile hybrid up -d
  curl -s localhost:8000/health | jq
  pytest -q
  ruff check .

Constraints reminder: no SDK imports outside server/providers/ (that folder is empty
this milestone), no Daily, no Piper, no secrets in code, all I/O async.

Start by listing the files you will create and any question you have. Then build.
```

---

## 🕸️ Prompt 0.5 — Knowledge Graph (Daksh)

```
Read CLAUDE.md. Milestone 0 is verified and working. Build Milestone 0.5 only.
This runs BEFORE retrieval (Milestone 1) by explicit decision — accept that this
milestone carries the most schedule risk in the project and keep scope tight.

MILESTONE 0.5 — Knowledge graph of entities and relations
Goal: augment retrieval for multi-hop questions (rulers, dynasties, battles, dates)
that plain vector search answers poorly. This is ADDITIVE — vector retrieval must
keep working standalone with zero dependency on this milestone (see CLAUDE.md C11).

  1. docker-compose.yml — add a `neo4j` service (Neo4j 5.x Community) to the
     `hybrid` and `onprem` profiles ONLY. Never `cloud` — see CLAUDE.md C11/C12.
     Expose bolt://7687. Persist data in a named volume.
  2. server/providers/graph_neo4j.py — implements GraphProvider from CLAUDE.md §4.
     health() must return False (not raise) if Neo4j is unreachable. Prove this
     with a test that stops the container and calls health().
  3. scripts/build_graph.py — reads data/corpus.jsonl (from build_corpus.py, already
     built), runs an LLM extraction pass per chunk to pull (entity, relation, entity)
     triples — e.g. (Shah Jahan, FATHER_OF, Aurangzeb), (Aurangzeb, IMPRISONED, Shah Jahan).
     Requirements:
       - Store source chunk_id(s) on every relationship, so graph facts stay citable.
       - Deduplicate entities by normalized name (case, diacritics) before writing —
         "Akbar" and "akbar" must not become two nodes.
       - Idempotent and resumable, same as index_corpus.py.
       - Log extraction failures per chunk; do not silently drop them.
       - This uses an LLM (LLMProvider from Milestone 0's stub is fine to call here,
         or a direct Groq call if the provider isn't wired yet — your call, note which).
  4. server/rag/graph_retrieve.py — given a question, extract the entity name(s)
     mentioned (simple NER or LLM-based, your choice — document it), query the graph
     up to GRAPH_MAX_HOPS, and return GraphHit objects. Only invoked when the
     question's type is in GRAPH_TRIGGER_TYPES (multi_hop, comparative) — do not
     query the graph on every request (C12).
  5. A small eval: use the golden_set.jsonl questions where "type" is multi_hop or
     comparative (there are 9 of these — check the file). For each, run graph
     traversal and report whether it surfaced the correct connecting entity/relation.
     This is a small, honest signal, not a full benchmark — report it as such.

ACCEPTANCE:
  docker compose --profile hybrid up -d
  python scripts/build_graph.py
  → cypher-shell or Neo4j Browser at localhost:7474 shows populated nodes/relationships
  pytest tests/test_graph_provider.py -v   # includes the "Neo4j down → health() False" test
  python scripts/eval_graph.py
  → prints hit/miss per multi-hop/comparative golden question

Do NOT touch server/rag/retrieve.py (vector retrieval) or any ASR/LLM/TTS provider
this session — those are later milestones. If entity extraction quality is poor on
more than ~30% of chunks, stop and report that honestly rather than tuning prompts
indefinitely — flag it as a finding for the report.
```

---

## 🔎 Prompt 1 — Retrieval (Daksh)

```

Read CLAUDE.md. Milestones 0 and 0.5 are verified and working. Build Milestone 1 only.

MILESTONE 1 — Cross-lingual retrieval (vector side)
  1. server/providers/base.py — the exact ABCs from CLAUDE.md §4, including
     GraphProvider (Milestone 0.5 already implements it against this — if the file
     exists, extend it, don't recreate it).
  2. server/providers/embed_cohere.py — Cohere embed-multilingual-v3.0 API,
     implementing EmbeddingProvider. This is the DEFAULT (EMBED_PROVIDER=cohere) —
     no local model, no CPU bottleneck, dev machine has no GPU. Also add
     server/providers/embed_local.py (BAAI/bge-m3 via sentence-transformers) but
     ONLY as the onprem fallback — do not make it the default path.
  3. server/providers/rerank_cohere.py — Cohere rerank-multilingual-v3.0 API. Same
     pattern: default, cloud, free tier. Add rerank_local.py (bge-reranker-v2-m3)
     as the onprem-only fallback, not the default.
  4. scripts/index_corpus.py — reads data/corpus.jsonl, embeds via the Cohere
     embedding provider (batched — Cohere's API accepts up to 96 texts/call, use
     that batch size), upserts into the chunks table. Idempotent: re-running must
     not duplicate rows. Must be resumable if it crashes halfway. This should now
     run in well under a minute for 2371 chunks, not 35 — if it's still slow,
     something's wrong, don't just accept it.
  5. server/rag/retrieve.py — hybrid retrieval:
       - dense: pgvector cosine, top-K = RETRIEVE_TOP_K
       - sparse: Postgres full-text BM25 over the same table
       - fuse with Reciprocal Rank Fusion
       - rerank via the Cohere rerank provider down to RERANK_TOP_K
     Return chunk_id, article, section, text, and score for each hit.
     This function does NOT call the graph — that fusion happens in a thin
     orchestrator, not inside retrieve.py, so retrieve.py stays testable standalone.
  6. server/rag/orchestrate.py — thin wrapper: calls retrieve.py always; calls
     graph_retrieve.py (Milestone 0.5) only when question type is in
     GRAPH_TRIGGER_TYPES; merges both into one context list for the prompt.
  7. scripts/eval_retrieval.py — runs every question in data/golden_set.jsonl that
     has a non-empty gold_articles list, and reports context-recall@5 BROKEN DOWN
     BY the `flavour` field AND separately by vector-only vs. vector+graph for the
     multi_hop/comparative subset. This dual view is the point — it's your first
     real ablation.

ACCEPTANCE:
  python scripts/build_corpus.py
  python scripts/index_corpus.py
  python scripts/eval_retrieval.py
  → prints per-flavour recall@5, plus the vector-only vs vector+graph split

Do NOT touch ASR, TTS, LLM, or the voice pipeline this session.
Report the real recall numbers you measured. If something scores badly, say so —
do not tune the test to make it look good.
```


---

## 🔒 Prompt 2 — Language Fidelity Contract (Daksh)

```
Read CLAUDE.md. Milestones 0-1 verified. Build Milestone 2 only.
This is the project's core technical contribution. Take it seriously.

MILESTONE 2 — detect → constrain → validate → repair
  1. server/lang/detect.py
       detect(text) -> (lang, script, confidence)
       - script via Unicode block histogram (Latn / Deva / Beng / Taml / Gujr / Telu ...)
       - language via fastText lid.176
       - CRITICAL: Hindi and Hinglish share language id "hi". They are distinguished
         ONLY by script. Romanised Hindi must resolve to lang="hi-en", script="Latn".
         Write tests that lock this behaviour in.
  2. server/lang/contract.py
       build_contract(transcript) -> LangContract
       constrain(contract) -> str            # the system-prompt fragment
       validate(text, contract) -> Violation | None
       repair(text, contract, llm) -> str    # ONE retry, then give up and log
     Violation triggers when the foreign-script character ratio exceeds
     FIDELITY_SCRIPT_THRESHOLD.
  3. server/providers/llm_groq.py and llm_ollama.py implementing LLMProvider.
     stream() must be cancellable — an asyncio.CancelledError has to actually stop
     the HTTP stream, not just abandon the coroutine. Test this explicitly.
  4. server/rag/prompt.py — answer template that forces inline citations of chunk_id
     and forces an explicit "I don't know from my sources" when context is weak.
  5. scripts/eval_fidelity.py — runs all 50 golden questions text-only (no audio yet)
     and reports:
       - Language Fidelity Rate per flavour
       - false-premise rejection: 4/4?
       - out-of-scope refusal, AND whether the refusal stayed in the user's language

ACCEPTANCE:
  pytest tests/test_detect.py -v
  python scripts/eval_fidelity.py
  → LFR table per flavour + trap results

Report real numbers. An LFR below target is a finding, not a failure to hide.
```

---

## 🎙️ Prompt 3 — Voice loop, turn-based (Arpit)

```
Read CLAUDE.md. Milestones 0-2 verified. Build Milestone 3 only.
Turn-based (push-to-talk) ONLY. Streaming and barge-in are Milestone 4 — do not start them.

MILESTONE 3 — end-to-end voice, turn-based
  1. server/providers/asr_groq.py — whisper-large-v3-turbo via Groq
     server/providers/asr_local.py — faster-whisper INT8
     server/providers/asr_sarvam.py — Sarvam Saarika
     All implement ASRProvider and populate Transcript.lang + .script using
     server/lang/detect.py. Do not trust the ASR's own language label alone.
  2. server/providers/tts_sarvam.py (Bulbul v3) and tts_kokoro.py (local).
     Both implement TTSProvider and yield PCM chunks.
  3. server/asr_router.py — runs a cheap language-ID pass on the first ~1s of audio
     and routes to the Indic or the English/mixed ASR. Document the routing rule in
     a docstring. Log every routing decision so it can be audited later.
  4. server/pipeline.py — Pipecat pipeline wired onto self-hosted LiveKit transport:
     mic → VAD (silero, VAD_SILENCE_MS) → router → ASR → contract → RAG → LLM →
     TTS → speaker. Turn-based: finalise on silence, then process.
  5. web/ — minimal React + Vite client. Push-to-talk button, LiveKit connection,
     AudioWorklet capture at 16kHz mono. Show the transcript, the detected
     lang/script, the answer text, and the cited sources. Ugly is fine. Working is not optional.
  6. Per-stage latency logged via OpenTelemetry.

ACCEPTANCE:
  docker compose --profile cloud up -d
  open https://localhost:8443
  → hold the button, ask "Ashoka ne Kalinga ke baad kya kiya?" in Hinglish
  → get a spoken Hinglish answer with visible citations
  → server logs show a per-stage latency breakdown

Note: dev machine has no dedicated GPU (integrated graphics, 16GB RAM). Use the
`cloud` profile (Groq + Sarvam) for this milestone, not hybrid/onprem — local
ASR/LLM/TTS would be too slow to iterate against on this hardware.

Report the actual measured TTFA. Do not estimate it.
```

---

## ⚡ Prompt 4 — Streaming + barge-in (Chayan)

```
Read CLAUDE.md. Milestones 0-3 verified. Build Milestone 4 only.

MILESTONE 4 — streaming and true barge-in
  1. server/pipeline.py — convert to streaming:
       - partial ASR transcripts drive early retrieval
       - LLM output is chunked on sentence boundaries and sent to TTS immediately
       - audio starts before generation finishes
  2. server/barge_in.py — the cancellation bus. When VAD detects user speech while
     the agent is speaking:
       a. cancel the in-flight LLM stream (real HTTP cancellation)
       b. flush the TTS queue and stop playback
       c. REWIND conversation state to what was ACTUALLY SPOKEN ALOUD, not what was
          generated. Track spoken-character offsets from TTS chunk acks to do this.
     Step (c) is the part everyone gets wrong. It is the reason this milestone exists.
  3. tests/test_barge_in.py — simulate an interrupt at 30%, 60% and 90% through a
     response. Assert conversation state contains only the spoken prefix, and that
     no orphaned LLM or TTS task is still running afterwards.
  4. Instrument TTFA (p50/p95) and barge-in latency. Log both per turn.

ACCEPTANCE:
  pytest tests/test_barge_in.py -v
  → live: interrupt the agent mid-sentence, then ask a follow-up that depends on
    what it had said BEFORE you cut it off. The follow-up must be answered correctly.
  → report measured TTFA p50/p95 and barge-in latency

Targets are TTFA p50 < 700ms and barge-in < 100ms. If you miss them, report the real
numbers and where the time went. Do not adjust the measurement to hit the target.
```

---

## 🧪 Prompt 5 — Full evaluation (All)

```
Read CLAUDE.md. Milestones 0-4 verified. Build Milestone 5 only.

MILESTONE 5 — the results section
  1. scripts/evaluate.py — one command, full harness over data/golden_set.jsonl:
       - text-mode for RAG metrics; audio-mode via TTS-synthesised input for WER/CER
       - metrics: WER/CER per flavour, Language Fidelity Rate, RAGAS faithfulness,
         RAGAS answer relevance, context recall@5, false-premise rejection,
         out-of-scope refusal, TTFA p50/p95, barge-in latency
  2. ABLATIONS — this is the actual contribution. Run and tabulate:
       - single Whisper-only ASR   vs  routed dual-path ASR
       - no fidelity contract       vs  contract enabled
       - no RAG  vs  dense-only RAG  vs  hybrid+rerank RAG
  3. Emit results to results/YYYY-MM-DD.json AND a markdown table for the report.
  4. Update README.md: replace the 🎯 Target column with measured values, and flip
     the repo-layout status markers.

ACCEPTANCE:
  python scripts/evaluate.py --all --ablations
  → results/*.json + a printed markdown table

Report every number exactly as measured, including the bad ones. The deltas between
ablation arms ARE the project's finding — a small or negative delta is still a result
and must be reported honestly.
```

---

## 🔐 Prompt 6 — Air-gap hardening (All)

```
Read CLAUDE.md. Milestones 0-5 verified. Build Milestone 6 only.

Note: the dev machine has no dedicated GPU and cannot run this profile end-to-end
at usable speed. Build and unit-test this milestone for correctness. Live air-gap
verification (the final ACCEPTANCE step) happens on GPU-equipped hardware when
available — say so explicitly rather than attempting it on this machine.

MILESTONE 6 — onprem profile, provably air-gapped
  1. Make `docker compose --profile onprem up` work with NO internet access at all.
     Models must be pre-baked into images or mounted from a local volume — no
     HuggingFace download at runtime.
  2. scripts/preflight.py — verifies before startup: models present, DB reachable,
     GPU/VRAM sufficient, and that no cloud provider is selected.
  3. Provider fallback: if a cloud provider errors or times out, fall back to the
     local equivalent and log the switch. Test by pointing a key at a dead endpoint.
  4. docs/INSTALL.md — a runbook a client's sysadmin can follow without asking us
     anything. Include hardware requirements and a rollback procedure.
  5. Prove the air-gap: run the stack with networking disabled and complete a full
     voice turn.

ACCEPTANCE:
  python scripts/preflight.py
  docker compose --profile onprem up -d
  → disconnect the network, complete a voice conversation, show no outbound attempts
    (capture with tcpdump or docker network inspect)

This milestone is the client sales pitch. It has to actually be true.
```

---

## 🔧 Course-Correction Block

Paste this verbatim if Claude drifts, over-explains, or starts inventing.

```
Stop. Re-read CLAUDE.md §2 (hard constraints) and §7 (working style).

Then answer these, briefly:
1. Which hard constraint did the last change violate, if any?
2. Is there any code you wrote that returns mock/placeholder data, or any number you
   reported that you did not actually measure? Name it.
3. Are we still inside the current milestone's scope, or did you start the next one?

Then fix only what's broken. Do not refactor anything that already works, and do not
explain unless I ask.
```

---

## 📌 Session hygiene

| Do | Don't |
|---|---|
| One milestone per session | Chain three milestones in one context window |
| `/clear` between milestones | Let 200k tokens of old build logs accumulate |
| Commit after every verified milestone | Build for two weeks then commit once |
| Paste the real error output | Paraphrase the error from memory |
| Say "it's still broken, here's the log" | Say "it doesn't work" |