# CLAUDE.md — BHASHA Project Rules

> This file is loaded automatically at the start of every session. Treat everything
> in it as a standing constraint, not a suggestion. If a user request conflicts with
> a rule here, say so before writing code.

---

## 1. What we are building

**BHASHA** — a self-hostable, real-time **speech-to-speech** voice agent that answers
questions about **Indian history**, grounded in a cited Wikipedia corpus, and replies
in the **same language *and script*** the user spoke.

The five things that make this project non-generic. Never simplify any of them away:

1. **Dual-path ASR with a language router** — Indic speech and English/code-mixed speech go to different engines.
2. **Language Fidelity Contract** — detect → constrain → validate → repair. Output script is enforced, not hoped for.
3. **Cross-lingual grounded RAG** — English corpus, any-language question, cited any-language answer.
4. **Knowledge-graph-augmented retrieval** — a Neo4j graph of entities/relations (rulers, dynasties, battles, dates) that vector retrieval alone can't traverse, used specifically for multi-hop questions.
5. **True barge-in** — cancel LLM stream + flush TTS + rewind conversation state to what was actually *spoken*.

The end customer is a **client who will self-host this inside their own network**.
Every design decision must survive that constraint.

---

## 2. Hard constraints — never violate without asking

| # | Rule | Why |
|---|---|---|
| C1 | **Every ASR / LLM / TTS / embedding call goes through a provider interface** in `server/providers/`. No SDK import outside that folder. | The whole deployment story depends on swapping providers by env var. |
| C2 | **No hosted-only dependency in the critical path.** No Daily transport. Use self-hosted LiveKit. | Client runs this on-prem. SaaS transport is a non-starter. |
| C3 | **Never use Piper TTS.** It was archived in Oct 2025. Use Sarvam Bulbul v3 (API) or Kokoro-82M (local). | Dead upstream. |
| C4 | **No secrets in code.** Everything via `.env` + `pydantic-settings`. `.env` is gitignored; `.env.example` is committed with dummy values. | |
| C5 | **The `onprem` profile must make zero outbound network calls.** If you add a dependency that phones home, flag it. | Air-gap is the selling point. |
| C6 | **All I/O is `async`.** No blocking calls inside the Pipecat pipeline. CPU-bound work goes in `asyncio.to_thread`. | One blocking call destroys the latency budget. |
| C7 | **Never invent a benchmark number.** If something is unmeasured, write `TODO: measure`. Do not write a plausible-looking figure. | The evaluation table is the project's actual contribution. |
| C8 | **Never fabricate a citation.** RAG answers cite `chunk_id` from retrieved context or say they don't know. | |
| C9 | **Never silently stub.** If you can't implement something, `raise NotImplementedError` with a comment and tell the user in your reply. Do not return fake data that makes a test pass. | |
| C10 | **No `localStorage` / `sessionStorage` assumptions in shared code paths.** Session state lives in Postgres. | |
| C11 | **The knowledge graph is additive, never a single point of failure.** Every query must produce a valid answer from vector retrieval alone. If Neo4j is unreachable, log a warning and fall back to vector-only — never fail the request. | Multi-hop enrichment should never take down the whole assistant over a graph outage. |
| C12 | **Graph traversal is only attempted when it's likely to help.** Route to the graph using the question's `type` signal (multi_hop, comparative) — not on every query. Don't query both stores unconditionally "just in case." | Keeps latency predictable and keeps the ablation (vector-only vs vector+graph) meaningful. |

---

## 2a. Developer hardware constraint

The primary developer's machine has **no dedicated GPU (integrated graphics only)
and 16GB RAM**. Consequences, binding for all sessions until stated otherwise:

- **Default `BHASHA_PROFILE=cloud` for all local development.** Never default examples,
  docker compose invocations, or "try this" commands to `hybrid` or `onprem` — CPU-only
  faster-whisper, a local LLM, and local TTS are all too slow on this machine to iterate against.
- Milestone 3 (voice loop) and Milestone 4 (streaming/barge-in) must be built and verified
  against `cloud` (Groq + Sarvam APIs). Do not ask the developer to benchmark local ASR/LLM/TTS
  latency on this machine — the numbers would be meaningless.
- Milestone 6 (`onprem`, air-gapped) is still a required deliverable, but treat it as
  **code that must be correct and reviewable**, not something to be interactively debugged
  on the primary machine. If GPU-equipped hardware becomes available later, flag that
  `onprem` validation should happen there.
- Keep local Docker services light: Postgres + pgvector + LiveKit + Caddy is fine on 16GB.
  Do not add a locally-run LLM or ASR container to any compose profile the developer will
  routinely run.

## 3. Pinned stack — do not substitute silently

```
Python           3.11+
Orchestration    pipecat-ai >= 1.0
Transport        livekit-server (self-hosted, Apache-2.0) + livekit-agents
VAD              silero-vad
ASR (EN/mixed)   whisper-large-v3-turbo  — via Groq API OR faster-whisper (local, INT8)
ASR (Indic)      Sarvam Saarika (API) OR AI4Bharat IndicWhisper (local)
Lang/script ID   fasttext lid.176 + Unicode block histogram
LLM              Groq (llama-4-scout / gpt-oss-120b) OR Ollama/vLLM (qwen3:4b)
Embeddings       Cohere embed-multilingual-v3.0 (default, free tier) | BAAI/bge-m3 (onprem, local)
Reranker         Cohere rerank-multilingual-v3.0 (default, free tier) | bge-reranker-v2-m3 (onprem, local)
Store            PostgreSQL 16 + pgvector (HNSW)
Knowledge graph  Neo4j 5.x Community (self-hosted) OR Neo4j AuraDB Free (cloud)
Backend          FastAPI + uvicorn + websockets
Frontend         React 18 + Vite + AudioWorklet   (NOT ScriptProcessorNode)
Config           pydantic-settings
Lint/format      ruff
Tests            pytest + pytest-asyncio
Tracing          OpenTelemetry + self-hosted Langfuse
```

---

## 4. The provider interface — build to exactly this

`server/providers/base.py` defines these. Every concrete provider implements one.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

@dataclass(frozen=True)
class Transcript:
    text: str
    lang: str            # BCP-47-ish: "en", "hi", "hi-en", "mr"
    script: str          # ISO 15924: "Latn", "Deva", "Beng", "Taml"
    confidence: float
    is_final: bool
    latency_ms: float

@dataclass(frozen=True)
class LangContract:
    lang: str
    script: str
    register: str = "neutral"   # "formal" | "neutral" | "casual"

class ASRProvider(ABC):
    name: str
    @abstractmethod
    async def transcribe(self, pcm: bytes, sample_rate: int,
                         hint_lang: str | None = None) -> Transcript: ...
    @abstractmethod
    async def health(self) -> bool: ...

class LLMProvider(ABC):
    name: str
    @abstractmethod
    async def stream(self, messages: list[dict], contract: LangContract,
                     context: list[dict]) -> AsyncIterator[str]:
        """Yields token deltas. MUST be cancellable via asyncio.CancelledError."""
    @abstractmethod
    async def health(self) -> bool: ...

class TTSProvider(ABC):
    name: str
    @abstractmethod
    async def synthesize(self, text: str, lang: str,
                         voice: str | None = None) -> AsyncIterator[bytes]:
        """Yields PCM chunks. MUST be cancellable mid-stream."""
    @abstractmethod
    async def health(self) -> bool: ...

class EmbeddingProvider(ABC):
    name: str
    @abstractmethod
    async def embed(self, texts: list[str],
                    is_query: bool = False) -> list[list[float]]: ...

@dataclass(frozen=True)
class GraphHit:
    path: list[dict]      # ordered nodes+edges forming the traversal, e.g.
                           # [{"entity": "Shah Jahan"}, {"rel": "FATHER_OF"}, {"entity": "Aurangzeb"}]
    source_chunk_ids: list[str]   # which corpus chunks this triple was extracted from
    confidence: float

class GraphProvider(ABC):
    name: str
    @abstractmethod
    async def query_entity(self, entity_name: str, max_hops: int = 2) -> list[GraphHit]: ...
    @abstractmethod
    async def health(self) -> bool:
        """MUST return False gracefully (no exception) if Neo4j is unreachable — see C11."""
```

Providers are selected by a factory reading env vars. Never `if provider == "groq"`
scattered through business logic — resolve once at startup.

---

## 5. Environment contract

```bash
BHASHA_PROFILE=hybrid            # cloud | hybrid | onprem

ASR_EN_PROVIDER=groq             # groq | local
ASR_INDIC_PROVIDER=sarvam        # sarvam | local
LLM_PROVIDER=groq                # groq | ollama | vllm
TTS_PROVIDER=sarvam              # sarvam | kokoro
EMBED_PROVIDER=cohere            # cohere (default, cloud, free tier) | local (bge-m3, CPU-slow)
RERANK_PROVIDER=cohere           # cohere (default, cloud, free tier) | local (bge-reranker, CPU-slow)
COHERE_API_KEY=

GROQ_API_KEY=
SARVAM_API_KEY=
DATABASE_URL=postgresql+asyncpg://bhasha:bhasha@localhost:5432/bhasha
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

VAD_SILENCE_MS=600
RETRIEVE_TOP_K=20
RERANK_TOP_K=5
FIDELITY_SCRIPT_THRESHOLD=0.15   # foreign-script ratio that triggers repair

GRAPH_PROVIDER=neo4j_local        # neo4j_local | neo4j_aura | none
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
GRAPH_MAX_HOPS=2
GRAPH_TRIGGER_TYPES=multi_hop,comparative   # question types that attempt a graph query
```

`onprem` profile forces `GRAPH_PROVIDER=neo4j_local` (self-hosted container) — never
`neo4j_aura`, since that's a cloud dependency and would break the air-gap guarantee.

Setting `BHASHA_PROFILE=onprem` must **override** every `*_PROVIDER` to a local value,
regardless of what else is in `.env`. Fail loudly at startup if a cloud key is set
while profile is `onprem`.

---

## 6. Repository layout

```
bhasha/
├── docker-compose.yml          # profiles: cloud | hybrid | onprem
├── .env.example                # committed, dummy values
├── server/
│   ├── main.py                 # FastAPI app + LiveKit token endpoint
│   ├── config.py                # pydantic-settings, profile resolution
│   ├── pipeline.py             # Pipecat pipeline assembly
│   ├── barge_in.py             # cancellation bus + conversation state rewind
│   ├── providers/              # C1 lives here — no SDK imports outside
│   │   └── graph_neo4j.py      # implements GraphProvider, C11 fallback baked in
│   ├── lang/
│   │   ├── detect.py           # fastText + Unicode block histogram
│   │   └── contract.py         # detect → constrain → validate → repair
│   └── rag/
│       ├── retrieve.py         # hybrid BM25 + dense → rerank
│       ├── graph_retrieve.py   # entity extraction from question → graph traversal → merge
│       ├── prompt.py           # citation-forced answer template, cites chunks AND graph paths
│       └── schema.sql          # chunks, embeddings, conversations, turns
├── web/                        # React + Vite + AudioWorklet
├── scripts/
│   ├── build_corpus.py         # DONE — do not rewrite
│   ├── build_graph.py          # LLM-based entity/relation extraction → Neo4j
│   ├── index_corpus.py
│   └── evaluate.py
├── data/
│   ├── corpus_articles.txt     # DONE — 80 articles
│   └── golden_set.jsonl        # DONE — 50 stratified questions
└── tests/
```

`scripts/build_corpus.py`, `data/corpus_articles.txt` and `data/golden_set.jsonl`
already exist and are tested. **Read them before writing anything that consumes them.**
Do not regenerate or "improve" them unless asked.

---

## 7. Working style

**The user is a final-year student who is new to most of this stack. So:**

- **Do the task first, then explain — and only explain if asked.** Ship working code, then stop.
- Work in **one milestone at a time**. Do not jump ahead to milestone 4 while milestone 2 is unverified.
- After every milestone, print the **exact command** the user should run to verify it, and what they should see.
- Keep files under ~300 lines. Split when they grow.
- Comment the *why*, not the *what*. Assume the reader knows Python but not Pipecat, pgvector, or WebRTC.
- When you make a judgement call the user didn't specify, say so in one line at the end. Don't bury it.

**Ask before proceeding when:**
- A pinned dependency doesn't exist at the expected version
- A hard constraint (§2) would need to be broken
- The request is ambiguous in a way that changes the schema or an interface
- You're about to add a new service to `docker-compose.yml`

**Never do these:**
- ❌ Write a `try/except: pass`
- ❌ Return mock data to make a test green
- ❌ Add a dependency not in §3 without saying why
- ❌ Rewrite a file that already works because you'd have structured it differently
- ❌ Claim something is measured/tested when you didn't run it
- ❌ Produce a wall of explanation when the user asked for code

---

## 8. Definition of done

A milestone is done when **all** of these hold:

1. `ruff check .` passes
2. `pytest` passes, and the tests actually exercise the new code
3. The verification command in the milestone's acceptance criteria runs and produces the stated output
4. No `NotImplementedError` remains in the milestone's scope
5. `.env.example` lists any new variable
6. The README's repo-layout status markers are updated (`⬜` → `✅`)
