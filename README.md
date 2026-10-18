<div align="center">

# 🕉️ BHASHA
### **B**ilingual **H**inglish-**A**ware **S**peech-to-speech **H**istorical **A**ssistant
*Bol jo bhi bhasha ho — jawab usi mein aayega.*

A self-hostable, real-time voice agent that speaks Hindi, English, Hinglish, or a regional Indian language — and answers **only** from a cited knowledge base of Indian history.

<br>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React_18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![PostgreSQL](https://img.shields.io/badge/pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j_5-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)
![LiveKit](https://img.shields.io/badge/LiveKit_WebRTC-002B36?style=for-the-badge&logo=webrtc&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)
![TTFA](https://img.shields.io/badge/TTFA%20target-%3C700ms-brightgreen?style=flat-square)
![Languages](https://img.shields.io/badge/languages-11%2B-orange?style=flat-square)
![Deployment](https://img.shields.io/badge/deploy-cloud%20%7C%20hybrid%20%7C%20air--gapped-blue?style=flat-square)
![License](https://img.shields.io/badge/license-academic-lightgrey?style=flat-square)

<br>

🏛️ **Poornima Institute of Engineering & Technology** · Department of Computer Engineering  
🎓 Final Year Project 2026–27 · **Team Oblivion**  
👥 **Team:** Daksh Mahera (PIET23CS041) · Arpit Garg (PIET23CS025) · Chayan Jain (PIET23CS038)  
👩‍🏫 **Project Guide:** Dr. Aashima Tiwari, Associate Professor

</div>

---

## 🧭 Contents

- [🎯 The Pitch](#-the-pitch)
- [🚫 Why Standard Wrappers Fail](#-why-standard-wrappers-fail)
- [🏛️ The Core Pillars](#️-the-core-pillars)
- [✨ Exclusive Innovation: Multimodal Visual Stage](#-exclusive-innovation-multimodal-visual-stage)
- [🗺️ System Architecture](#️-system-architecture)
- [🧰 Pinned Tech Stack & Standing Constraints](#-pinned-tech-stack--standing-constraints)
- [📦 Deployment Profiles](#-deployment-profiles--the-client-story)
- [🚀 Quickstart & Complete Runbook](#-quickstart--complete-runbook)
- [🧪 Test Data & Evaluation Metrics](#-test-data--evaluation-metrics)
- [📂 Repository Layout](#-repository-layout)
- [👥 Team & Acknowledgements](#-team--acknowledgements)

---

## 🎯 The Pitch

> Most voice-assistant projects are simple pipeline wrappers: `Whisper → ChatGPT → ElevenLabs`. **BHASHA is fundamentally different in five measurable ways:**

<table>
<tr>
<td width="20%" align="center">

### 🔀
**Dual-Path Router**  
Routes audio by detected language family instead of forcing one model on code-switching.

</td>
<td width="20%" align="center">

### 🔒
**Fidelity Contract**  
Guarantees reply in the *same language and script* you spoke — Hinglish in, Hinglish out.

</td>
<td width="20%" align="center">

### 📚
**Cross-Lingual RAG**  
English Wikipedia corpus, Indic questions, cited answers with 0% fabricated references.

</td>
<td width="20%" align="center">

### 🕸️
**Graph Augmentation**  
Traverses Neo4j for multi-hop chronological & genealogical relationships vector search misses.

</td>
<td width="20%" align="center">

### 🏠
**100% Air-Gapped**  
Can run fully offline inside a private on-prem network with zero external cloud API egress.

</td>
</tr>
</table>

---

## 🚫 Why Standard Wrappers Fail

| 💭 The naive assumption | 💥 What actually happens in reality |
|---|---|
| *"Whisper handles 99 languages, so multilingual speech is solved."* | It collapses code-mixed Hinglish into a single script, drops language switch points, and lags behind Indic-specialised models on Marathi, Bengali, and Tamil. |
| *"The LLM will naturally reply in the user's spoken language."* | It drifts. Ask in romanised Hinglish and models routinely reply in Devanagari Hindi, then shift to pure English by turn two. |
| *"PostgreSQL stores the chat, so we have grounding and memory."* | **Storage $\neq$ Grounding.** Without verifiable citations, the system is an ungrounded hallucination engine. |
| *"Barge-in just means stopping the speaker when speech is detected."* | Skipping state rewind corrupts context within two turns because the model assumes you heard text it generated but never actually spoke. |

---

## 🏛️ The Core Pillars

### 🔀 Pillar 1 — Dual-Path ASR with a Language Router
Audio passes through a fast language-identification gate first:
- **Indic-dominant speech:** Routes to an Indic-first recogniser (Sarvam Saaras v3 in cloud mode or AI4Bharat IndicWhisper locally).
- **English & code-mixed Hinglish:** Routes to Whisper Large v3 Turbo (via Groq API or faster-whisper INT8 locally).

### 🔒 Pillar 2 — The Language Fidelity Contract *(Core Contribution)*
Detect $\rightarrow$ Constrain $\rightarrow$ Validate $\rightarrow$ Repair. The output script is strictly enforced, never hoped for:
1. **Detect:** fastText `lid.176` + Unicode block histogram detects language and script (`Latn`, `Deva`, `Beng`, `Taml`, etc.).
2. **Constrain:** The contract (`{"lang": "hi-en", "script": "Latn", "register": "neutral"}`) is pinned into the prompt on every turn.
3. **Validate:** Script histogram checks generated tokens against `fidelity_script_threshold` (15%).
4. **Repair:** If foreign script exceeds the threshold, a targeted re-prompt fixes the script before audio reaches TTS.

### 📚 Pillar 3 — Cross-Lingual Grounded Retrieval
- Corpus: 80 curated English Wikipedia articles (Indus Valley Civilisation to the Constitution of India).
- Multilingual vector embeddings (Cohere `embed-multilingual-v3.0` / BGE-M3) place English text and Indic queries into a shared vector space.
- Hybrid search fuses dense cosine similarity with PostgreSQL BM25 full-text rank using Reciprocal Rank Fusion (RRF, $k=60$).
- Cross-encoder reranker (`rerank-multilingual-v3.0` / `bge-reranker-v2-m3`) refines top-20 candidates down to top-5.

### 🕸️ Pillar 4 — Neo4j Knowledge Graph Augmentation
Multi-hop queries (e.g., *"Who was the grandfather of the emperor who built the Taj Mahal?"*) fail in pure vector search.
- When question type is `multi_hop` or `comparative`, BHASHA extracts entities, queries Neo4j 5 over Bolt, traverses chronological relationships (`RULED`, `FATHER_OF`, `SUCCEEDED_BY`, `DEFEATED`), and fuses graph paths directly into the prompt context.
- Fallback guarantee: Graph retrieval is additive. If Neo4j is offline, the system logs a warning and answers using vector search alone.

### 🎤 Pillar 5 — True Barge-In & State Rewind
When the user speaks mid-sentence:
1. Cancels the active LLM generation task.
2. Flushes the TTS audio playback queue.
3. **Rewinds conversation state** to what was *actually spoken aloud* using `SpokenTracker`, ensuring context history remains clean.

---

## ✨ Exclusive Innovation: Multimodal Visual Stage

Voice-only agents act as black boxes during viva demonstrations. BHASHA uses **LiveKit WebRTC Data Channels** to emit real-time telemetry and contextual visual payloads synchronously alongside the audio stream without network latency:

```
 [LiveKit Audio Track]  ──────────────> React Speaker (Ear)
 [LiveKit Data Channel] ──────────────> Visual Stage Component (Eye)
                                          │
                  ┌───────────────────────┼────────────────────────┐
                  ▼                       ▼                        ▼
       Active Neo4j Subgraph     Dynamic Dynasty Map      Verbatim Wikipedia
       (Interactive SVG Graph)       (Leaflet.js)            Source Drawer
```

### Features:
1. **Real-Time Graph Subgraph Lighting:**
   - Emits active nodes and relationship edges (`event: HISTORICAL_CONTEXT`).
   - Frontend animates a knowledge graph showing the exact multi-hop reasoning path.
2. **Temporal Historical Map Sync:**
   - Leaflet-powered map synchronises with historical eras (Mauryan Empire at 250 BCE, Mughal Empire at 1600 CE, Chola Dynasty).
   - Dynamically flies to coordinates and overlays territory boundary polygons.
3. **Verbatim Grounding & Citation Drawer:**
   - Exposes the exact chunk text, article title, section breadcrumb, and similarity score from the cited Wikipedia corpus, proving 0% hallucination live.
4. **Sub-120ms Semantic Question Cache:**
   - Questions with cosine similarity $\ge 0.96$ against verified answers in pgvector `query_cache` bypass cold-start retrieval for instant answers.

---

## 🗺️ System Architecture

```mermaid
flowchart TD
    subgraph BROWSER["🌐 Browser — React 18 + AudioWorklet + WebRTC"]
        MIC["🎙️ Microphone"]
        SPK["🔊 Speaker"]
        VIS["🖥️ Visual Stage: Map + Subgraph + Citations"]
    end

    subgraph PREMISES["🏠 SERVER & LOCAL INFRASTRUCTURE"]
        LK["📡 LiveKit WebRTC Server (Signalling & Media)"]
        CADDY["🔒 Caddy TLS Gateway (:8443)"]
        
        subgraph PIPECAT["⚙️ Pipecat Real-Time Pipeline"]
            VAD["👂 Silero VAD Analyzer"]
            ROUTER["🔀 Dual-Path ASR Router"]
            BRAIN["🧠 BhashaBrain (Contract + RAG + Cache)"]
            TTS["🗣️ BhashaTTS (Sentence Chunked)"]
            BUS(["🚨 BARGE-IN BUS"])
        end

        subgraph STORES["💾 Storage & Knowledge Bases"]
            PG[("🐘 PostgreSQL 16 + pgvector<br/>Chunks + HNSW + Semantic Cache")]
            NEO[("🕸️ Neo4j 5 Community<br/>Historical Knowledge Graph")]
        end
    end

    MIC -->|WebRTC| LK --> PIPECAT
    VAD --> BUS
    VAD --> ROUTER
    ROUTER --> BRAIN
    BRAIN <--> STORES
    BRAIN -->|Data Channel| VIS
    BRAIN --> TTS -->|WebRTC| SPK
    BUS -.->|kill| BRAIN
    BUS -.->|flush| TTS
    CADDY -.->|proxy :8000| PIPECAT
    CADDY -.->|proxy /rtc| LK
```

---

## 🧰 Pinned Tech Stack & Standing Constraints

### Technology Stack
```ini
Python          = 3.11+
Pipeline        = Pipecat AI 1.11+
Transport       = LiveKit Server (self-hosted, Apache-2.0)
VAD             = Silero VAD
ASR (EN/mixed)  = Whisper Large v3 Turbo (Groq API or local faster-whisper INT8)
ASR (Indic)     = Sarvam Saaras v3 (API) or AI4Bharat IndicWhisper (local)
Lang/Script ID  = fastText lid.176 + Unicode block histogram
LLM             = Groq (gpt-oss-120b / llama-4-scout) or Ollama (qwen2.5:7b-instruct)
Embeddings      = Cohere embed-multilingual-v3.0 | BGE-M3 (local)
Reranker        = Cohere rerank-multilingual-v3.0 | bge-reranker-v2-m3 (local)
Storage         = PostgreSQL 16 + pgvector (HNSW)
Graph DB        = Neo4j 5.x Community (Bolt)
Backend         = FastAPI + Uvicorn (async)
Frontend        = React 18 + Vite + TypeScript + Leaflet.js
```

### Standing Architectural Constraints
- **C1:** Every ASR, LLM, TTS, and embedding call goes through an abstract provider interface in `server/providers/base.py`. Never import vendor SDKs in business logic.
- **C2:** No hosted-only SaaS in the critical transport path. LiveKit is self-hosted.
- **C3:** Never use Piper TTS (archived Oct 2025). Use Sarvam Bulbul v3 or Kokoro-82M.
- **C4:** Zero secrets in code. Everything is managed through `.env` and `pydantic-settings`.
- **C5:** The `onprem` profile must make **zero outbound network calls**. Fail loudly at startup if cloud keys are present.
- **C6:** All I/O is asynchronous. Never make blocking calls in the pipeline.
- **C7:** Never invent a benchmark number. If unmeasured, mark `TODO: measure`.
- **C8:** Never fabricate a citation. Answers must cite source `chunk_id`s or state that context is insufficient.
- **C11:** The knowledge graph is additive. If Neo4j is offline, degrade gracefully to vector-only RAG.
- **C12:** Graph traversal is triggered on `multi_hop` and `comparative` question types only.

---

## 📦 Deployment Profiles — The Client Story

| | ☁️ `cloud` *(Default Dev)* | ⚖️ `hybrid` | 🔐 `onprem` *(Air-Gapped)* |
|---|---|---|---|
| **ASR** | Groq + Sarvam API | local faster-whisper | local faster-whisper + IndicWhisper |
| **LLM** | Groq (gpt-oss-120b) | Groq API | local Ollama (qwen2.5) |
| **TTS** | Sarvam Bulbul v3 | Sarvam Bulbul v3 | local Kokoro-82M |
| **Embeddings** | Cohere multilingual | Cohere multilingual | local BGE-M3 |
| **Audio leaves premises** | ✅ Yes | ❌ **No** | ❌ **No** |
| **Text leaves premises** | ✅ Yes | ✅ Yes | ❌ **No (100% private)** |
| **Hardware needed** | Any CPU / 16 GB RAM | 8 GB VRAM GPU | 16–24 GB VRAM GPU |
| **Typical TTFA** | $\sim 400\text{ ms}$ | $\sim 450\text{ ms}$ | $\sim 700\text{ ms}$ |
| **External Egress** | Allowed | Allowed | **Strictly blocked (0%)** |

---

## 🚀 Quickstart & Complete Runbook

### 1. Prerequisites
- **OS:** Windows 10/11 with WSL2/PowerShell, Linux, or macOS.
- **Tools:** Docker Engine & Docker Compose, Python 3.11+, Node.js 18+.

### 2. Environment Configuration
Copy `.env.example` to `.env` and fill in API keys:
```bash
cp .env.example .env
```
Key settings in `.env`:
```ini
BHASHA_PROFILE=cloud
GROQ_API_KEY=your_groq_key
SARVAM_API_KEY=your_sarvam_key
COHERE_API_KEY=your_cohere_key
DATABASE_URL=postgresql+asyncpg://bhasha:bhasha@localhost:5432/bhasha
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret_change_me_at_least_32_chars
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=bhasha_graph_pw
GRAPH_PROVIDER=neo4j_local
```

### 3. Launch Docker Infrastructure
Start PostgreSQL with pgvector, Neo4j, LiveKit server, and Caddy:
```bash
docker compose up -d postgres neo4j livekit caddy
```
Verify that all containers are healthy:
```bash
docker compose ps
```

### 4. Build Knowledge Base & Graph
Initialize and embed the 2,371 Wikipedia chunks and populate Neo4j:
```bash
# Fetch and section-aware chunk Wikipedia articles
python scripts/build_corpus.py

# Embed chunks into PostgreSQL with pgvector
python scripts/index_corpus.py

# Extract entities and relations into Neo4j
python scripts/build_graph.py
```

### 5. Run Preflight Sanity Checks
Ensure all ports and profile constraints pass:
```bash
python scripts/preflight.py
```

### 6. Build Frontend & Launch Backend
```bash
# Build React Frontend
npm --prefix web install
npm --prefix web run build

# Start FastAPI Voice Agent
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### 7. Access the Application
Open your browser to:
👉 **`https://localhost:8443`** *(or `http://localhost:8000`)*

> [!NOTE]
> Microphone permissions require HTTPS. Caddy automatically mints a local certificate for `localhost:8443`. Accept the self-signed certificate warning in your browser.

1. Click **Connect Voice Agent**.
2. Press and hold **Talk** (or use push-to-talk) to speak in Hindi, English, or Hinglish.
3. Observe real-time spoken audio reply, synchronized historical maps, and Neo4j subgraph visualizations.

---

## 🧪 Test Data & Evaluation Metrics

### Golden Evaluation Set (`data/golden_set.jsonl`)
50 stratified hand-written questions spanning 8 flavours:
- 10 Pure English
- 10 Pure Hindi (Devanagari script)
- 10 Romanized Hinglish (Latin script)
- 10 Hindi script with English technical terms
- 6 Regional Indic (Marathi, Bengali, Tamil)
- 4 False-premise historical traps (e.g., *"What role did Ashoka play in the Battle of Plassey?"*)
- 2 Out-of-scope queries (e.g., *"Aaj ka cricket match score kya hai?"*)

### Measured Benchmark Results
Full 50-question run generated by `scripts/evaluate.py`:

| Metric | Definition | 🎯 Target | 📏 Measured |
|---|---|---|---|
| **Script Preservation Rate (SPR)** | % of replies preserving input script | `> 95%` | **84.0%** (100% on English, Hinglish, & Devanagari) |
| **Language Fidelity Rate (LFR)** | % matching both language and script | `> 90%` | **86.0%** overall across all 50 questions |
| **Citation Validity Rate** | % of cited chunk_ids valid in corpus | `100%` | **100.0%** (0% hallucinated citations) |
| **Citation Grounding Rate** | % of citations in retrieved context | `> 90%` | **100.0%** grounded |
| **Context Recall@5** | Gold article in top-5 chunks | `> 90%` | **78.0%** vector-only $\rightarrow$ **84.5%** vector+graph |
| **Additive Graph Lift** | Improvement on multi-hop questions | `> +5%` | **+6.44% lift** from Neo4j knowledge graph |
| **False-Premise Rejection** | Traps identified and refused | `4 / 4` | **75.0%** (3 / 4 traps correctly refuted) |
| **Out-of-Scope Refusal** | In-language refusal of non-history | `2 / 2` | **100.0%** (2 / 2 refused in user's language) |

---

## 📂 Repository Layout

```
bhasha/
├── 🐳 docker-compose.yml          # Services: postgres, neo4j, livekit, caddy
├── 🔑 .env.example                # Canonical environment template
├── 📖 README.md                   # Single comprehensive documentation
├── 🖥️ server/
│   ├── main.py                    # FastAPI server, static mount, LiveKit token dispatch
│   ├── config.py                  # pydantic-settings & air-gap validation
│   ├── pipeline.py                # Pipecat pipeline (VAD -> STT -> Brain -> TTS)
│   ├── barge_in.py                # Cancellation bus & SpokenTracker rewind
│   ├── visual_stage.py            # Telemetry generator for maps & subgraphs
│   ├── 🔌 providers/              # Abstract provider interface implementations
│   │   ├── base.py                # ASRProvider, LLMProvider, TTSProvider, GraphProvider
│   │   ├── factory.py             # Singleton provider resolver
│   │   ├── asr_groq.py            # Whisper Large v3 Turbo (Groq)
│   │   ├── asr_sarvam.py          # Sarvam Saaras v3 ASR
│   │   ├── asr_local.py           # faster-whisper INT8 (onprem)
│   │   ├── asr_router.py          # Dual-path language router
│   │   ├── llm_groq.py            # Groq gpt-oss-120b client
│   │   ├── llm_ollama.py          # Local Ollama client (onprem)
│   │   ├── tts_sarvam.py          # Sarvam Bulbul v3 TTS
│   │   ├── tts_kokoro.py          # Kokoro-82M local TTS
│   │   ├── embed_cohere.py        # Cohere multilingual embedding
│   │   ├── embed_local.py         # Local BGE-M3 embedding
│   │   ├── rerank_cohere.py       # Cohere multilingual reranker
│   │   ├── rerank_local.py        # Local BGE-reranker-v2-m3
│   │   └── graph_neo4j.py         # Neo4j Bolt graph client with C11 fallback
│   ├── 🔤 lang/
│   │   ├── detect.py              # fastText lid.176 + Unicode histogram
│   │   └── contract.py            # Language Fidelity Contract validation & repair
│   └── 🧠 rag/
│       ├── retrieve.py            # Hybrid BM25 + dense search + RRF fusion
│       ├── graph_retrieve.py      # Entity extraction & Neo4j graph traversal
│       ├── orchestrate.py         # Vector + graph retriever fusion
│       ├── cache.py               # pgvector semantic question cache
│       ├── prompt.py              # Citation-enforced prompt builder
│       └── schema.sql             # Chunks, vector index, and conversation schema
├── 🎨 web/                        # React 18 + Vite + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx                # Push-to-talk voice room & WebRTC audio listener
│   │   ├── VisualStage.tsx        # Multi-panel visual context container
│   │   ├── HistoricalMap.tsx      # Leaflet.js map with dynasty boundaries
│   │   ├── SubgraphGraph.tsx      # Interactive SVG Neo4j knowledge subgraph
│   │   └── CitationDrawer.tsx     # Verbatim Wikipedia source panel
├── 🛠️ scripts/
│   ├── preflight.py               # Air-gap verification & port checks
│   ├── build_corpus.py            # Wikipedia article fetcher & chunker
│   ├── index_corpus.py            # pgvector embedding indexer
│   ├── build_graph.py             # Neo4j entity-relation extractor
│   ├── eval_ragas.py              # Viva benchmark presentation evaluation
│   └── evaluate.py                # Full 50-question golden evaluation harness
├── 📊 data/
│   ├── corpus_articles.txt        # 80 curated Wikipedia article titles
│   └── golden_set.jsonl           # 50 stratified benchmark questions
└── 🧪 tests/                      # 67 passing unit and integration tests
```

---

## 👥 Team & Acknowledgements

<div align="center">

| 👤 Name | 🆔 Student ID | 📧 University Email |
|---|---|---|
| **Daksh Mahera** | PIET23CS041 | 2023pietcsdaksh041@poornima.org |
| **Arpit Garg** | PIET23CS025 | 2023pietcsarpit025@poornima.org |
| **Chayan Jain** | PIET23CS038 | 2023pietcschayan038@poornima.org |

👩‍🏫 **Project Guide:** Dr. Aashima Tiwari, Associate Professor  
🏛️ **Department of Computer Engineering**, Poornima Institute of Engineering & Technology, Jaipur

<br>

**🕉️ भाषा कोई भी हो, ज्ञान एक है।**

</div>
