# BHASHA

**Bilingual, Hinglish-Aware Speech-to-Speech Historical Assistant**

A self-hostable, language-agnostic voice agent that holds a real-time spoken conversation in Hindi, English, Hinglish, or a regional Indian language — and answers only from a curated, cited knowledge base of Indian history.

> Poornima Institute of Engineering & Technology · Department of Computer Engineering
> Final Year Project 2026–27 · Team Oblivion
> Guide: Dr. Aashima Tiwari, Associate Professor

---

## 1. The one-line pitch

Every voice assistant demo is Whisper → GPT → TTS. BHASHA is different in four specific, measurable ways: it **routes** between ASR engines by detected language, it **guarantees** the reply comes back in the same language *and script* the user spoke, it **grounds** every answer in retrieved sources so it can be scored for hallucination, and it **ships** as a single Compose file that a client can run inside their own network with zero data leaving the premises.

---

## 2. Why the obvious approach isn't good enough

| The default assumption | What actually happens |
|---|---|
| "Whisper handles 99 languages, so multilingual is solved." | Whisper collapses code-mixed Hinglish into a single script and drops the switch points. It also trails Indic-specialised models on Marathi, Bengali, and Tamil. |
| "The LLM will just reply in whatever language the user used." | It won't, reliably. Ask in romanised Hinglish and models frequently answer in Devanagari or drift to pure English by the third turn. |
| "PostgreSQL stores the conversation, so we have memory." | Storage is not retrieval. Without grounding the system is an API wrapper with no measurable contribution. |
| "Accuracy improved." | Against what baseline, on what test set, measured how? |

BHASHA is built to survive each of those questions.

---

## 3. The four pillars

### 3.1 Dual-path ASR with a language router
Audio hits a fast language-identification pass first. Indic-dominant speech routes to an Indic-first recogniser; English and code-mixed speech routes to Whisper Large v3 Turbo. One model for everything is a compromise; two models plus a router is an architecture.

### 3.2 The Language Fidelity Contract
The core engineering contribution. The system detects **language *and* script** on input, pins both into a contract, constrains generation to that contract, then *validates the output against it* and repairs on violation.

```python
contract = {"lang": "hi-en", "script": "Latn", "register": "casual"}

# 1. DETECT — fastText lid.176 + Unicode block histogram over the transcript
# 2. CONSTRAIN — inject the contract into the system prompt, per turn, every turn
# 3. VALIDATE — re-run script detection over the generated text
#    if Devanagari ratio > 0.15 while contract.script == "Latn": violation
# 4. REPAIR — one transliteration-and-retry pass before the text reaches TTS
```

This is what turns "the bot sometimes replies in the wrong language" from an anecdote into a number: **Language Fidelity Rate (LFR)**, reported per flavour in §8.

### 3.3 Cross-lingual grounded retrieval
The knowledge corpus is English Wikipedia. The user speaks Hinglish. Multilingual embeddings put both in the same vector space, so a Hindi question retrieves English passages and the answer is generated back in Hindi with citations. Every claim traces to a `chunk_id`, an article, and a section breadcrumb.

### 3.4 True barge-in
The user can interrupt mid-sentence. This is not "stop the speaker" — it requires cancelling the in-flight LLM token stream, flushing the TTS queue, and **rewinding conversation state to what was actually spoken aloud**, not what was generated. Skipping that last step is the most common bug in voice agents and it corrupts context within three turns.

---

## 4. Architecture

```
                          ┌──────────────────────────────┐
   Browser (React)        │      CLIENT PREMISES         │
   ┌──────────────┐       │                              │
   │ AudioWorklet │◄─WebRTC──► LiveKit Media Server       │
   │  mic + spkr  │       │   (self-hosted, Apache-2.0)  │
   └──────────────┘       │            │                 │
                          │            ▼                 │
                          │   ┌────────────────────┐     │
                          │   │  Pipecat Pipeline  │     │
                          │   └────────────────────┘     │
                          │            │                 │
                          │   ┌────────┴─────────┐       │
                          │   ▼                  ▼       │
                          │ Silero VAD ──► BARGE-IN BUS  │
                          │  (~1 ms)        (cancel all) │
                          │   │                          │
                          │   ▼                          │
                          │ ┌───────────────────────┐    │
                          │ │ Language + Script ID  │    │
                          │ │  fastText + Unicode   │    │
                          │ └───────────┬───────────┘    │
                          │      ┌──────┴──────┐         │
                          │      ▼             ▼         │
                          │  ASR: Indic    ASR: Whisper  │
                          │   (Saarika /    large-v3-    │
                          │  IndicWhisper)    turbo      │
                          │      └──────┬──────┘         │
                          │             ▼                │
                          │   ┌──────────────────┐       │
                          │   │  RAG ORCHESTRATOR│       │
                          │   │ query translate  │       │
                          │   │ hybrid BM25+dense│       │
                          │   │ rerank (top-k=5) │       │
                          │   └────────┬─────────┘       │
                          │            │                 │
                          │   ┌────────▼─────────┐       │
                          │   │ PostgreSQL 16    │       │
                          │   │  + pgvector      │       │
                          │   │ chunks · history │       │
                          │   └────────┬─────────┘       │
                          │            ▼                 │
                          │   ┌──────────────────┐       │
                          │   │ LLM (streaming)  │       │
                          │   │ + fidelity       │       │
                          │   │   contract       │       │
                          │   └────────┬─────────┘       │
                          │            ▼                 │
                          │   sentence chunker           │
                          │            ▼                 │
                          │   TTS (Bulbul v3 / Kokoro)   │
                          │            │                 │
                          └────────────┼─────────────────┘
                                       ▼
                                  audio out
```

**One design rule governs everything:** each stage sits behind an abstract interface and is selected at runtime by an environment variable. Swapping the TTS vendor is a config change, never a code change. That rule is what makes §7 possible.

---

## 5. Latency budget

Target: **time-to-first-audio (TTFA) under 700 ms** on the hybrid profile. Reference open-source builds on this exact shape of pipeline reach 165 ms TTFA on a dedicated GPU, so the target has real headroom.

| Stage | Budget | Notes |
|---|---|---|
| Silero VAD end-of-turn | ~1 ms | Finalise on 600 ms of trailing silence — better than a fixed timer or a raw amplitude threshold |
| Language + script ID | ~5 ms | fastText, CPU, runs on the partial transcript |
| ASR (Whisper Turbo) | 55–150 ms | Turbo's 4-layer decoder replaces Large v3's 32; roughly 48% faster for a small WER cost |
| Embedding + hybrid retrieval | ~40 ms | pgvector HNSW index |
| Rerank (top-20 → top-5) | ~25 ms | bge-reranker-v2-m3 |
| LLM to first clause | ~150 ms | Streaming; we do not wait for the full response |
| TTS first chunk | 50–120 ms | Sentence-chunked so audio starts before generation finishes |
| **TTFA total** | **~330–500 ms** | Barge-in cancels the whole chain in under 50 ms |

---

## 6. Technology stack

| Layer | Choice | Why this and not the obvious alternative |
|---|---|---|
| **Pipeline orchestration** | **Pipecat** (Python, BSD-2) | Models the agent as swappable frame processors; hit v1.0 in April 2026 with the widest plugin library. Chosen over writing our own async loop. |
| **Transport** | **LiveKit** media server, self-hosted (Apache-2.0) | Pipecat's default WebRTC partner is Daily, which is hosted SaaS — unusable inside a client's private network. LiveKit self-hosts. We run Pipecat's pipeline *on* LiveKit's transport. |
| **VAD / turn-taking** | Silero VAD | ~1 ms on CPU. Energy thresholds break the moment the room has a fan in it. |
| **ASR — English / Hinglish** | Whisper `large-v3-turbo` (Groq API or local `faster-whisper` INT8) | Groq's free tier allows 2,000 requests and 28,800 audio seconds per day at 228× real-time — enough to develop the entire project without spending anything. |
| **ASR — Indic** | Sarvam **Saarika** (API) / AI4Bharat **IndicWhisper** (local, Apache-2.0) | Saarika does auto language detection and transcribe-plus-translate in a single call, ~₹30/hour. IndicWhisper is the air-gapped equivalent. |
| **Language + script ID** | fastText `lid.176` + Unicode block histogram | Language alone is insufficient — Hinglish and Hindi are the *same language ID*, different scripts. The script signal is what the fidelity contract runs on. |
| **LLM** | Groq (Llama 4 Scout / GPT-OSS 120B) · local vLLM or Ollama + Qwen3-4B | Groq for speed during development, local for air-gapped delivery. Same interface. |
| **Embeddings** | **BGE-M3** | Multilingual, and produces dense + sparse vectors from one model — hybrid retrieval without running two. |
| **Reranker** | `bge-reranker-v2-m3` | Cross-lingual reranking is where cross-lingual RAG is won or lost. |
| **Vector + relational store** | **PostgreSQL 16 + pgvector** | One database for chunks, embeddings, conversation history, and audit logs. A separate vector DB is one more service for a client to operate, for no gain at our corpus size. |
| **TTS** | Sarvam **Bulbul v3** (API) · **Kokoro-82M** (local) | Bulbul v3 covers 11 Indian languages with 30+ voices and is built for code-mixed Hinglish, at ~₹30 per 10,000 characters. **Do not build on Piper** — it was archived in October 2025. ElevenLabs and OpenAI TTS are dropped: expensive and weak on Indic. |
| **Backend** | FastAPI + WebSockets | |
| **Frontend** | React 18 + Vite + AudioWorklet | AudioWorklet, not the deprecated ScriptProcessorNode. |
| **Observability** | OpenTelemetry + Langfuse (self-hosted) | Per-stage latency traces. Without these, "it feels slow" is unactionable. |
| **Evaluation** | RAGAS faithfulness + custom LFR + WER/CER | See §8. |
| **Packaging** | Docker Compose with profiles | See §7. |

**Explicitly rejected:** Daily (SaaS transport), Piper (archived), ElevenLabs (cost + Indic quality), FAISS as primary store (pgvector consolidates), a standalone vector DB (operational overhead the client would inherit).

---

## 7. Deployment profiles — the client story

The same codebase ships in three configurations. The client picks one at install time by setting `BHASHA_PROFILE`.

| | `cloud` | `hybrid` *(recommended)* | `onprem` |
|---|---|---|---|
| ASR | Groq + Sarvam API | local faster-whisper | local faster-whisper + IndicWhisper |
| LLM | Groq API | Groq API | local vLLM / Ollama |
| TTS | Sarvam Bulbul v3 | Sarvam Bulbul v3 | local Kokoro-82M |
| Audio leaves premises | yes | no | **no** |
| Text leaves premises | yes | yes | **no** |
| Hardware needed | none | 8 GB VRAM | 16–24 GB VRAM |
| TTFA | ~400 ms | ~450 ms | ~700 ms |
| Cost at 1,000 conversations/month | ~₹900 | ~₹600 | ₹0 + electricity |

`onprem` is the differentiator in a client conversation. A government department, a university, or a healthcare provider cannot send citizen audio to a third-party API. Most competing products cannot offer that option at all.

```bash
git clone <repo> && cd bhasha
cp .env.example .env            # set BHASHA_PROFILE and keys
docker compose --profile hybrid up -d
python scripts/build_corpus.py  # fetch + chunk Wikipedia
python scripts/index_corpus.py  # embed into pgvector
open https://localhost:8443
```

> Microphone access requires HTTPS. Compose ships a Caddy sidecar that terminates TLS with a self-signed cert for LAN deployment.

---

## 8. Test data and evaluation

### 8.1 Knowledge corpus
`data/corpus_articles.txt` lists ~80 curated English Wikipedia articles spanning the Indus Valley Civilisation through the Constitution of India, organised by era. `scripts/build_corpus.py` fetches them, drops non-factual sections (References, See also, External links), and performs **section-aware chunking** — 512 tokens with 64-token sentence-level overlap, preserving the heading breadcrumb (`Ashoka > Kalinga War > Aftermath`) as retrieval metadata. Character-count chunking destroys that breadcrumb, which is the single most useful piece of metadata the source gives us. Expected output: roughly 4,000–6,000 chunks.

Every chunk carries `revision_id` and `fetched_at`, so the corpus is reproducible and the exact snapshot used in the report can be cited.

### 8.2 Golden evaluation set
`data/golden_set.jsonl` — 50 hand-written questions with gold answers and gold source articles, deliberately stratified:

| Flavour | n | Tests |
|---|---|---|
| Pure English | 10 | Baseline |
| Pure Hindi (Devanagari) | 10 | Cross-lingual retrieval |
| Hinglish (Roman script) | 10 | The hard case — ASR *and* script fidelity |
| Hindi script + English technical terms | 10 | Register preservation |
| Regional (Marathi, Bengali, Tamil) | 6 | Indic ASR routing + cross-lingual generation |
| Mid-sentence code-switching | 1 | Intra-sentential switching |
| Out-of-scope | 2 | Refusal without falling back to parametric memory |
| Multi-turn coreference | 1 | State across turns and languages |

Four of the 50 are **false-premise traps** ("What role did Ashoka play in the Battle of Plassey?"). A system that answers these fluently is hallucinating, and the trap catches it. The out-of-scope pair catches the other failure: answering from the model's own memory instead of the corpus.

### 8.3 Metrics

| Metric | Definition | Target |
|---|---|---|
| **WER / CER** | Word / character error rate per language flavour vs. reference transcripts | < 15% Hinglish, < 10% English |
| **Language Fidelity Rate (LFR)** | % of replies matching the input contract on *both* language and script | > 95% |
| **Faithfulness** | RAGAS — proportion of claims supported by retrieved context | > 0.90 |
| **Answer relevance** | RAGAS | > 0.85 |
| **Context recall@5** | Gold article present in top-5 retrieved chunks | > 0.90 |
| **False-premise rejection** | Traps correctly refused | 4 / 4 |
| **Out-of-scope refusal** | Refused *in the user's own language* | 2 / 2 |
| **TTFA p50 / p95** | Time to first audio byte | < 700 ms / < 1200 ms |
| **Barge-in latency** | Interrupt to silence | < 100 ms |

Report every metric as an **ablation**: single-ASR vs. routed; no-contract vs. contract; no-RAG vs. RAG vs. RAG+rerank. The deltas are the project's actual result, not the absolute numbers.

---

## 9. Repository layout

```
bhasha/
├── docker-compose.yml          # profiles: cloud | hybrid | onprem
├── .env.example
├── server/
│   ├── main.py                 # FastAPI + LiveKit token issuance
│   ├── pipeline.py             # Pipecat pipeline assembly
│   ├── barge_in.py             # cancellation bus + state rewind
│   ├── providers/              # ← the abstraction layer
│   │   ├── base.py             # ASRProvider · LLMProvider · TTSProvider
│   │   ├── asr_groq.py  asr_sarvam.py  asr_local.py
│   │   ├── llm_groq.py  llm_ollama.py
│   │   └── tts_sarvam.py  tts_kokoro.py
│   ├── lang/
│   │   ├── detect.py           # fastText + Unicode block histogram
│   │   └── contract.py         # detect → constrain → validate → repair
│   └── rag/
│       ├── retrieve.py         # hybrid BM25 + dense, then rerank
│       └── prompt.py           # citation-forced answer template
├── web/                        # React + Vite + AudioWorklet
├── scripts/
│   ├── build_corpus.py         # ✅ implemented
│   ├── index_corpus.py         # embed → pgvector
│   └── evaluate.py             # runs golden_set.jsonl, emits the metrics table
├── data/
│   ├── corpus_articles.txt     # ✅ 80 curated articles
│   ├── corpus.jsonl            # generated
│   └── golden_set.jsonl        # ✅ 50 stratified questions
└── docs/
    └── architecture.md
```

---

## 10. Known risks and mitigations

| Risk | Mitigation |
|---|---|
| Whisper hallucinates text during silence | Minimum audio-energy gate before ASR, plus a no-speech threshold; VAD-gated input rather than fixed-duration recording |
| Sarvam API unreachable at the client site | Provider abstraction falls back to the local engine automatically; `onprem` profile has no external dependency at all |
| Hinglish ground truth is subjective (two annotators romanise differently) | Score CER not WER for Hinglish, and have all three team members annotate a 20-question overlap to report inter-annotator agreement |
| Regional-language TTS quality varies | Sarvam publishes per-language voice recommendations; we pin the recommended voice per language and document the choice |
| Cloud rate limits during the demo | Pre-warm a local `hybrid` instance as the live fallback; never demo on a free tier |
| Corpus drifts as Wikipedia is edited | `revision_id` pinned per chunk; the evaluated snapshot is frozen and archived |

---

## 11. Execution plan

| Phase | Weeks | Deliverable | Owner |
|---|---|---|---|
| 0 · Foundation | 1–2 | Compose stack up, Postgres + pgvector, corpus built and indexed | All |
| 1 · Retrieval | 3–5 | Hybrid retrieval + reranking, context recall baseline measured | Daksh |
| 2 · Voice loop | 4–7 | LiveKit + Pipecat + VAD + ASR + TTS, turn-based end-to-end | Arpit |
| 3 · Fidelity contract | 6–8 | detect → constrain → validate → repair, LFR measured | Daksh |
| 4 · Streaming + barge-in | 8–10 | Sentence chunking, cancellation bus, state rewind, TTFA measured | Chayan |
| 5 · Evaluation | 10–12 | Full ablation table across all 50 golden questions | All |
| 6 · Hardening | 12–14 | `onprem` profile verified air-gapped, install runbook written | All |
| 7 · Report & defence | 14–16 | Paper draft, demo script, viva prep | All |

---

## 12. Team

| Name | Registration No. | Email |
|---|---|---|
| Daksh Mahera | PIET23CS041 | 2023pietcsdaksh041@poornima.org |
| Arpit Garg | PIET23CS025 | 2023pietcsarpit025@poornima.org |
| Chayan Jain | PIET23CS038 | 2023pietcschayan038@poornima.org |

**Project Guide:** Dr. Aashima Tiwari, Associate Professor, Department of Computer Engineering

---

## 13. Acknowledgements and licensing

Knowledge corpus derived from English Wikipedia (CC BY-SA 4.0) — attribution is preserved per chunk and surfaced in every cited answer. Pipecat (BSD-2), LiveKit (Apache-2.0), Silero VAD (MIT), Whisper (MIT), BGE-M3 (MIT), Kokoro-82M (Apache-2.0), AI4Bharat IndicWhisper (Apache-2.0). Sarvam AI Saarika and Bulbul v3 are commercial APIs used under their published terms.
