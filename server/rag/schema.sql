-- BHASHA schema. Columns of `chunks` mirror the JSONL emitted by
-- scripts/build_corpus.py exactly — do not drift from it.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- corpus ----
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,          -- "ashoka::0007"
    article      TEXT NOT NULL,
    url          TEXT NOT NULL,
    section      TEXT NOT NULL,             -- "Kalinga War > Aftermath"
    text         TEXT NOT NULL,
    n_tokens     INTEGER NOT NULL,
    lang         TEXT NOT NULL DEFAULT 'en',
    revision_id  BIGINT,
    fetched_at   TIMESTAMPTZ NOT NULL,

    -- bge-m3 is 1024-dimensional. Nullable so index_corpus.py can be resumable:
    -- a row may exist before its embedding has been computed.
    embedding    vector(1024),

    -- BM25 side of the hybrid search. Generated, so it can never go stale.
    tsv          tsvector GENERATED ALWAYS AS (
                     setweight(to_tsvector('english', coalesce(article, '')), 'A') ||
                     setweight(to_tsvector('english', coalesce(section, '')), 'B') ||
                     setweight(to_tsvector('english', coalesce(text, '')),    'C')
                 ) STORED
);

-- Cosine distance, because bge-m3 embeddings are normalised.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_article ON chunks (article);

-- ----------------------------------------------------------- sessions -------
-- Session state lives here, never in the browser (C10).
CREATE TABLE IF NOT EXISTS conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    room        TEXT NOT NULL,
    lang        TEXT,                       -- contract language, e.g. "hi-en"
    script      TEXT,                       -- ISO 15924, e.g. "Latn"
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS turns (
    id               BIGSERIAL PRIMARY KEY,
    conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_index       INTEGER NOT NULL,
    role             TEXT NOT NULL CHECK (role IN ('user', 'agent')),

    transcript       TEXT,                  -- user: ASR text; agent: text generated
    -- What was actually played out of the speaker. Diverges from `transcript`
    -- when the user barges in mid-sentence; Milestone 4 rewinds state to this.
    spoken_text      TEXT,
    detected_lang    TEXT,
    detected_script  TEXT,

    cited_chunk_ids  TEXT[] NOT NULL DEFAULT '{}',
    latency_ms       JSONB  NOT NULL DEFAULT '{}',   -- {"asr": 210, "llm_ttft": 180, ...}
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (conversation_id, turn_index)
);

CREATE INDEX IF NOT EXISTS turns_conversation ON turns (conversation_id, turn_index);
