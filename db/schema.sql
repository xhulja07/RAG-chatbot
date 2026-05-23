-- ============================================================
-- Project 1: Knowledge Base Chatbot — Postgres + pgvector schema
-- Run once against your Postgres instance:
--   psql $DATABASE_URL -f db/schema.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------
-- documents: one row per source file
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    title         TEXT NOT NULL,
    filename      TEXT NOT NULL UNIQUE,
    category      TEXT NOT NULL,          -- e.g. 'HR Policy', 'SOP', 'Finance'
    owner_email   TEXT,
    department    TEXT,
    source_path   TEXT,
    last_updated  TIMESTAMPTZ DEFAULT NOW(),
    is_active     BOOLEAN DEFAULT TRUE,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------
-- chunks: one row per text chunk, with its embedding
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id            SERIAL PRIMARY KEY,
    document_id   INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,       -- position within the document
    content       TEXT NOT NULL,
    token_count   INTEGER,
    embedding     VECTOR(1536),           -- OpenAI text-embedding-3-small
                                          -- change to 3072 for -large
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Vector similarity index (cosine distance, IVFFlat for scale)
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Fast lookup by document
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);

-- ---------------------------------------------------------------
-- query_log: every user query + cost tracking
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_log (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT,
    user_email      TEXT,
    question        TEXT NOT NULL,
    answer          TEXT,
    source_doc_ids  INTEGER[],            -- which documents were cited
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    latency_ms      INTEGER,
    was_refused     BOOLEAN DEFAULT FALSE,
    refusal_reason  TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------
-- employees: used to join department context onto documents
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS employees (
    email       TEXT PRIMARY KEY,
    name        TEXT,
    department  TEXT,
    role        TEXT
);

-- ---------------------------------------------------------------
-- Useful views
-- ---------------------------------------------------------------

-- Documents overdue for review (older than 180 days)
CREATE OR REPLACE VIEW stale_documents AS
SELECT
    d.id,
    d.title,
    d.category,
    d.last_updated,
    NOW() - d.last_updated AS age,
    d.owner_email,
    e.department
FROM documents d
LEFT JOIN employees e ON d.owner_email = e.email
WHERE d.last_updated < NOW() - INTERVAL '180 days'
  AND d.is_active = TRUE
ORDER BY d.last_updated ASC;

-- Per-category query volume and average cost
CREATE OR REPLACE VIEW query_cost_by_category AS
SELECT
    d.category,
    COUNT(DISTINCT ql.id)                          AS total_queries,
    ROUND(AVG(ql.input_tokens + ql.output_tokens)) AS avg_tokens_per_query,
    -- Claude Sonnet pricing: $3/M input, $15/M output
    ROUND(
        SUM(ql.input_tokens  * 0.000003 +
            ql.output_tokens * 0.000015)::NUMERIC, 4
    )                                               AS total_cost_usd
FROM query_log ql
JOIN LATERAL UNNEST(ql.source_doc_ids) AS src_id ON TRUE
JOIN documents d ON d.id = src_id
GROUP BY d.category
ORDER BY total_cost_usd DESC;
