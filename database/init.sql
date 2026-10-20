-- 1. activate the vector extension for PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Table for storing documents (e.g., PDFs, text files, etc.)
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Table for storing Markdown chapters
CREATE TABLE chapters (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    chapter_index INT NOT NULL,
    UNIQUE (document_id, chapter_index)
);

-- 4. Table for storing chunks (the actual content + vector)
-- Note: 768 is the typical dimension for nomic-embed-text.
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    chapter_id INT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(768),
    UNIQUE (chapter_id, chunk_index)
);

-- 5. One spaced-repetition schedule per chapter
CREATE TABLE reviews (
    id SERIAL PRIMARY KEY,
    chapter_id INT NOT NULL UNIQUE REFERENCES chapters(id) ON DELETE CASCADE,
    review_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    next_review_date TIMESTAMPTZ NOT NULL,
    interval_days INT DEFAULT 0,
    repetition_count INT DEFAULT 0,
    ease_factor REAL DEFAULT 2.5,
    last_quality_score INT                -- Last rating (0 to 5)
);

-- 6. Vector index for similarity search
CREATE INDEX chunks_embedding_hnsw_idx ON chunks 
USING hnsw (embedding vector_cosine_ops);