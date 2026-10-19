-- 1. activate the vector extension for PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Table for storing documents (e.g., PDFs, text files, etc.)
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Table for storing chunks (the actual content + vector)
-- Note: 768 is the typical dimension for many local embedding models 
-- (e.g., nomic-embed-text uses 768 dimensions). If you use a different model, adjust this accordingly.
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB, -- flexible field for e.g., chapter titles, page numbers
    embedding VECTOR(768) 
);

-- 4. Table for spaced repetition (the learning memory)
CREATE TABLE reviews (
    id SERIAL PRIMARY KEY,
    chunk_id INT REFERENCES chunks(id) ON DELETE CASCADE,
    review_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    next_review_date TIMESTAMP NOT NULL,
    interval_days INT DEFAULT 0,          -- Current interval in days
    repetition_count INT DEFAULT 0,       -- How often has it been reviewed?
    ease_factor REAL DEFAULT 2.5,         -- Difficulty factor according to SM-2 algorithm
    last_quality_score INT                -- Last rating (0 to 5)
);

-- 5. Create a vector index for ultra-fast similarity search
-- HNSW (Hierarchical Navigable Small World) is currently the gold standard for vector indexes.
CREATE INDEX chunks_embedding_hnsw_idx ON chunks 
USING hnsw (embedding vector_cosine_ops);