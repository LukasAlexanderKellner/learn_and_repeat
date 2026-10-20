from datetime import datetime
import logging
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

from src.processing.chunker import semantic_chunk_markdown
from src.processing.embedder import generate_embedding

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / "docker" / ".env")


def _connect_to_database() -> psycopg.Connection:
    """Connect using DATABASE_URL or the POSTGRES_* variables from docker/.env."""
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return psycopg.connect(database_url)

    required_variables = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
    missing_variables = [
        variable for variable in required_variables if not os.environ.get(variable)
    ]
    if missing_variables:
        missing = ", ".join(missing_variables)
        raise RuntimeError(
            f"Missing required database environment variables: {missing}"
        )

    return psycopg.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5433"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def ingest_markdown_file(markdown_path: str, filename: str) -> int | None:
    """Embed and store a Markdown file, skipping paths already ingested.

    Returns the new document ID, or None if this path was already ingested.
    """
    source_path = Path(markdown_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Markdown-Datei nicht gefunden: {source_path}")

    with source_path.open("r", encoding="utf-8") as markdown_file:
        markdown_text = markdown_file.read()

    chunks = semantic_chunk_markdown(markdown_text, max_chunk_size=800)
    if not chunks:
        raise ValueError(f"Markdown-Datei enthält keinen Text: {source_path}")
    logger.info("Markdown in %s Chunks aufgeteilt.", len(chunks))

    # Prepare every embedding before opening a database transaction.
    embedded_chunks: list[tuple[int, str, Vector]] = []
    for idx, chunk_text in enumerate(chunks, start=1):
        logger.info("Generiere Embedding für Chunk %s/%s...", idx, len(chunks))
        embedding = generate_embedding(chunk_text)
        if embedding is None:
            logger.error(
                "Ingestion abgebrochen: Embedding für Chunk %s/%s fehlgeschlagen.",
                idx,
                len(chunks),
            )
            raise RuntimeError(
                f"Embedding für Chunk {idx}/{len(chunks)} fehlgeschlagen; "
                "es wurden keine Daten gespeichert."
            )
        embedded_chunks.append((idx, chunk_text, Vector(embedding)))

    resolved_path = str(source_path.resolve())
    legacy_relative_path = os.path.relpath(resolved_path, PROJECT_ROOT)

    with _connect_to_database() as conn:
        register_vector(conn)
        conn.commit()

        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pg_advisory_xact_lock(hashtextextended(%s, 0));
                    """,
                    (resolved_path,),
                )
                cur.execute(
                    """
                    SELECT id
                    FROM documents
                    WHERE file_path = %s OR file_path = %s OR file_path = %s
                    LIMIT 1;
                    """,
                    (str(source_path), resolved_path, legacy_relative_path),
                )
                existing_document = cur.fetchone()
                if existing_document is not None:
                    logger.info(
                        "Dokument bereits vorhanden (Document-ID: %s); "
                        "überspringe erneute Ingestion.",
                        existing_document[0],
                    )
                    return None

                cur.execute(
                    """
                    INSERT INTO documents (filename, file_path)
                    VALUES (%s, %s)
                    RETURNING id;
                    """,
                    (filename, resolved_path),
                )
                document_row = cur.fetchone()
                if document_row is None:
                    raise RuntimeError("Datenbank lieferte keine neue Document-ID.")
                document_id = document_row[0]
                today = datetime.now()

                for idx, chunk_text, embedding in embedded_chunks:
                    cur.execute(
                        """
                        INSERT INTO chunks (document_id, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (document_id, idx, chunk_text, embedding),
                    )
                    chunk_row = cur.fetchone()
                    if chunk_row is None:
                        raise RuntimeError(
                            f"Datenbank lieferte keine ID für Chunk {idx}."
                        )
                    chunk_id = chunk_row[0]
                    cur.execute(
                        """
                        INSERT INTO reviews (
                            chunk_id, next_review_date, interval_days,
                            repetition_count, ease_factor
                        )
                        VALUES (%s, %s, %s, %s, %s);
                        """,
                        (chunk_id, today, 0, 0, 2.5),
                    )

    logger.info("Erfolgreich in DB gespeichert! Document-ID: %s", document_id)
    return document_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_md = PROJECT_ROOT / "data/output_markdown/vork_math_cleaned.md"
    ingest_markdown_file(str(test_md), "vork_math.pdf")
