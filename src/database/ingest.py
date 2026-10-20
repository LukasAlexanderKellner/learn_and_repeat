import logging
import os
from pathlib import Path

from pgvector import Vector
from pgvector.psycopg import register_vector

from src.database.connection import PROJECT_ROOT, connect_to_database
from src.processing.chunker import (
    semantic_chunk_markdown,
    split_markdown_into_chapters,
)
from src.processing.embedder import generate_embedding

logger = logging.getLogger(__name__)


def ingest_markdown_file(markdown_path: str, filename: str) -> int | None:
    """Embed and store a Markdown file, skipping paths already ingested.

    Returns the new document ID, or None if this path was already ingested.
    """
    source_path = Path(markdown_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Markdown-Datei nicht gefunden: {source_path}")

    with source_path.open("r", encoding="utf-8") as markdown_file:
        markdown_text = markdown_file.read()

    chapters = split_markdown_into_chapters(markdown_text)
    if not chapters:
        raise ValueError(f"Markdown-Datei enthält keinen Text: {source_path}")
    logger.info("Markdown in %s Kapitel aufgeteilt.", len(chapters))

    chunked_chapters = [
        (
            chapter_index,
            chapter_title,
            semantic_chunk_markdown(chapter_content, max_chunk_size=800),
        )
        for chapter_index, (chapter_title, chapter_content) in enumerate(
            chapters, start=1
        )
    ]
    for _, chapter_title, chunks in chunked_chapters:
        if not chunks:
            raise ValueError(f"Kapitel enthält keinen Text: {chapter_title}")

    total_chunk_count = sum(len(chunks) for _, _, chunks in chunked_chapters)
    embedded_chapters: list[tuple[int, str, list[tuple[int, str, Vector]]]] = []
    total_chunks = 0
    for chapter_index, chapter_title, chunks in chunked_chapters:
        embedded_chunks: list[tuple[int, str, Vector]] = []
        for chunk_index, chunk_text in enumerate(chunks, start=1):
            total_chunks += 1
            logger.info(
                "Generiere Embedding für Chunk %s/%s (Kapitel %s/%s)...",
                total_chunks,
                total_chunk_count,
                chapter_index,
                len(chapters),
            )
            embedding = generate_embedding(chunk_text)
            if embedding is None:
                logger.error(
                    "Ingestion abgebrochen: Embedding für Kapitel %s, Chunk %s "
                    "fehlgeschlagen.",
                    chapter_index,
                    chunk_index,
                )
                raise RuntimeError(
                    f"Embedding für Kapitel {chapter_index}, Chunk {chunk_index} "
                    "fehlgeschlagen; es wurden keine Daten gespeichert."
                )
            embedded_chunks.append((chunk_index, chunk_text, Vector(embedding)))
        embedded_chapters.append((chapter_index, chapter_title, embedded_chunks))

    resolved_path = str(source_path.resolve())
    legacy_relative_path = os.path.relpath(resolved_path, PROJECT_ROOT)

    with connect_to_database() as conn:
        register_vector(conn)
        conn.commit()

        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0));",
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
                for chapter_index, chapter_title, embedded_chunks in embedded_chapters:
                    cur.execute(
                        """
                        INSERT INTO chapters (document_id, title, chapter_index)
                        VALUES (%s, %s, %s)
                        RETURNING id;
                        """,
                        (document_id, chapter_title, chapter_index),
                    )
                    chapter_row = cur.fetchone()
                    if chapter_row is None:
                        raise RuntimeError(
                            f"Datenbank lieferte keine ID für Kapitel {chapter_index}."
                        )
                    chapter_id = chapter_row[0]
                    cur.execute(
                        """
                        INSERT INTO reviews (
                            chapter_id, next_review_date, interval_days,
                            repetition_count, ease_factor
                        )
                        VALUES (%s, CURRENT_TIMESTAMP, %s, %s, %s);
                        """,
                        (chapter_id, 0, 0, 2.5),
                    )
                    for chunk_index, chunk_text, embedding in embedded_chunks:
                        cur.execute(
                            """
                            INSERT INTO chunks (chapter_id, chunk_index, content, embedding)
                            VALUES (%s, %s, %s, %s);
                            """,
                            (chapter_id, chunk_index, chunk_text, embedding),
                        )

    logger.info("Erfolgreich in DB gespeichert! Document-ID: %s", document_id)
    logger.info(
        "%s Kapitel und %s Chunks inklusive Kapitel-Reviews gespeichert.",
        len(embedded_chapters),
        total_chunk_count,
    )
    return document_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_md = PROJECT_ROOT / "data/output_markdown/vork_math_cleaned.md"
    ingest_markdown_file(str(test_md), "vork_math.pdf")
