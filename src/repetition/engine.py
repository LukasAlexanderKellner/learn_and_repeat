from datetime import datetime

from psycopg.rows import dict_row

from src.database.connection import connect_to_database


def get_due_reviews(limit: int = 5) -> list[dict]:
    """Fetch due chapter reviews and their assembled Markdown content."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    with connect_to_database() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT r.id AS review_id, r.chapter_id,
                       chapter.title AS chapter_title,
                       (
                           SELECT string_agg(
                               chunk.content,
                               chr(10) || chr(10) ORDER BY chunk.chunk_index
                           )
                           FROM chunks AS chunk
                           WHERE chunk.chapter_id = chapter.id
                       ) AS content,
                       r.repetition_count, r.interval_days, r.ease_factor,
                       r.next_review_date
                FROM reviews r
                JOIN chapters AS chapter ON r.chapter_id = chapter.id
                WHERE r.next_review_date <= CURRENT_TIMESTAMP
                ORDER BY r.next_review_date ASC, chapter.chapter_index ASC
                LIMIT %s;
                """,
                (limit,),
            )
            return cur.fetchall()


def update_review_state(
    review_id: int,
    quality_score: int,
    new_rep: int,
    new_interval: int,
    new_ef: float,
    next_date: datetime,
) -> None:
    """Persist a chapter's updated review schedule and rating."""
    if not 0 <= quality_score <= 5:
        raise ValueError("quality_score must be between 0 and 5")

    with connect_to_database() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE reviews
                SET repetition_count = %s,
                    interval_days = %s,
                    ease_factor = %s,
                    next_review_date = %s,
                    last_quality_score = %s,
                    review_date = CURRENT_TIMESTAMP
                WHERE id = %s AND next_review_date <= CURRENT_TIMESTAMP;
                """,
                (new_rep, new_interval, new_ef, next_date, quality_score, review_id),
            )
            if cur.rowcount != 1:
                raise LookupError(
                    f"Review mit ID {review_id} wurde nicht gefunden oder "
                    "zwischenzeitlich bereits bewertet."
                )
