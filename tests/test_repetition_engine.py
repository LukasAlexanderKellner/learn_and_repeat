from datetime import datetime
import unittest
from unittest.mock import patch

from src.repetition import engine


class FakeCursor:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount
        self.query = ""
        self.params: tuple[object, ...] = ()

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...]) -> None:
        self.query = query
        self.params = params

    def fetchall(self) -> list[dict]:
        return []


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.fake_cursor = cursor

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def cursor(self, **kwargs: object) -> FakeCursor:
        return self.fake_cursor


class RepetitionEngineTests(unittest.TestCase):
    def test_rating_update_rejects_stale_reviews(self) -> None:
        cursor = FakeCursor(rowcount=0)
        connection = FakeConnection(cursor)

        with patch.object(engine, "connect_to_database", return_value=connection):
            with self.assertRaisesRegex(LookupError, "bereits bewertet"):
                engine.update_review_state(1, 4, 2, 6, 2.5, datetime.now())

        self.assertIn("next_review_date <= CURRENT_TIMESTAMP", cursor.query)

    def test_due_review_limit_must_be_positive(self) -> None:
        with patch.object(engine, "connect_to_database") as connect:
            with self.assertRaises(ValueError):
                engine.get_due_reviews(limit=0)

        connect.assert_not_called()

    def test_due_review_query_returns_whole_chapter_content(self) -> None:
        cursor = FakeCursor(rowcount=0)
        connection = FakeConnection(cursor)

        with patch.object(engine, "connect_to_database", return_value=connection):
            result = engine.get_due_reviews(limit=20)

        self.assertEqual(result, [])
        self.assertIn("string_agg", cursor.query)
        self.assertIn("chunk.chapter_id = chapter.id", cursor.query)


if __name__ == "__main__":
    unittest.main()
