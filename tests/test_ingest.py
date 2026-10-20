from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import src.database.ingest as ingest


class FakeCursor:
    def __init__(self, existing_document: tuple[int] | None = None) -> None:
        self.existing_document = existing_document
        self.last_query = ""
        self.queries: list[str] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self.last_query = query
        self.queries.append(query)

    def fetchone(self) -> tuple[int] | None:
        if "FROM documents" in self.last_query:
            return self.existing_document
        if "INSERT INTO documents" in self.last_query:
            return (42,)
        if "INSERT INTO chunks" in self.last_query:
            return (101,)
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.db_cursor = cursor

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def commit(self) -> None:
        pass

    def transaction(self):
        return nullcontext()

    def cursor(self) -> FakeCursor:
        return self.db_cursor


class IngestMarkdownFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.markdown_path = Path(self.temp_dir.name) / "sample.md"
        self.markdown_path.write_text("A short section.", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_embedding_failure_stops_before_database_connection(self) -> None:
        with patch.object(ingest, "generate_embedding", return_value=None), patch.object(
            ingest, "_connect_to_database"
        ) as connect:
            with self.assertRaisesRegex(RuntimeError, "keine Daten gespeichert"):
                ingest.ingest_markdown_file(str(self.markdown_path), "sample.pdf")

        connect.assert_not_called()

    def test_embeddings_are_generated_before_database_connection(self) -> None:
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        events: list[str] = []

        def generate_embedding(_text: str) -> list[float]:
            events.append("embedding")
            return [0.1] * 768

        def connect() -> FakeConnection:
            events.append("connect")
            return connection

        with patch.object(
            ingest, "generate_embedding", side_effect=generate_embedding
        ), patch.object(ingest, "_connect_to_database", side_effect=connect), patch.object(
            ingest, "register_vector"
        ):
            document_id = ingest.ingest_markdown_file(
                str(self.markdown_path), "sample.pdf"
            )

        self.assertEqual(document_id, 42)
        self.assertEqual(events, ["embedding", "connect"])
        self.assertTrue(any("INSERT INTO chunks" in query for query in cursor.queries))

    def test_existing_path_is_not_inserted_again(self) -> None:
        cursor = FakeCursor(existing_document=(9,))
        connection = FakeConnection(cursor)

        with patch.object(
            ingest, "generate_embedding", return_value=[0.1] * 768
        ), patch.object(
            ingest, "_connect_to_database", return_value=connection
        ), patch.object(ingest, "register_vector"):
            result = ingest.ingest_markdown_file(
                str(self.markdown_path), "sample.pdf"
            )

        self.assertIsNone(result)
        self.assertFalse(any("INSERT INTO" in query for query in cursor.queries))


if __name__ == "__main__":
    unittest.main()
