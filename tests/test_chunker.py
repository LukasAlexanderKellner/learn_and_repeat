import unittest

from src.processing.chunker import semantic_chunk_markdown


class SemanticChunkMarkdownTests(unittest.TestCase):
    def test_oversized_sections_are_split_at_the_limit(self) -> None:
        chunks = semantic_chunk_markdown("x" * 2500, max_chunk_size=800)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 800 for chunk in chunks))

    def test_markdown_sections_respect_limit_after_combining(self) -> None:
        text = "# Heading\n\n" + ("word " * 250) + "\n\n" + ("y" * 900)

        chunks = semantic_chunk_markdown(text, max_chunk_size=800)

        self.assertTrue(all(0 < len(chunk) <= 800 for chunk in chunks))

    def test_non_positive_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            semantic_chunk_markdown("text", max_chunk_size=0)


if __name__ == "__main__":
    unittest.main()
