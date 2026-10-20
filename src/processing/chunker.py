import re


def split_markdown_into_chapters(markdown_text: str) -> list[tuple[str, str]]:
    """Split Markdown at level-one and level-two headings.

    Level-two sections use the most recent level-one heading as context. Deeper
    headings remain part of their containing chapter.
    """
    chapters: list[tuple[str, str]] = []
    current_title = "Einleitung"
    current_parent_title: str | None = None
    current_lines: list[str] = []

    def finish_chapter() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            chapters.append((current_title, content))

    for line in markdown_text.splitlines():
        heading_match = re.match(r"^(#{1,2})[ \t]+(.+?)\s*#*\s*$", line)
        if heading_match is None:
            current_lines.append(line)
            continue

        finish_chapter()
        heading_level = len(heading_match.group(1))
        heading_title = heading_match.group(2).strip()
        if heading_level == 1:
            current_parent_title = heading_title
            current_title = heading_title
        else:
            current_title = (
                f"{current_parent_title} / {heading_title}"
                if current_parent_title
                else heading_title
            )
        current_lines = [line]

    finish_chapter()
    return chapters


def semantic_chunk_markdown(
    markdown_text: str, max_chunk_size: int = 1000
) -> list[str]:
    """Split Markdown at logical boundaries such as headings and paragraphs.

    Keep each resulting chunk within the requested maximum size.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")

    # Split primarily at Markdown headings or paragraph boundaries.
    raw_sections = re.split(r"\n(?=#{1,3}\s)|\n\n", markdown_text)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for section in raw_sections:
        section_clean = section.strip()
        if not section_clean:
            continue

        sections = _split_oversized_section(section_clean, max_chunk_size)
        for part in sections:
            part_length = len(part)
            separator_length = 2 if current_chunk else 0
            if current_length + separator_length + part_length <= max_chunk_size:
                current_chunk.append(part)
                current_length += separator_length + part_length
                continue

            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [part]
            current_length = part_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def _split_oversized_section(section: str, max_chunk_size: int) -> list[str]:
    """Split long sections on whitespace where possible, with a hard size cap."""
    parts: list[str] = []
    remaining = section

    while len(remaining) > max_chunk_size:
        split_at = max(
            (
                index
                for index, character in enumerate(
                    remaining[: max_chunk_size + 1]
                )
                if character.isspace()
            ),
            default=0,
        )
        if split_at == 0:
            split_at = max_chunk_size

        part = remaining[:split_at].strip()
        if part:
            parts.append(part)
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)
    return parts


if __name__ == "__main__":
    sample_md = """# Lineare Algebra

## 1. Vektoren und Matrizen
Ein Vektor ist ein Element eines Vektorraums.

$$\\mathbf{x} = \\begin{pmatrix} x_1 \\\\ x_2 \\end{pmatrix}$$

## 2. Matrixmultiplikation
Die Multiplikation zweier Matrizen $A$ und $B$ ist definiert als...
"""
    results = semantic_chunk_markdown(sample_md, max_chunk_size=300)
    for i, c in enumerate(results, 1):
        print(f"--- CHUNK {i} ({len(c)} Zeichen) ---\n{c}\n")