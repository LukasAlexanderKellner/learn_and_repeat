import re


def semantic_chunk_markdown(
    markdown_text: str, max_chunk_size: int = 1000
) -> list[str]:
    """Spaltet ein Markdown-Dokument an logischen Grenzen (Überschriften, Paragraphen)

    in semantisch zusammenhängende Chunks auf.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")

    # Spalte primär an Markdown-Überschriften (#, ##, ###) oder Absätzen (\n\n)
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