from src.database.connection import PROJECT_ROOT
from src.database.ingest import ingest_markdown_file


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "output_markdown"
    markdown_files = sorted(output_dir.glob("*_cleaned.md"))
    if not markdown_files:
        print(f"No cleaned Markdown files found in {output_dir}")
        return

    for markdown_path in markdown_files:
        pdf_name = f"{markdown_path.stem.removesuffix('_cleaned')}.pdf"
        print(f"Ingesting {markdown_path.name}")
        ingest_markdown_file(str(markdown_path), pdf_name)


if __name__ == "__main__":
    main()
