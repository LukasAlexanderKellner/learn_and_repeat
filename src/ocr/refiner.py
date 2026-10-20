from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

gemini_client = genai.Client()


def refine_document_layout_and_content(raw_markdown: str) -> str:
    """Refine document layout and content in two passes."""

    # Pass 1: layout and structure.
    prompt_layout = (
        "Du bist ein Experte für Dokumenten-Layout und Markdown-Formatierung. "
        "Du erhältst ein rohes, aus mehreren Seiten zusammengefügtes Markdown-Dokument. "
        "Aufgaben:\n"
        "1. Entferne künstliche Seitenumbrüche (z. B. '--- Seite X ---') und verbinde zerrissene Sätze/Absätze flüssig.\n"
        "2. Vereinheitliche die Überschriften-Hierarchie (#, ##, ###).\n"
        '3. Stelle sicher, dass Tabellen und Bildverweise bitte mit <img src="assets/sketch_p1_1.png" alt="Skizze" width="500" style="display: block; margin: 0 auto; max-width: 60%; height: auto;" /> sauber im Fließtext verankert sind. Zentriere die Bilder.\n'
        "4. Gib AUSSCHLIESSLICH das überarbeitete Markdown zurück. Füge allerdings keine zusätzlichen Markdown-Tags hinzu, du wirst ausschließlich Markdwon-Files bekommen, die zusätzlichen Tags würden dann nicht korrekt darstellen."
    )

    response_layout = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt_layout, raw_markdown],
    )
    layout_clean_md = response_layout.text

    # Pass 2: factual and mathematical corrections.
    prompt_content = (
        "Du bist ein brillanter Professor für Mathematik und Machine Learning. "
        "Überarbeite das folgende Markdown-Dokument inhaltlich und fachlich:\n"
        "1. Korrigiere alle LaTeX-Formeln penibel auf korrekte Syntax und Lesbarkeit ($...$ und $$...$$).\n"
        "2. Behebe offensichtliche mathematische Tippfehler oder Notationsfehler im Text.\n"
        "3. Achte darauf, dass Fachbegriffe im Bereich Machine Learning / Mathe präzise verwendet werden.\n"
        "4. Verändere nicht den Sinn des Textes und lösche keine Inhalte oder Bildverweise.\n"
        "5. Gib AUSSCHLIESSLICH das korrigierte Markdown zurück."
    )

    response_content = gemini_client.models.generate_content(
        model="gemini-2.5-flash",  # An Anthropic API client could be used here instead.
        contents=[prompt_content, layout_clean_md],
    )

    return response_content.text