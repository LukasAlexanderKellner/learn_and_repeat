from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

gemini_client = genai.Client()


def refine_document_layout_and_content(raw_markdown: str) -> str:
    """Führt eine zweistufige Korrekturschleife für Layout und Inhalt durch."""

    # Pass 1: Layout & Strukturierung
    prompt_layout = (
        "Du bist ein Experte für Dokumenten-Layout und Markdown-Formatierung. "
        "Du erhältst ein rohes, aus mehreren Seiten zusammengefügtes Markdown-Dokument. "
        "Aufgaben:\n"
        "1. Entferne künstliche Seitenumbrüche (z. B. '--- Seite X ---') und verbinde zerrissene Sätze/Absätze flüssig.\n"
        "2. Vereinheitliche die Überschriften-Hierarchie (#, ##, ###).\n"
        "3. Stelle sicher, dass Tabellen und Bildverweise (![Skizze...](assets/...)) sauber im Fließtext verankert sind.\n"
        "4. Gib AUSSCHLIESSLICH das überarbeitete Markdown zurück."
    )

    response_layout = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt_layout, raw_markdown],
    )
    layout_clean_md = response_layout.text

    # Pass 2: Inhaltliche & Mathematische Korrektur
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
        model="gemini-2.5-flash",  # Hier könnte alternativ auch ein Claude-Modell via Anthropic API genutzt werden
        contents=[prompt_content, layout_clean_md],
    )

    return response_content.text