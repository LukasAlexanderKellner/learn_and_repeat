import base64
import json
import logging
import os
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.genai import types
from refiner import refine_document_layout_and_content

# Lädt Umgebungsvariablen aus der .env-Datei (inkl. GEMINI_API_KEY)
load_dotenv()

# Konfiguration
GEMINI_MODEL = "gemini-2.5-flash"

logger = logging.getLogger(__name__)

# Initialisiere den Gemini Client
gemini_client = genai.Client()


def pdf_page_to_bytes(page: fitz.Page, zoom: int = 2) -> tuple[bytes, int, int]:
    """Konvertiert eine PDF-Seite in PNG-Rohbytes und liefert die Pixel-Dimensionen."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    return img_bytes, pix.width, pix.height


def process_page_with_gemini(
    img_bytes: bytes, page_num: int
) -> tuple[str, list[dict]]:
    """Sendet das Bild an Gemini 2.5 Flash, um Text/LaTeX zu transkribieren

    und Skizzen direkt inline mit Platzhaltern im Lesefluss zu positionieren.
    """
    prompt = (
        "Du bist ein exzellenter OCR- und Layout-Assistent für mathematische und maschinelle Lern-Dokumente. "
        f"Analysiere dieses Dokumentenbild von Seite {page_num} und führe zwei Aufgaben aus:\n"
        "1. Transkribiere den Text und die Formeln präzise in Markdown. Achte penibel auf LaTeX-Formeln ($...$ und $$...$$).\n"
        "2. Identifiziere alle handgezeichneten Skizzen, Diagramme oder Graphen. "
        "Füge für jede Skizze einen Markdown-Bild-Tag im exakten Format "
        f"`![Skizze Seite {page_num} Nr. X](assets/sketch_p{page_num}_X.png)` ein "
        "(wobei X durchnummeriert ist: 1, 2, 3...), und zwar **genau an der vertikalen Stelle im Text**, "
        "an der die Skizze im Originaldokument steht.\n\n"
        "Gib deine Antwort ausschließlich als valides JSON-Objekt mit folgender Struktur zurück:\n"
        "{\n"
        '  "markdown_content": "Der vollständige Markdown-Text inklusive der eingebetteten Bild-Tags an den richtigen Stellen",\n'
        '  "sketches": [\n'
        '    {"index": 1, "box_2d": [y_min, x_min, y_max, x_max]}\n'
        "  ]\n"
        "}\n"
        "Die Koordinaten in 'box_2d' müssen auf eine Skala von 0 bis 1000 normalisiert sein. "
        "Falls keine Skizzen vorhanden sind, gib eine leere Liste 'sketches: []' zurück."
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.1, response_mime_type="application/json"
            ),
        )

        data = json.loads(response.text)
        markdown_content = data.get("markdown_content", "")
        sketches = data.get("sketches", [])
        return markdown_content, sketches
    except Exception:
        logger.exception(
            "Fehler bei der Gemini-Verarbeitung auf Seite %s", page_num
        )
        return f"[Fehler bei der OCR-Verarbeitung auf Seite {page_num}]", []


def crop_and_save_sketches(
    page: fitz.Page,
    sketches: list[dict],
    page_num: int,
    assets_dir: str,
    image_width: int,
    image_height: int,
):
    """Schneidet die Skizzen basierend auf den von Gemini gelieferten normalisierten

    Koordinaten aus und speichert sie unter dem exakten Dateinamen ab, den Gemini im Markdown referenziert hat.
    """
    os.makedirs(assets_dir, exist_ok=True)
    page_rect = page.rect
    padding_x = image_width * 0.02
    padding_y = image_height * 0.02

    for item in sketches:
        idx = item.get("index", 1)
        box = item.get("box_2d")
        if not box or len(box) != 4:
            continue

        # Denormalisierung: Skala 0-1000 auf tatsächliche Bildpixel umrechnen
        y_min = (box[0] / 1000.0) * image_height
        x_min = (box[1] / 1000.0) * image_width
        y_max = (box[2] / 1000.0) * image_height
        x_max = (box[3] / 1000.0) * image_width

        if x_max <= x_min or y_max <= y_min:
            continue

        # Padding hinzufügen
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(image_width, x_max + padding_x)
        y_max = min(image_height, y_max + padding_y)

        # Auf PDF-Seitenkoordinaten übertragen
        rect = fitz.Rect(
            page_rect.x0 + x_min * page_rect.width / image_width,
            page_rect.y0 + y_min * page_rect.height / image_height,
            page_rect.x0 + x_max * page_rect.width / image_width,
            page_rect.y0 + y_max * page_rect.height / image_height,
        )
        rect &= page_rect

        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            continue

        # Dateiname muss exakt zu dem passen, den wir im Prompt vorgegeben haben!
        asset_filename = f"sketch_p{page_num}_{idx}.png"
        asset_path = os.path.join(assets_dir, asset_filename)

        try:
            pix = page.get_pixmap(clip=rect, dpi=150)
            pix.save(asset_path)
            logger.info("Inlinierte Skizze erfolgreich gespeichert: %s", asset_path)
        except Exception:
            logger.exception(
                "Fehler beim Ausschneiden der Skizze %s auf Seite %s",
                idx,
                page_num,
            )


def process_pdf_to_markdown(pdf_path: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    logger.info("Starte Pipeline für '%s' (%s Seiten)...", base_name, len(doc))

    full_markdown = []

    for page_num in range(len(doc)):
        current_page_idx = page_num + 1
        logger.info("Verarbeite Seite %s/%s...", current_page_idx, len(doc))
        page = doc[page_num]

        # 1. Seite als Bytes rendern + Dimensionen holen
        img_bytes, image_width, image_height = pdf_page_to_bytes(page)

        # 2. Gemini verarbeitet Text, LaTeX UND platziert die Bild-Tags im Textfluss
        markdown_content, sketches = process_page_with_gemini(
            img_bytes, current_page_idx
        )

        # 3. Skizzen exakt nach den von Gemini definierten Indizes ausschneiden und abspeichern
        if sketches:
            crop_and_save_sketches(
                page,
                sketches,
                current_page_idx,
                assets_dir,
                image_width,
                image_height,
            )

        full_markdown.append(f"\n\n--- Seite {current_page_idx} ---\n\n")
        full_markdown.append(markdown_content)

    raw_markdown = "".join(full_markdown)
    final_markdown = refine_document_layout_and_content(raw_markdown)

    output_path = os.path.join(output_dir, f"{base_name}_cleaned.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    logger.info(
        "Fertig! Markdown mit perfekt inlinierten Skizzen gespeichert unter: %s",
        output_path,
    )
    return output_path


def load_processed_files(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()

    with state_path.open("r", encoding="utf-8") as state_file:
        state = json.load(state_file)

    if not isinstance(state, dict):
        raise ValueError(f"State file must contain a JSON object: {state_path}")

    processed_files = state.get("processed_files", [])
    if not isinstance(processed_files, list) or not all(
        isinstance(path, str) for path in processed_files
    ):
        raise ValueError(
            f"'processed_files' must be a list of paths in {state_path}"
        )

    return set(processed_files)


def save_processed_files(state_path: Path, processed_files: set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"processed_files": sorted(processed_files)}
    temporary_path = state_path.with_suffix(f"{state_path.suffix}.tmp")

    with temporary_path.open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, ensure_ascii=False, indent=2)
        state_file.write("\n")

    temporary_path.replace(state_path)


def process_untracked_pdfs(project_root: Path) -> None:
    project_root = project_root.resolve()
    input_dir = project_root / "data" / "input_pdf"
    output_dir = project_root / "data" / "output_markdown"
    state_path = project_root / "data" / "state.json"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"PDF input directory does not exist: {input_dir}")

    processed_files = load_processed_files(state_path)
    pdf_paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.lower(),
    )

    if not pdf_paths:
        logger.info("No PDF files found in %s", input_dir)
        return

    for pdf_path in pdf_paths:
        relative_path = pdf_path.resolve().relative_to(project_root).as_posix()
        if relative_path in processed_files:
            logger.info("Skipping already processed PDF: %s", relative_path)
            continue

        logger.info("Processing new PDF: %s", relative_path)
        process_pdf_to_markdown(str(pdf_path), str(output_dir))
        processed_files.add(relative_path)
        save_processed_files(state_path, processed_files)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repository_root = Path(__file__).resolve().parents[2] 
    process_untracked_pdfs(repository_root)
