import base64
import json
import logging
import os
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from the .env file, including GEMINI_API_KEY.
load_dotenv()

# Configuration.
OLLAMA_URL = "http://localhost:11434/api/generate"
OCR_MODEL = "richardyoung/olmocr2:7b-q8"
GEMINI_LAYOUT_MODEL = "gemini-2.5-flash"

logger = logging.getLogger(__name__)
OLLAMA_TIMEOUT = (10, 600)

# Initialize the Gemini client, which reads GEMINI_API_KEY from the environment.
gemini_client = genai.Client()


def pdf_page_to_base64_and_bytes(
    page: fitz.Page, zoom: int = 2
) -> tuple[str, bytes, int, int]:
    """Convert a PDF page to Base64 and bytes, and return its pixel dimensions."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_b64, img_bytes, pix.width, pix.height


def run_olmocr(image_base64: str) -> str | None:
    """Send an image to the local olmocr2 model for text and LaTeX transcription."""
    prompt = (
        "You are an excellent OCR assistant for mathematical and machine learning documents. "
        "Analyze this document image and transcribe the entire text accurately in Markdown. "
        "Pay extreme attention to correctly formatting mathematical formulas and variables in LaTeX notation "
        "(Inline with $...$ and displayed with $$...$$). "
        "Ignore artifacts and output only the pure transcription."
    )

    payload = {
        "model": OCR_MODEL,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        result = response.json().get("response")
        if not isinstance(result, str) or not result.strip():
            logger.error("Ollama gab ein leeres OCR-Ergebnis zurück")
            return None
        return result
    except requests.exceptions.RequestException:
        logger.exception("Fehler bei der Kommunikation mit Ollama (olmocr2)")
        return None


def run_gemini_layout_detection(img_bytes: bytes) -> list[list[float]]:
    """Use Gemini 2.5 Flash to detect sketches, diagrams, and drawings as

    structured JSON with normalized coordinates from 0 to 1000.
    """
    prompt = (
        "Analysiere dieses Dokumentenbild. Identifiziere alle handgezeichneten Skizzen, "
        "Diagramme, Graphen oder visuellen Illustrationen. "
        "Gib ausschließlich ein JSON-Objekt zurück mit dem Schlüssel 'sketches', das eine Liste von Objekten enthält. "
        "Jedes Objekt muss das Format haben: {\"box_2d\": [y_min, x_min, y_max, x_max]}. "
        "Die Koordinaten müssen auf einer Skala von 0 bis 1000 normalisiert sein. "
        "Falls keine Skizzen vorhanden sind, gib {\"sketches\": []} zurück."
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_LAYOUT_MODEL,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,  # Keep coordinate detection deterministic.
                response_mime_type="application/json",  # Require valid JSON output.
            ),
        )

        data = json.loads(response.text)
        # Accept either a list or a dictionary response from Gemini.
        if isinstance(data, dict):
            boxes = data.get("sketches", data.get("boxes", []))
        elif isinstance(data, list):
            boxes = data
        else:
            boxes = []

        extracted_boxes = []
        for item in boxes:
            if isinstance(item, dict) and "box_2d" in item:
                box = item["box_2d"]
                if isinstance(box, list) and len(box) == 4:
                    extracted_boxes.append([float(c) for c in box])
            elif isinstance(item, list) and len(item) == 4:
                extracted_boxes.append([float(c) for c in item])

        return extracted_boxes
    except Exception:
        logger.exception("Fehler bei der Gemini Layout-Erkennung")
        return []


def crop_gemini_sketches(
    page: fitz.Page,
    normalized_boxes: list[list[float]],
    page_num: int,
    assets_dir: str,
    image_width: int,
    image_height: int,
) -> list[str]:
    """Convert Gemini's normalized boxes (0-1000) to pixels

    and crop the sketches with PyMuPDF.
    """
    os.makedirs(assets_dir, exist_ok=True)
    asset_paths = []
    page_rect = page.rect
    padding_x = image_width * 0.02
    padding_y = image_height * 0.02

    logger.info(
        "Seite %s: %s Skizzen von Gemini erkannt.",
        page_num,
        len(normalized_boxes),
    )

    for idx, box in enumerate(normalized_boxes, start=1):
        # Convert normalized 0-1000 coordinates to image pixels.
        y_min = (box[0] / 1000.0) * image_height
        x_min = (box[1] / 1000.0) * image_width
        y_max = (box[2] / 1000.0) * image_height
        x_max = (box[3] / 1000.0) * image_width

        if x_max <= x_min or y_max <= y_min:
            logger.warning(
                "Ungültige Bounding Box auf Seite %s übersprungen: %s",
                page_num,
                box,
            )
            continue

        # Add padding while keeping coordinates within the image bounds.
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(image_width, x_max + padding_x)
        y_max = min(image_height, y_max + padding_y)

        # Convert image coordinates to PDF page coordinates.
        rect = fitz.Rect(
            page_rect.x0 + x_min * page_rect.width / image_width,
            page_rect.y0 + y_min * page_rect.height / image_height,
            page_rect.x0 + x_max * page_rect.width / image_width,
            page_rect.y0 + y_max * page_rect.height / image_height,
        )
        rect &= page_rect

        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            continue

        asset_filename = f"sketch_p{page_num}_{idx}.png"
        asset_path = os.path.join(assets_dir, asset_filename)

        try:
            pix = page.get_pixmap(clip=rect, dpi=150)
            pix.save(asset_path)
            asset_paths.append(asset_filename)
            logger.info("Skizze erfolgreich gespeichert: %s", asset_path)
        except Exception:
            logger.exception(
                "Fehler beim Ausschneiden der Skizze auf Seite %s", page_num
            )

    return asset_paths


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

        # 1. Render the page for Ollama, Gemini, and coordinate conversion.
        img_b64, img_bytes, image_width, image_height = (
            pdf_page_to_base64_and_bytes(page)
        )

        # 2. Use Gemini 2.5 Flash for layout analysis and sketch detection.
        normalized_boxes = run_gemini_layout_detection(img_bytes)
        extracted_image_paths = crop_gemini_sketches(
            page,
            normalized_boxes,
            current_page_idx,
            assets_dir,
            image_width,
            image_height,
        )

        # 3. Run text OCR locally with Ollama (olmocr2).
        text_markdown = run_olmocr(img_b64)
        if text_markdown is None:
            text_markdown = f"[OCR fehlgeschlagen für Seite {current_page_idx}]"

        # 4. Embed extracted sketches in the Markdown.
        if extracted_image_paths:
            text_markdown += "\n\n### Skizzen dieser Seite:\n"
            for asset_filename in extracted_image_paths:
                text_markdown += (
                    f"\n![Skizze von Seite {current_page_idx}]"
                    f"(assets/{asset_filename})\n"
                )

        full_markdown.append(f"\n\n--- Seite {current_page_idx} ---\n\n")
        full_markdown.append(text_markdown)

    raw_markdown = "".join(full_markdown)

    output_path = os.path.join(output_dir, f"{base_name}_cleaned.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(raw_markdown)

    logger.info("Fertig! Markdown mit Gemini-Skizzen gespeichert unter: %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_pdf = "data/input_pdf/math_vorkurs.pdf"

    if os.path.exists(test_pdf):
        process_pdf_to_markdown(test_pdf, "data/output_markdown/")
    else:
        logger.warning(
            "Bitte lege eine Test-PDF unter '%s' ab, um den Test zu starten.",
            test_pdf,
        )