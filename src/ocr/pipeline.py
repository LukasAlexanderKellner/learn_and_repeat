import base64
from io import BytesIO
import logging
import os
import fitz  # PyMuPDF
import requests

# Konfiguration für Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OCR_MODEL = "richardyoung/olmocr2:7b-q8"

logger = logging.getLogger(__name__)


def pdf_page_to_base64(page: fitz.Page, zoom: int = 2) -> str:
    """Converts a PDF-Page in a PNG Picture and returns it as Base64-String.
    """
    # Zoomfactor increases the resolution, so that formulas stay sharp (DPI increase)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")


def run_olmocr(image_base64: str) -> str:
    """Sends the picture to the local Ollama olmocr2 model."""
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
            "temperature": 0.1,  # Low temperature for factually accurate OCR
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException:
        logger.exception("Fehler bei der Kommunikation mit Ollama")
        return ""


def process_pdf_to_markdown(pdf_path: str, output_dir: str):
    """ Mainfunction: converts a full Goodnotes-PDF pagewise to Markdown."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = fitz.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    logger.info("Starte OCR für '%s' (%s Seiten)...", base_name, len(doc))

    full_markdown = []

    for page_num in range(len(doc)):
        logger.info("Verarbeite Seite %s/%s...", page_num + 1, len(doc))
        page = doc[page_num]

        # 1. Page to Base64 image
        img_b64 = pdf_page_to_base64(page)

        # 2. Ocr with Ollama Model
        ocr_text = run_olmocr(img_b64)

        full_markdown.append(f"\n\n--- Seite {page_num + 1} ---\n\n")
        full_markdown.append(ocr_text)

    # safe the full markdown to a file
    output_path = os.path.join(output_dir, f"{base_name}_cleaned.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(full_markdown))

    logger.info("Fertig! Markdown gespeichert unter: %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_pdf = "../../data/input_pdfs/beispiel_mathe.pdf"

    if os.path.exists(test_pdf):
        process_pdf_to_markdown(test_pdf, "../../data/processed/")
    else:
        logger.warning(
            "Bitte lege eine Test-PDF unter '%s' ab, um den Test zu starten.",
            test_pdf,
        )