import fitz

from app.services.ocr_service import (
    extract_text_from_scanned_pdf
)


def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    return text


def get_text_with_ocr_fallback(pdf_path):

    text = extract_text_from_pdf(pdf_path)

    # OCR si texte vide/faible
    if not text or len(text.strip()) < 50:

        print("🔍 OCR ACTIVATED")

        text = extract_text_from_scanned_pdf(
            pdf_path
        )

    return text