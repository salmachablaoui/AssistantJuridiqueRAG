# ============================================================
# app/services/pdf_service.py
# CONSERVÉ — extraction PDF avec fallback OCR (logique existante)
# ============================================================

import fitz
import structlog
from app.services.ocr_service import extract_text_from_scanned_pdf

logger = structlog.get_logger()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extraction texte natif depuis PDF (PyMuPDF)"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def get_text_with_ocr_fallback(pdf_path: str) -> str:
    """
    Logique originale conservée :
    1. Tente extraction texte natif
    2. Si résultat < 50 chars → active l'OCR (EasyOCR)
    """
    text = extract_text_from_pdf(pdf_path)

    if not text or len(text.strip()) < 50:
        logger.info("ocr_activated", pdf_path=pdf_path)
        text = extract_text_from_scanned_pdf(pdf_path)

    logger.info(
        "pdf_extracted",
        pdf_path=pdf_path,
        char_count=len(text),
        used_ocr=len(text.strip()) < 50
    )
    return text