# ============================================================
# app/services/ocr_service.py
# CONSERVÉ tel quel — EasyOCR fonctionne bien
# ============================================================

import fitz
import easyocr

# Initialisation du lecteur (FR + EN pour documents marocains)
reader = easyocr.Reader(['fr', 'en'])


def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    """
    Extrait le texte d'un PDF scanné via OCR (EasyOCR).
    Utilisé quand extract_text_from_pdf retourne < 50 caractères.
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        result = reader.readtext(img_bytes, detail=0)
        full_text += " ".join(result) + "\n"

    return full_text