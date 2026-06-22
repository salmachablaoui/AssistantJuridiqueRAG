import fitz
import easyocr

reader = easyocr.Reader(['fr', 'en'])

def extract_text_from_scanned_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        pix = page.get_pixmap(dpi=300)

        img_bytes = pix.tobytes("png")

        result = reader.readtext(img_bytes, detail=0)

        full_text += " ".join(result) + "\n"

    return full_text