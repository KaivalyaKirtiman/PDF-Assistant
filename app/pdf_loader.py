from pathlib import Path
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extracts and returns text from a PDF file.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"{pdf_path} not found.")
    
    reader = PdfReader(pdf_path)
    full_text = ""

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text.strip() + "\n"
        else:
            print(f"No extractable text found on page {i + 1}")

    if not full_text.strip():
        raise ValueError("No extractable text found in the PDF.")

    return full_text.strip()
