from app.pdf_loader import extract_text_from_pdf

pdf_path = "data/sample.pdf"  # Make sure this file exists
text = extract_text_from_pdf(pdf_path)
print(text[:1000])  # Print first 1000 characters of extracted text
