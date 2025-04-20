from app.chunk_text import chunk_text
from app.pdf_loader import extract_text_from_pdf

pdf_path = "data/sample.pdf"
text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text, max_length=700, overlap=100)

for i, chunk in enumerate(chunks[:3]):  # show first 3 chunks
    print(f"\n--- Chunk {i+1} ---\n{chunk}\n")
