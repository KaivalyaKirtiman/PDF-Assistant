from typing import List

def chunk_text(text: str, max_length: int = 700, overlap: int = 100) -> List[str]:
    """
    Splits text into overlapping chunks to preserve context.
    """
    if not text.strip():
        raise ValueError("Input text is empty. Cannot chunk.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_length, text_length)
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start += max_length - overlap

    return chunks
