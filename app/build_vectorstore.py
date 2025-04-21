import os
from pdf_loader import extract_text_from_pdf
from chunk_text import chunk_text


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def build_vectorstore_from_pdf(pdf_path: str, vectorstore_path: str = "vectorstore/faiss_index"):
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("Chunking text with metadata...")
    chunks = chunk_text(text)

    documents = [
        Document(page_content=chunk, metadata={"source": os.path.basename(pdf_path), "chunk_index": idx})
        for idx, chunk in enumerate(chunks)
    ]

    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating FAISS vectorstore...")
    db = FAISS.from_documents(documents, embedding=embeddings)

    os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
    db.save_local(vectorstore_path)

    print(f"Vectorstore saved to {vectorstore_path}")
