import streamlit as st
import os
from pdf_loader import extract_text_from_pdf
from build_vectorstore import build_vectorstore_from_pdf

# Ensure data/ folder exists
os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="PDF Assistant", layout="wide")
st.title("📚 PDF Assistant")
st.write("Upload a PDF, build a knowledge base, and ask questions!")

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    file_path = f"data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully.")
    st.subheader("Extracted Text Preview:")
    text = extract_text_from_pdf(file_path)
    st.text_area("PDF Text", value=text[:1500] + "...", height=200)

    if st.button("🔨 Build Vectorstore"):
        build_vectorstore_from_pdf(file_path)
        st.success("Vectorstore built successfully!")

# Use updated imports for LangChain v0.2+
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = None

if uploaded_file and st.session_state.get("vectorstore_uploaded") != uploaded_file.name:
    build_vectorstore_from_pdf(file_path)
    st.session_state.vectorstore_uploaded = uploaded_file.name
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    if os.path.exists("vectorstore/faiss_index/index.faiss"):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)


from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

st.subheader("🧠 Ask a Question")
query = st.text_input("Your question about the PDF:")

if query:
    st.write("🔍 Searching knowledge base...")
    results = vectorstore.similarity_search(query, k=3)

    context = "\n\n---\n\n".join([f"[Chunk {res.metadata.get('chunk_index')}] {res.page_content}" for res in results])

    prompt = f"""You are an intelligent assistant. Use the provided context to answer the user's question.
If the answer is not found in the context, say "Sorry, I couldn't find the answer in the document."

Context:
{context}

Question: {query}
Answer:"""

    st.write("🤖 Thinking...")
    response = llm.invoke(prompt)

    st.subheader("📌 Answer:")
    st.write(response)

    with st.expander("📄 See Context Used"):
        st.markdown(context)
