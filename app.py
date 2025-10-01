import io
import requests
import pdfplumber
import numpy as np
import faiss
import streamlit as st
from openai import OpenAI

# ---------------------
# Ask for API Key
# ---------------------
st.set_page_config(page_title="Financial Sustainability Toolkit Bot")
st.title("💬 Financial Sustainability Toolkit Q&A")

api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if not api_key:
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------------
# Load PDF + chunk
# ---------------------
@st.cache_resource
def build_index():
    PDF_URL = "https://globalschoolsforum.org/sites/default/files/2025-08/financial_sustainability_toolkit.pdf"
    resp = requests.get(PDF_URL)
    resp.raise_for_status()
    pdf_bytes = resp.content

    all_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text()
            if txt:
                all_text.append(f"[Page {page_num}]\n{txt}")
    full_text = "\n".join(all_text)

    # split into overlapping chunks
    chunk_size, overlap = 1500, 200
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start = end - overlap

    # embed chunks
    resp = client.embeddings.create(model="text-embedding-3-large", input=chunks)
    embs = [d.embedding for d in resp.data]

    # build faiss index
    index = faiss.IndexFlatL2(len(embs[0]))
    index.add(np.array(embs).astype("float32"))
    return chunks, index, embs

chunks, index, embs = build_index()

# ---------------------
# Helper: Answer question
# ---------------------
def answer_question(question):
    q_emb = client.embeddings.create(model="text-embedding-3-large", input=[question]).data[0].embedding
    q_emb = np.array([q_emb]).astype("float32")
    distances, indices = index.search(q_emb, 3)
    context = "\n---\n".join([chunks[i] for i in indices[0]])

    prompt = f"""
You are an assistant that answers questions using the “Financial Sustainability Toolkit”.

Context:
{context}

Question: {question}

Answer clearly and concisely. Cite page numbers when available.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------------------
# Chat UI
# ---------------------
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask me about the Toolkit..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer = answer_question(question)
        st.markdown(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})
