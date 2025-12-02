import os
import math
import tempfile
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from together import Together
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
API_KEY = os.getenv("TOGETHER_API_KEY")

st.set_page_config(page_title="RAG Chatbot (Python 3.13)", page_icon="🤖")
st.title("RAG Chatbot using Together.ai")

if not API_KEY:
    st.error("⚠ Add TOGETHER_API_KEY in a .env file.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "LLM Model",
        [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "microsoft/Phi-3-mini-128k-instruct",
        ],
    )
    chunk_size = st.slider("Chunk size", 300, 2000, 800)
    overlap = st.slider("Overlap", 0, 500, 200)
    k = st.slider("Top K retrieved chunks", 1, 10, 4)

uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)

if "store" not in st.session_state:
    st.session_state.store = None

client = Together()

# HELPERS
def load_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)
    return ""


def embed_texts(texts):
    response = client.embeddings.create(
        model="togethercomputer/m2-bert-80M-2k-retrieval",
        input=texts
    )
    return [d.embedding for d in response.data]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0


# BUILD KB 
if uploaded and st.button("Build Knowledge Base"):
    st.session_state.store = []
    texts = []

    for f in uploaded:
        raw = load_text(f)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_text(raw)

        for c in chunks:
            st.session_state.store.append({"source": f.name, "text": c})
            texts.append(c)

    embeddings = embed_texts(texts)
    for i, emb in enumerate(embeddings):
        st.session_state.store[i]["embedding"] = emb

    st.success(f"Indexed {len(st.session_state.store)} document chunks.")


# CHAT 
query = st.text_input("Ask a question:")

if st.button("Ask") and query:
    if not st.session_state.store:
        st.error("Upload files first")
    else:
        q_emb = embed_texts([query])[0]

        ranked = sorted(
            st.session_state.store,
            key=lambda item: cosine(q_emb, item["embedding"]),
            reverse=True
        )[:k]

        context = "\n\n".join(
            f"[{item['source']}]\n{item['text']}" for item in ranked
        )

        prompt = f"""
You are a helpful assistant. Answer using ONLY the context below.

Context:
{context}

Question: {query}

Answer:
        """

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        answer = response.choices[0].message["content"]
        st.write("### Answer:")
        st.write(answer)

        with st.expander("Sources"):
            for r in ranked:
                st.write(f"- {r['source']}")
