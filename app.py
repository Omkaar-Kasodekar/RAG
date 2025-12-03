import os
import math
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ----------- Setup -----------

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(page_title="RAG Chatbot (OpenRouter)", page_icon="🤖")
st.title("📚 RAG Chatbot using OpenRouter")

if not API_KEY:
    st.error("⚠ Add your OPENROUTER_API_KEY to .env")
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


with st.sidebar:
    st.header("⚙️ Settings")
    llm_model = st.selectbox(
        "Chat Model",
        [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
    )
    embedding_model = "text-embedding-3-small"   # fixed good embedding model

    chunk_size = st.slider("Chunk Size", 300, 2000, 800)
    overlap = st.slider("Overlap", 0, 500, 200)
    k = st.slider("Top K Retrieved Chunks", 1, 10, 4)

uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)

if "store" not in st.session_state:
    st.session_state.store = None


# ----------- Helper Functions -----------

def load_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""


def embed_texts(texts):
    """Embeds multiple texts using OpenRouter."""
    response = client.embeddings.create(
        model=embedding_model,
        input=texts
    )

    # If 1 item → return single vector
    if isinstance(texts, str) or len(texts) == 1:
        return [response.data[0].embedding]

    return [item.embedding for item in response.data]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0


# ----------- Build Knowledge Base -----------

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

    with st.spinner("Generating embeddings via OpenRouter..."):
        embeddings = embed_texts(texts)

    for i, emb in enumerate(embeddings):
        st.session_state.store[i]["embedding"] = emb

    st.success(f"Indexed {len(st.session_state.store)} chunks!")


# ----------- Chat -----------

query = st.text_input("Ask a question:")

if st.button("Ask") and query:
    if not st.session_state.store:
        st.error("⚠ Build the knowledge base first")
    else:
        q_emb = embed_texts([query])[0]

        ranked = sorted(
            st.session_state.store,
            key=lambda item: cosine(q_emb, item["embedding"]),
            reverse=True
        )[:k]

        context = "\n\n".join(f"[{item['source']}]\n{item['text']}" for item in ranked)

        prompt = f"""
Use ONLY the information below to answer the question.
If the answer isn't found in the context, reply: "I don't know."

Context:
{context}

Question: {query}

Answer:
"""

        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )

        bot_reply = response.choices[0].message.content

        st.write("### 🤖 Answer:")
        st.write(bot_reply)

        with st.expander("📎 Sources used"):
            for r in ranked:
                st.write(f"- {r['source']}")
