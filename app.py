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
st.title("RAG Chatbot using OpenRouter")

if not API_KEY:
    st.error("⚠ Add your OPENROUTER_API_KEY to .env")
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    llm_model = st.selectbox(
        "Chat Model",
        [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
    )
    embedding_model = "text-embedding-3-small"

    chunk_size = st.slider("Chunk Size", 300, 2000, 800)
    overlap = st.slider("Chunk Overlap", 0, 500, 200)
    k = st.slider("Top K Retrieved Chunks", 1, 10, 4)

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.experimental_rerun()

uploaded = st.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# --- Session state ---
if "store" not in st.session_state:
    st.session_state.store = None
if "chat_history" not in st.session_state:
    # list of {"role": "user"/"assistant", "content": "..."}
    st.session_state.chat_history = []


# ----------- Helper Functions -----------

def load_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""


def embed_texts(texts):
    resp = client.embeddings.create(
        model=embedding_model,
        input=texts
    )
    return [item.embedding for item in resp.data]


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
        if not raw.strip():
            continue

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        chunks = splitter.split_text(raw)

        for c in chunks:
            st.session_state.store.append({"source": f.name, "text": c})
            texts.append(c)

    if not texts:
        st.error("No readable text found in uploaded files.")
    else:
        with st.spinner("Generating embeddings via OpenRouter..."):
            embeddings = embed_texts(texts)

        for i, emb in enumerate(embeddings):
            st.session_state.store[i]["embedding"] = emb

        st.success(f"Indexed {len(st.session_state.store)} chunks!")


# ----------- Chat UI with bubbles -----------

st.subheader("Chat with your documents")

# 1) Render previous messages as chat bubbles
for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# 2) Input for new message
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    if not st.session_state.store:
        with st.chat_message("assistant"):
            st.markdown("⚠ Please upload files and build the knowledge base first.")
    else:
        # Show user's bubble immediately
        with st.chat_message("user"):
            st.markdown(user_query)

        # --- Retrieval ---
        q_emb = embed_texts([user_query])[0]
        ranked = sorted(
            st.session_state.store,
            key=lambda item: cosine(q_emb, item["embedding"]),
            reverse=True
        )[:k]

        context = "\n\n".join(
            f"[{item['source']}]\n{item['text']}" for item in ranked
        )

        system_prompt = (
            "You are a helpful assistant answering questions about a set of documents. "
            "Use ONLY the information in the provided context to answer the user. "
            "If the answer is not in the context, say \"I don't know.\" "
            "Keep answers concise but clear."
        )

        history_messages = st.session_state.chat_history.copy()
        current_user_message = {
            "role": "user",
            "content": f"Context:\n{context}\n\nUser question: {user_query}"
        }
        messages = (
            [{"role": "system", "content": system_prompt}]
            + history_messages
            + [current_user_message]
        )

        # --- Model call ---
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=messages,
                    max_tokens=400
                )
                bot_reply = response.choices[0].message.content
                st.markdown(bot_reply)

                with st.expander("📎 Sources used this turn"):
                    for r in ranked:
                        st.write(f"- {r['source']}")

        # --- Update memory AFTER rendering ---
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
