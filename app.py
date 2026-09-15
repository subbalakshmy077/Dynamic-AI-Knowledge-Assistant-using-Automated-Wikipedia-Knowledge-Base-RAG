from pathlib import Path

import pandas as pd
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Wikipedia RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("Dynamic Wikipedia Knowledge Assistant")


# =====================================================
# Configuration
# =====================================================

ARTICLES_DIR = Path("AI_Knowledge_Base")
METADATA_FILE = ARTICLES_DIR / "metadata.csv"
FAISS_DIR = Path("faiss-db")


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:
    st.header("Knowledge Base")

    if FAISS_DIR.exists():
        st.success("FAISS database loaded")

        if METADATA_FILE.exists():
            metadata_df = pd.read_csv(METADATA_FILE)
            st.metric("Articles", len(metadata_df))

        txt_files = list(ARTICLES_DIR.glob("*.txt"))
        st.metric("Text Files", len(txt_files))

    else:
        st.error("FAISS database not found. Please run the notebook first.")

    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("- Ask questions about AI, ML, DL, NLP, etc.")
    st.markdown("- Answers are generated from Wikipedia articles.")
    st.markdown("- Sources are displayed below each answer.")


# =====================================================
# Check if FAISS exists
# =====================================================

if not FAISS_DIR.exists():
    st.error("FAISS database not found in 'faiss-db' folder.")
    st.info("Please run the '02_rag_pipeline.ipynb' notebook first to create the vector database.")
    st.stop()


# =====================================================
# Load Embeddings and Vector Database
# =====================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vector_db():
    embeddings = load_embeddings()

    vector_db = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db


vector_db = load_vector_db()


# =====================================================
# Chat Interface
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input("Ask a question about your knowledge base")


if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            results = vector_db.similarity_search(query, k=2)

            context = "\n\n".join(
                f"Source: {doc.metadata['title']}\n{doc.page_content}"
                for doc in results
            )

            prompt = f"""
You are an educational AI assistant.

Answer the question using only the following context.
If the answer is not available in the context, say:
"I could not find the answer in the knowledge base."

Do not invent information.
Give a clear explanation suitable for a beginner.

Context:
{context}

Question:
{query}

Answer:
"""

            llm = ChatOllama(
                model="tinyllama:1.1b",
                temperature=0
            )

            response = llm.invoke(prompt)

            st.markdown(response.content)

            st.markdown("### Sources")

            shown_sources = set()

            for doc in results:
                title = doc.metadata["title"]
                url = doc.metadata["url"]

                if title not in shown_sources:
                    st.markdown(f"- [{title}]({url})")
                    shown_sources.add(title)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content
            })