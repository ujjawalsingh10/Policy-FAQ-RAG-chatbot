import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

load_dotenv()

st.set_page_config(page_title="Policy FAQ RAG Chatbot", page_icon="📄", layout="wide")

# Root path for finding the folder
PROJECT_ROOT = Path(__file__).resolve().parent

with st.sidebar:
    st.title("⚙️ Configuration")
    llm_name = st.text_input("LLM (Groq model)", "llama-3.1-8b-instant")
    embed_model = st.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")
    db_dir = st.text_input("Vector DB directory", "vectordb")
    top_k = st.slider("Top-K retrieved chunks", 1, 8, 4)
    
    st.markdown("---")
    
    # Reset Chat Option
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.session_state.pipeline = None
        st.rerun()  # Forces the app to reload with fresh state
        
    # Path check relative to project root
    index_path = PROJECT_ROOT / db_dir / "index.faiss"
    if index_path.exists():
        st.success("✅ Vector DB found")
    else:
        st.warning(f"Vector DB not found at {index_path}")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

st.title("📄 Policy FAQ RAG Chatbot")

if query := st.chat_input("Ask a question about the document…"):
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY missing.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.pipeline is None:
        with st.spinner("Loading RAG pipeline…"):
            st.session_state.pipeline = RAGPipeline(db_dir=db_dir, embed_model=embed_model, llm_name=llm_name, top_k=top_k)

    with st.chat_message("assistant"):
        sources = st.session_state.pipeline.get_sources(query, top_k=top_k)
        answer = st.write_stream(st.session_state.pipeline.stream(query))
        
        with st.expander("📎 Source passages"):
            for src in sources:
                st.markdown(f"**{src['chunk_id']}**")
                st.caption(src["text"])
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})