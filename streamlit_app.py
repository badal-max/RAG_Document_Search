"""Streamlit UI for Agentic RAG System (Groq-powered)"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstores.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


# ---------------- Page config ---------------- #

st.set_page_config(
    page_title="🤖 Agentic RAG (Groq)",
    page_icon="🔍",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- Session state ---------------- #

def init_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []


# ---------------- RAG initialization ---------------- #

@st.cache_resource
def initialize_rag():
    try:
        # LLM (Groq)
        llm = Config.get_llm()

        # Document processor
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

        # Vector store
        vector_store = VectorStore()

        # Sources (URLs / files)
        sources = Config.DEFAULT_URLS

        # Load + split documents
        documents = doc_processor.process_sources(sources)

        # Build vector index
        vector_store.create_vectorstore(documents)

        # Build LangGraph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()

        return graph_builder, len(documents)

    except Exception as e:
        st.error("❌ Failed to initialize RAG system")
        st.exception(e)
        return None, 0


# ---------------- Main app ---------------- #

def main():
    init_session_state()

    st.title("🔍 Agentic RAG Search (Groq)")
    st.markdown("Ask questions about the loaded documents.")

    # Initialize system once
    if not st.session_state.initialized:
        with st.spinner("Loading RAG system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")

    st.markdown("---")

    # Search form
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")

    # Handle query
    if submit and question and st.session_state.rag_system:
        with st.spinner("Thinking..."):
            start_time = time.time()

            result = st.session_state.rag_system.run(question)

            elapsed_time = time.time() - start_time

            st.session_state.history.append({
                "question": question,
                "answer": result["answer"],
                "time": elapsed_time,
            })

            st.markdown("### 💡 Answer")
            st.success(result["answer"])

            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(result["retrieved_docs"], 1):
                    st.text_area(
                        f"Document {i}",
                        doc.page_content[:300] + "...",
                        height=100,
                        disabled=True
                    )

            st.caption(f"⏱️ Response time: {elapsed_time:.2f}s")

    # History
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")

        for item in reversed(st.session_state.history[-3:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer'][:200]}...")
            st.caption(f"Time: {item['time']:.2f}s")


if __name__ == "__main__":
    main()
