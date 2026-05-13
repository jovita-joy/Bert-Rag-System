
import streamlit as st
import os
from retriever import load_vectorstore, answer_question
from ingest import run_ingestion
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BERT Paper — AI Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🧠 BERT Paper — High-Precision Research Assistant")
st.caption("Layout-Aware RAG System | Revin Techno Solutions Technical Assessment | Built by Alphin Das")

st.markdown("""
> **Document:** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (Devlin et al., 2019)
> **System:** Stateless single-turn QA with layout-aware ingestion, MMR retrieval, and Gemini 2.0 Flash
""")

st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — System Info & Setup
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System")

    # Check if vector store exists
    chroma_exists = Path("./chroma_db").exists()

    if chroma_exists:
        st.success("✅ Vector store ready")
    else:
        st.warning("⚠️ Vector store not built yet")
        if st.button("🚀 Build Vector Store", type="primary"):
            with st.spinner("Ingesting BERT paper — this takes 2-3 minutes..."):
                try:
                    run_ingestion()
                    st.success("✅ Done! Refresh the page.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.header("📋 Challenge Queries")
    st.caption("Click to auto-fill:")

    challenge_queries = [
        "What are the two pre-training tasks used in BERT and how do they work?",
        "Compare the BERT-BASE and BERT-LARGE model sizes — how many parameters, layers, and hidden dimensions does each have?",
        "What does the input representation diagram show about how BERT constructs its input embeddings?"
    ]

    for i, q in enumerate(challenge_queries):
        if st.button(f"Query {i+1}", key=f"cq_{i}"):
            st.session_state["prefill_query"] = q

    st.divider()
    st.header("ℹ️ Architecture")
    st.markdown("""
    **Ingestion:**
    - PyMuPDF layout-aware parsing
    - 2-column reading order fix
    - Gemini Vision figure analysis

    **Retrieval:**
    - Google embedding-001
    - ChromaDB vector store
    - MMR search (k=6, fetch_k=20)

    **Generation:**
    - Gemini 2.0 Flash
    - Temperature: 0.1
    - Strict context-only prompt
    """)

# ─────────────────────────────────────────────
# LOAD VECTOR STORE
# ─────────────────────────────────────────────
@st.cache_resource
def get_vectorstore():
    return load_vectorstore()

# ─────────────────────────────────────────────
# MAIN QUERY INTERFACE
# ─────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    prefill = st.session_state.get("prefill_query", "")
    query = st.text_area(
        "Ask a question about the BERT paper:",
        value=prefill,
        height=100,
        placeholder="e.g. What are the two pre-training tasks in BERT?"
    )

with col2:
    st.write("")
    st.write("")
    search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state["prefill_query"] = ""
    st.rerun()

# ─────────────────────────────────────────────
# RUN QUERY
# ─────────────────────────────────────────────
if search_btn and query.strip():
    if not Path("./chroma_db").exists():
        st.error("Please build the vector store first using the sidebar button.")
    else:
        with st.spinner("🔍 Retrieving and reasoning..."):
            try:
                vs = get_vectorstore()
                result = answer_question(query, vs)

                # ── ANSWER ──
                st.subheader("📝 Answer")
                st.markdown(result["answer"])

                st.caption(f"Retrieved {result['chunks_retrieved']} source chunks")
                st.divider()

                # ── RETRIEVED SOURCES ──
                st.subheader("📚 Retrieved Source Chunks")
                st.caption("Expand each chunk to see exactly what the backend retrieved to construct the answer.")

                for i, source in enumerate(result["sources"]):
                    chunk_type = "🖼️ Figure" if source["type"] == "figure" else "📄 Text"
                    label = f"{chunk_type} | {source['source']}"

                    with st.expander(f"Chunk {i+1} — {label}"):
                        st.markdown(f"**Page:** {source['page']}")
                        st.markdown(f"**Type:** {source['type']}")
                        st.markdown("**Content:**")
                        st.text(source["content"])

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure your GOOGLE_API_KEY is set in your .env file")

elif search_btn and not query.strip():
    st.warning("Please enter a question first.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("Built by Jovita Joy | Revin Techno Solutions Technical Assessment | Layout-Aware RAG with Gemini 2.0 Flash + ChromaDB")
