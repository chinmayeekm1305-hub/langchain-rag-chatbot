"""
Tax Document Assistant - Streamlit web app (Version 2: big PDFs)
Run with:  streamlit run app.py
It re-uses the RAG pipeline from qa_agent.py (same folder).
"""
from pathlib import Path

import streamlit as st

from qa_agent import (
    get_vectorstore,
    build_agent,
    load_pdf_pages,
    split_into_chunks,
    add_in_batches,
    format_sources,
    DOCS_DIR,
    DB_DIR,
)

# ---------------- Page setup ----------------
st.set_page_config(page_title="Tax Document Assistant", page_icon="📄")
st.title("📄 Tax Document Assistant")
st.caption(
    "Ask questions about your GST / Income Tax documents. "
    "Answers come only from the documents in the knowledge base, with page references."
)

Path(DOCS_DIR).mkdir(exist_ok=True)


# ---------------- Helpers ----------------
@st.cache_resource(show_spinner=False)
def load_system():
    """Load (or build) the vector DB and the agent once, then reuse it."""
    vector_store = get_vectorstore()
    agent = build_agent(vector_store)
    return vector_store, agent


def has_documents():
    return any(
        p.suffix.lower() in {".pdf", ".txt", ".md"}
        for p in Path(DOCS_DIR).rglob("*")
    )


# ---------------- Sidebar: upload documents ----------------
if "processed" not in st.session_state:
    st.session_state.processed = set()

with st.sidebar:
    st.header("📁 Documents")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is not None and uploaded.name not in st.session_state.processed:
        st.session_state.processed.add(uploaded.name)
        save_path = Path(DOCS_DIR) / uploaded.name

        if save_path.exists():
            st.info(f"'{uploaded.name}' is already in the knowledge base.")
        else:
            save_path.write_bytes(uploaded.getvalue())
            pages = load_pdf_pages(save_path)

            if not pages:
                save_path.unlink()
                st.error("No text found in this PDF. It looks like a scanned PDF - please upload a text-based one.")
            elif Path(DB_DIR).exists():
                # Knowledge base already exists -> add only the new file's chunks
                chunks = split_into_chunks(pages)
                bar = st.progress(0, text=f"Embedding {len(chunks)} chunks...")
                vector_store, _ = load_system()
                add_in_batches(
                    vector_store, chunks,
                    on_progress=lambda done, total: bar.progress(
                        done / total, text=f"Embedded {done}/{total} chunks"
                    ),
                )
                st.success(f"Added '{uploaded.name}': {len(pages)} pages, {len(chunks)} chunks.")
            else:
                st.success(f"Saved '{uploaded.name}'. Building the knowledge base...")

    st.subheader("In the knowledge base")
    files = sorted(p.name for p in Path(DOCS_DIR).glob("*") if p.is_file())
    if files:
        for name in files:
            st.write("•", name)
    else:
        st.write("No documents yet.")

# ---------------- Main area: chat ----------------
if not has_documents():
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

with st.spinner("Loading the knowledge base (first time with big PDFs can take several minutes)..."):
    vector_store, agent = load_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.caption("📎 Sources: " + " | ".join(m["sources"]))

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents and writing the answer..."):
            result = agent.invoke({
                "messages": [{"role": "user", "content": question}],
                "context": [],
            })
        answer = str(result["messages"][-1].content)
        sources = format_sources(result.get("context", []))
        st.markdown(answer)
        if sources:
            st.caption("📎 Sources: " + " | ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
