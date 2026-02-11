import json
import os
import sys
import webbrowser
from pathlib import Path

import streamlit as st

from src.config import CHUNKS_FILE, EMBEDDINGS_FILE, INDEX_DIR, RAW_PAPERS_DIR
from src.rag_chain import answer_question

METADATA_PATH = RAW_PAPERS_DIR / "metadata.json"


@st.cache_data
def load_source_links() -> dict[str, str]:
    if not METADATA_PATH.exists():
        return {}

    try:
        rows = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    links: dict[str, str] = {}
    for row in rows:
        pdf_file = row.get("pdf_file")
        arxiv_url = row.get("arxiv_url")
        if pdf_file and arxiv_url:
            links[pdf_file] = arxiv_url
    return links


def local_pdf_path(source_file: str) -> Path | None:
    pdf_path = (RAW_PAPERS_DIR / source_file).resolve()
    return pdf_path if pdf_path.exists() else None


def open_local_pdf(pdf_path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(pdf_path))  # type: ignore[attr-defined]
    else:
        webbrowser.open(pdf_path.as_uri())


SOURCE_LINKS = load_source_links()

st.set_page_config(page_title="AI Papers RAG", page_icon="books", layout="wide")

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

st.title("AI Research Papers RAG")
st.caption("Single-turn Q&A over your indexed AI papers dataset.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Papers folder", str(RAW_PAPERS_DIR))
with col2:
    st.metric("Local index", str(INDEX_DIR))

if not CHUNKS_FILE.exists() or not EMBEDDINGS_FILE.exists():
    st.error("No index found. Run: python -m src.ingest --reset")
    st.stop()

question = st.text_input(
    "Ask a question",
    placeholder="What are the key contributions of transformer-based models?",
)
ask = st.button("Get answer", type="primary")

if ask:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Searching papers and generating answer..."):
            try:
                answer, docs = answer_question(question)
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        st.session_state.last_answer = answer
        st.session_state.last_docs = docs

if st.session_state.last_answer is not None:
    st.subheader("Answer")
    st.write(st.session_state.last_answer)

    st.subheader("Top Sources")
    docs = st.session_state.last_docs
    if not docs:
        st.info("No sources returned.")

    for i, doc in enumerate(docs, start=1):
        source = doc.get("source_file", "unknown")
        page = doc.get("page", "?")
        snippet = " ".join(doc.get("text", "").split())
        snippet = snippet[:450] + ("..." if len(snippet) > 450 else "")

        source_url = SOURCE_LINKS.get(source)
        pdf_path = local_pdf_path(source)
        key_base = f"{i}_{source}_{page}".replace(" ", "_").replace(".", "_")

        st.markdown(f"**{i}. {source} (p.{page})**")

        c1, c2, c3 = st.columns(3)
        with c1:
            if source_url:
                st.link_button("Open on arXiv", source_url)
        with c2:
            if pdf_path is not None:
                if st.button("Open local PDF", key=f"open_{key_base}"):
                    try:
                        open_local_pdf(pdf_path)
                        st.success(f"Opened: {pdf_path.name}")
                    except Exception as exc:
                        st.error(f"Failed to open local PDF: {exc}")
        with c3:
            if pdf_path is not None:
                st.download_button(
                    "Download PDF",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=f"download_{key_base}",
                )

        st.write(snippet)
