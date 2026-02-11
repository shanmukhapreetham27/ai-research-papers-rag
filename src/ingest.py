"""Build a local embeddings index from AI research PDFs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from tqdm import tqdm

from .config import CHUNKS_FILE, EMBEDDINGS_FILE, INDEX_DIR, RAW_PAPERS_DIR, get_settings


def sanitize_text(text: str) -> str:
    # Some PDFs include invalid surrogate chars that break JSON/HTTP encoding.
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = " ".join(sanitize_text(text).split())
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def build_chunks(pdf_dir: Path, chunk_size: int, chunk_overlap: int, max_papers: int | None) -> list[dict]:
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if max_papers:
        pdf_files = pdf_files[:max_papers]

    chunks: list[dict] = []
    for pdf_file in tqdm(pdf_files, desc="Reading PDFs"):
        reader = PdfReader(str(pdf_file))
        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for part_idx, chunk_text in enumerate(split_text(page_text, chunk_size, chunk_overlap), start=1):
                chunks.append(
                    {
                        "source_file": pdf_file.name,
                        "page": page_idx,
                        "chunk_id": part_idx,
                        "text": chunk_text,
                    }
                )
    return chunks


def embed_texts(client: OpenAI, model: str, texts: list[str], batch_size: int) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=np.float32)


def build_index(reset: bool = False, max_papers: int | None = None) -> None:
    settings = get_settings()
    if not RAW_PAPERS_DIR.exists():
        raise FileNotFoundError(f"Missing papers directory: {RAW_PAPERS_DIR}")

    if reset and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks(
        pdf_dir=RAW_PAPERS_DIR,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_papers=max_papers,
    )
    if not chunks:
        raise ValueError(f"No text chunks could be created from PDFs in {RAW_PAPERS_DIR}")

    client = OpenAI(api_key=settings.api_key, base_url=settings.api_base)
    texts = [item["text"] for item in chunks]
    embeddings = embed_texts(client, settings.embedding_model, texts, settings.embed_batch_size)

    with CHUNKS_FILE.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=True) + "\n")
    np.save(EMBEDDINGS_FILE, embeddings)

    print(f"Indexed {len(chunks)} chunks from {len(set(c['source_file'] for c in chunks))} papers.")
    print(f"Saved chunks at: {CHUNKS_FILE}")
    print(f"Saved embeddings at: {EMBEDDINGS_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create embeddings index for research papers.")
    parser.add_argument("--reset", action="store_true", help="Delete existing index first")
    parser.add_argument("--max-papers", type=int, default=None, help="Limit number of papers (for quick tests)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(reset=args.reset, max_papers=args.max_papers)


if __name__ == "__main__":
    main()
