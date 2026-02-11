"""Single-turn RAG utilities over local embeddings index."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from openai import OpenAI

from .config import CHUNKS_FILE, EMBEDDINGS_FILE, get_settings


def _load_index() -> tuple[list[dict[str, Any]], np.ndarray]:
    if not CHUNKS_FILE.exists() or not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError("Index files are missing. Run: python -m src.ingest --reset")

    with CHUNKS_FILE.open("r", encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    embeddings = np.load(EMBEDDINGS_FILE)
    return chunks, embeddings


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def retrieve(question: str, k: int) -> list[dict[str, Any]]:
    settings = get_settings()
    chunks, embeddings = _load_index()
    if len(chunks) != len(embeddings):
        raise ValueError("Index mismatch: chunks and embeddings length differ.")

    client = OpenAI(api_key=settings.api_key, base_url=settings.api_base)
    q_vec = client.embeddings.create(model=settings.embedding_model, input=question).data[0].embedding
    q_arr = np.asarray(q_vec, dtype=np.float32)

    normalized_embeddings = _normalize_rows(embeddings)
    q_norm = q_arr / max(np.linalg.norm(q_arr), 1e-12)
    scores = normalized_embeddings @ q_norm

    top_idx = np.argsort(scores)[-k:][::-1]
    return [
        {
            "source_file": chunks[i]["source_file"],
            "page": chunks[i]["page"],
            "text": chunks[i]["text"],
            "score": float(scores[i]),
        }
        for i in top_idx
    ]


def _build_context(docs: list[dict[str, Any]]) -> str:
    parts = []
    for idx, doc in enumerate(docs, start=1):
        parts.append(
            f"[{idx}] Source: {doc['source_file']} | Page: {doc['page']}\n"
            f"{doc['text']}"
        )
    return "\n\n".join(parts)


def answer_question(question: str, k: int | None = None) -> tuple[str, list[dict[str, Any]]]:
    settings = get_settings()
    top_k = k or settings.retrieve_k
    docs = retrieve(question, top_k)
    if not docs:
        return "I could not find relevant content in the indexed papers.", []

    prompt = f"""
You are an AI research assistant. Answer only from the context below.
If the answer is not present in context, say: "I don't know based on the indexed papers."

Question: {question}

Context:
{_build_context(docs)}

Answer with concise points and include source citations like [source_file p.X].
""".strip()

    client = OpenAI(api_key=settings.api_key, base_url=settings.api_base)
    result = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = result.choices[0].message.content or "No answer returned."
    return answer, docs
