# AI Research Papers RAG

A single-turn RAG assistant over AI research papers (100-150 papers scale), built for internship/demo use.

## Demo

![RAG UI Demo](docs/screenshots/latest-demo.png)

## What This Project Shows

- End-to-end RAG pipeline over real research PDFs.
- Grounded answers with source snippets and paper links.
- Local paper actions from UI: open local PDF and download PDF.
- Reproducible data/index pipeline for interviewers.

## Features

- `arXiv downloader`: fetches recent AI papers into `data/papers/`.
- `ingestion/indexing`: PDF parsing, chunking, embedding, and local vector index.
- `retrieval + generation`: cosine similarity retrieval + constrained answer generation.
- `single-turn UI`: one question box, no chat history, top source chunks with links.

## Tech Stack

- Python 3.14
- Streamlit
- OpenRouter (LLM + embeddings API via OpenAI-compatible client)
- NumPy + local JSONL/NPY index
- arXiv + pypdf

## Project Structure

```text
RAG project/
  app.py
  requirements.txt
  .env.example
  data/
    papers/            # local PDFs (not committed)
    index/             # local index files (not committed)
  docs/
    screenshots/
      latest-demo.png
  src/
    config.py
    download_arxiv.py
    ingest.py
    rag_chain.py
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
```

## Build Dataset + Index

Download papers:

```powershell
python -m src.download_arxiv --max-results 120
```

Build index:

```powershell
python -m src.ingest --reset
```

Quick smoke test (smaller index):

```powershell
python -m src.ingest --reset --max-papers 20
```

## Run App

```powershell
streamlit run app.py
```

If port `8501` is busy:

```powershell
streamlit run app.py --server.port 8502
```

## How to Demo in Interview

1. Ask one paper-specific question.
2. Show generated answer with citations.
3. Open one top source via arXiv/local PDF.
4. Explain chunking (`CHUNK_SIZE`, `CHUNK_OVERLAP`) and retrieval depth (`RETRIEVE_K`).

## Reproducibility Notes

- `data/papers/` and `data/index/` are intentionally not committed due size/cost.
- Any evaluator can regenerate them with the commands above.
- `.env` is excluded from git; use `.env.example` template.

## Resume Bullet (Example)

Built a Retrieval-Augmented Generation assistant over 120 AI research papers using Python, Streamlit, and OpenRouter embeddings/LLM APIs, with local indexing, source-grounded answers, and paper-level citation links for transparent retrieval.
