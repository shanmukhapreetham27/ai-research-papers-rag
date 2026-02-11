# AI Research Papers RAG

Single-turn Retrieval-Augmented Generation (RAG) assistant for AI research papers.
The application retrieves relevant chunks from indexed papers and generates grounded answers with source references.

## Demo

![RAG UI Demo](docs/screenshots/latest-demo.png)

## Key Capabilities

- End-to-end RAG pipeline over research PDFs.
- Source-grounded responses with retrieved evidence.
- Source actions in UI: open arXiv page, open local PDF, download PDF.
- Reproducible indexing workflow for local evaluation.

## Tech Stack

- Python 3.14
- Streamlit
- OpenRouter (OpenAI-compatible API)
- NumPy (local vector index: JSONL + NPY)
- arXiv API + pypdf

## Repository Structure

```text
.
+-- app.py
+-- requirements.txt
+-- .env.example
+-- data
|   +-- sample_papers        # tiny bundled demo set (committed)
|   +-- papers               # downloaded papers (not committed)
|   +-- index                # generated index (not committed)
+-- docs
|   +-- screenshots
|       +-- latest-demo.png
+-- src
    +-- config.py
    +-- download_arxiv.py
    +-- ingest.py
    +-- rag_chain.py
    +-- use_sample_dataset.py
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set your key in `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
```

## Quick Start (Bundled Sample Dataset)

Use this path for fastest verification after cloning:

```powershell
python -m src.use_sample_dataset --clean
python -m src.ingest --reset --max-papers 3
streamlit run app.py --server.port 8502
```

## Full Dataset Run (100-150 Papers)

```powershell
python -m src.download_arxiv --max-results 120
python -m src.ingest --reset
streamlit run app.py --server.port 8502
```

## Configuration

Main settings are read from `.env`:

- `OPENROUTER_API_KEY`
- `API_BASE_URL`
- `EMBEDDING_MODEL`
- `LLM_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVE_K`
- `EMBED_BATCH_SIZE`

## Operational Notes

- `data/papers/` and `data/index/` are intentionally excluded from git because they are large/generated artifacts.
- `.env` is excluded from git; keep secrets in local environment files only.
- If port `8501` is in use, run Streamlit on another port (example above uses `8502`).
