"""Download AI research papers from arXiv as PDFs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import arxiv
from tqdm import tqdm

from .config import RAW_PAPERS_DIR


DEFAULT_QUERY = "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV)"


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._ -]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:90]


def download_papers(query: str, max_results: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(page_size=100, delay_seconds=2.5, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    metadata = []
    results = list(client.results(search))
    for result in tqdm(results, desc="Downloading PDFs"):
        paper_id = result.get_short_id().replace("/", "_")
        title_stub = sanitize_filename(result.title)
        pdf_name = f"{paper_id}_{title_stub}.pdf"

        try:
            result.download_pdf(dirpath=str(out_dir), filename=pdf_name)
            metadata.append(
                {
                    "paper_id": paper_id,
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "published": result.published.isoformat() if result.published else None,
                    "pdf_file": pdf_name,
                    "summary": result.summary,
                    "arxiv_url": result.entry_id,
                }
            )
        except Exception as exc:  # continue on occasional network/pdf failures
            print(f"[WARN] Failed to download {paper_id}: {exc}")

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download AI papers from arXiv.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="arXiv query string")
    parser.add_argument("--max-results", type=int, default=120, help="Number of papers to download")
    parser.add_argument("--out-dir", default=str(RAW_PAPERS_DIR), help="Folder for downloaded PDFs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out_dir)
    metadata_path = download_papers(args.query, args.max_results, output_dir)
    print(f"Download complete. Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
