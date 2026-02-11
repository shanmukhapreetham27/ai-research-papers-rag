"""Copy bundled tiny sample dataset into data/papers for quick demos."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .config import BASE_DIR, RAW_PAPERS_DIR

SAMPLE_DIR = BASE_DIR / "data" / "sample_papers"


def install_sample_dataset(clean: bool) -> None:
    if not SAMPLE_DIR.exists():
        raise FileNotFoundError(f"Sample dataset folder not found: {SAMPLE_DIR}")

    RAW_PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    if clean:
        for existing in RAW_PAPERS_DIR.glob("*.pdf"):
            existing.unlink()
        metadata_file = RAW_PAPERS_DIR / "metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()

    copied = 0
    for src in SAMPLE_DIR.glob("*.pdf"):
        shutil.copy2(src, RAW_PAPERS_DIR / src.name)
        copied += 1

    sample_metadata = SAMPLE_DIR / "metadata.json"
    if sample_metadata.exists():
        shutil.copy2(sample_metadata, RAW_PAPERS_DIR / "metadata.json")

    print(f"Copied {copied} sample PDFs into {RAW_PAPERS_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install bundled tiny sample dataset.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing PDFs from data/papers before copying sample files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_sample_dataset(clean=args.clean)


if __name__ == "__main__":
    main()
