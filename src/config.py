from pathlib import Path
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_PAPERS_DIR = DATA_DIR / "papers"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    api_base: str = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retrieve_k: int = int(os.getenv("RETRIEVE_K", "3"))
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))


def get_settings() -> Settings:
    settings = Settings()
    if not settings.api_key:
        raise ValueError("OPENROUTER_API_KEY is missing. Put it in your .env file.")
    return settings
