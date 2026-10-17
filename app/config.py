"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _configured_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    generation_model: str
    embedding_model: str
    documents_dir: Path
    faiss_index_path: Path
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    project_root: Path

    @classmethod
    def from_env(cls, project_root: Path = PROJECT_ROOT) -> "Settings":
        root = project_root.resolve()
        load_dotenv(root / ".env")

        chunk_size = int(os.getenv("CHUNK_SIZE", "700"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        retrieval_k = int(os.getenv("RETRIEVAL_K", "6"))
        if chunk_size < 1:
            raise ValueError("CHUNK_SIZE must be greater than zero.")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("CHUNK_OVERLAP must be at least zero and less than CHUNK_SIZE.")
        if retrieval_k < 1:
            raise ValueError("RETRIEVAL_K must be greater than zero.")

        api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
        return cls(
            api_key=api_key,
            generation_model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash").strip(),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ).strip(),
            documents_dir=_configured_path(os.getenv("DOCUMENTS_DIR", "documents"), root),
            faiss_index_path=_configured_path(os.getenv("FAISS_INDEX_PATH", "vectorstore"), root),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_k=retrieval_k,
            project_root=root,
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your Gemini API key."
            )
        return self.api_key
