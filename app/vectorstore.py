"""Build, persist, load, and query the combined local FAISS index."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def build_faiss_index(
    chunks: list[Document],
    embeddings: Embeddings,
    index_path: Path,
) -> FAISS:
    if not chunks:
        raise ValueError("Cannot build the FAISS index: no text chunks were produced.")
    index = FAISS.from_documents(chunks, embeddings)
    index_path.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_path))
    if not (index_path / "index.faiss").is_file() or not (index_path / "index.pkl").is_file():
        raise RuntimeError(f"FAISS index artifacts were not written to: {index_path}")
    return index


def load_faiss_index(index_path: Path, embeddings: Embeddings) -> FAISS:
    validate_index_files(index_path)
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def validate_index_files(index_path: Path) -> None:
    index_path = Path(index_path)
    required_files = (index_path / "index.faiss", index_path / "index.pkl")
    missing_files = [path.name for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"FAISS index is missing from '{index_path}' ({', '.join(missing_files)}). "
            "Build it first with: python -m scripts.build_index"
        )


def retrieve_chunks(
    index: FAISS,
    question: str,
    k: int = 6,
) -> list[tuple[Document, float]]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if k < 1:
        raise ValueError("k must be greater than zero.")
    return [
        (document, float(score))
        for document, score in index.similarity_search_with_score(question, k=k)
    ]
