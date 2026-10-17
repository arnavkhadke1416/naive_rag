"""Build one FAISS index from all PDFs in the configured documents folder."""

from __future__ import annotations

import logging

from app.chunker import chunk_documents
from app.config import Settings
from app.document_loader import load_pdf_documents
from app.embeddings import create_embeddings
from app.vectorstore import build_faiss_index


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = Settings.from_env()
    ingestion = load_pdf_documents(settings.documents_dir)
    if not ingestion.documents:
        raise ValueError(
            f"No extractable text found in {ingestion.pdf_count} PDF(s); "
            "scanned documents may require OCR."
        )
    chunks = chunk_documents(
        ingestion.documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise ValueError("No indexable chunks were produced from the PDF collection.")
    index = build_faiss_index(chunks, create_embeddings(settings), settings.faiss_index_path)
    print(
        f"Indexed {ingestion.pdf_count} PDF(s), {ingestion.page_count} page(s), "
        f"{len(chunks)} chunk(s) in one FAISS index at {settings.faiss_index_path} "
        f"({index.index.ntotal} vectors)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
