"""Discover PDFs and extract one metadata-bearing document per non-empty page."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)
SHORT_PAGE_CHARACTERS = 40


class DocumentLoadError(RuntimeError):
    """Raised when a source PDF cannot be read completely."""


@dataclass(frozen=True)
class IngestionReport:
    documents: list[Document]
    pdf_count: int
    page_count: int
    empty_page_count: int
    short_page_count: int


def load_pdf_documents(documents_dir: Path) -> IngestionReport:
    """Load every PDF in a folder, retaining its filename and one-based page."""
    directory = Path(documents_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"PDF folder does not exist: {directory}")

    pdf_paths = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {directory}")

    documents: list[Document] = []
    page_count = 0
    empty_page_count = 0
    short_page_count = 0

    for pdf_path in pdf_paths:
        try:
            document_identity = pdf_path.name.encode("utf-8") + b"\0" + pdf_path.read_bytes()
            document_id = hashlib.sha256(document_identity).hexdigest()
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, start=1):
                page_count += 1
                text = (page.extract_text() or "").strip()
                if not text:
                    empty_page_count += 1
                    logger.warning("Empty page: %s, page %d (scanned PDFs may need OCR).", pdf_path.name, page_number)
                    continue
                if len(text) < SHORT_PAGE_CHARACTERS:
                    short_page_count += 1
                    logger.warning(
                        "Suspiciously short page: %s, page %d (%d characters).",
                        pdf_path.name,
                        page_number,
                        len(text),
                    )
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_number,
                            "document_id": document_id,
                        },
                    )
                )
        except Exception as exc:
            raise DocumentLoadError(f"Failed to load PDF '{pdf_path.name}': {exc}") from exc

    logger.info(
        "Processed %d PDF(s), %d page(s): %d text page(s), %d empty page(s), %d short page(s).",
        len(pdf_paths),
        page_count,
        len(documents),
        empty_page_count,
        short_page_count,
    )
    return IngestionReport(
        documents=documents,
        pdf_count=len(pdf_paths),
        page_count=page_count,
        empty_page_count=empty_page_count,
        short_page_count=short_page_count,
    )
