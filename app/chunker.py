"""Split page documents into traceable overlapping character chunks."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split page documents; sizes and overlap are characters, not tokens."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least zero and less than chunk_size.")
    for document in documents:
        if not document.metadata.get("source") or not document.metadata.get("page"):
            raise ValueError("Every page document must include source and page metadata.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(documents))
    counts: Counter[str] = Counter()
    for chunk in chunks:
        source = str(chunk.metadata["source"])
        counts[source] += 1
        chunk_index = counts[source]
        identity = (
            f"{chunk.metadata.get('document_id', source)}:"
            f"{source}:{chunk.metadata['page']}:{chunk_index}:{chunk.page_content}"
        )
        chunk.metadata["chunk_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    logger.info(
        "Created %d chunk(s) across %d PDF(s): %s",
        len(chunks),
        len(counts),
        ", ".join(f"{source}={counts[source]}" for source in sorted(counts, key=str.casefold)),
    )
    return chunks
