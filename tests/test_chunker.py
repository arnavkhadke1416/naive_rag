from langchain_core.documents import Document
import pytest

from app.chunker import chunk_documents


def test_chunking_preserves_page_and_source_and_assigns_stable_ids() -> None:
    source_documents = [
        Document(
            page_content=("alpha " * 180),
            metadata={"source": "one.pdf", "page": 2, "document_id": "doc-one"},
        ),
        Document(
            page_content="Evidence from another source document.",
            metadata={"source": "two.pdf", "page": 7, "document_id": "doc-two"},
        ),
    ]

    chunks = chunk_documents(source_documents, chunk_size=200, chunk_overlap=30)

    assert len(chunks) > 2
    assert all(len(chunk.metadata["chunk_id"]) == 64 for chunk in chunks)
    assert {chunk.metadata["source"] for chunk in chunks} == {"one.pdf", "two.pdf"}
    assert {chunk.metadata["page"] for chunk in chunks if chunk.metadata["source"] == "one.pdf"} == {2}
    assert {chunk.metadata["page"] for chunk in chunks if chunk.metadata["source"] == "two.pdf"} == {7}
    repeated = chunk_documents(source_documents, chunk_size=200, chunk_overlap=30)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        chunk.metadata["chunk_id"] for chunk in repeated
    ]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, 10), (10, -1)],
)
def test_chunk_configuration_is_validated(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_chunking_rejects_pages_without_source_metadata() -> None:
    with pytest.raises(ValueError, match="source and page"):
        chunk_documents([Document(page_content="text", metadata={})])


def test_chunk_ids_distinguish_identical_pages_from_different_files() -> None:
    documents = [
        Document(page_content="Identical page text.", metadata={"source": source, "page": 1, "document_id": "same"})
        for source in ("first.pdf", "second.pdf")
    ]

    chunks = chunk_documents(documents)

    assert chunks[0].page_content == chunks[1].page_content
    assert chunks[0].metadata["chunk_id"] != chunks[1].metadata["chunk_id"]
