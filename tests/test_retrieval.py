from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.vectorstore import retrieve_chunks


class SynonymEmbeddings(Embeddings):
    concepts = {
        "deadline": {"deadline", "due", "date", "submission"},
        "sales": {"sales", "revenue", "income"},
    }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        terms = set(text.casefold().split())
        return [float(bool(terms & synonyms)) for synonyms in self.concepts.values()]


def test_query_retrieves_supporting_chunks_from_multiple_pdfs() -> None:
    documents = [
        Document(
            page_content="The submission deadline is October 10.",
            metadata={"source": "schedule.pdf", "page": 4},
        ),
        Document(
            page_content="Annual revenue increased by twelve percent.",
            metadata={"source": "results.pdf", "page": 8},
        ),
        Document(
            page_content="The archive contains scanned forms.",
            metadata={"source": "archive.pdf", "page": 1},
        ),
    ]
    index = FAISS.from_documents(documents, SynonymEmbeddings())

    matches = retrieve_chunks(index, "What is the due date and sales change?", k=2)

    assert {document.metadata["source"] for document, _ in matches} == {
        "schedule.pdf",
        "results.pdf",
    }
    assert {document.metadata["page"] for document, _ in matches} == {4, 8}
    assert all(document.page_content for document, _ in matches)
