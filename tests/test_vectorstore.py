from pathlib import Path
import re
import subprocess
import sys

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import pytest

from app.vectorstore import build_faiss_index, load_faiss_index, retrieve_chunks


class KeywordEmbeddings(Embeddings):
    vocabulary = ("orchid", "granite", "revenue", "policy")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [float(bool(re.search(rf"\b{word}\b", lowered))) for word in self.vocabulary]


def test_build_save_reload_and_retrieve_from_combined_index(tmp_path: Path) -> None:
    docs = [
        Document(page_content="Orchid cultivation requires shade.", metadata={"source": "plants.pdf", "page": 2}),
        Document(page_content="Granite is an igneous rock.", metadata={"source": "geology.pdf", "page": 5}),
        Document(
            page_content="Orchid exports and granite supply are discussed together.",
            metadata={"source": "summary.pdf", "page": 1},
        ),
    ]

    build_faiss_index(docs, KeywordEmbeddings(), tmp_path)
    loaded = load_faiss_index(tmp_path, KeywordEmbeddings())
    results = retrieve_chunks(loaded, "orchid granite", k=3)

    assert loaded.index.ntotal == 3
    assert {document.metadata["source"] for document, _ in results} == {
        "plants.pdf",
        "geology.pdf",
        "summary.pdf",
    }
    assert all(isinstance(score, float) for _, score in results)


def test_saved_index_loads_in_a_fresh_python_process(tmp_path: Path) -> None:
    build_faiss_index(
        [Document(page_content="Fresh process check.", metadata={"source": "check.pdf", "page": 1})],
        KeywordEmbeddings(),
        tmp_path,
    )
    code = """
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import sys

class TestEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
    def embed_query(self, text):
        return [float(bool(text))]

index = FAISS.load_local(sys.argv[1], TestEmbeddings(), allow_dangerous_deserialization=True)
print(index.index.ntotal)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )

    assert completed.stdout.strip() == "1"


def test_index_build_and_retrieval_reject_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no text chunks"):
        build_faiss_index([], KeywordEmbeddings(), tmp_path)

    with pytest.raises(FileNotFoundError, match="Build it first"):
        load_faiss_index(tmp_path, KeywordEmbeddings())

    index = build_faiss_index(
        [Document(page_content="A text.", metadata={"source": "one.pdf", "page": 1})],
        KeywordEmbeddings(),
        tmp_path,
    )
    with pytest.raises(ValueError, match="empty"):
        retrieve_chunks(index, "  ")
    with pytest.raises(ValueError, match="greater than zero"):
        retrieve_chunks(index, "text", k=0)
