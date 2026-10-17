from pathlib import Path

import pytest

from app import embeddings
from app.config import Settings


def make_settings(api_key: str | None = "test-key") -> Settings:
    root = Path.cwd()
    return Settings(
        api_key=api_key,
        generation_model="gemini-test-generation",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        documents_dir=root / "documents",
        faiss_index_path=root / "vectorstore",
        chunk_size=700,
        chunk_overlap=100,
        retrieval_k=6,
        project_root=root,
    )


def test_embedding_factory_uses_sentence_transformers_model_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions: list[dict[str, str]] = []

    class FakeHuggingFaceEmbeddings:
        def __init__(self, model_name: str) -> None:
            constructions.append({"model_name": model_name})

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    monkeypatch.setattr(embeddings, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)

    configured = embeddings.create_embeddings(make_settings(api_key=None))

    assert constructions == [{"model_name": "sentence-transformers/all-MiniLM-L6-v2"}]
    assert configured.embed_documents(["chunk"]) == [[1.0]]
    assert configured.embed_query("question") == [1.0]


def test_generation_factory_uses_separate_configured_gemini_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions: list[dict[str, object]] = []

    class FakeChatModel:
        def __init__(self, **kwargs: object) -> None:
            constructions.append(kwargs)

    monkeypatch.setattr(embeddings, "ChatGoogleGenerativeAI", FakeChatModel)

    embeddings.create_chat_model(make_settings())

    assert constructions == [
        {"model": "gemini-test-generation", "api_key": "test-key", "temperature": 0}
    ]


def test_generation_factory_requires_a_gemini_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        embeddings.create_chat_model(make_settings(api_key=None))
