from pathlib import Path

from langchain_core.documents import Document

from app import cli
from app.config import Settings


class FakeIndex:
    def similarity_search_with_score(self, question: str, k: int) -> list[tuple[Document, float]]:
        return [(Document(page_content="A fact.", metadata={"source": "source.pdf", "page": 3}), 0.1)]


class FakeModel:
    def invoke(self, prompt: str) -> str:
        return "A fact is documented [source.pdf, page 3]."


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key",
        generation_model="test-model",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        documents_dir=tmp_path / "documents",
        faiss_index_path=tmp_path / "vectorstore",
        chunk_size=700,
        chunk_overlap=100,
        retrieval_k=6,
        project_root=tmp_path,
    )


def test_cli_answers_multiple_questions_and_exits(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli.Settings, "from_env", lambda: make_settings(tmp_path))
    monkeypatch.setattr(cli, "validate_index_files", lambda path: None)
    monkeypatch.setattr(cli, "create_embeddings", lambda settings: object())
    monkeypatch.setattr(cli, "load_faiss_index", lambda path, embeddings: FakeIndex())
    monkeypatch.setattr(cli, "create_chat_model", lambda settings: FakeModel())
    questions = iter(["First question?", "Second question?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(questions))

    status = cli.main()

    output = capsys.readouterr().out
    assert status == 0
    assert output.count("Assistant: A fact is documented") == 2
    assert output.count("- source.pdf, page 3") == 2


def test_cli_reports_missing_index(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.Settings, "from_env", lambda: make_settings(tmp_path))
    monkeypatch.setattr(cli, "create_embeddings", lambda settings: object())

    status = cli.main()

    assert status == 1
    assert "python -m scripts.build_index" in capsys.readouterr().err
