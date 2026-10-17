from pathlib import Path

import pytest

from app.config import Settings


def test_settings_use_project_relative_paths_and_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for name in (
        "GEMINI_MODEL",
        "EMBEDDING_MODEL",
        "DOCUMENTS_DIR",
        "FAISS_INDEX_PATH",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVAL_K",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env(tmp_path)

    assert settings.documents_dir == tmp_path / "documents"
    assert settings.faiss_index_path == tmp_path / "vectorstore"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.generation_model == "gemini-3.7-flash"
    assert settings.api_key is None
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        settings.require_api_key()


def test_settings_read_environment_and_validate_chunk_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("DOCUMENTS_DIR", "pdfs")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "80")

    settings = Settings.from_env(tmp_path)

    assert settings.api_key == "test-key"
    assert settings.documents_dir == tmp_path / "pdfs"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 80

    monkeypatch.setenv("CHUNK_OVERLAP", "500")
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings.from_env(tmp_path)
