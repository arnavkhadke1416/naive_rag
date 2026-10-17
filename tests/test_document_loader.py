from pathlib import Path

from pypdf import PdfWriter
import pytest

from app import document_loader
from app.document_loader import DocumentLoadError, load_pdf_documents


class FakePage:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


class FakeReader:
    pages_by_name: dict[str, list[FakePage]] = {}

    def __init__(self, path: str) -> None:
        name = Path(path).name
        if name == "broken.pdf":
            raise ValueError("malformed PDF")
        self.pages = self.pages_by_name[name]


def test_loads_all_pdfs_in_order_with_page_metadata_and_reports_empty_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "B.pdf").touch()
    (tmp_path / "a.PDF").touch()
    (tmp_path / "notes.txt").touch()
    FakeReader.pages_by_name = {
        "a.PDF": [FakePage("A meaningful page containing more than forty characters."), FakePage(" ")],
        "B.pdf": [FakePage("Another meaningful page containing more than forty characters.")],
    }
    monkeypatch.setattr(document_loader, "PdfReader", FakeReader)

    report = load_pdf_documents(tmp_path)

    assert report.pdf_count == 2
    assert report.page_count == 3
    assert report.empty_page_count == 1
    assert [doc.metadata["source"] for doc in report.documents] == ["a.PDF", "B.pdf"]
    assert [doc.metadata["page"] for doc in report.documents] == [1, 1]
    assert len(report.documents[0].metadata["document_id"]) == 64
    assert report.documents[0].metadata["document_id"] != report.documents[1].metadata["document_id"]
    assert load_pdf_documents(tmp_path).documents[0].metadata["document_id"] == report.documents[0].metadata["document_id"]
    assert "Empty page" in caplog.text


@pytest.mark.parametrize(("directory_exists", "expected"), [(False, "does not exist"), (True, "No PDF files")])
def test_requires_existing_folder_with_pdfs(tmp_path: Path, directory_exists: bool, expected: str) -> None:
    directory = tmp_path / "documents"
    if directory_exists:
        directory.mkdir()

    with pytest.raises(FileNotFoundError, match=expected):
        load_pdf_documents(directory)


def test_does_not_silently_skip_a_broken_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "broken.pdf").touch()
    monkeypatch.setattr(document_loader, "PdfReader", FakeReader)

    with pytest.raises(DocumentLoadError, match="broken.pdf"):
        load_pdf_documents(tmp_path)


def test_reads_a_real_pdf_and_reports_pages_without_text(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with (tmp_path / "blank.pdf").open("wb") as pdf_file:
        writer.write(pdf_file)

    report = load_pdf_documents(tmp_path)

    assert report.pdf_count == 1
    assert report.page_count == 1
    assert report.empty_page_count == 1
    assert report.documents == []
