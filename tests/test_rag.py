from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from app.rag import INSUFFICIENT_EVIDENCE, answer_question


class FakeIndex:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results

    def similarity_search_with_score(self, question: str, k: int) -> list[tuple[Document, float]]:
        return self.results[:k]


@dataclass
class FakeResponse:
    content: str


class FakeModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(self.response)


def make_evidence() -> list[tuple[Document, float]]:
    return [
        (
            Document(
                page_content="Revenue increased by 12%.",
                metadata={"source": "results.pdf", "page": 2},
            ),
            0.2,
        ),
        (
            Document(
                page_content="The policy attributes growth to exports.",
                metadata={"source": "policy.pdf", "page": 4},
            ),
            0.4,
        ),
    ]


def test_answer_synthesizes_multiple_sources_and_returns_valid_citations() -> None:
    model = FakeModel(
        "Revenue increased by 12% [results.pdf, page 2]. "
        "Exports contributed to growth [policy.pdf, page 4]."
    )

    result = answer_question("What drove the revenue increase?", FakeIndex(make_evidence()), model, k=2)

    assert "results.pdf" in result.answer
    assert [source.source for source in result.sources] == ["results.pdf", "policy.pdf"]
    assert [source.page for source in result.sources] == [2, 4]
    assert "Revenue increased by 12%." in model.prompts[0]
    assert "The policy attributes growth to exports." in model.prompts[0]
    assert "sources conflict" in model.prompts[0]


def test_conflicting_evidence_can_be_reported_with_both_citations() -> None:
    evidence = [
        (
            Document(
                page_content="The policy permits remote work.",
                metadata={"source": "policy-a.pdf", "page": 3},
            ),
            0.2,
        ),
        (
            Document(
                page_content="The policy prohibits remote work.",
                metadata={"source": "policy-b.pdf", "page": 7},
            ),
            0.3,
        ),
    ]
    model = FakeModel(
        "The documents conflict: remote work is permitted [policy-a.pdf, page 3] "
        "but also prohibited [policy-b.pdf, page 7]."
    )

    result = answer_question("What is the remote-work policy?", FakeIndex(evidence), model)

    assert "conflict" in result.answer
    assert [(source.source, source.page) for source in result.sources] == [
        ("policy-a.pdf", 3),
        ("policy-b.pdf", 7),
    ]
    assert "sources conflict" in model.prompts[0]


@pytest.mark.parametrize(
    "response",
    [
        "Revenue increased by 12%.",
        "Revenue increased [invented.pdf, page 9].",
        "Revenue increased [results.pdf, page 3].",
    ],
)
def test_uncited_or_unretrieved_references_abstain(response: str) -> None:
    result = answer_question("What changed?", FakeIndex(make_evidence()), FakeModel(response))

    assert result.answer == INSUFFICIENT_EVIDENCE
    assert result.sources == []


def test_empty_retrieval_abstains_without_calling_model() -> None:
    model = FakeModel("This should never be used.")

    result = answer_question("What is known?", FakeIndex([]), model)

    assert result.answer == INSUFFICIENT_EVIDENCE
    assert result.sources == []
    assert model.prompts == []


def test_empty_question_is_rejected_before_retrieval() -> None:
    with pytest.raises(ValueError, match="empty"):
        answer_question(" ", FakeIndex([]), FakeModel("unused"))
