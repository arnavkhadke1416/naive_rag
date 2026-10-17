"""Retrieve evidence and generate source-validated, grounded answers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.prompts import build_answer_prompt
from app.vectorstore import retrieve_chunks

INSUFFICIENT_EVIDENCE = "I don't have enough evidence in the indexed documents to answer that question."
CITATION_PATTERN = re.compile(r"\[([^\[\]\n]+?),\s*page\s+(\d+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class SourceReference:
    source: str
    page: int


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[SourceReference]


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        text = "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ).strip()
    else:
        raise TypeError("Gemini returned a response without text content.")
    if not text:
        raise ValueError("Gemini returned an empty answer.")
    return text


def answer_question(question: str, index: Any, model: Any, k: int = 6) -> AnswerResult:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    retrieved = retrieve_chunks(index, question, k=k)
    if not retrieved:
        return AnswerResult(INSUFFICIENT_EVIDENCE, [])

    documents = [document for document, _ in retrieved]
    allowed_references = {
        (str(document.metadata["source"]), int(document.metadata["page"]))
        for document in documents
    }
    generated = _response_text(model.invoke(build_answer_prompt(question, documents)))
    citation_matches = list(CITATION_PATTERN.finditer(generated))
    cited_references = [
        (match.group(1).strip(), int(match.group(2)))
        for match in citation_matches
    ]
    if not cited_references or any(reference not in allowed_references for reference in cited_references):
        return AnswerResult(INSUFFICIENT_EVIDENCE, [])

    sources: list[SourceReference] = []
    seen: set[tuple[str, int]] = set()
    for source, page in cited_references:
        reference = (source, page)
        if reference not in seen:
            sources.append(SourceReference(source=source, page=page))
            seen.add(reference)
    return AnswerResult(generated, sources)
