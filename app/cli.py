"""Interactive terminal interface for the Naive RAG application."""

from __future__ import annotations

import sys

from app.config import Settings
from app.embeddings import create_chat_model, create_embeddings
from app.rag import answer_question
from app.vectorstore import load_faiss_index, validate_index_files


def main() -> int:
    try:
        settings = Settings.from_env()
        validate_index_files(settings.faiss_index_path)
        embeddings = create_embeddings(settings)
        index = load_faiss_index(settings.faiss_index_path, embeddings)
        model = create_chat_model(settings)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Naive RAG — type 'exit' or 'quit' to quit")
    while True:
        try:
            question = input("\nYou: ").strip()
        except EOFError:
            print()
            return 0
        if question.casefold() in {"exit", "quit"}:
            return 0
        if not question:
            print("Please enter a question.")
            continue

        result = answer_question(question, index, model, k=settings.retrieval_k)
        print(f"Assistant: {result.answer}")
        if result.sources:
            print("Sources:")
            for source in result.sources:
                print(f"- {source.source}, page {source.page}")


if __name__ == "__main__":
    raise SystemExit(main())
