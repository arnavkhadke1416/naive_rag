"""Local sentence-transformer embeddings and Gemini answer generation."""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Settings


def create_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def create_chat_model(settings: Settings) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.generation_model,
        api_key=settings.require_api_key(),
        temperature=0,
    )
