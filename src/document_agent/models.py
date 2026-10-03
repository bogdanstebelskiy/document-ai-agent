from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings

from document_agent.config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embed_model,
    )
