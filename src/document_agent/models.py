from functools import lru_cache

from document_agent.config import settings

from langchain_ollama import ChatOllama, OllamaEmbeddings


@lru_cache(maxsize = 1)
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
    )

@lru_cache(maxsize = 1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embed_model,
    )