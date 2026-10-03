from langchain_chroma import Chroma

from document_agent.config import settings
from document_agent.domain import Chunk


class VectorStore:
    def __init__(self, embedding_function):
        self.vector_store = Chroma(
            collection_name="document_agent",
            embedding_function=embedding_function,
            persist_directory=str(settings.data_dir / "chroma"),
        )

    def add(self, chunks: list[Chunk]):
        self.vector_store.add_texts(
            texts=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    **chunk.metadata,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ],
            ids=[chunk.chunk_id for chunk in chunks],
        )

    def delete_doc(self, doc_id: str):
        self.vector_store.delete(
            where={"doc_id": doc_id},
        )

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k,
        )

        return [
            (
                Chunk(
                    chunk_id=document.metadata["chunk_id"],
                    doc_id=document.metadata["doc_id"],
                    index=document.metadata["index"],
                    text=document.page_content,
                    metadata=document.metadata
                ),
                score,
            )
            for document, score in results
        ]

    def count(self) -> int:
        return self.vector_store._collection.count()