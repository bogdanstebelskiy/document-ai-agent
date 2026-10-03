from pathlib import Path

from document_agent.domain import Document
from document_agent.ingest.chunkers.base import Chunker
from document_agent.ingest.loaders.base import Loader


class LoaderRegistry:
    def __init__(self, loaders: list[Loader]) -> None:
        self._loaders = loaders

    def for_path(self, path: Path | str) -> Loader | None:
        for loader in self._loaders:
            if loader.supports(path):
                return loader
        return None

    @staticmethod
    def default() -> "LoaderRegistry":
        from document_agent.ingest.loaders.text import TextLoader

        return LoaderRegistry([TextLoader()])


class ChunkerRegistry:
    def __init__(self, chunkers: list[Chunker]) -> None:
        self._chunkers = chunkers

    def for_doc(self, doc: Document) -> Chunker | None:
        for chunker in self._chunkers:
            if chunker.supports(doc):
                return chunker
        return None

    @staticmethod
    def default() -> "ChunkerRegistry":
        from document_agent.ingest.chunkers.text import TextChunker

        return ChunkerRegistry([TextChunker()])
