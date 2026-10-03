from abc import ABC, abstractmethod

from document_agent.config import Settings
from document_agent.domain import Document, Chunk

class BaseChunker(ABC):
    @abstractmethod
    def split(self, doc: Document, settings: Settings) -> list[Chunk]:
        ...