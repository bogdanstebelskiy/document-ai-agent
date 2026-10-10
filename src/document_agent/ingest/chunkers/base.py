from typing import ClassVar, Protocol, runtime_checkable

from document_agent.config import Settings
from document_agent.domain import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    def supports(self, doc: Document) -> bool: ...
    def split(self, doc: Document, settings: Settings) -> list[Chunk]: ...


class ContentTypeMixin:
    content_types: ClassVar[frozenset[str]]

    def supports(self, doc: Document) -> bool:
        return doc.content_type in self.content_types
