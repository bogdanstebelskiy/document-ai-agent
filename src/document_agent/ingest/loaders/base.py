from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from document_agent.domain import Document


@runtime_checkable
class Loader(Protocol):
    def supports(self, path: Path | str) -> bool: ...
    def load(self, path: Path | str) -> Document: ...


class ExtensionMixin:
    extensions: ClassVar[frozenset[str]]

    def supports(self, path: Path | str) -> bool:
        return Path(path).suffix.lower() in self.extensions
