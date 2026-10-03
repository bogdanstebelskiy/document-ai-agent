from pathlib import Path
from typing import Protocol, runtime_checkable

from document_agent.domain import Document


@runtime_checkable
class Loader(Protocol):
    def supports(self, path: Path | str) -> bool: ...
    def load(self, path: Path | str) -> Document: ...


class ExtensionMixin:
    """Mixin: implements supports() by checking path suffix against extensions."""

    extensions: set[str]

    def supports(self, path: Path | str) -> bool:
        return Path(path).suffix.lower() in self.extensions
