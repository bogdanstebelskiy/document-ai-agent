from abc import ABC, abstractmethod
from pathlib import Path

from document_agent.domain import Document


class BaseLoader(ABC):
    name: str
    extensions: set[str]

    def supports(self, path: Path | str) -> bool:
        return Path(path).suffix.lower() in self.extensions

    @abstractmethod
    def load(self, path: Path | str) -> Document:
        ...