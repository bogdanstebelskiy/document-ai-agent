from pathlib import Path
from typing import ClassVar

from document_agent.domain import Document, doc_id_for, hash_text
from document_agent.ingest.formats.text import EXTENSION_MAP
from document_agent.ingest.loaders.base import ExtensionMixin


class TextLoader(ExtensionMixin):
    name = "text"
    extensions: ClassVar[frozenset[str]] = frozenset(EXTENSION_MAP)

    def load(self, path: Path | str) -> Document:
        path = Path(path)

        text = path.read_text(encoding="utf-8", errors="replace")

        return Document(
            doc_id=doc_id_for(path),
            source_uri=str(path.resolve()),
            content_hash=hash_text(text),
            content_type=EXTENSION_MAP[path.suffix.lower()],
            text=text,
            metadata={
                "title": TextLoader._extract_title(text, path),
                "mtime": path.stat().st_mtime,
            },
        )

    @staticmethod
    def _extract_title(text: str, path: Path) -> str:
        for line in text.splitlines():
            if line.startswith("# "):
                return line[2:].strip()

        return path.name
