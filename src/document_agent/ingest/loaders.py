from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from document_agent.domain import Document, doc_id_for, hash_text

EXTENSION_MAP: dict[str, str] = {
    ".txt": "text/plain",
    ".md": "text/markdown",
}


@runtime_checkable
class Loader(Protocol):
    def supports(self, path: Path | str) -> bool: ...
    def load(self, path: Path | str) -> Document: ...


class TextLoader:
    name = "text"
    extensions: ClassVar[frozenset[str]] = frozenset(EXTENSION_MAP)

    def supports(self, path: Path | str) -> bool:
        return Path(path).suffix.lower() in self.extensions

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
                "title": _extract_title(text, path),
                "mtime": path.stat().st_mtime,
            },
        )


def _extract_title(text: str, path: Path) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()

    return path.name
