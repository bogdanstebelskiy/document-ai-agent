from pathlib import Path

from document_agent.domain import Document, doc_id_for, hash_text
from document_agent.ingest.loaders.base import BaseLoader


class TextLoader(BaseLoader):
    name = "text"
    extensions = {".txt", ".md"}

    def load(self, path: Path | str) -> Document:
        path = Path(path)

        text = path.read_text(encoding="utf-8", errors="replace")

        title = TextLoader._extract_title(text, path)

        return Document(
            doc_id=doc_id_for(path),
            source_uri=str(path.resolve()),
            content_hash=hash_text(text),
            text=text,
            metadata={
                "title": title,
                "mtime": path.stat().st_mtime,
            },
        )

    @staticmethod
    def _extract_title(text: str, path: Path) -> str:
        for line in text.splitlines():
            if line.startswith("# "):
                return line[2:].strip()

        return path.name