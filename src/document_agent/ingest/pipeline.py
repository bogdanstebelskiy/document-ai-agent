from dataclasses import dataclass
from pathlib import Path

from document_agent.config import Settings
from document_agent.config import settings as default_settings
from document_agent.domain import Document
from document_agent.ingest.chunkers import Chunker, TextChunker
from document_agent.ingest.loaders import Loader, TextLoader
from document_agent.store.manifest import Manifest
from document_agent.store.vector import VectorStore


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0
    unsupported: int = 0


class IngestPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        loaders: list[Loader] | None = None,
        chunkers: list[Chunker] | None = None,
        vector_store: VectorStore | None = None,
        manifest: Manifest | None = None,
    ) -> None:
        self._settings = settings or default_settings
        self._loaders = loaders or [TextLoader()]
        self._chunkers = chunkers or [TextChunker()]
        self._vector_store = vector_store or VectorStore.default(self._settings)
        self._manifest = manifest or Manifest.default(self._settings)

    def _find_loader(self, path: Path) -> Loader | None:
        return next((loader for loader in self._loaders if loader.supports(path)), None)

    def _find_chunker(self, doc: Document) -> Chunker | None:
        return next((chunker for chunker in self._chunkers if chunker.supports(doc)), None)

    def ingest(self, folder: Path | str) -> IngestReport:
        report = IngestReport()
        folder = Path(folder)
        seen_ids: set[str] = set()

        for path in folder.rglob("*"):
            if not path.is_file():
                continue

            loader = self._find_loader(path)
            if not loader:
                report.unsupported += 1
                continue

            doc = loader.load(path)
            seen_ids.add(doc.doc_id)

            existing = self._manifest.get(doc.doc_id)

            if existing is not None and existing.content_hash == doc.content_hash:
                report.skipped += 1
                continue

            chunker = self._find_chunker(doc)
            if not chunker:
                report.unsupported += 1
                continue

            chunks = chunker.split(doc, self._settings)

            if existing is None:
                self._vector_store.add(chunks)
                report.added += 1
            else:
                self._vector_store.delete_doc(doc.doc_id)
                self._vector_store.add(chunks)
                report.updated += 1

            self._manifest.upsert(doc.doc_id, doc.source_uri, doc.content_hash)

        for record in self._manifest.all():
            if record.doc_id not in seen_ids and not Path(record.source_uri).exists():
                self._vector_store.delete_doc(record.doc_id)
                self._manifest.delete(record.doc_id)
                report.removed += 1

        return report
