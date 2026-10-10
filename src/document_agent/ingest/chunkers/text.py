from typing import ClassVar

from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_agent.config import Settings
from document_agent.config import settings as default_settings
from document_agent.domain import Chunk, Document, chunk_id_for
from document_agent.ingest.chunkers.base import ContentTypeMixin
from document_agent.ingest.formats.text import CONTENT_TYPES


class TextChunker(ContentTypeMixin):
    content_types: ClassVar[frozenset[str]] = CONTENT_TYPES

    def split(
        self, doc: Document, settings: Settings = default_settings
    ) -> list[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n# ",
                "\n## ",
                "\n### ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        texts = splitter.split_text(doc.text)

        return [
            Chunk(
                chunk_id=chunk_id_for(doc.doc_id, chunk_idx),
                doc_id=doc.doc_id,
                index=chunk_idx,
                text=text,
                metadata={
                    **doc.metadata,
                    "index": chunk_idx,
                    "doc_id": doc.doc_id,
                    "source_uri": doc.source_uri,
                },
            )
            for chunk_idx, text in enumerate(texts)
        ]
