from pathlib import Path
import hashlib
import os

from pydantic import BaseModel


MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]

class Document(BaseModel):
    doc_id: str
    source_uri: str
    content_hash: str
    text: str
    metadata: Metadata

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    index: int
    text: str
    metadata: Metadata

def doc_id_for(path: str | Path) -> str:
    resolved = Path(path).resolve()
    normalized = os.path.normcase(str(resolved))

    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

def chunk_id_for(doc_id: str, index: int) -> str:
    return f"{doc_id}:{index}"

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()