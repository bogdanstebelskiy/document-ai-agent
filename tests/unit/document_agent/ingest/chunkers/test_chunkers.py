from document_agent.config import Settings
from document_agent.domain import Document, hash_text
from document_agent.ingest.chunkers.text import TextChunker


def make_document(text: str) -> Document:
    return Document(
        doc_id="test-doc-id",
        source_uri="test.txt",
        content_hash=hash_text(text),
        text=text,
        metadata={"title": "Test Document"}
    )

def test_long_document_produces_multiple_chunks():
    text = "A" * 2000
    doc = make_document(text)
    settings = Settings(chunk_size=1000, chunk_overlap=150)

    chunks = TextChunker().split(doc, settings)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 1000 for chunk in chunks)

def test_consecutive_chunks_share_overlapping_text():
    text = "".join(f"{i:04d} " for i in range(500))
    doc = make_document(text)
    settings = Settings(chunk_size=1000, chunk_overlap=150)

    chunks = TextChunker().split(doc, settings)

    assert len(chunks) > 1

    for previous, current in zip(chunks[:-1], chunks[1:]):
        has_overlap = any(
            previous.text.endswith(current.text[:overlap_length])
            for overlap_length in range(1, settings.chunk_overlap + 1)
        )

        assert has_overlap

def test_short_document_produces_one_chunk():
    text = "This is a short document."
    doc = make_document(text)
    settings = Settings(chunk_size=1000, chunk_overlap=150)

    chunks = TextChunker().split(doc, settings)

    assert len(chunks) == 1
    assert chunks[0].text == text

def test_chunk_ids_are_sequential():
    text = "A " * 5000
    doc = make_document(text)
    settings = Settings(chunk_size=1000, chunk_overlap=150)

    chunks = TextChunker().split(doc, settings)

    assert [chunk.chunk_id for chunk in chunks] == [
        f"{doc.doc_id}:{index}"
        for index in range(len(chunks))
    ]

def test_chunks_preserve_document_metadata():
    text = "This is a short document."
    doc = make_document(text)
    settings = Settings(chunk_size=1000, chunk_overlap=150)

    chunks = TextChunker().split(doc, settings)

    assert chunks[0].metadata == {
        "title": "Test Document",
        "index": 0,
        "doc_id": "test-doc-id",
        "source_uri": "test.txt",
    }