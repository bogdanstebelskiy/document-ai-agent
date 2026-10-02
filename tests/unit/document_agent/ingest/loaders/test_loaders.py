from document_agent.domain import Document, doc_id_for, hash_text
from document_agent.ingest.loaders.text import TextLoader


def test_load_returns_document_with_correct_metadata(tmp_path):
    path = tmp_path / "document.md"
    text = "# Title\n\nHello, world."
    path.write_text(text, encoding="utf-8")

    document = TextLoader().load(path)

    assert isinstance(document, Document)
    assert document.doc_id == doc_id_for(path)
    assert document.content_hash == hash_text(text)
    assert document.metadata["title"] == "Title"

def test_invalid_utf8_loads_without_crashing(tmp_path):
    path = tmp_path / "document.txt"

    path.write_bytes(b"Hello,\xff World")

    document = TextLoader().load(path)

    assert isinstance(document, Document)
    assert "Hello" in document.text
    assert "World" in document.text