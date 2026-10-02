import os

import pytest

from document_agent.domain import doc_id_for, hash_text


def test_document_id_same_for_relative_and_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    path = tmp_path / "document.txt"
    path.touch()

    relative_path = "document.txt"
    absolute_path = str(path.resolve())

    assert doc_id_for(relative_path) == doc_id_for(absolute_path)

@pytest.mark.skipif(os.name != "nt", reason="Windows-specific path casing")
def test_document_id_same_for_different_casing(tmp_path):
    path = tmp_path / "Document.txt"
    path.touch()

    lower = str(path).lower()
    upper = str(path).upper()

    assert doc_id_for(lower) == doc_id_for(upper)

def test_hash_text_changes_when_text_changes():
    text = "Hello, world!"

    original_hash = hash_text(text)
    changed_hash = hash_text("Goodbye, world!")

    assert original_hash != changed_hash