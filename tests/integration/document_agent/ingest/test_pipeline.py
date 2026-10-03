from unittest.mock import MagicMock

import pytest

from document_agent.config import Settings
from document_agent.domain import doc_id_for
from document_agent.ingest.pipeline import IngestPipeline
from document_agent.ingest.registry import ChunkerRegistry, LoaderRegistry
from document_agent.store.manifest import Manifest
from document_agent.store.vector import VectorStore


@pytest.fixture
def notes_dir(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.txt").write_text("Note A. " * 20, encoding="utf-8")
    (folder / "b.txt").write_text("Note B. " * 20, encoding="utf-8")
    (folder / "c.txt").write_text("Note C. " * 20, encoding="utf-8")
    return folder


@pytest.fixture
def vector_store():
    return MagicMock(spec=VectorStore)


@pytest.fixture
def pipeline(tmp_path, vector_store):
    return IngestPipeline(
        settings=Settings(data_dir=tmp_path, chunk_size=500, chunk_overlap=50),
        loaders=LoaderRegistry.default(),
        chunkers=ChunkerRegistry.default(),
        vector_store=vector_store,
        manifest=Manifest(tmp_path / "manifest"),
    )


def test_first_run_adds_all_files(pipeline, notes_dir, vector_store):
    report = pipeline.ingest(notes_dir)

    assert report.added == 3
    assert report.updated == 0
    assert report.skipped == 0
    assert report.removed == 0
    assert vector_store.add.call_count == 3


def test_second_run_skips_all(pipeline, notes_dir):
    pipeline.ingest(notes_dir)
    report = pipeline.ingest(notes_dir)

    assert report.added == 0
    assert report.updated == 0
    assert report.skipped == 3
    assert report.removed == 0


def test_edited_file_reports_updated(pipeline, notes_dir, vector_store):
    pipeline.ingest(notes_dir)
    vector_store.reset_mock()

    (notes_dir / "b.txt").write_text("Changed content. " * 20, encoding="utf-8")
    report = pipeline.ingest(notes_dir)

    assert report.updated == 1
    assert report.skipped == 2
    assert report.added == 0
    assert vector_store.delete_doc.call_count == 1
    assert vector_store.add.call_count == 1


def test_deleted_file_reports_removed(pipeline, notes_dir, vector_store):
    pipeline.ingest(notes_dir)
    vector_store.reset_mock()

    doc_id = doc_id_for(notes_dir / "c.txt")
    (notes_dir / "c.txt").unlink()
    report = pipeline.ingest(notes_dir)

    assert report.removed == 1
    assert report.skipped == 2
    assert vector_store.delete_doc.assert_called_once_with(doc_id) is None
