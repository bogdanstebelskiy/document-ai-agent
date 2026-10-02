from pathlib import Path

from document_agent.ingest.loaders.base import BaseLoader
from document_agent.ingest.registry import LoaderRegistry
from document_agent.ingest.loaders.text import TextLoader

class FakeLoader(BaseLoader):
    name = "fake"
    extensions = {".fake"}

    def load(self, path: Path | str):
        pass

def test_registry_picks_matching_loader():
    text_loader = TextLoader()
    fake_loader = FakeLoader()

    registry = LoaderRegistry([text_loader, fake_loader])

    assert registry.for_path("document.fake") is fake_loader

def test_unsupported_file_returns_none():
    registry = LoaderRegistry([TextLoader()])

    assert registry.for_path("image.png") is None