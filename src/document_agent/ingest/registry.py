from pathlib import Path

from document_agent.ingest.loaders.base import BaseLoader
from document_agent.ingest.loaders.text import TextLoader


class LoaderRegistry:
    def __init__(self, loaders: list[BaseLoader]) -> None:
        self.loaders = loaders

    def for_path(self, path: Path | str) -> BaseLoader | None:
        for loader in self.loaders:
            if loader.supports(path):
                return loader

        return None

def default_registry() -> LoaderRegistry:
    return LoaderRegistry([
        TextLoader()
    ])