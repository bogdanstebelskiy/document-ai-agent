EXTENSION_MAP: dict[str, str] = {
    ".txt": "text/plain",
    ".md": "text/markdown",
}

CONTENT_TYPES: frozenset[str] = frozenset(EXTENSION_MAP.values())
