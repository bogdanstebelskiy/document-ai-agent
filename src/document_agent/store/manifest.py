import sqlite3
from pathlib import Path

from document_agent.config import Settings
from document_agent.config import settings as default_settings


class Manifest:
    def __init__(self, data_dir: Path):
        self.db_path = data_dir / "document_agent.sqlite"

        data_dir.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.db_path)

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents
            (
                doc_id TEXT PRIMARY KEY,
                source_uri TEXT NOT NULL,
                content_hash TEXT NOT NULL
            )
            """
        )

        self.connection.commit()

    def get(self, doc_id: str) -> tuple[str, str, str] | None:
        cursor = self.connection.execute(
            """
            SELECT doc_id, source_uri, content_hash
            FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,),
        )

        return cursor.fetchone()

    def upsert(self, doc_id: str, source_uri: str, content_hash: str) -> None:
        self.connection.execute(
            """
            INSERT INTO documents (doc_id, source_uri, content_hash)
            VALUES (?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_uri = excluded.source_uri,
                content_hash = excluded.content_hash
            """,
            (doc_id, source_uri, content_hash),
        )

        self.connection.commit()

    def delete(self, doc_id: str) -> None:
        self.connection.execute(
            """
            DELETE FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,),
        )

        self.connection.commit()

    def all(self) -> list[tuple[str, str, str]]:
        cursor = self.connection.execute(
            """
            SELECT doc_id, source_uri, content_hash
            FROM documents
            """
        )

        return cursor.fetchall()

    @staticmethod
    def default(settings: Settings = default_settings) -> "Manifest":
        return Manifest(settings.data_dir)
