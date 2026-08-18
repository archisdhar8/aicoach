import sqlite3
from pathlib import Path
from uuid import UUID

from app.schemas.domain import Play


class SQLitePlayRepository:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS plays (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save(self, play: Play) -> Play:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO plays (id, name, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    str(play.id),
                    play.name,
                    play.model_dump_json(by_alias=True),
                    play.created_at.isoformat(),
                    play.updated_at.isoformat(),
                ),
            )
        return play

    def get(self, play_id: UUID) -> Play | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM plays WHERE id = ?", (str(play_id),)
            ).fetchone()
        return None if row is None else Play.model_validate_json(row[0])

    def list(self) -> list[Play]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM plays ORDER BY updated_at DESC"
            ).fetchall()
        return [Play.model_validate_json(row[0]) for row in rows]

    def delete(self, play_id: UUID) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM plays WHERE id = ?", (str(play_id),))
        return cursor.rowcount > 0

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path)
