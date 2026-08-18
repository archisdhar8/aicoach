from dataclasses import dataclass
from os import getenv
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "NBA Play Lab API"
    database_path: Path = Path(getenv("NBA_DATABASE_PATH", "data/nba_play_lab.sqlite3"))
    frontend_origin: str = getenv("NBA_FRONTEND_ORIGIN", "http://localhost:3001")


settings = Settings()
