from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = DATA_DIR / "models"
REPORT_DIR = DATA_DIR / "reports"
ODDS_DIR = DATA_DIR / "odds"
DB_PATH = Path(os.environ.get("MLB_WINNERS_DB_PATH", str(DATA_DIR / "mlb_winners.duckdb")))


@dataclass(frozen=True)
class Settings:
    db_path: Path = DB_PATH
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    model_dir: Path = MODEL_DIR
    report_dir: Path = REPORT_DIR
    odds_dir: Path = ODDS_DIR
    edge_threshold: float = 0.03
    min_probability: float = 0.35


def ensure_dirs(settings: Settings = Settings()) -> None:
    for path in [
        settings.data_dir,
        settings.raw_dir,
        settings.model_dir,
        settings.report_dir,
        settings.odds_dir,
        settings.odds_dir / "historical",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
