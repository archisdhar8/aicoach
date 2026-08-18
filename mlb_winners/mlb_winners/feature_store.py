from __future__ import annotations

import hashlib
import json

import pandas as pd

from .db import upsert_df
from .features import FEATURE_COLUMNS


def write_engineered_features(
    con,
    frame: pd.DataFrame,
    feature_set_version: str = "rolling-v2",
    data_version: str = "duckdb-local",
) -> int:
    if frame.empty:
        return 0
    rows = []
    for row in frame.to_dict("records"):
        payload = {column: row.get(column) for column in FEATURE_COLUMNS if column in row}
        feature_id = hashlib.sha1(f"{row.get('game_pk')}:{feature_set_version}:{data_version}".encode()).hexdigest()
        rows.append(
            {
                "feature_id": feature_id,
                "game_pk": row.get("game_pk"),
                "game_date": pd.to_datetime(row.get("game_date")).date() if row.get("game_date") is not None else None,
                "feature_set_version": feature_set_version,
                "data_version": data_version,
                "target_home_win": row.get("target_home_win"),
                "raw_payload": json.dumps(payload, default=str),
            }
        )
    return upsert_df(con, "engineered_game_features", pd.DataFrame(rows))
