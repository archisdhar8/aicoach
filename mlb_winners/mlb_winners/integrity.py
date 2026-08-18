from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

import pandas as pd

from .db import upsert_df


def run_data_integrity_checks(
    con,
    games: pd.DataFrame,
    odds: pd.DataFrame,
    weather: pd.DataFrame,
    lineups: pd.DataFrame,
    check_date: date,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    slate = games[pd.to_datetime(games["game_date"]).dt.date == check_date].copy()
    odds_date = odds[pd.to_datetime(odds["snapshot_date"]).dt.date == check_date].copy() if not odds.empty else odds
    weather_games = set(weather["game_pk"].tolist()) if not weather.empty else set()
    lineup_games = set(lineups["game_pk"].tolist()) if not lineups.empty else set()

    for game in slate.to_dict("records"):
        game_pk = game["game_pk"]
        if not game.get("home_probable_pitcher") or not game.get("away_probable_pitcher"):
            checks.append(row(check_date, game_pk, "high", "missing_starter", "One or both probable starters are missing.", game))
        if game_pk not in weather_games:
            checks.append(row(check_date, game_pk, "medium", "missing_weather", "No weather snapshot cached for this game.", game))
        if game_pk not in lineup_games:
            checks.append(row(check_date, game_pk, "medium", "missing_lineup", "No lineup snapshot cached; using team-level defaults.", game))
        odds_match = odds_date[
            (odds_date["home_team"] == game.get("home_team")) & (odds_date["away_team"] == game.get("away_team"))
        ] if not odds_date.empty else pd.DataFrame()
        if odds_match.empty:
            checks.append(row(check_date, game_pk, "high", "missing_odds", "No moneyline odds snapshot cached.", game))
        else:
            suspicious = odds_match[
                odds_match["home_moneyline"].abs().gt(800) | odds_match["away_moneyline"].abs().gt(800)
            ]
            if not suspicious.empty:
                checks.append(row(check_date, game_pk, "medium", "suspicious_odds", "Extreme moneyline detected.", suspicious.iloc[0].to_dict()))

    if not checks:
        checks.append(row(check_date, None, "info", "slate_clean", "No data integrity issues found.", {"date": str(check_date)}))
    df = pd.DataFrame(checks)
    upsert_df(con, "data_quality_checks", df)
    return df


def row(check_date: date, game_pk: int | None, severity: str, name: str, message: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload, default=str, sort_keys=True)
    check_id = hashlib.sha1(f"{check_date}:{game_pk}:{name}:{raw}".encode()).hexdigest()
    return {
        "check_id": check_id,
        "game_pk": game_pk,
        "game_date": check_date,
        "severity": severity,
        "check_name": name,
        "message": message,
        "status": "open" if severity in {"high", "medium"} else "ok",
        "raw_payload": payload,
    }
