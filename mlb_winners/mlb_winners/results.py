from __future__ import annotations

import pandas as pd

from .db import upsert_df
from .odds import american_profit_per_dollar


def settle_bet_recommendations(con, games: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    if games.empty or recommendations.empty:
        return pd.DataFrame()
    game_lookup = {row["game_pk"]: row for row in games.to_dict("records")}
    rows = []
    for bet in recommendations.to_dict("records"):
        game = game_lookup.get(bet.get("game_pk"))
        if not game or pd.isna(game.get("home_won")):
            continue
        status = str(game.get("status") or "").lower()
        is_final = any(token in status for token in ["final", "completed", "game over"])
        if not is_final:
            continue
        if pd.isna(game.get("home_score")) or pd.isna(game.get("away_score")):
            continue
        selection = bet.get("selection")
        if selection == game.get("home_team"):
            won = bool(game.get("home_won"))
        elif selection == game.get("away_team"):
            won = not bool(game.get("home_won"))
        else:
            continue
        stake = float(bet.get("stake_units") or 1.0)
        odds = int(bet.get("odds"))
        profit = stake * american_profit_per_dollar(odds) if won else -stake
        rows.append(
            {
                "recommendation_id": bet["recommendation_id"],
                "result": "win" if won else "loss",
                "units_profit": profit,
                "closing_odds": None,
                "clv": None,
                "notes": "settled_from_final_score",
            }
        )
    df = pd.DataFrame(rows)
    upsert_df(con, "bet_results", df)
    return df
