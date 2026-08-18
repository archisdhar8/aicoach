from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_training_frame
from .modeling import predict_home_prob, train_model
from .odds import classify_edge, devig_two_way, expected_value_per_dollar, american_profit_per_dollar


def run_value_backtest(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    odds: pd.DataFrame,
    test_year: int,
    report_dir: Path,
    statcast_team: pd.DataFrame | None = None,
    statcast_pitchers: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    top_n: int = 3,
    min_confidence: str = "strong",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = build_training_frame(games, team_stats, statcast_team, statcast_pitchers, weather)
    train = frame[(frame["season"] < test_year) & frame["target_home_win"].notna()]
    test = frame[(frame["season"] == test_year) & frame["target_home_win"].notna()].copy()
    if train.empty or test.empty:
        raise ValueError(f"Not enough data for value backtest: train={len(train)} test={len(test)}")
    bundle = train_model(train)
    test["model_home_prob"] = predict_home_prob(bundle, test)
    test["model_away_prob"] = 1.0 - test["model_home_prob"]
    candidates = attach_historical_odds(test, odds)
    candidates = candidates[candidates["confidence"] == min_confidence].copy()
    candidates = candidates.sort_values(["game_date", "ev_per_dollar"], ascending=[True, False])
    bets = candidates.groupby("game_date", as_index=False, group_keys=False).head(top_n).copy()
    if bets.empty:
        summary = pd.DataFrame(
            [{"test_year": test_year, "bets": 0, "units_profit": 0.0, "roi": 0.0, "win_rate": np.nan}]
        )
    else:
        bets["won"] = np.where(
            bets["bet_side"] == bets["home_team"],
            bets["target_home_win"].astype(bool),
            ~bets["target_home_win"].astype(bool),
        )
        bets["units_profit"] = np.where(
            bets["won"],
            bets["bet_moneyline"].map(american_profit_per_dollar),
            -1.0,
        )
        summary = pd.DataFrame(
            [
                {
                    "test_year": test_year,
                    "days_with_bets": bets["game_date"].nunique(),
                    "bets": len(bets),
                    "wins": int(bets["won"].sum()),
                    "losses": int((~bets["won"]).sum()),
                    "win_rate": float(bets["won"].mean()),
                    "units_staked": float(len(bets)),
                    "units_profit": float(bets["units_profit"].sum()),
                    "roi": float(bets["units_profit"].sum() / len(bets)),
                    "avg_edge": float(bets["edge"].mean()),
                    "avg_ev": float(bets["ev_per_dollar"].mean()),
                }
            ]
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    bets.to_csv(report_dir / f"value_bets_{test_year}.csv", index=False)
    summary.to_csv(report_dir / f"value_backtest_{test_year}.csv", index=False)
    return summary, bets


def attach_historical_odds(test: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return pd.DataFrame()
    odds = odds.copy()
    odds["snapshot_date"] = pd.to_datetime(odds["snapshot_date"]).dt.normalize()
    odds = odds.sort_values(["snapshot_date", "home_team", "away_team", "bookmaker"])
    odds = odds.drop_duplicates(["snapshot_date", "home_team", "away_team"])
    rows = []
    for row in test.itertuples(index=False):
        game_date = pd.Timestamp(row.game_date).normalize()
        match = odds[
            (odds["snapshot_date"] == game_date)
            & (odds["home_team"] == row.home_team)
            & (odds["away_team"] == row.away_team)
        ]
        if match.empty:
            continue
        market = match.iloc[0]
        fair_home, fair_away = devig_two_way(market.home_moneyline, market.away_moneyline)
        home_edge = row.model_home_prob - fair_home
        away_edge = row.model_away_prob - fair_away
        home_ev = expected_value_per_dollar(row.model_home_prob, market.home_moneyline)
        away_ev = expected_value_per_dollar(row.model_away_prob, market.away_moneyline)
        if home_ev >= away_ev:
            bet_side = row.home_team
            bet_moneyline = market.home_moneyline
            edge = home_edge
            ev = home_ev
        else:
            bet_side = row.away_team
            bet_moneyline = market.away_moneyline
            edge = away_edge
            ev = away_ev
        confidence = classify_edge(edge, ev)
        rows.append(
            {
                "game_pk": row.game_pk,
                "game_date": game_date.date(),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "target_home_win": row.target_home_win,
                "model_home_prob": row.model_home_prob,
                "model_away_prob": row.model_away_prob,
                "bookmaker": market.bookmaker,
                "home_moneyline": market.home_moneyline,
                "away_moneyline": market.away_moneyline,
                "bet_side": bet_side,
                "bet_moneyline": bet_moneyline,
                "edge": edge,
                "ev_per_dollar": ev,
                "confidence": confidence,
            }
        )
    return pd.DataFrame(rows)
