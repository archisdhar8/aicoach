from __future__ import annotations

import numpy as np
import pandas as pd

from .odds import devig_two_way


def elo_probability(home_rating: float, away_rating: float, home_field: float = 24.0) -> float:
    return float(1.0 / (1.0 + 10 ** (-((home_rating + home_field) - away_rating) / 400.0)))


def margin_multiplier(home_score: int, away_score: int, rating_diff: float) -> float:
    margin = abs(int(home_score) - int(away_score))
    if margin <= 1:
        return 1.0
    return float(np.log(margin + 1.0) * (2.2 / ((abs(rating_diff) * 0.001) + 2.2)))


def build_elo_ratings(games: pd.DataFrame, k: float = 20.0, initial_rating: float = 1500.0) -> dict[int, float]:
    ratings: dict[int, float] = {}
    if games.empty:
        return ratings
    df = games[games["home_won"].notna()].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk"])
    for row in df.to_dict("records"):
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        home_rating = ratings.get(home_id, initial_rating)
        away_rating = ratings.get(away_id, initial_rating)
        expected_home = elo_probability(home_rating, away_rating)
        actual_home = 1.0 if row.get("home_won") else 0.0
        rating_diff = home_rating - away_rating
        mult = margin_multiplier(row.get("home_score", 0) or 0, row.get("away_score", 0) or 0, rating_diff)
        change = k * mult * (actual_home - expected_home)
        ratings[home_id] = home_rating + change
        ratings[away_id] = away_rating - change
    return ratings


def elo_probs_for_frame(frame: pd.DataFrame, ratings: dict[int, float], initial_rating: float = 1500.0) -> np.ndarray:
    probs = []
    for row in frame.to_dict("records"):
        home_rating = ratings.get(int(row["home_team_id"]), initial_rating)
        away_rating = ratings.get(int(row["away_team_id"]), initial_rating)
        starter_adj = (row.get("away_starter_fip_proxy", 4.5) - row.get("home_starter_fip_proxy", 4.5)) * 18.0
        probs.append(elo_probability(home_rating + starter_adj, away_rating))
    return np.array(probs, dtype=float)


def market_prior_from_odds(home_moneyline: int, away_moneyline: int) -> float:
    home, _ = devig_two_way(home_moneyline, away_moneyline)
    return home


def blend_probabilities(probabilities: dict[str, float], weights: dict[str, float] | None = None) -> float:
    clean = {k: float(v) for k, v in probabilities.items() if v is not None and not pd.isna(v)}
    if not clean:
        return 0.5
    if weights is None:
        weights = {k: 1.0 for k in clean}
    total_weight = sum(weights.get(k, 0.0) for k in clean)
    if total_weight <= 0:
        return float(np.mean(list(clean.values())))
    return float(sum(clean[k] * weights.get(k, 0.0) for k in clean) / total_weight)


def model_disagreement(probabilities: dict[str, float]) -> float:
    values = [float(v) for v in probabilities.values() if v is not None and not pd.isna(v)]
    return float(np.std(values)) if len(values) > 1 else 0.0


def uncertainty_score(disagreement: float, data_quality_penalty: float = 0.0, lineup_confirmed: bool = False) -> float:
    lineup_penalty = 0.0 if lineup_confirmed else 0.12
    return float(np.clip(disagreement * 2.0 + data_quality_penalty + lineup_penalty, 0.0, 1.0))
