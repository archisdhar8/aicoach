from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_prediction_frame
from .props import predict_props, write_props


def predict_nrfi(
    games: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    statcast_pitchers: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    lineups: pd.DataFrame | None,
    player_stats: pd.DataFrame | None,
) -> pd.DataFrame:
    frame = build_prediction_frame(games, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    rows = []
    for row in frame.itertuples(index=False):
        away_score = first_inning_score_probability(row, batting_side="away", pitching_side="home")
        home_score = first_inning_score_probability(row, batting_side="home", pitching_side="away")
        nrfi = float(np.clip((1.0 - away_score) * (1.0 - home_score), 0.05, 0.95))
        confirmed = min(float(getattr(row, "home_lineup_confirmed", 0.0)), float(getattr(row, "away_lineup_confirmed", 0.0))) >= 1.0
        rows.append(
            {
                "game_pk": row.game_pk,
                "game_date": row.game_date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "away_top1_score_probability": away_score,
                "home_bottom1_score_probability": home_score,
                "nrfi_probability": nrfi,
                "model_version": "nrfi-scaffold-v1",
                "confidence": "scaffold",
                "uncertainty_score": 0.55 if confirmed else 0.75,
                "skip_reason": "" if confirmed else "lineup_unconfirmed",
            }
        )
    return pd.DataFrame(rows)


def predict_f5(
    games: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    statcast_pitchers: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    lineups: pd.DataFrame | None,
    player_stats: pd.DataFrame | None,
) -> pd.DataFrame:
    frame = build_prediction_frame(games, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    rows = []
    for row in frame.itertuples(index=False):
        home = f5_runs(row, batting_side="home", pitching_side="away")
        away = f5_runs(row, batting_side="away", pitching_side="home")
        confirmed = min(float(getattr(row, "home_lineup_confirmed", 0.0)), float(getattr(row, "away_lineup_confirmed", 0.0))) >= 1.0
        rows.append(
            {
                "game_pk": row.game_pk,
                "game_date": row.game_date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "projected_home_f5_runs": home,
                "projected_away_f5_runs": away,
                "projected_f5_total": home + away,
                "projected_f5_run_diff": home - away,
                "model_version": "f5-scaffold-v1",
                "confidence": "scaffold",
                "uncertainty_score": 0.50 if confirmed else 0.72,
                "skip_reason": "" if confirmed else "lineup_unconfirmed",
            }
        )
    return pd.DataFrame(rows)


def predict_hitter_total_bases(
    games: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    statcast_pitchers: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    lineups: pd.DataFrame,
    player_stats: pd.DataFrame,
    prop_lines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return predict_props(games, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats, "total_bases", prop_lines)


def first_inning_score_probability(row, batting_side: str, pitching_side: str) -> float:
    top3_rating = float(getattr(row, f"{batting_side}_lineup_offense_rating", 0.7) or 0.7)
    offense = float(getattr(row, f"{batting_side}_team_xwoba_last14", 0.320) or 0.320)
    starter_fip = float(getattr(row, f"{pitching_side}_starter_fip_proxy", 4.2) or 4.2)
    starter_kbb = float(getattr(row, f"{pitching_side}_starter_kbb_prior", 2.2) or 2.2)
    starter_hr = float(getattr(row, f"{pitching_side}_starter_barrel_allowed", 0.075) or 0.075)
    park = float(getattr(row, "park_run_factor", 1.0) or 1.0)
    weather = 1.0 + min(float(getattr(row, "wind_speed_mph", 0.0) or 0.0), 20.0) * 0.003
    logit = -1.35 + 0.72 * (top3_rating - 0.7) + 3.2 * (offense - 0.320) + 0.16 * (starter_fip - 4.2) - 0.055 * (starter_kbb - 2.2) + 1.1 * (starter_hr - 0.075)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return float(np.clip(prob * park * weather, 0.08, 0.48))


def f5_runs(row, batting_side: str, pitching_side: str) -> float:
    offense_runs = float(getattr(row, f"{batting_side}_last14_runs_for", 4.4) or 4.4)
    offense_xwoba = float(getattr(row, f"{batting_side}_team_xwoba_last14", 0.320) or 0.320)
    lineup = float(getattr(row, f"{batting_side}_lineup_offense_rating", 0.7) or 0.7)
    starter_fip = float(getattr(row, f"{pitching_side}_starter_fip_proxy", 4.2) or 4.2)
    starter_xwoba = float(getattr(row, f"{pitching_side}_starter_xwoba_allowed", 0.320) or 0.320)
    park = float(getattr(row, "park_run_factor", 1.0) or 1.0)
    base = 2.15 + 0.22 * (offense_runs - 4.4) + 3.1 * (offense_xwoba - 0.320) + 0.70 * (lineup - 0.7)
    pitcher_adj = 0.18 * (starter_fip - 4.2) + 2.0 * (starter_xwoba - 0.320)
    return float(np.clip((base + pitcher_adj) * park, 0.5, 6.5))


def write_market_predictions(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
