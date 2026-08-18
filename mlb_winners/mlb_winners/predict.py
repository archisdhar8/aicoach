from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_prediction_frame, build_training_frame
from .ensemble import build_elo_ratings, blend_probabilities, elo_probs_for_frame, model_disagreement, uncertainty_score
from .market import apply_moneyline_tier_filter, latest_odds_by_game, line_move_for_side, market_adjusted_probability
from .modeling import load_bundle, predict_home_prob, train_model
from .odds import devig_two_way, expected_value_per_dollar, classify_edge


EXPLANATION_FEATURE_COLUMNS = [
    "home_win_pct",
    "away_win_pct",
    "home_run_diff_per_game",
    "away_run_diff_per_game",
    "home_last7_run_diff",
    "away_last7_run_diff",
    "home_last14_run_diff",
    "away_last14_run_diff",
    "home_last30_run_diff",
    "away_last30_run_diff",
    "home_last14_runs_for",
    "away_last14_runs_for",
    "home_ewma_runs_for",
    "away_ewma_runs_for",
    "home_ewma_runs_against",
    "away_ewma_runs_against",
    "home_bullpen_ip_last3",
    "away_bullpen_ip_last3",
    "home_bullpen_er_last3",
    "away_bullpen_er_last3",
    "home_bullpen_quality_score",
    "away_bullpen_quality_score",
    "home_starter_era_prior",
    "away_starter_era_prior",
    "home_starter_season_era",
    "away_starter_season_era",
    "home_starter_season_whip",
    "away_starter_season_whip",
    "home_starter_season_kbb",
    "away_starter_season_kbb",
    "home_starter_season_fip_proxy",
    "away_starter_season_fip_proxy",
    "home_starter_season_starts",
    "away_starter_season_starts",
    "home_starter_whip_prior",
    "away_starter_whip_prior",
    "home_starter_kbb_prior",
    "away_starter_kbb_prior",
    "home_starter_fip_proxy",
    "away_starter_fip_proxy",
    "home_starter_last5_fip",
    "away_starter_last5_fip",
    "home_starter_rest_days",
    "away_starter_rest_days",
    "home_starter_workload_trend",
    "away_starter_workload_trend",
    "home_starter_xwoba_allowed",
    "away_starter_xwoba_allowed",
    "home_starter_hard_hit_allowed",
    "away_starter_hard_hit_allowed",
    "home_starter_barrel_allowed",
    "away_starter_barrel_allowed",
    "home_team_xwoba_last14",
    "away_team_xwoba_last14",
    "home_team_hard_hit_last14",
    "away_team_hard_hit_last14",
    "home_team_barrel_last14",
    "away_team_barrel_last14",
    "home_team_k_rate_last14",
    "away_team_k_rate_last14",
    "home_team_bb_rate_last14",
    "away_team_bb_rate_last14",
    "home_pitching_xwoba_allowed_last14",
    "away_pitching_xwoba_allowed_last14",
    "home_bullpen_kbb_last7",
    "away_bullpen_kbb_last7",
    "home_lineup_offense_rating",
    "away_lineup_offense_rating",
    "home_lineup_platoon_score",
    "away_lineup_platoon_score",
    "temperature_f",
    "park_run_factor",
]


DAILY_PREDICTION_COLUMNS = [
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "home_probable_pitcher_id",
    "away_probable_pitcher_id",
    "home_starter_games_prior",
    "away_starter_games_prior",
    "home_lineup_confirmed",
    "away_lineup_confirmed",
    "home_bullpen_fatigue_score",
    "away_bullpen_fatigue_score",
    "precipitation_in",
    "wind_speed_mph",
    "xgb_home_prob",
    "logistic_home_prob",
    "elo_home_prob",
    "model_home_prob",
    "model_away_prob",
    "model_disagreement",
    "uncertainty_score",
    "model_pick",
    "bet_side",
    "bet_moneyline",
    "edge",
    "ev_per_dollar",
    "confidence",
    "skip_reason",
    "tier",
    "risk_flags",
    "stake_units",
]


def empty_daily_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_PREDICTION_COLUMNS)


def train_or_load_model(games: pd.DataFrame, team_stats: pd.DataFrame, model_path: Path, train_through: int | None = None):
    if model_path.exists():
        return load_bundle(model_path)
    frame = build_training_frame(games, team_stats)
    if train_through is not None:
        frame = frame[frame["season"] <= train_through]
    return train_model(frame)


def make_daily_predictions(
    today_games: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    statcast_pitchers: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    odds: pd.DataFrame | None,
    model_bundle,
    lineups: pd.DataFrame | None = None,
    player_stats: pd.DataFrame | None = None,
    baseline_model=None,
    edge_threshold: float = 0.03,
) -> pd.DataFrame:
    if today_games.empty or "game_pk" not in today_games.columns:
        return empty_daily_prediction_frame()
    frame = build_prediction_frame(today_games, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    if frame.empty:
        return empty_daily_prediction_frame()
    frame["xgb_home_prob"] = predict_home_prob(model_bundle, frame)
    frame["logistic_home_prob"] = baseline_model.predict_proba(frame[model_bundle.feature_columns])[:, 1] if baseline_model is not None else np.nan
    elo_ratings = build_elo_ratings(history_games)
    frame["elo_home_prob"] = elo_probs_for_frame(frame, elo_ratings)
    frame["model_home_prob"] = frame.apply(
        lambda r: blend_probabilities(
            {
                "xgb": r["xgb_home_prob"],
                "logistic": r["logistic_home_prob"],
                "elo": r["elo_home_prob"],
            },
            {"xgb": 0.5, "logistic": 0.2, "elo": 0.2},
        ),
        axis=1,
    )
    frame["model_away_prob"] = 1.0 - frame["model_home_prob"]
    output_columns = [
            "game_pk",
            "game_date",
            "home_team",
            "away_team",
            "home_probable_pitcher_id",
            "away_probable_pitcher_id",
            "home_starter_games_prior",
            "away_starter_games_prior",
            "home_lineup_confirmed",
            "away_lineup_confirmed",
            "home_bullpen_fatigue_score",
            "away_bullpen_fatigue_score",
            "precipitation_in",
            "wind_speed_mph",
            "xgb_home_prob",
            "logistic_home_prob",
            "elo_home_prob",
            "model_home_prob",
            "model_away_prob",
    ]
    output_columns.extend([column for column in EXPLANATION_FEATURE_COLUMNS if column in frame.columns and column not in output_columns])
    output = frame[output_columns].copy()
    output["model_disagreement"] = frame.apply(
        lambda r: model_disagreement({"xgb": r["xgb_home_prob"], "logistic": r["logistic_home_prob"], "elo": r["elo_home_prob"]}),
        axis=1,
    )
    output["uncertainty_score"] = output.apply(
        lambda r: uncertainty_score(
            r["model_disagreement"],
            data_quality_penalty=float((1.0 - min(r["home_lineup_confirmed"], r["away_lineup_confirmed"])) * 0.10),
            lineup_confirmed=bool(min(r["home_lineup_confirmed"], r["away_lineup_confirmed"])),
        ),
        axis=1,
    )
    output["model_pick"] = np.where(output["model_home_prob"] >= 0.5, output["home_team"], output["away_team"])

    if odds is not None and not odds.empty:
        output = attach_odds(output, odds)
    else:
        output["bet_side"] = "no odds"
        output["bet_moneyline"] = np.nan
        output["edge"] = np.nan
        output["ev_per_dollar"] = np.nan
        output["confidence"] = "no odds"
        output["skip_reason"] = "missing_odds"
        output["tier"] = "no odds"
        output["risk_flags"] = ""
        output["stake_units"] = 0.0
    return output.sort_values(["confidence", "ev_per_dollar"], ascending=[True, False])


def attach_odds(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Prefer one bookmaker row per game. If multiple books exist, keep the first stable row.
    all_odds = odds.copy()
    all_odds = all_odds[all_odds["home_moneyline"].notna() & all_odds["away_moneyline"].notna()].copy()
    if all_odds.empty:
        predictions = predictions.copy()
        predictions["bet_side"] = "no odds"
        predictions["bet_moneyline"] = np.nan
        predictions["edge"] = np.nan
        predictions["ev_per_dollar"] = np.nan
        predictions["confidence"] = "no odds"
        predictions["skip_reason"] = "missing_odds"
        predictions["tier"] = "no odds"
        predictions["risk_flags"] = ""
        predictions["stake_units"] = 0.0
        return predictions
    odds = latest_odds_by_game(all_odds).sort_values(["home_team", "away_team", "bookmaker"]).drop_duplicates(["home_team", "away_team"])
    odds_lookup = {(r.home_team, r.away_team): r for r in odds.itertuples(index=False)}
    for row in predictions.itertuples(index=False):
        result = row._asdict()
        market = odds_lookup.get((row.home_team, row.away_team))
        if not market:
            result.update(
                {
                    "bet_side": "no odds",
                    "bet_moneyline": np.nan,
                    "edge": np.nan,
                    "ev_per_dollar": np.nan,
                    "confidence": "no odds",
                    "skip_reason": "missing_odds",
                    "tier": "no odds",
                    "risk_flags": "",
                    "stake_units": 0.0,
                }
            )
            rows.append(result)
            continue
        fair_home, fair_away = devig_two_way(market.home_moneyline, market.away_moneyline)
        home_market_blend = market_adjusted_probability(row.model_home_prob, fair_home)
        away_market_blend = 1.0 - home_market_blend
        home_ev = expected_value_per_dollar(home_market_blend, market.home_moneyline)
        away_ev = expected_value_per_dollar(away_market_blend, market.away_moneyline)
        home_edge = home_market_blend - fair_home
        away_edge = away_market_blend - fair_away
        if home_ev >= away_ev:
            edge, ev, side, line, side_prob = home_edge, home_ev, row.home_team, market.home_moneyline, home_market_blend
        else:
            edge, ev, side, line, side_prob = away_edge, away_ev, row.away_team, market.away_moneyline, away_market_blend
        line_move = line_move_for_side(all_odds, row.home_team, row.away_team, side)
        missing_starter = bool(pd.isna(row.home_probable_pitcher_id) or pd.isna(row.away_probable_pitcher_id))
        missing_lineup = bool(min(row.home_lineup_confirmed, row.away_lineup_confirmed) < 1.0)
        weather_uncertain = bool((row.precipitation_in or 0.0) > 0.12 or (row.wind_speed_mph or 0.0) > 22)
        bullpen_uncertain = bool(max(row.home_bullpen_fatigue_score, row.away_bullpen_fatigue_score) > 0.85)
        thin_starter_history = bool(min(row.home_starter_games_prior, row.away_starter_games_prior) < 2)
        filter_result = apply_moneyline_tier_filter(
            edge=edge,
            ev=ev,
            moneyline=line,
            uncertainty_score=row.uncertainty_score,
            missing_starter=missing_starter,
            missing_lineup=missing_lineup,
            weather_uncertain=weather_uncertain,
            adverse_line_move=line_move,
            bullpen_uncertainty=bullpen_uncertain,
            thin_starter_history=thin_starter_history,
        )
        confidence = filter_result.confidence
        skip_reason = filter_result.skip_reason
        tier = filter_result.tier
        risk_flags = "|".join(filter_result.risk_flags)
        stake_units = filter_result.stake_units
        elite_starter_flag = elite_opposing_starter_flag(row, side, edge)
        if confidence != "no bet" and elite_starter_flag:
            confidence = "no bet"
            skip_reason = elite_starter_flag
            tier = "no bet"
            stake_units = 0.0
        legacy_confidence = classify_edge(edge, ev)
        if confidence == "thin" and legacy_confidence == "no bet":
            confidence = "no bet"
            skip_reason = "edge_below_legacy_threshold"
            tier = "no bet"
            stake_units = 0.0
        if confidence == "no bet":
            side = "no bet"
        result.update(
            {
                "bookmaker": market.bookmaker,
                "home_moneyline": market.home_moneyline,
                "away_moneyline": market.away_moneyline,
                "fair_home_prob": fair_home,
                "fair_away_prob": fair_away,
                "market_adjusted_home_prob": home_market_blend,
                "market_adjusted_away_prob": away_market_blend,
                "bet_probability": side_prob if side != "no bet" else np.nan,
                "bet_side": side,
                "bet_moneyline": line if side != "no bet" else np.nan,
                "edge": edge,
                "ev_per_dollar": ev,
                "confidence": confidence,
                "skip_reason": skip_reason,
                "tier": tier,
                "risk_flags": risk_flags,
                "stake_units": stake_units,
                "line_move_for_side": line_move,
            }
        )
        rows.append(result)
    return pd.DataFrame(rows)


def elite_opposing_starter_flag(row, side: str, edge: float) -> str:
    if side in {"no bet", "no odds", None}:
        return ""
    opponent_prefix = "away" if side == row.home_team else "home"
    starts = float(getattr(row, f"{opponent_prefix}_starter_season_starts", 0.0) or 0.0)
    if starts < 5:
        return ""
    era = float(getattr(row, f"{opponent_prefix}_starter_season_era", 4.5) or 4.5)
    fip = float(getattr(row, f"{opponent_prefix}_starter_season_fip_proxy", 4.5) or 4.5)
    kbb = float(getattr(row, f"{opponent_prefix}_starter_season_kbb", 2.2) or 2.2)
    last5_fip = float(getattr(row, f"{opponent_prefix}_starter_last5_fip", fip) or fip)
    elite = era <= 2.25 and (fip <= 3.25 or last5_fip <= 3.00 or kbb >= 4.0)
    if elite and edge < 0.12:
        return f"elite_opposing_starter_season_era_{era:.2f}"
    return ""
