from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import build_prediction_frame


@dataclass(frozen=True)
class SimulationConfig:
    sims: int = 20000
    seed: int = 42
    starter_innings_weight: float = 0.62


def simulate_game_from_row(row: pd.Series | dict[str, Any], config: SimulationConfig = SimulationConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = dict(row)
    rng = np.random.default_rng(config.seed + int(r.get("game_pk", 0) or 0))
    home_runs = np.zeros(config.sims, dtype=int)
    away_runs = np.zeros(config.sims, dtype=int)
    home_first5 = np.zeros(config.sims, dtype=int)
    away_first5 = np.zeros(config.sims, dtype=int)
    home_innings = np.zeros((config.sims, 9), dtype=int)
    away_innings = np.zeros((config.sims, 9), dtype=int)
    home_lambda, away_lambda = expected_team_runs(r)
    for inning in range(9):
        inning_weight = 1.08 if inning < 5 else 0.92
        h_rate = max(home_lambda / 9.0 * inning_weight, 0.05)
        a_rate = max(away_lambda / 9.0 * inning_weight, 0.05)
        away_innings[:, inning] = rng.poisson(a_rate, size=config.sims)
        home_innings[:, inning] = rng.poisson(h_rate, size=config.sims)
    home_runs = home_innings.sum(axis=1)
    away_runs = away_innings.sum(axis=1)
    home_first5 = home_innings[:, :5].sum(axis=1)
    away_first5 = away_innings[:, :5].sum(axis=1)
    draws = home_runs == away_runs
    if draws.any():
        home_runs[draws] += rng.binomial(1, 0.54, size=draws.sum())
        away_runs[draws] += rng.binomial(1, 0.46, size=draws.sum())
    sims = pd.DataFrame(
        {
            "game_pk": r.get("game_pk"),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "home_runs": home_runs,
            "away_runs": away_runs,
            "total_runs": home_runs + away_runs,
            "home_first5": home_first5,
            "away_first5": away_first5,
        }
    )
    summary = summarize_simulations(sims)
    summary["expected_home_runs"] = home_lambda
    summary["expected_away_runs"] = away_lambda
    return summary, sims


def simulate_slate(
    games: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    statcast_pitchers: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    config: SimulationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = build_prediction_frame(games, history_games, team_stats, statcast_team, statcast_pitchers, weather)
    summaries = []
    samples = []
    for _, row in frame.iterrows():
        summary, sims = simulate_game_from_row(row, config)
        summaries.append(summary)
        samples.append(sims)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(), pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()


def add_simulation_columns(
    predictions: pd.DataFrame,
    config: SimulationConfig = SimulationConfig(sims=10000, seed=42),
) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    summaries = []
    for _, row in predictions.iterrows():
        summary, _ = simulate_game_from_row(row, config)
        summaries.append(summary)
    simulation = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if simulation.empty:
        return predictions.copy()
    sim_columns = [
        "game_pk",
        "sim_home_win_prob",
        "sim_away_win_prob",
        "mean_total_runs",
        "total_p10",
        "total_p50",
        "total_p90",
        "home_team_total_mean",
        "away_team_total_mean",
        "home_first5_mean",
        "away_first5_mean",
        "first5_total_mean",
        "expected_home_runs",
        "expected_away_runs",
    ]
    out = predictions.merge(simulation[[column for column in sim_columns if column in simulation.columns]], on="game_pk", how="left")
    out["sim_pick"] = np.where(out["sim_home_win_prob"] >= 0.5, out["home_team"], out["away_team"])
    out["model_sim_disagreement"] = (out["model_home_prob"] - out["sim_home_win_prob"]).abs()
    out["model_sim_agree"] = out["model_pick"].eq(out["sim_pick"])
    return out


def expected_team_runs(r: dict[str, Any]) -> tuple[float, float]:
    base = 4.35
    park = float(r.get("park_run_factor", 1.0) or 1.0)
    temp_boost = (float(r.get("temperature_f", 72.0) or 72.0) - 72.0) * 0.006
    wind_boost = min(float(r.get("wind_out_proxy", 0.0) or 0.0), 20.0) * 0.004
    env = np.clip(park + temp_boost + wind_boost, 0.75, 1.35)
    home_off = true_talent_offense_index(r, "home", "away")
    away_off = true_talent_offense_index(r, "away", "home")
    home_pitch = true_talent_run_prevention_index(r, "home", "away")
    away_pitch = true_talent_run_prevention_index(r, "away", "home")
    home_runs = base * env * home_off * away_pitch * 1.03
    away_runs = base * env * away_off * home_pitch * 0.97
    return float(np.clip(home_runs, 2.0, 8.5)), float(np.clip(away_runs, 2.0, 8.5))


def true_talent_offense_index(r: dict[str, Any], side: str, opponent_side: str) -> float:
    raw = offensive_index(r, side)
    opponent_quality = opponent_pitching_quality_adjustment(r, opponent_side)
    lineup = lineup_index(r, side)
    recent = recent_run_environment_index(r, side)
    return float(np.clip(raw * opponent_quality * lineup * recent, 0.68, 1.38))


def true_talent_run_prevention_index(r: dict[str, Any], pitching_side: str, opponent_side: str) -> float:
    raw = run_prevention_index(r, pitching_side)
    opponent_quality = opponent_offense_quality_adjustment(r, opponent_side)
    starter_share = starter_innings_share(r, pitching_side)
    bullpen = bullpen_leverage_index(r, pitching_side)
    starter = starter_quality_index(r, pitching_side)
    mixed = starter_share * starter + (1.0 - starter_share) * bullpen
    return float(np.clip(raw * opponent_quality * mixed, 0.68, 1.42))


def offensive_index(r: dict[str, Any], side: str) -> float:
    xwoba = float(r.get(f"{side}_team_xwoba_last14", 0.320) or 0.320)
    barrel = float(r.get(f"{side}_team_barrel_last14", 0.075) or 0.075)
    k_rate = float(r.get(f"{side}_team_k_rate_last14", 0.22) or 0.22)
    bb_rate = float(r.get(f"{side}_team_bb_rate_last14", 0.085) or 0.085)
    rd = float(r.get(f"{side}_run_diff_per_game", 0.0) or 0.0)
    idx = 1.0 + 1.8 * (xwoba - 0.320) + 1.5 * (barrel - 0.075) - 0.8 * (k_rate - 0.22) + 0.9 * (bb_rate - 0.085) + 0.025 * rd
    return float(np.clip(idx, 0.72, 1.32))


def run_prevention_index(r: dict[str, Any], pitching_side: str) -> float:
    starter_fip = float(r.get(f"{pitching_side}_starter_fip_proxy", 4.5) or 4.5)
    starter_xwoba = float(r.get(f"{pitching_side}_starter_xwoba_allowed", 0.320) or 0.320)
    bullpen_er = float(r.get(f"{pitching_side}_bullpen_er_last7", 3.5) or 3.5)
    bullpen_kbb = float(r.get(f"{pitching_side}_bullpen_kbb_last7", 2.2) or 2.2)
    idx = 1.0 + 0.09 * (starter_fip - 4.2) + 1.3 * (starter_xwoba - 0.320) + 0.02 * (bullpen_er - 3.5) - 0.03 * (bullpen_kbb - 2.2)
    return float(np.clip(idx, 0.72, 1.35))


def opponent_pitching_quality_adjustment(r: dict[str, Any], opponent_side: str) -> float:
    allowed_xwoba = float(r.get(f"{opponent_side}_pitching_xwoba_allowed_last14", 0.320) or 0.320)
    starter_fip = float(r.get(f"{opponent_side}_starter_fip_proxy", 4.2) or 4.2)
    bullpen_kbb = float(r.get(f"{opponent_side}_bullpen_kbb_last7", 2.2) or 2.2)
    idx = 1.0 + 0.9 * (allowed_xwoba - 0.320) + 0.035 * (starter_fip - 4.2) - 0.025 * (bullpen_kbb - 2.2)
    return float(np.clip(idx, 0.86, 1.16))


def opponent_offense_quality_adjustment(r: dict[str, Any], opponent_side: str) -> float:
    xwoba = float(r.get(f"{opponent_side}_team_xwoba_last14", 0.320) or 0.320)
    barrel = float(r.get(f"{opponent_side}_team_barrel_last14", 0.075) or 0.075)
    k_rate = float(r.get(f"{opponent_side}_team_k_rate_last14", 0.22) or 0.22)
    idx = 1.0 + 1.0 * (xwoba - 0.320) + 0.8 * (barrel - 0.075) - 0.35 * (k_rate - 0.22)
    return float(np.clip(idx, 0.88, 1.14))


def lineup_index(r: dict[str, Any], side: str) -> float:
    rating = float(r.get(f"{side}_lineup_offense_rating", 0.7) or 0.7)
    platoon = float(r.get(f"{side}_lineup_platoon_score", 0.0) or 0.0)
    confirmed = float(r.get(f"{side}_lineup_confirmed", 0.0) or 0.0)
    idx = 1.0 + 0.18 * (rating - 0.7) + 0.10 * platoon
    if confirmed < 1.0:
        idx *= 0.985
    return float(np.clip(idx, 0.90, 1.12))


def recent_run_environment_index(r: dict[str, Any], side: str) -> float:
    last14_for = float(r.get(f"{side}_last14_runs_for", 4.35) or 4.35)
    ewma_for = float(r.get(f"{side}_ewma_runs_for", 4.35) or 4.35)
    idx = 1.0 + 0.018 * ((last14_for + ewma_for) / 2.0 - 4.35)
    return float(np.clip(idx, 0.90, 1.12))


def starter_innings_share(r: dict[str, Any], pitching_side: str) -> float:
    games = float(r.get(f"{pitching_side}_starter_games_prior", 8.0) or 0.0)
    rest = float(r.get(f"{pitching_side}_starter_rest_days", 5.0) or 5.0)
    workload = float(r.get(f"{pitching_side}_starter_workload_trend", 0.0) or 0.0)
    share = 0.60 + min(games, 20.0) * 0.004 + np.clip(rest - 4.0, -2.0, 3.0) * 0.015 + np.clip(workload, -25.0, 25.0) * 0.001
    return float(np.clip(share, 0.48, 0.74))


def starter_quality_index(r: dict[str, Any], pitching_side: str) -> float:
    fip = float(r.get(f"{pitching_side}_starter_fip_proxy", 4.2) or 4.2)
    last5_fip = float(r.get(f"{pitching_side}_starter_last5_fip", fip) or fip)
    xwoba = float(r.get(f"{pitching_side}_starter_xwoba_allowed", 0.320) or 0.320)
    hard_hit = float(r.get(f"{pitching_side}_starter_hard_hit_allowed", 0.39) or 0.39)
    barrel = float(r.get(f"{pitching_side}_starter_barrel_allowed", 0.075) or 0.075)
    kbb = float(r.get(f"{pitching_side}_starter_kbb_prior", 2.4) or 2.4)
    idx = (
        1.0
        + 0.055 * (fip - 4.2)
        + 0.035 * (last5_fip - 4.2)
        + 1.05 * (xwoba - 0.320)
        + 0.30 * (hard_hit - 0.39)
        + 0.65 * (barrel - 0.075)
        - 0.025 * (kbb - 2.4)
    )
    return float(np.clip(idx, 0.78, 1.28))


def bullpen_leverage_index(r: dict[str, Any], pitching_side: str) -> float:
    fatigue = float(r.get(f"{pitching_side}_bullpen_fatigue_score", 0.0) or 0.0)
    ip3 = float(r.get(f"{pitching_side}_bullpen_ip_last3", 10.0) or 10.0)
    er3 = float(r.get(f"{pitching_side}_bullpen_er_last3", 3.0) or 3.0)
    quality = float(r.get(f"{pitching_side}_bullpen_quality_score", 0.0) or 0.0)
    kbb = float(r.get(f"{pitching_side}_bullpen_kbb_last7", 2.2) or 2.2)
    idx = 1.0 + 0.050 * fatigue + 0.012 * (ip3 - 10.0) + 0.018 * (er3 - 3.0) - 0.035 * quality - 0.020 * (kbb - 2.2)
    return float(np.clip(idx, 0.82, 1.30))


def summarize_simulations(sims: pd.DataFrame) -> pd.DataFrame:
    home_wins = sims["home_runs"] > sims["away_runs"]
    total = sims["total_runs"]
    row = {
        "game_pk": sims["game_pk"].iloc[0],
        "home_team": sims["home_team"].iloc[0],
        "away_team": sims["away_team"].iloc[0],
        "sim_home_win_prob": float(home_wins.mean()),
        "sim_away_win_prob": float(1.0 - home_wins.mean()),
        "mean_total_runs": float(total.mean()),
        "total_p10": float(total.quantile(0.10)),
        "total_p50": float(total.quantile(0.50)),
        "total_p90": float(total.quantile(0.90)),
        "home_team_total_mean": float(sims["home_runs"].mean()),
        "away_team_total_mean": float(sims["away_runs"].mean()),
        "home_first5_mean": float(sims["home_first5"].mean()),
        "away_first5_mean": float(sims["away_first5"].mean()),
        "first5_total_mean": float((sims["home_first5"] + sims["away_first5"]).mean()),
        "home_run_p10": float(sims["home_runs"].quantile(0.10)),
        "home_run_p50": float(sims["home_runs"].quantile(0.50)),
        "home_run_p90": float(sims["home_runs"].quantile(0.90)),
        "away_run_p10": float(sims["away_runs"].quantile(0.10)),
        "away_run_p50": float(sims["away_runs"].quantile(0.50)),
        "away_run_p90": float(sims["away_runs"].quantile(0.90)),
    }
    return pd.DataFrame([row])
