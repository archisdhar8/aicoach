from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, lgamma, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from .features import build_prediction_frame, build_training_frame
from .market import BetFilterConfig, apply_bet_filter, latest_odds_by_game
from .odds import devig_two_way, expected_value_per_dollar


PROP_MARKETS = {"strikeouts", "hits_allowed", "total_bases", "hr", "hits", "rbi", "runs", "earned_runs", "outs_recorded"}

ODDS_MARKET_BY_PROP = {
    "strikeouts": "pitcher_strikeouts",
    "hits_allowed": "pitcher_hits_allowed",
    "earned_runs": "pitcher_earned_runs",
    "outs_recorded": "pitcher_outs",
    "total_bases": "batter_total_bases",
    "hits": "batter_hits",
    "hrr": "batter_hits_runs_rbis",
    "hr": "batter_home_runs",
    "rbi": "batter_rbis",
    "runs": "batter_runs_scored",
}

PROP_BET_FILTER = BetFilterConfig(
    base_edge_threshold=0.03,
    favorite_edge_threshold=0.05,
    underdog_edge_threshold=0.025,
    max_official_underdog_moneyline=10000,
    max_official_favorite_moneyline=10000,
    max_reasonable_edge=1.0,
    max_reasonable_ev=10.0,
)

DEFAULT_PROP_LINES = {
    "strikeouts": 4.5,
    "hits_allowed": 5.5,
    "earned_runs": 2.5,
    "outs_recorded": 16.5,
    "total_bases": 1.5,
    "hits": 0.5,
    "hr": 0.5,
    "rbi": 0.5,
    "runs": 0.5,
}

PROP_PROBABILITY_FLOOR = 0.12
PROP_PROBABILITY_CEILING = 0.88

PITCHER_K_FEATURES = [
    "expected_batters_faced",
    "projected_k_rate",
    "projected_starter_ip",
    "pitcher_kbb",
    "pitcher_recent_k_rate",
    "opponent_k_rate",
    "lineup_k_rate",
    "starter_velocity",
    "starter_spin_rate",
    "starter_fip_proxy",
    "starter_last5_fip",
    "starter_season_era",
    "starter_season_whip",
    "starter_season_fip_proxy",
    "starter_season_starts",
    "starter_rest_days",
    "starter_workload_trend",
    "opponent_xwoba",
    "park_run_factor",
    "temperature_f",
    "wind_speed_mph",
]


@dataclass(frozen=True)
class PitcherKModelBundle:
    model: object
    feature_columns: list[str]
    train_rows: int
    target_min: float
    target_max: float
    target_mean: float
    target_std: float
    baseline_mae: float
    baseline_rmse: float
    mae: float
    rmse: float
    prediction_std: float
    enabled: bool
    reason: str


@dataclass(frozen=True)
class PitcherKBacktestResult:
    training_frame: pd.DataFrame
    metrics: dict[str, float | str | int]
    sample_predictions: pd.DataFrame
    feature_missing_rates: pd.DataFrame


def poisson_over_probability(mean: float, line: float) -> float:
    mean = max(float(mean), 0.001)
    threshold = int(np.floor(line))
    cdf = sum(exp(-mean + k * np.log(mean) - lgamma(k + 1)) for k in range(threshold + 1))
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def normal_over_probability(mean: float, line: float, variance: float | None = None) -> float:
    mean = max(float(mean), 0.001)
    variance = float(variance if variance is not None else max(mean, 1.0))
    std = max(sqrt(variance), 0.25)
    z = (float(line) + 0.5 - mean) / std
    return float(np.clip(0.5 * (1.0 - erf(z / sqrt(2.0))), 0.0, 1.0))


def conservative_prop_probability(probability: float) -> float:
    return float(np.clip(probability, PROP_PROBABILITY_FLOOR, PROP_PROBABILITY_CEILING))


def predict_props(
    games: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    statcast_pitchers: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    lineups: pd.DataFrame,
    player_stats: pd.DataFrame,
    market: str,
    prop_lines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if market not in PROP_MARKETS:
        raise ValueError(f"market must be one of {sorted(PROP_MARKETS)}")
    dated_team_stats = add_game_dates_to_team_stats(team_stats, pd.concat([history_games, games], ignore_index=True))
    frame = build_prediction_frame(games, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    if market in {"strikeouts", "hits_allowed", "earned_runs", "outs_recorded"}:
        props = pitcher_props(frame, market, player_stats, dated_team_stats)
        if market == "strikeouts":
            training_frame = build_training_frame(history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
            props = apply_pitcher_k_regression(props, training_frame, dated_team_stats, player_stats)
    else:
        props = batter_props(frame, lineups, player_stats, market)
    return attach_prop_lines(props, prop_lines, market) if prop_lines is not None else mark_props_without_odds(props)


def add_game_dates_to_team_stats(team_stats: pd.DataFrame | None, games: pd.DataFrame | None) -> pd.DataFrame:
    if team_stats is None or team_stats.empty:
        return pd.DataFrame()
    out = team_stats.copy()
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"])
        return out
    if games is None or games.empty or "game_pk" not in games.columns or "game_date" not in games.columns:
        return out
    game_dates = games[["game_pk", "game_date"]].drop_duplicates("game_pk").copy()
    game_dates["game_date"] = pd.to_datetime(game_dates["game_date"])
    return out.merge(game_dates, on="game_pk", how="left")


def pitcher_props(
    frame: pd.DataFrame,
    market: str,
    player_stats: pd.DataFrame | None = None,
    team_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        for side, opp in [("home", "away"), ("away", "home")]:
            pitcher_id = getattr(row, f"{side}_probable_pitcher_id", None)
            pitcher_name = getattr(row, f"{side}_probable_pitcher", None)
            feature_values = pitcher_k_feature_values(row, side, opp, player_stats, team_stats)
            ip = feature_values["projected_starter_ip"]
            batters = feature_values["expected_batters_faced"]
            k_prob = feature_values["projected_k_rate"]
            contact_allowed = float(getattr(row, f"{side}_starter_xwoba_allowed", 0.320) or 0.320)
            projection_by_market = {
                "strikeouts": max(1.0, batters * k_prob),
                "hits_allowed": max(2.0, batters * (0.23 + 0.48 * contact_allowed)),
                "earned_runs": max(0.5, ip * max(float(getattr(row, f"{side}_starter_era_prior", 4.5) or 4.5), 2.0) / 9.0),
                "outs_recorded": max(9.0, ip * 3.0),
            }
            projection = projection_by_market[market]
            line = DEFAULT_PROP_LINES[market]
            over_probability = conservative_prop_probability(poisson_over_probability(projection, line))
            rows.append(
                {
                    "game_pk": row.game_pk,
                    "game_date": row.game_date,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "team": getattr(row, f"{side}_team"),
                    "opponent": getattr(row, f"{opp}_team"),
                    "market": market,
                    "odds_market": ODDS_MARKET_BY_PROP[market],
                    "player_id": pitcher_id,
                    "player_name": pitcher_name or f"{getattr(row, f'{side}_team')} starter",
                    "selection": f"{pitcher_name or getattr(row, f'{side}_team') + ' starter'} over {line}",
                    "projection": projection,
                    "baseline_projection": projection,
                    "xgb_projection": np.nan,
                    **feature_values,
                    "line": line,
                    "over_probability": over_probability,
                    "under_probability": 1.0 - over_probability,
                    "lineup_required": False,
                    "lineup_confirmed": 1.0,
                    "missing_starter": bool(pd.isna(pitcher_id)),
                    "thin_history": bool(float(getattr(row, f"{side}_starter_games_prior", 0) or 0) < 2),
                    "weather_uncertain": bool((getattr(row, "precipitation_in", 0.0) or 0.0) > 0.12 or (getattr(row, "wind_speed_mph", 0.0) or 0.0) > 22),
                    "bullpen_uncertain": False,
                    "uncertainty_score": pitcher_prop_uncertainty(row, side),
                    "short_workload_role": bool(ip < 4.0 or batters < 17.0),
                    "insufficient_data": bool(feature_values.get("feature_missing_rate", 0.0) > 0.25),
                }
            )
    return pd.DataFrame(rows)


def batter_props(frame: pd.DataFrame, lineups: pd.DataFrame, player_stats: pd.DataFrame, market: str) -> pd.DataFrame:
    if lineups is None or lineups.empty:
        return pd.DataFrame()
    stats = summarize_player_stats(player_stats)
    rows = []
    for row in frame.itertuples(index=False):
        game_lineup = lineups[lineups["game_pk"] == row.game_pk].sort_values("batting_order")
        for player in game_lineup.itertuples(index=False):
            player_summary = stats.get(int(player.player_id), {})
            pa = projected_pa(player.batting_order)
            park = float(getattr(row, "park_run_factor", 1.0) or 1.0)
            tb_mean = max(0.15, pa * player_summary.get("tb_per_pa", 0.34) * park)
            hr_mean = max(0.005, pa * player_summary.get("hr_per_pa", 0.03) * park)
            hits_mean = max(0.20, pa * player_summary.get("hit_per_pa", 0.23) * park)
            rbi_mean = max(0.05, pa * player_summary.get("rbi_per_pa", 0.10) * park)
            runs_mean = max(0.05, pa * player_summary.get("run_per_pa", 0.12) * park)
            projection_by_market = {
                "total_bases": tb_mean,
                "hr": hr_mean,
                "hits": hits_mean,
                "rbi": rbi_mean,
                "runs": runs_mean,
            }
            projection = projection_by_market[market]
            line = DEFAULT_PROP_LINES[market]
            confirmed = bool(getattr(player, "confirmed", False))
            over_probability = conservative_prop_probability(poisson_over_probability(projection, line))
            rows.append(
                {
                    "game_pk": row.game_pk,
                    "game_date": row.game_date,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "team": player.team_name,
                    "opponent": row.away_team if player.team_name == row.home_team else row.home_team,
                    "market": market,
                    "odds_market": ODDS_MARKET_BY_PROP[market],
                    "player_id": player.player_id,
                    "player_name": player.player_name,
                    "selection": f"{player.player_name} over {line}",
                    "projection": projection,
                    "line": line,
                    "over_probability": over_probability,
                    "under_probability": 1.0 - over_probability,
                    "lineup_required": True,
                    "lineup_confirmed": float(confirmed),
                    "missing_starter": bool(pd.isna(getattr(row, "home_probable_pitcher_id", None)) or pd.isna(getattr(row, "away_probable_pitcher_id", None))),
                    "thin_history": bool(player_summary.get("pa", 0) < 25),
                    "weather_uncertain": bool((getattr(row, "precipitation_in", 0.0) or 0.0) > 0.12 or (getattr(row, "wind_speed_mph", 0.0) or 0.0) > 22),
                    "bullpen_uncertain": bool(max(getattr(row, "home_bullpen_fatigue_score", 0.0), getattr(row, "away_bullpen_fatigue_score", 0.0)) > 0.85),
                    "uncertainty_score": batter_prop_uncertainty(player_summary, confirmed),
                }
            )
    return pd.DataFrame(rows)


def attach_prop_lines(props: pd.DataFrame, prop_lines: pd.DataFrame | None, market: str) -> pd.DataFrame:
    if props.empty:
        return props
    if prop_lines is None or prop_lines.empty:
        return mark_props_without_odds(props)
    odds_market = ODDS_MARKET_BY_PROP[market]
    lines = prop_lines[prop_lines["market"] == odds_market].copy()
    if lines.empty:
        return mark_props_without_odds(props)
    lines["fetched_at"] = pd.to_datetime(lines["fetched_at"])
    lines["player_key"] = lines["player_name"].map(player_key)
    line_moves = prop_line_moves(lines)
    lines = (
        lines.sort_values(["home_team", "away_team", "player_key", "bookmaker", "fetched_at"])
        .drop_duplicates(["home_team", "away_team", "player_key", "bookmaker"], keep="last")
        .sort_values(["home_team", "away_team", "player_key", "bookmaker"])
        .drop_duplicates(["home_team", "away_team", "player_key"], keep="first")
    )
    lookup = {(r.home_team, r.away_team, r.player_key): r for r in lines.itertuples(index=False)}
    rows = []
    for row in props.itertuples(index=False):
        result = row._asdict()
        line = lookup.get((row.home_team, row.away_team, player_key(row.player_name)))
        if not line or pd.isna(line.over_odds) or pd.isna(line.under_odds):
            result.update(no_odds_columns("missing_prop_odds"))
            rows.append(result)
            continue
        over_prob = conservative_prop_probability(poisson_over_probability(row.projection, float(line.line)))
        under_prob = 1.0 - over_prob
        market_over, market_under = devig_two_way(line.over_odds, line.under_odds)
        over_ev = expected_value_per_dollar(over_prob, line.over_odds)
        under_ev = expected_value_per_dollar(under_prob, line.under_odds)
        if over_ev >= under_ev:
            decision = "over"
            selected_prob = over_prob
            market_prob = market_over
            odds_price = line.over_odds
            ev = over_ev
        else:
            decision = "under"
            selected_prob = under_prob
            market_prob = market_under
            odds_price = line.under_odds
            ev = under_ev
        edge = selected_prob - market_prob
        adverse_move = line_moves.get((row.home_team, row.away_team, player_key(row.player_name), decision), 0.0)
        stale = prop_line_is_stale(getattr(line, "fetched_at", None), getattr(line, "snapshot_date", None))
        confidence, skip_reason = apply_bet_filter(
            edge=edge,
            ev=ev,
            moneyline=odds_price,
            uncertainty_score=row.uncertainty_score,
            missing_starter=bool(row.missing_starter),
            missing_lineup=bool(row.lineup_required and row.lineup_confirmed < 1.0),
            weather_uncertain=bool(row.weather_uncertain),
            adverse_line_move=adverse_move,
            bullpen_uncertainty=bool(row.bullpen_uncertain),
            thin_starter_history=bool(row.thin_history),
            config=PROP_BET_FILTER,
        )
        if stale:
            confidence = "no bet"
            skip_reason = "stale_prop_odds"
        if bool(getattr(row, "insufficient_data", False)):
            confidence = "no bet"
            skip_reason = "insufficient_pitcher_k_data"
        if bool(getattr(row, "short_workload_role", False)):
            confidence = "no bet"
            skip_reason = "short_workload_role"
        if confidence == "no bet":
            decision = "no bet"
        result.update(
            {
                "bookmaker": line.bookmaker,
                "line": float(line.line),
                "over_odds": int(line.over_odds),
                "under_odds": int(line.under_odds),
                "over_probability": over_prob,
                "under_probability": under_prob,
                "market_over_probability": market_over,
                "market_under_probability": market_under,
                "bet_probability": selected_prob if decision != "no bet" else np.nan,
                "bet_odds": int(odds_price) if decision != "no bet" else np.nan,
                "edge": edge,
                "ev_per_dollar": ev,
                "decision": decision,
                "confidence": confidence,
                "skip_reason": skip_reason,
                "line_move_for_side": adverse_move,
            }
        )
        rows.append(result)
    return pd.DataFrame(rows)


def mark_props_without_odds(props: pd.DataFrame) -> pd.DataFrame:
    if props.empty:
        return props
    rows = []
    for row in props.to_dict("records"):
        row.update(no_odds_columns("missing_prop_odds"))
        rows.append(row)
    return pd.DataFrame(rows)


def no_odds_columns(reason: str) -> dict[str, object]:
    return {
        "bookmaker": None,
        "over_odds": np.nan,
        "under_odds": np.nan,
        "market_over_probability": np.nan,
        "market_under_probability": np.nan,
        "bet_probability": np.nan,
        "bet_odds": np.nan,
        "edge": np.nan,
        "ev_per_dollar": np.nan,
        "decision": "no odds",
        "confidence": "no odds",
        "skip_reason": reason,
    }


def expected_batters_faced(projected_ip: float, workload_trend: float = 0.0) -> float:
    workload_adjustment = np.clip(float(workload_trend or 0.0) / 120.0, -0.35, 0.25)
    return float(np.clip((projected_ip + workload_adjustment) * 4.25, 6.0, 32.0))


def projected_starter_ip_from_context(fip: float, workload_trend: float, season_starts: float, season_era: float, opponent_xwoba: float) -> float:
    starts = float(season_starts or 0.0)
    workload = float(workload_trend or 0.0)
    if starts < 2 and workload < -8:
        role_baseline = 2.2
    elif starts < 4 and workload < -4:
        role_baseline = 3.3
    else:
        role_baseline = 5.45
    role_bonus = np.clip((starts - 4.0) * 0.035, -0.25, 0.35)
    workload_bonus = np.clip(workload / 180.0, -0.35, 0.25)
    quality_bonus = np.clip((4.2 - float(fip or 4.2)) * 0.14 + (4.2 - float(season_era or 4.2)) * 0.08, -0.45, 0.50)
    opponent_penalty = np.clip((float(opponent_xwoba or 0.320) - 0.320) * 2.2, -0.25, 0.35)
    return float(np.clip(role_baseline + role_bonus + workload_bonus + quality_bonus - opponent_penalty, 1.4, 7.1))


def pitcher_strikeout_probability(pitcher_kbb: float, opponent_k_rate: float, starter_velocity: float = 92.5, starter_spin: float = 2250.0) -> float:
    pitcher_component = np.tanh((float(pitcher_kbb or 2.2) - 2.2) / 2.5)
    velocity_boost = np.clip((float(starter_velocity or 92.5) - 92.5) * 0.006, -0.025, 0.035)
    spin_boost = np.clip((float(starter_spin or 2250.0) - 2250.0) / 10000.0, -0.025, 0.025)
    prob = 0.215 + 0.55 * (float(opponent_k_rate or 0.22) - 0.22) + 0.035 * pitcher_component + velocity_boost + spin_boost
    return float(np.clip(prob, 0.14, 0.34))


def pitcher_recent_k_rate_from_team_stats(team_stats: pd.DataFrame | None, pitcher_id: int | float | None, before_game_date=None, starts: int = 5) -> float | None:
    if team_stats is None or team_stats.empty or pitcher_id is None or pd.isna(pitcher_id):
        return None
    df = team_stats[team_stats["starter_id"].eq(int(pitcher_id))].copy()
    if before_game_date is not None and "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df[df["game_date"] < pd.Timestamp(before_game_date)]
    df = df[pd.to_numeric(df.get("starter_ip"), errors="coerce").fillna(0).gt(0)]
    if df.empty:
        return None
    df = df.sort_values("game_date").tail(starts)
    strikeouts = float(pd.to_numeric(df.get("starter_so"), errors="coerce").fillna(0).sum())
    innings = float(pd.to_numeric(df.get("starter_ip"), errors="coerce").fillna(0).sum())
    batters = innings * 4.25
    if batters <= 0:
        return None
    return float(np.clip(strikeouts / batters, 0.08, 0.45))


def _raw_missing(row, attr: str) -> bool:
    return not hasattr(row, attr) or pd.isna(getattr(row, attr))


def pitcher_k_feature_values(
    row,
    side: str,
    opp: str,
    player_stats: pd.DataFrame | None = None,
    team_stats: pd.DataFrame | None = None,
) -> dict[str, float]:
    required_raw_attrs = [
        f"{side}_probable_pitcher_id",
        f"{side}_starter_kbb_prior",
        f"{opp}_team_k_rate_last14",
        f"{side}_starter_fip_proxy",
        f"{side}_starter_season_era",
        f"{side}_starter_season_starts",
        f"{side}_starter_rest_days",
        f"{side}_starter_workload_trend",
        f"{opp}_team_xwoba_last14",
    ]
    missing = sum(1 for attr in required_raw_attrs if _raw_missing(row, attr))
    feature_missing_rate = missing / len(required_raw_attrs)
    pitcher_kbb = float(getattr(row, f"{side}_starter_kbb_prior", 2.2) or 2.2)
    opponent_k_rate = float(getattr(row, f"{opp}_team_k_rate_last14", 0.22) or 0.22)
    lineup_k_rate = opponent_k_rate
    starter_velocity = float(getattr(row, f"{side}_starter_velocity", 92.5) or 92.5)
    starter_spin = float(getattr(row, f"{side}_starter_spin_rate", 2250.0) or 2250.0)
    starter_fip = float(getattr(row, f"{side}_starter_fip_proxy", 4.2) or 4.2)
    starter_last5_fip = float(getattr(row, f"{side}_starter_last5_fip", starter_fip) or starter_fip)
    season_era = float(getattr(row, f"{side}_starter_season_era", getattr(row, f"{side}_starter_era_prior", 4.5)) or 4.5)
    season_whip = float(getattr(row, f"{side}_starter_season_whip", getattr(row, f"{side}_starter_whip_prior", 1.3)) or 1.3)
    season_fip = float(getattr(row, f"{side}_starter_season_fip_proxy", starter_fip) or starter_fip)
    season_starts = float(getattr(row, f"{side}_starter_season_starts", getattr(row, f"{side}_starter_games_prior", 0.0)) or 0.0)
    rest_days = float(getattr(row, f"{side}_starter_rest_days", 5.0) or 5.0)
    workload_trend = float(getattr(row, f"{side}_starter_workload_trend", 0.0) or 0.0)
    opponent_xwoba = float(getattr(row, f"{opp}_team_xwoba_last14", 0.320) or 0.320)
    recent = pitcher_recent_k_rate_from_team_stats(team_stats, getattr(row, f"{side}_probable_pitcher_id", None), getattr(row, "game_date", None))
    base_k = pitcher_strikeout_probability(pitcher_kbb, opponent_k_rate, starter_velocity, starter_spin)
    recent_k = float(recent if recent is not None else base_k)
    projected_k_rate = float(np.clip(0.58 * base_k + 0.27 * recent_k + 0.15 * lineup_k_rate, 0.12, 0.38))
    ip = projected_starter_ip_from_context(starter_fip, workload_trend, season_starts, season_era, opponent_xwoba)
    batters = expected_batters_faced(ip, workload_trend)
    return {
        "expected_batters_faced": batters,
        "projected_k_rate": projected_k_rate,
        "projected_starter_ip": ip,
        "pitcher_kbb": pitcher_kbb,
        "pitcher_recent_k_rate": recent_k,
        "opponent_k_rate": opponent_k_rate,
        "lineup_k_rate": lineup_k_rate,
        "starter_velocity": starter_velocity,
        "starter_spin_rate": starter_spin,
        "starter_fip_proxy": starter_fip,
        "starter_last5_fip": starter_last5_fip,
        "starter_season_era": season_era,
        "starter_season_whip": season_whip,
        "starter_season_fip_proxy": season_fip,
        "starter_season_starts": season_starts,
        "starter_rest_days": rest_days,
        "starter_workload_trend": workload_trend,
        "opponent_xwoba": opponent_xwoba,
        "park_run_factor": float(getattr(row, "park_run_factor", 1.0) or 1.0),
        "temperature_f": float(getattr(row, "temperature_f", 72.0) or 72.0),
        "wind_speed_mph": float(getattr(row, "wind_speed_mph", 0.0) or 0.0),
        "feature_missing_rate": float(feature_missing_rate),
    }


def make_pitcher_k_regressor() -> object:
    try:
        from xgboost import XGBRegressor

        regressor = XGBRegressor(
            n_estimators=260,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    except Exception:
        regressor = HistGradientBoostingRegressor(max_iter=220, learning_rate=0.045, random_state=42)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", regressor)])


def build_pitcher_k_training_frame(
    feature_frame: pd.DataFrame,
    team_stats: pd.DataFrame | None,
    player_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if feature_frame.empty or team_stats is None or team_stats.empty:
        return pd.DataFrame()
    stats = team_stats.copy()
    if "game_date" in stats.columns:
        stats["game_date"] = pd.to_datetime(stats["game_date"])
    stats_lookup = {
        (int(row.game_pk), str(row.side)): row
        for row in stats.itertuples(index=False)
        if pd.notna(getattr(row, "starter_so", np.nan)) and float(getattr(row, "starter_ip", 0.0) or 0.0) > 0
    }
    rows = []
    for row in feature_frame.itertuples(index=False):
        if pd.isna(getattr(row, "target_home_win", np.nan)):
            continue
        for side, opp in [("home", "away"), ("away", "home")]:
            pitcher_id = getattr(row, f"{side}_probable_pitcher_id", None)
            if pitcher_id is None or pd.isna(pitcher_id):
                continue
            stat = stats_lookup.get((int(row.game_pk), side))
            if stat is None:
                continue
            starter_id = getattr(stat, "starter_id", None)
            if starter_id is not None and pd.notna(starter_id) and int(starter_id) != int(pitcher_id):
                continue
            target = safe_float(getattr(stat, "starter_so", np.nan), np.nan)
            if pd.isna(target) or target < 0 or target > 20:
                continue
            values = pitcher_k_feature_values(row, side, opp, player_stats, team_stats)
            values.update(
                {
                    "game_pk": row.game_pk,
                    "game_date": row.game_date,
                    "player_id": int(pitcher_id),
                    "player_name": getattr(row, f"{side}_probable_pitcher", None),
                    "side": side,
                    "target_strikeouts": target,
                    "baseline_projection": values["expected_batters_faced"] * values["projected_k_rate"],
                }
            )
            rows.append(values)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["target_strikeouts"] = pd.to_numeric(out["target_strikeouts"], errors="coerce")
        out = out.dropna(subset=["target_strikeouts"])
    return out


def safe_float(value, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def evaluate_pitcher_k_regressor(train: pd.DataFrame) -> tuple[PitcherKModelBundle | None, pd.DataFrame]:
    if len(train) < 80:
        return None, pd.DataFrame()
    train = train.sort_values(["game_date", "game_pk"]).copy()
    split = max(int(len(train) * 0.82), 1)
    fit = train.iloc[:split]
    valid = train.iloc[split:] if split < len(train) else train.iloc[:0]
    target = pd.to_numeric(train["target_strikeouts"], errors="coerce").dropna()
    if target.empty or target.max() < 3 or target.std() < 0.75:
        return None, pd.DataFrame()
    model = make_pitcher_k_regressor()
    model.fit(fit[PITCHER_K_FEATURES], fit["target_strikeouts"])
    if valid.empty:
        pred = model.predict(fit[PITCHER_K_FEATURES])
        y = fit["target_strikeouts"]
        baseline = fit["baseline_projection"]
        eval_frame = fit.copy()
    else:
        pred = model.predict(valid[PITCHER_K_FEATURES])
        y = valid["target_strikeouts"]
        baseline = valid["baseline_projection"]
        eval_frame = valid.copy()
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 14.0)
    mae = float(mean_absolute_error(y, pred))
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    baseline_mae = float(mean_absolute_error(y, baseline))
    baseline_rmse = float(np.sqrt(mean_squared_error(y, baseline)))
    pred_std = float(np.std(pred))
    enabled, reason = pitcher_k_ml_gate(
        xgb_mae=mae,
        baseline_mae=baseline_mae,
        prediction_std=pred_std,
        predictions=pred,
    )
    eval_frame = eval_frame.copy()
    eval_frame["xgb_projection"] = pred
    eval_frame["baseline_error"] = np.abs(eval_frame["target_strikeouts"] - eval_frame["baseline_projection"])
    eval_frame["xgb_error"] = np.abs(eval_frame["target_strikeouts"] - eval_frame["xgb_projection"])
    bundle = PitcherKModelBundle(
        model=model,
        feature_columns=PITCHER_K_FEATURES.copy(),
        train_rows=len(train),
        target_min=float(target.min()),
        target_max=float(target.max()),
        target_mean=float(target.mean()),
        target_std=float(target.std()),
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        mae=mae,
        rmse=rmse,
        prediction_std=pred_std,
        enabled=enabled,
        reason=reason,
    )
    return bundle, eval_frame


def pitcher_k_ml_gate(xgb_mae: float, baseline_mae: float, prediction_std: float, predictions) -> tuple[bool, str]:
    preds = np.asarray(predictions, dtype=float)
    if prediction_std < 0.50:
        return False, "disabled: near-zero XGB prediction variance"
    if np.nanmin(preds) < 0.0 or np.nanmax(preds) > 14.0:
        return False, "disabled: unrealistic XGB projection range"
    if not np.isfinite(xgb_mae) or not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return False, "disabled: invalid validation metrics"
    improvement = (baseline_mae - xgb_mae) / baseline_mae
    if improvement < 0.03:
        return False, f"disabled: XGB MAE improvement {improvement:.1%} below 3% threshold"
    return True, f"enabled: XGB MAE improvement {improvement:.1%}"


def train_pitcher_k_model(feature_frame: pd.DataFrame, team_stats: pd.DataFrame | None, player_stats: pd.DataFrame | None = None) -> PitcherKModelBundle | None:
    train = build_pitcher_k_training_frame(feature_frame, team_stats, player_stats)
    bundle, _ = evaluate_pitcher_k_regressor(train)
    return bundle


def apply_pitcher_k_regression(
    props: pd.DataFrame,
    feature_frame: pd.DataFrame,
    team_stats: pd.DataFrame | None,
    player_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if props.empty:
        return props
    train = build_pitcher_k_training_frame(feature_frame, team_stats, player_stats)
    bundle, _ = evaluate_pitcher_k_regressor(train)
    out = props.copy()
    out["model_version"] = "pitcher-k-baseline-v1"
    out["ml_guardrail"] = "baseline production default"
    out["projection"] = out["baseline_projection"].astype(float)
    out["over_probability"] = [conservative_prop_probability(poisson_over_probability(mean, line)) for mean, line in zip(out["projection"], out["line"])]
    out["under_probability"] = 1.0 - out["over_probability"]
    if bundle is None:
        out["ml_guardrail"] = "disabled: insufficient or invalid training data"
        return out
    mask = out["market"].eq("strikeouts")
    if not mask.any():
        return out
    missing_mask = out.loc[mask, "feature_missing_rate"].astype(float).gt(0.25)
    if missing_mask.any():
        target_index = out.loc[mask].index[missing_mask]
        out.loc[target_index, "insufficient_data"] = True
    preds = np.clip(bundle.model.predict(out.loc[mask, bundle.feature_columns]), 0.0, 14.0)
    out.loc[mask, "xgb_projection"] = preds
    if bundle.enabled:
        baseline = out.loc[mask, "baseline_projection"].astype(float).to_numpy()
        blended = 0.35 * preds + 0.65 * baseline
        out.loc[mask, "projection"] = blended
        out.loc[mask, "over_probability"] = [conservative_prop_probability(poisson_over_probability(mean, line)) for mean, line in zip(blended, out.loc[mask, "line"])]
        out.loc[mask, "under_probability"] = 1.0 - out.loc[mask, "over_probability"]
        out.loc[mask, "model_version"] = "pitcher-k-baseline-xgb-blend-v1"
    out.loc[mask, "ml_guardrail"] = bundle.reason
    out.loc[mask, "training_rows"] = bundle.train_rows
    out.loc[mask, "target_min"] = bundle.target_min
    out.loc[mask, "target_max"] = bundle.target_max
    out.loc[mask, "target_mean"] = bundle.target_mean
    out.loc[mask, "target_std"] = bundle.target_std
    out.loc[mask, "baseline_validation_mae"] = bundle.baseline_mae
    out.loc[mask, "baseline_validation_rmse"] = bundle.baseline_rmse
    out.loc[mask, "validation_mae"] = bundle.mae
    out.loc[mask, "validation_rmse"] = bundle.rmse
    out.loc[mask, "xgb_prediction_std"] = bundle.prediction_std
    return out


def run_pitcher_k_backtest(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    statcast_pitchers: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    lineups: pd.DataFrame,
    player_stats: pd.DataFrame,
    report_dir: Path,
    docs_dir: Path,
    *,
    start_year: int = 2021,
    end_year: int = 2025,
) -> PitcherKBacktestResult:
    history_games = games[
        games["season"].between(start_year, end_year)
        & games["home_score"].notna()
        & games["away_score"].notna()
    ].copy()
    dated_team_stats = add_game_dates_to_team_stats(team_stats, history_games)
    feature_frame = build_training_frame(history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    train = build_pitcher_k_training_frame(feature_frame, dated_team_stats, player_stats)
    bundle, eval_frame = evaluate_pitcher_k_regressor(train)
    if train.empty:
        metrics: dict[str, float | str | int] = {"training_rows": 0, "recommendation": "baseline only: no training rows"}
        sample = pd.DataFrame()
    elif bundle is None or eval_frame.empty:
        target = pd.to_numeric(train["target_strikeouts"], errors="coerce").dropna()
        metrics = {
            "training_rows": int(len(train)),
            "target_min": float(target.min()) if not target.empty else np.nan,
            "target_max": float(target.max()) if not target.empty else np.nan,
            "target_mean": float(target.mean()) if not target.empty else np.nan,
            "target_std": float(target.std()) if not target.empty else np.nan,
            "recommendation": "baseline only: invalid or insufficient XGB training data",
        }
        sample = train.tail(20).copy()
    else:
        y = eval_frame["target_strikeouts"].astype(float)
        baseline = eval_frame["baseline_projection"].astype(float)
        xgb = eval_frame["xgb_projection"].astype(float)
        metrics = {
            "training_rows": int(bundle.train_rows),
            "holdout_rows": int(len(eval_frame)),
            "target_min": bundle.target_min,
            "target_max": bundle.target_max,
            "target_mean": bundle.target_mean,
            "target_std": bundle.target_std,
            "baseline_mae": bundle.baseline_mae,
            "baseline_rmse": bundle.baseline_rmse,
            "xgb_mae": bundle.mae,
            "xgb_rmse": bundle.rmse,
            "xgb_prediction_std": bundle.prediction_std,
            "baseline_correlation": safe_corr(y, baseline),
            "xgb_correlation": safe_corr(y, xgb),
            "xgb_enabled": int(bundle.enabled),
            "recommendation": "blend allowed" if bundle.enabled else f"baseline only: {bundle.reason}",
        }
        sample = eval_frame[[
            "game_pk",
            "game_date",
            "player_id",
            "player_name",
            "side",
            "target_strikeouts",
            "baseline_projection",
            "xgb_projection",
            "baseline_error",
            "xgb_error",
        ]]
    report_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(report_dir / "pitcher_k_training_diagnostic.csv", index=False)
    sample.to_csv(report_dir / "pitcher_k_backtest_sample.csv", index=False)
    missing = pitcher_k_feature_missing_rates(train)
    missing.to_csv(report_dir / "pitcher_k_feature_missing_rates.csv", index=False)
    write_pitcher_k_diagnostic_report(docs_dir / "PITCHER_K_DIAGNOSTIC_REPORT.md", train, metrics, sample, missing)
    return PitcherKBacktestResult(train, metrics, sample, missing)


def safe_corr(a, b) -> float:
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    if len(a) < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float("nan")
    return float(a.corr(b))


def pitcher_k_feature_missing_rates(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in PITCHER_K_FEATURES:
        if col not in train.columns:
            rows.append({"feature": col, "missing_rate": 1.0, "default_like_rate": 1.0})
            continue
        s = pd.to_numeric(train[col], errors="coerce")
        default_like = s.isin([0.0, 1.0, 2.2, 4.2, 4.5, 5.0, 72.0, 92.5, 2250.0]).mean() if len(s) else 0.0
        rows.append({"feature": col, "missing_rate": float(s.isna().mean()), "default_like_rate": float(default_like)})
    return pd.DataFrame(rows)


def projection_bucket_report(df: pd.DataFrame, projection_col: str) -> pd.DataFrame:
    if df.empty or projection_col not in df.columns:
        return pd.DataFrame()
    bins = [0, 3, 4, 5, 6, 7, 20]
    labels = ["0-3", "3-4", "4-5", "5-6", "6-7", "7+"]
    tmp = df.copy()
    tmp["bucket"] = pd.cut(tmp[projection_col].astype(float), bins=bins, labels=labels, include_lowest=True, right=False)
    return (
        tmp.groupby("bucket", observed=False)
        .agg(rows=("target_strikeouts", "size"), projected_avg=(projection_col, "mean"), actual_avg=("target_strikeouts", "mean"))
        .reset_index()
    )


def write_pitcher_k_diagnostic_report(
    path: Path,
    train: pd.DataFrame,
    metrics: dict[str, float | str | int],
    sample: pd.DataFrame,
    missing: pd.DataFrame,
) -> None:
    target = pd.to_numeric(train.get("target_strikeouts", pd.Series(dtype=float)), errors="coerce").dropna()
    target_summary = {
        "rows": int(len(target)),
        "min": float(target.min()) if not target.empty else np.nan,
        "p25": float(target.quantile(0.25)) if not target.empty else np.nan,
        "mean": float(target.mean()) if not target.empty else np.nan,
        "p75": float(target.quantile(0.75)) if not target.empty else np.nan,
        "max": float(target.max()) if not target.empty else np.nan,
        "std": float(target.std()) if not target.empty else np.nan,
    }
    lines = [
        "# Pitcher K Diagnostic Report",
        "",
        "This report audits the pitcher strikeout prop model after the constant-0.50 XGB prediction bug.",
        "",
        "## Final Recommendation",
        "",
        f"- {metrics.get('recommendation', 'baseline only')}",
        "- Production pitcher-K predictions use the baseline projection unless the ML guardrail explicitly enables a blend.",
        "",
        "## Target Distribution",
        "",
        markdown_table(pd.DataFrame([target_summary])),
        "",
        "## Metrics",
        "",
        markdown_table(pd.DataFrame([metrics])),
        "",
        "## Feature Missing / Default-Filled Rates",
        "",
        markdown_table(missing.sort_values(["missing_rate", "default_like_rate"], ascending=False).head(25)),
        "",
        "## Calibration By Projection Bucket",
        "",
    ]
    if not sample.empty and "xgb_projection" in sample.columns:
        baseline_buckets = projection_bucket_report(sample, "baseline_projection")
        xgb_buckets = projection_bucket_report(sample, "xgb_projection")
        lines.extend(["### Baseline", "", markdown_table(baseline_buckets), "", "### XGB", "", markdown_table(xgb_buckets), ""])
    else:
        lines.extend(["No holdout sample available.", ""])
    lines.extend(["## Sample Predictions", ""])
    if sample.empty:
        lines.append("No sample predictions available.")
    else:
        lines.append(markdown_table(sample.head(20)))
    path.write_text("\n".join(lines) + "\n")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(lambda v: "" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v)))
    header = "| " + " | ".join(clean.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in clean.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def prop_line_moves(lines: pd.DataFrame) -> dict[tuple[str, str, str, str], float]:
    if lines.empty:
        return {}
    out: dict[tuple[str, str, str, str], float] = {}
    grouped = lines.sort_values("fetched_at").groupby(["home_team", "away_team", "player_key"], dropna=False)
    for (home, away, key), group in grouped:
        first = group.iloc[0]
        last = group.iloc[-1]
        if pd.notna(first.get("over_odds")) and pd.notna(first.get("under_odds")) and pd.notna(last.get("over_odds")) and pd.notna(last.get("under_odds")):
            first_over, first_under = devig_two_way(first.over_odds, first.under_odds)
            last_over, last_under = devig_two_way(last.over_odds, last.under_odds)
            out[(home, away, key, "over")] = float(last_over - first_over)
            out[(home, away, key, "under")] = float(last_under - first_under)
    return out


def prop_line_is_stale(fetched_at, snapshot_date=None, max_age_hours: float = 8.0) -> bool:
    if fetched_at is None or pd.isna(fetched_at):
        return True
    if snapshot_date is not None and not pd.isna(snapshot_date):
        if pd.Timestamp(snapshot_date).date() != pd.Timestamp.now(tz="UTC").date():
            return False
    ts = pd.Timestamp(fetched_at)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    age = pd.Timestamp.now(tz="UTC") - ts
    return age.total_seconds() > max_age_hours * 3600


def pitcher_prop_uncertainty(row, side: str) -> float:
    games_prior = float(getattr(row, f"{side}_starter_games_prior", 0) or 0)
    thin_history = 0.20 if games_prior < 2 else 0.0
    lineup_penalty = (1.0 - min(float(getattr(row, "home_lineup_confirmed", 0.0)), float(getattr(row, "away_lineup_confirmed", 0.0)))) * 0.10
    weather_penalty = 0.10 if bool((getattr(row, "precipitation_in", 0.0) or 0.0) > 0.12 or (getattr(row, "wind_speed_mph", 0.0) or 0.0) > 22) else 0.0
    return float(np.clip(0.25 + thin_history + lineup_penalty + weather_penalty, 0.0, 1.0))


def batter_prop_uncertainty(summary: dict[str, float], confirmed: bool) -> float:
    pa = float(summary.get("pa", 0.0))
    sample_penalty = 0.20 if pa < 25 else 0.08 if pa < 75 else 0.0
    lineup_penalty = 0.20 if not confirmed else 0.0
    return float(np.clip(0.25 + sample_penalty + lineup_penalty, 0.0, 1.0))


def summarize_player_stats(player_stats: pd.DataFrame) -> dict[int, dict[str, float]]:
    if player_stats is None or player_stats.empty:
        return {}
    grouped = player_stats.groupby("player_id", dropna=True).agg(
        total_bases=("total_bases", "sum"),
        home_runs=("home_runs", "sum"),
        hits=("hits", "sum"),
        rbi=("rbi", "sum"),
        runs=("runs", "sum"),
        at_bats=("at_bats", "sum"),
        walks=("walks", "sum"),
    )
    out = {}
    for player_id, row in grouped.iterrows():
        pa = max(float(row["at_bats"] or 0) + float(row["walks"] or 0), 1.0)
        out[int(player_id)] = {
            "pa": pa,
            "tb_per_pa": float(row["total_bases"] or 0) / pa,
            "hr_per_pa": float(row["home_runs"] or 0) / pa,
            "hit_per_pa": float(row["hits"] or 0) / pa,
            "rbi_per_pa": float(row["rbi"] or 0) / pa,
            "run_per_pa": float(row["runs"] or 0) / pa,
        }
    return out


def projected_starter_ip(fip: float) -> float:
    return float(np.clip(5.8 - 0.25 * (float(fip or 4.2) - 4.2), 4.2, 6.7))


def projected_pa(order: int | None) -> float:
    if not order:
        return 3.7
    return float(np.clip(4.85 - 0.13 * (order - 1), 3.5, 4.9))


def player_key(name: str | None) -> str:
    return "".join(ch.lower() for ch in str(name or "") if ch.isalnum())


def write_props(props: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    props.to_csv(path, index=False)
