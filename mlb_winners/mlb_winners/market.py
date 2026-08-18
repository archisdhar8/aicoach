from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .odds import devig_two_way


@dataclass(frozen=True)
class BetFilterConfig:
    base_edge_threshold: float = 0.035
    favorite_edge_threshold: float = 0.06
    underdog_edge_threshold: float = 0.055
    high_uncertainty_threshold: float = 0.45
    max_adverse_move: float = -0.025
    max_official_underdog_moneyline: int = 250
    max_official_favorite_moneyline: int = 300
    max_reasonable_edge: float = 0.18
    max_reasonable_ev: float = 0.35
    market_adjustment_alpha: float = 0.30


@dataclass(frozen=True)
class MoneylineFilterResult:
    confidence: str
    skip_reason: str
    risk_flags: tuple[str, ...]
    stake_units: float
    tier: str


def latest_odds_by_game(odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return odds
    df = odds.copy()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    return (
        df.sort_values(["home_team", "away_team", "bookmaker", "fetched_at"])
        .drop_duplicates(["home_team", "away_team", "bookmaker"], keep="last")
        .sort_values(["home_team", "away_team", "bookmaker"])
        .reset_index(drop=True)
    )


def market_report(odds: pd.DataFrame, report_date: date) -> pd.DataFrame:
    if odds.empty:
        return pd.DataFrame()
    df = odds[pd.to_datetime(odds["snapshot_date"]).dt.date == report_date].copy()
    if df.empty:
        return pd.DataFrame()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    rows = []
    for (home, away, book), group in df.sort_values("fetched_at").groupby(["home_team", "away_team", "bookmaker"]):
        first = group.iloc[0]
        last = group.iloc[-1]
        has_moneyline = pd.notna(first.home_moneyline) and pd.notna(first.away_moneyline) and pd.notna(last.home_moneyline) and pd.notna(last.away_moneyline)
        home_open, away_open = devig_two_way(first.home_moneyline, first.away_moneyline) if has_moneyline else (np.nan, np.nan)
        home_latest, away_latest = devig_two_way(last.home_moneyline, last.away_moneyline) if has_moneyline else (np.nan, np.nan)
        hours = max((last.fetched_at - first.fetched_at).total_seconds() / 3600.0, 1e-6)
        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "bookmaker": book,
                "snapshots": len(group),
                "first_fetched_at": first.fetched_at,
                "latest_fetched_at": last.fetched_at,
                "home_open_moneyline": first.home_moneyline,
                "away_open_moneyline": first.away_moneyline,
                "home_latest_moneyline": last.home_moneyline,
                "away_latest_moneyline": last.away_moneyline,
                "home_line_move_prob": home_latest - home_open if has_moneyline else np.nan,
                "away_line_move_prob": away_latest - away_open if has_moneyline else np.nan,
                "home_latest_fair_prob": home_latest,
                "line_velocity_per_hour": abs(home_latest - home_open) / hours if has_moneyline else np.nan,
                "total_points": last.get("total_points"),
                "home_spread": last.get("home_spread"),
            }
        )
    report = pd.DataFrame(rows)
    disagreement = (
        report.groupby(["home_team", "away_team"])
        .agg(
            book_count=("bookmaker", "nunique"),
            home_prob_std=("home_latest_fair_prob", "std"),
            line_velocity_max=("line_velocity_per_hour", "max"),
        )
        .reset_index()
    )
    report = report.merge(disagreement, on=["home_team", "away_team"], how="left")
    report["home_prob_std"] = report["home_prob_std"].fillna(0.0)
    report["line_velocity_max"] = report["line_velocity_max"].fillna(0.0)
    report["market_flag"] = np.where(
        report["line_velocity_max"].fillna(0.0).gt(0.03) | report["home_prob_std"].gt(0.02),
        "watch",
        "normal",
    )
    return report


def line_move_for_side(market_rows: pd.DataFrame, home_team: str, away_team: str, side: str) -> float:
    if market_rows.empty or side in {"no bet", "no odds", None}:
        return 0.0
    group = market_rows[(market_rows["home_team"] == home_team) & (market_rows["away_team"] == away_team)].copy()
    if group.empty:
        return 0.0
    group["fetched_at"] = pd.to_datetime(group["fetched_at"])
    group = group.sort_values("fetched_at")
    first = group.iloc[0]
    last = group.iloc[-1]
    home_open, away_open = devig_two_way(first.home_moneyline, first.away_moneyline)
    home_latest, away_latest = devig_two_way(last.home_moneyline, last.away_moneyline)
    if side == home_team:
        return float(home_latest - home_open)
    if side == away_team:
        return float(away_latest - away_open)
    return 0.0


def dynamic_edge_threshold(moneyline: int | float, uncertainty_score: float, config: BetFilterConfig = BetFilterConfig()) -> float:
    line = float(moneyline)
    threshold = config.favorite_edge_threshold if line < 0 else config.underdog_edge_threshold
    threshold = max(threshold, config.base_edge_threshold)
    if uncertainty_score >= config.high_uncertainty_threshold:
        threshold += 0.025
    return float(threshold)


def market_adjusted_probability(model_prob: float, market_prob: float, config: BetFilterConfig = BetFilterConfig()) -> float:
    adjusted = float(market_prob) + config.market_adjustment_alpha * (float(model_prob) - float(market_prob))
    return float(np.clip(adjusted, 0.01, 0.99))


def apply_bet_filter(
    *,
    edge: float,
    ev: float,
    moneyline: int | float,
    uncertainty_score: float,
    missing_starter: bool,
    missing_lineup: bool,
    weather_uncertain: bool,
    adverse_line_move: float,
    bullpen_uncertainty: bool,
    thin_starter_history: bool = False,
    config: BetFilterConfig = BetFilterConfig(),
) -> tuple[str, str]:
    line = float(moneyline)
    if missing_starter:
        return "no bet", "missing_starter"
    if line > config.max_official_underdog_moneyline:
        return "no bet", "longshot_market_outlier"
    if line < -config.max_official_favorite_moneyline:
        return "no bet", "heavy_favorite_price"
    if edge >= config.max_reasonable_edge or ev >= config.max_reasonable_ev:
        return "no bet", "market_model_dislocation"
    if missing_lineup and edge < 0.07:
        return "no bet", "missing_lineup"
    if weather_uncertain:
        return "no bet", "weather_uncertainty"
    if bullpen_uncertainty and edge < 0.06:
        return "no bet", "bullpen_uncertainty"
    if thin_starter_history and edge < 0.12:
        return "no bet", "thin_starter_history"
    if uncertainty_score >= 0.70:
        return "no bet", "high_uncertainty"
    if adverse_line_move <= config.max_adverse_move:
        return "no bet", "market_moving_against_pick"
    threshold = dynamic_edge_threshold(moneyline, uncertainty_score, config)
    if ev <= 0 or edge < threshold:
        return "no bet", f"edge_below_dynamic_threshold_{threshold:.3f}"
    if edge >= 0.10 and ev >= 0.10 and uncertainty_score < 0.45:
        return "strong", ""
    if edge >= 0.06 and ev >= 0.05:
        return "medium", ""
    return "thin", ""


def apply_moneyline_tier_filter(
    *,
    edge: float,
    ev: float,
    moneyline: int | float,
    uncertainty_score: float,
    missing_starter: bool,
    missing_lineup: bool,
    weather_uncertain: bool,
    adverse_line_move: float,
    bullpen_uncertainty: bool,
    thin_starter_history: bool = False,
    config: BetFilterConfig = BetFilterConfig(),
) -> MoneylineFilterResult:
    line = float(moneyline)
    risk_flags: list[str] = []
    if missing_lineup:
        risk_flags.append("missing_lineup")
    if bullpen_uncertainty:
        risk_flags.append("bullpen_uncertainty")
    if thin_starter_history:
        risk_flags.append("thin_starter_history")
    if weather_uncertain:
        risk_flags.append("weather_uncertainty")
    if uncertainty_score >= config.high_uncertainty_threshold:
        risk_flags.append("high_uncertainty")

    if missing_starter:
        return MoneylineFilterResult("no bet", "missing_starter", tuple(risk_flags), 0.0, "no bet")
    if line > config.max_official_underdog_moneyline:
        return MoneylineFilterResult("no bet", "longshot_market_outlier", tuple(risk_flags), 0.0, "no bet")
    if line < -config.max_official_favorite_moneyline:
        return MoneylineFilterResult("no bet", "heavy_favorite_price", tuple(risk_flags), 0.0, "no bet")
    if edge >= config.max_reasonable_edge or ev >= config.max_reasonable_ev:
        return MoneylineFilterResult("no bet", "market_model_dislocation", tuple(risk_flags), 0.0, "no bet")
    if adverse_line_move <= config.max_adverse_move:
        return MoneylineFilterResult("no bet", "market_moving_against_pick", tuple(risk_flags), 0.0, "no bet")
    if ev <= 0:
        return MoneylineFilterResult("no bet", "negative_ev", tuple(risk_flags), 0.0, "no bet")

    threshold = dynamic_edge_threshold(moneyline, uncertainty_score, config)
    soft_risk_count = len(risk_flags)
    severe_weather = weather_uncertain and (uncertainty_score >= 0.70)
    if severe_weather:
        return MoneylineFilterResult("no bet", "weather_uncertainty", tuple(risk_flags), 0.0, "no bet")

    if edge >= 0.10 and ev >= 0.10 and uncertainty_score < 0.45 and soft_risk_count == 0:
        return MoneylineFilterResult("strong", "", tuple(risk_flags), 1.0, "official")
    if edge >= 0.055 and ev >= 0.04 and soft_risk_count == 0:
        return MoneylineFilterResult("medium", "", tuple(risk_flags), 1.0, "official")
    if edge >= 0.04 and ev >= 0.02 and soft_risk_count <= 1:
        reason = "soft_filter_watchlist" if soft_risk_count else f"edge_below_official_threshold_{threshold:.3f}"
        return MoneylineFilterResult("watchlist", reason, tuple(risk_flags), 0.25, "watchlist")
    if edge < min(threshold, 0.04):
        return MoneylineFilterResult("no bet", f"edge_below_dynamic_threshold_{threshold:.3f}", tuple(risk_flags), 0.0, "no bet")
    if soft_risk_count > 1:
        return MoneylineFilterResult("no bet", "multiple_soft_risks", tuple(risk_flags), 0.0, "no bet")
    return MoneylineFilterResult("thin", "edge_below_official_threshold", tuple(risk_flags), 0.0, "thin")
