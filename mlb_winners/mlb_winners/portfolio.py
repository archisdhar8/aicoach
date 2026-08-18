from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .odds import american_profit_per_dollar


@dataclass(frozen=True)
class PortfolioConfig:
    bankroll_units: float = 100.0
    max_daily_exposure: float = 6.0
    max_market_exposure: float = 3.0
    max_game_exposure: float = 2.0
    kelly_fraction: float = 0.25
    max_bet_units: float = 1.0


def kelly_stake_units(probability: float, odds: int, config: PortfolioConfig) -> float:
    b = american_profit_per_dollar(odds)
    q = 1.0 - probability
    fraction = (b * probability - q) / b
    if fraction <= 0:
        return 0.0
    units = config.bankroll_units * fraction * config.kelly_fraction
    return float(np.clip(units, 0.0, config.max_bet_units))


def optimize_portfolio(candidates: pd.DataFrame, staking: str = "flat", config: PortfolioConfig = PortfolioConfig()) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    df = candidates.copy()
    if "probability" not in df.columns:
        df["probability"] = df.get("model_probability", np.nan)
    df = df.sort_values("ev_per_unit" if "ev_per_unit" in df.columns else "ev_per_dollar", ascending=False)
    accepted = []
    daily_exposure = 0.0
    market_exposure: dict[str, float] = {}
    game_exposure: dict[int, float] = {}
    for row in df.to_dict("records"):
        market = row.get("market", "moneyline")
        game_pk = row.get("game_pk")
        odds = int(row.get("odds") or row.get("bet_moneyline"))
        probability = float(row.get("probability") or row.get("model_probability") or row.get("over_probability") or 0.0)
        stake = 1.0 if staking == "flat" else kelly_stake_units(probability, odds, config)
        stake = min(stake, config.max_bet_units)
        skip_reason = None
        if stake <= 0:
            skip_reason = "non_positive_kelly"
        elif daily_exposure + stake > config.max_daily_exposure:
            skip_reason = "daily_exposure_cap"
        elif market_exposure.get(market, 0.0) + stake > config.max_market_exposure:
            skip_reason = "market_exposure_cap"
        elif game_pk is not None and game_exposure.get(game_pk, 0.0) + stake > config.max_game_exposure:
            skip_reason = "same_game_correlation_cap"
        record = row.copy()
        record["stake_units"] = 0.0 if skip_reason else stake
        record["portfolio_status"] = "skip" if skip_reason else "accepted"
        record["skip_reason"] = skip_reason
        accepted.append(record)
        if not skip_reason:
            daily_exposure += stake
            market_exposure[market] = market_exposure.get(market, 0.0) + stake
            if game_pk is not None:
                game_exposure[game_pk] = game_exposure.get(game_pk, 0.0) + stake
    return pd.DataFrame(accepted)


def max_drawdown(units: pd.Series) -> float:
    if units.empty:
        return 0.0
    curve = units.cumsum()
    peak = curve.cummax()
    return float((curve - peak).min())
