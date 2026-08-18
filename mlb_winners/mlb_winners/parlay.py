from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .odds import american_profit_per_dollar


@dataclass(frozen=True)
class LottoParlay:
    legs: pd.DataFrame
    stake_units: float
    hit_probability: float
    combined_decimal_odds: float
    combined_american_odds: int
    break_even_probability: float
    ev_per_unit: float
    potential_profit_units: float
    total_payout_units: float
    parlay_id: str


def american_to_decimal(odds: int | float) -> float:
    return 1.0 + american_profit_per_dollar(odds)


def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1")
    profit = decimal_odds - 1.0
    if decimal_odds >= 2.0:
        return int(round(profit * 100))
    return int(round(-100 / profit))


def build_lotto_parlay(
    predictions: pd.DataFrame,
    min_legs: int = 4,
    max_legs: int = 8,
    stake_units: float = 0.1,
) -> LottoParlay | None:
    if predictions.empty:
        return None
    candidates = build_lotto_candidates(predictions)
    if len(candidates) < min_legs:
        return None

    candidates["decimal_odds"] = candidates["bet_moneyline"].apply(american_to_decimal)
    candidates["positive_ev"] = candidates["ev_per_dollar"].fillna(-1.0).gt(0.0)
    candidates["official_value"] = candidates["confidence"].isin(["strong", "medium"])
    candidates["lotto_score"] = (
        candidates["ev_per_dollar"].fillna(0.0)
        + candidates["edge"].fillna(0.0) * 0.35
        + candidates["bet_probability"].fillna(0.0) * 0.08
        + np.log(candidates["decimal_odds"]) * 0.12
        + candidates["positive_ev"].astype(float) * 0.15
        + candidates["official_value"].astype(float) * 0.10
    )
    legs = candidates.sort_values("lotto_score", ascending=False).head(max_legs).copy()
    if len(legs) < min_legs:
        return None

    hit_probability = float(np.prod(legs["bet_probability"].astype(float).clip(0.001, 0.999)))
    combined_decimal = float(np.prod(legs["decimal_odds"].astype(float)))
    combined_american = decimal_to_american(combined_decimal)
    break_even = 1.0 / combined_decimal
    ev_per_unit = hit_probability * (combined_decimal - 1.0) - (1.0 - hit_probability)
    potential_profit = stake_units * (combined_decimal - 1.0)
    total_payout = stake_units * combined_decimal
    key_payload = [
        f"{int(row.game_pk)}:{row.bet_side}:{int(row.bet_moneyline)}"
        for row in legs.sort_values("game_pk").itertuples(index=False)
    ]
    parlay_id = hashlib.sha1(json.dumps(key_payload).encode()).hexdigest()
    return LottoParlay(
        legs=legs,
        stake_units=stake_units,
        hit_probability=hit_probability,
        combined_decimal_odds=combined_decimal,
        combined_american_odds=combined_american,
        break_even_probability=break_even,
        ev_per_unit=float(ev_per_unit),
        potential_profit_units=float(potential_profit),
        total_payout_units=float(total_payout),
        parlay_id=parlay_id,
    )


def build_lotto_candidates(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for row in predictions.to_dict("records"):
        candidate = row.copy()
        if is_usable_official_leg(candidate):
            candidate["lotto_source"] = "official"
            rows.append(candidate)
            continue
        fallback = model_pick_leg(candidate)
        if fallback is not None:
            rows.append(fallback)
    if not rows:
        return pd.DataFrame()
    candidates = pd.DataFrame(rows)
    candidates = candidates.dropna(subset=["game_pk", "bet_side", "bet_moneyline", "bet_probability"])
    candidates = candidates[candidates["bet_side"].ne("no bet")]
    candidates = candidates[candidates["bet_probability"].astype(float).between(0.34, 0.82)]
    candidates = candidates[candidates["ev_per_dollar"].fillna(-1.0).ge(-0.15)]
    candidates = candidates.sort_values("ev_per_dollar", ascending=False).drop_duplicates("game_pk", keep="first")
    return candidates


def is_usable_official_leg(row: dict) -> bool:
    confidence = row.get("confidence")
    skip_reason = row.get("skip_reason")
    return (
        confidence in {"strong", "medium", "thin"}
        and pd.notna(row.get("bet_side"))
        and row.get("bet_side") != "no bet"
        and pd.notna(row.get("bet_probability"))
        and pd.notna(row.get("bet_moneyline"))
        and (pd.isna(skip_reason) or skip_reason == "")
        and float(row.get("ev_per_dollar", -1.0)) > -0.02
    )


def model_pick_leg(row: dict) -> dict | None:
    pick = row.get("model_pick")
    if not pick or pd.isna(pick):
        return None
    home_team = row.get("home_team")
    away_team = row.get("away_team")
    if pick == home_team:
        moneyline = row.get("home_moneyline")
        probability = first_number(row.get("market_adjusted_home_prob"), row.get("model_home_prob"))
        market_probability = row.get("fair_home_prob")
    elif pick == away_team:
        moneyline = row.get("away_moneyline")
        probability = first_number(row.get("market_adjusted_away_prob"), row.get("model_away_prob"))
        market_probability = row.get("fair_away_prob")
    else:
        return None
    if pd.isna(moneyline) or pd.isna(probability):
        return None
    edge = float(probability) - float(market_probability) if pd.notna(market_probability) else 0.0
    ev = float(probability) * american_profit_per_dollar(float(moneyline)) - (1.0 - float(probability))
    fallback = row.copy()
    fallback.update(
        {
            "bet_side": pick,
            "bet_moneyline": float(moneyline),
            "bet_probability": float(probability),
            "edge": edge,
            "ev_per_dollar": ev,
            "confidence": "lotto",
            "lotto_source": "model_pick",
            "lotto_note": row.get("skip_reason") or "model_pick",
        }
    )
    return fallback


def first_number(*values) -> float | None:
    for value in values:
        if pd.notna(value):
            return float(value)
    return None


def format_lotto_parlay_alert(parlay: LottoParlay | None, alert_date: str) -> str:
    if parlay is None:
        return f"MLB {alert_date} lotto parlay: no 4+ leg upcoming parlay qualifies right now."
    lines = [
        f"MLB {alert_date} lotto parlay ({len(parlay.legs)} legs, {parlay.stake_units:.2f}u):",
        f"Odds: {parlay.combined_american_odds:+d} | hit p={parlay.hit_probability * 100:.2f}% | "
        f"break-even={parlay.break_even_probability * 100:.2f}% | EV/u={parlay.ev_per_unit:.2f}",
        f"Potential profit: +{parlay.potential_profit_units:.2f}u | total payout={parlay.total_payout_units:.2f}u",
    ]
    for row in parlay.legs.to_dict("records"):
        prob = float(row["bet_probability"]) * 100
        edge = float(row["edge"]) * 100
        source = row.get("lotto_source", "official")
        note = f" [{source}]" if source != "official" else ""
        lines.append(f"- {row['bet_side']} {int(row['bet_moneyline']):+d} p={prob:.1f}% edge={edge:.1f}%{note}")
    return "\n".join(lines)
