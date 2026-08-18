from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3"


def build_play_context(row: dict[str, Any]) -> dict[str, Any]:
    selection = row.get("bet_side")
    home_team = row.get("home_team")
    away_team = row.get("away_team")
    is_home_pick = selection == home_team
    market_prob = row.get("fair_home_prob") if is_home_pick else row.get("fair_away_prob")
    model_side_prob = row.get("bet_probability")
    return {
        "game": f"{away_team} at {home_team}",
        "selection": selection,
        "moneyline": row.get("bet_moneyline"),
        "confidence": row.get("confidence"),
        "model_probability": _pct(model_side_prob),
        "market_no_vig_probability": _pct(market_prob),
        "edge": _pct(row.get("edge")),
        "expected_value_per_dollar": _num(row.get("ev_per_dollar"), 3),
        "model_pick": row.get("model_pick"),
        "xgb_home_probability": _pct(row.get("xgb_home_prob")),
        "logistic_home_probability": _pct(row.get("logistic_home_prob")),
        "elo_home_probability": _pct(row.get("elo_home_prob")),
        "model_disagreement": _num(row.get("model_disagreement"), 3),
        "uncertainty_score": _num(row.get("uncertainty_score"), 3),
        "line_move_for_side": _num(row.get("line_move_for_side"), 3),
        "home_lineup_confirmed": _flag(row.get("home_lineup_confirmed")),
        "away_lineup_confirmed": _flag(row.get("away_lineup_confirmed")),
        "home_starter_games_prior": _num(row.get("home_starter_games_prior"), 0),
        "away_starter_games_prior": _num(row.get("away_starter_games_prior"), 0),
        "home_bullpen_fatigue_score": _num(row.get("home_bullpen_fatigue_score"), 3),
        "away_bullpen_fatigue_score": _num(row.get("away_bullpen_fatigue_score"), 3),
        "precipitation_inches": _num(row.get("precipitation_in"), 3),
        "wind_speed_mph": _num(row.get("wind_speed_mph"), 1),
        "positive_factors": _factor_text(row, is_home_pick, positive=True),
        "risk_factors": _factor_text(row, is_home_pick, positive=False),
    }


def fallback_play_reason(row: dict[str, Any]) -> str:
    context = build_play_context(row)
    parts = [
        f"model prices {context['selection']} at {context['model_probability']} vs market {context['market_no_vig_probability']}",
        f"edge {context['edge']} and EV {context['expected_value_per_dollar']}",
    ]
    if context["positive_factors"] != "none clear":
        parts.append(f"drivers: {context['positive_factors']}")
    if context["risk_factors"] != "none clear":
        parts.append(f"risks: {context['risk_factors']}")
    if context["home_lineup_confirmed"] == "yes" and context["away_lineup_confirmed"] == "yes":
        parts.append("both lineups are confirmed")
    if context["uncertainty_score"] not in {"NA", None}:
        parts.append(f"uncertainty score {context['uncertainty_score']}")
    return "Reason: " + "; ".join(parts) + "."


def explain_play_with_ollama(
    row: dict[str, Any],
    model: str | None = None,
    url: str | None = None,
    timeout: float = 20.0,
) -> str:
    model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    url = url or os.getenv("OLLAMA_URL") or DEFAULT_OLLAMA_URL
    context = build_play_context(row)
    prompt = _build_prompt(context)
    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 140},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    text = str(payload.get("response", "")).strip()
    if not text:
        raise RuntimeError("Ollama returned an empty explanation")
    return _clean_reason(text)


def add_ollama_reasons(
    predictions: pd.DataFrame,
    model: str | None = None,
    url: str | None = None,
    timeout: float = 20.0,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    enriched = predictions.copy()
    reasons = []
    for row in enriched.to_dict("records"):
        if row.get("confidence") not in {"strong", "medium"}:
            reasons.append("")
            continue
        try:
            reasons.append(explain_play_with_ollama(row, model=model, url=url, timeout=timeout))
        except Exception:
            reasons.append(fallback_play_reason(row))
    enriched["reason"] = reasons
    return enriched


def _build_prompt(context: dict[str, Any]) -> str:
    lines = "\n".join(f"- {key}: {value}" for key, value in context.items())
    return (
        "You are explaining an MLB moneyline value bet to a bettor. "
        "Use only the data below. Do not invent injuries, news, weather, or stats. "
        "Do not infer team quality beyond the listed probabilities and feature values. "
        "Do not say the bet is guaranteed. Avoid raw feature names when writing. "
        "You must cite concrete factors from positive_factors and risk_factors when they are present. "
        "In 1-2 concise sentences, explain why the play is attractive "
        "and mention the main risk if visible.\n\n"
        f"Data:\n{lines}\n\nExplanation:"
    )


def _clean_reason(text: str) -> str:
    text = " ".join(text.replace("\n", " ").split())
    if len(text) > 700:
        text = text[:697].rstrip() + "..."
    if not text.lower().startswith("reason:"):
        text = "Reason: " + text
    return text


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.1f}%"


def _num(value: Any, digits: int) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _flag(value: Any) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return "yes" if float(value) >= 1.0 else "no"


def _factor_text(row: dict[str, Any], is_home_pick: bool, positive: bool) -> str:
    side = "home" if is_home_pick else "away"
    opp = "away" if is_home_pick else "home"
    selection = row.get("home_team") if is_home_pick else row.get("away_team")
    opponent = row.get("away_team") if is_home_pick else row.get("home_team")
    factors = _compare_factors(row, side, opp, selection, opponent)
    wanted = [factor for factor in factors if factor["supports_pick"] == positive]
    wanted = sorted(wanted, key=lambda factor: factor["strength"], reverse=True)[:4]
    if not wanted:
        return "none clear"
    return "; ".join(factor["text"] for factor in wanted)


def _compare_factors(
    row: dict[str, Any],
    side: str,
    opp: str,
    selection: Any,
    opponent: Any,
) -> list[dict[str, Any]]:
    specs = [
        ("win_pct", "season win rate", "higher", 0.04, 100.0),
        ("run_diff_per_game", "run differential per game", "higher", 0.25, 1.0),
        ("last7_run_diff", "last 7 run differential", "higher", 4.0, 1.0),
        ("last14_run_diff", "last 14 run differential", "higher", 6.0, 1.0),
        ("last30_run_diff", "last 30 run differential", "higher", 10.0, 1.0),
        ("last14_runs_for", "last 14 offense", "higher", 0.35, 1.0),
        ("ewma_runs_for", "recent scoring form", "higher", 0.25, 1.0),
        ("ewma_runs_against", "recent run prevention", "lower", 0.25, 1.0),
        ("bullpen_ip_last3", "bullpen workload last 3 days", "lower", 1.5, 1.0),
        ("bullpen_er_last3", "bullpen runs allowed last 3 days", "lower", 1.5, 1.0),
        ("bullpen_fatigue_score", "bullpen fatigue", "lower", 0.10, 1.0),
        ("bullpen_quality_score", "bullpen quality", "higher", 0.08, 1.0),
        ("bullpen_kbb_last7", "bullpen K/BB last 7 days", "higher", 0.25, 1.0),
        ("starter_era_prior", "starter ERA", "lower", 0.45, 1.0),
        ("starter_whip_prior", "starter WHIP", "lower", 0.08, 1.0),
        ("starter_kbb_prior", "starter K/BB", "higher", 0.35, 1.0),
        ("starter_fip_proxy", "starter FIP proxy", "lower", 0.45, 1.0),
        ("starter_last5_fip", "starter last 5 FIP proxy", "lower", 0.55, 1.0),
        ("starter_rest_days", "starter rest", "higher", 1.0, 1.0),
        ("starter_workload_trend", "starter workload trend", "lower", 0.75, 1.0),
        ("starter_xwoba_allowed", "starter contact quality allowed", "lower", 0.025, 100.0),
        ("starter_hard_hit_allowed", "starter hard-hit allowed", "lower", 0.05, 100.0),
        ("starter_barrel_allowed", "starter barrel rate allowed", "lower", 0.02, 100.0),
        ("team_xwoba_last14", "team xwOBA last 14", "higher", 0.020, 100.0),
        ("team_hard_hit_last14", "hard-hit rate last 14", "higher", 0.050, 100.0),
        ("team_barrel_last14", "barrel rate last 14", "higher", 0.018, 100.0),
        ("team_bb_rate_last14", "walk rate last 14", "higher", 0.018, 100.0),
        ("team_k_rate_last14", "strikeout rate last 14", "lower", 0.030, 100.0),
        ("pitching_xwoba_allowed_last14", "staff xwOBA allowed last 14", "lower", 0.020, 100.0),
        ("lineup_offense_rating", "confirmed lineup offense rating", "higher", 0.040, 1.0),
        ("lineup_platoon_score", "lineup platoon score", "higher", 0.040, 1.0),
    ]
    factors: list[dict[str, Any]] = []
    for suffix, label, direction, threshold, display_scale in specs:
        picked_value = _float_or_none(row.get(f"{side}_{suffix}"))
        opp_value = _float_or_none(row.get(f"{opp}_{suffix}"))
        if picked_value is None or opp_value is None:
            continue
        diff = picked_value - opp_value
        if abs(diff) < threshold:
            continue
        supports_pick = diff > 0 if direction == "higher" else diff < 0
        better_word = "better" if supports_pick else "worse"
        picked_display = _display_value(picked_value, display_scale)
        opp_display = _display_value(opp_value, display_scale)
        text = f"{selection} has {better_word} {label} ({picked_display} vs {opponent} {opp_display})"
        factors.append({"text": text, "supports_pick": supports_pick, "strength": abs(diff) / threshold})
    return factors


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _display_value(value: float, scale: float) -> str:
    display = value * scale
    if scale == 100.0:
        return f"{display:.1f}%"
    return f"{display:.2f}"
