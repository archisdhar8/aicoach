from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, lgamma
from pathlib import Path

import numpy as np
import pandas as pd

from .odds import (
    american_profit_per_dollar,
    devig_two_way,
    expected_value_per_dollar,
    implied_prob_to_moneyline,
)
from .props import (
    pitcher_props,
    apply_pitcher_k_regression,
    add_game_dates_to_team_stats,
    conservative_prop_probability,
    normal_over_probability,
    projected_pa,
)


HRR_MARKET = "hrr"
HRR_ODDS_MARKET = "batter_hits_runs_rbis"
QUALIFIER_MODEL_VERSION = "player-prop-qualifier-v1"


@dataclass(frozen=True)
class PropQualificationConfig:
    min_edge: float = 0.04
    min_ev: float = 0.05
    min_data_quality: float = 0.75
    max_odds_age_hours: float = 8.0
    require_confirmed_lineup: bool = True
    max_parlay_legs: int = 4
    min_parlay_legs: int = 2
    max_same_game_legs: int = 2


def qualify_player_props(
    slate: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    statcast_pitchers: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    lineups: pd.DataFrame,
    player_stats: pd.DataFrame,
    prop_lines: pd.DataFrame | None,
    *,
    config: PropQualificationConfig = PropQualificationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build qualified player props, rejected candidates, and optional parlays.

    This is intentionally conservative. It only qualifies a leg when model
    probability and sportsbook price both clear thresholds; otherwise the row is
    retained as a rejection with a concise reason.
    """
    dated_team_stats = add_game_dates_to_team_stats(team_stats, pd.concat([history_games, slate], ignore_index=True))
    batter = build_hrr_candidates(slate, history_games, dated_team_stats, statcast_team, weather, lineups, player_stats, prop_lines, config=config)
    pitcher = build_pitcher_k_candidates(slate, history_games, dated_team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats, prop_lines, config=config)
    all_rows = pd.concat([batter, pitcher], ignore_index=True, sort=False)
    if all_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), parlay_outputs(pd.DataFrame(), config)
    qualified = all_rows[all_rows["qualified"].astype(bool)].copy()
    rejected = all_rows[~all_rows["qualified"].astype(bool)].copy()
    qualified = qualified.sort_values(["ev_per_dollar", "edge"], ascending=False).reset_index(drop=True)
    rejected = rejected.sort_values(["market", "player_name"]).reset_index(drop=True)
    return qualified, rejected, parlay_outputs(qualified, config)


def build_hrr_candidates(
    slate: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    lineups: pd.DataFrame,
    player_stats: pd.DataFrame,
    prop_lines: pd.DataFrame | None,
    *,
    config: PropQualificationConfig,
) -> pd.DataFrame:
    if slate.empty:
        return pd.DataFrame()
    if lineups is None or lineups.empty:
        return pd.DataFrame()
    slate = slate.copy()
    slate["game_date"] = pd.to_datetime(slate["game_date"]).dt.date
    target_date = pd.to_datetime(slate["game_date"]).min()
    game_ids = set(pd.to_numeric(slate["game_pk"], errors="coerce").dropna().astype(int))
    lineup_rows = lineups[pd.to_numeric(lineups["game_pk"], errors="coerce").isin(game_ids)].copy()
    if lineup_rows.empty:
        return pd.DataFrame()
    summaries = player_hrr_summaries(player_stats, before_date=target_date)
    statcast = player_statcast_summaries(statcast_team, before_date=target_date)
    team_context = team_run_context(team_stats, before_date=target_date)
    game_context = game_context_lookup(slate, weather)
    base_rows: list[dict] = []
    for player in lineup_rows.sort_values(["game_pk", "team_name", "batting_order"]).itertuples(index=False):
        game_pk = int(player.game_pk)
        game = slate[slate["game_pk"].eq(game_pk)].iloc[0]
        team = str(player.team_name)
        opponent = game["away_team"] if team == game["home_team"] else game["home_team"]
        confirmed = bool(getattr(player, "confirmed", False))
        summary = summaries.get(int(player.player_id), {})
        stat = statcast.get(int(player.player_id), {})
        projection, probability = hrr_projection_probability(
            batting_order=int(getattr(player, "batting_order", 9) or 9),
            player_summary=summary,
            player_statcast=stat,
            team_context=team_context.get(team, {}),
            opponent_context=team_context.get(opponent, {}),
            game_context=game_context.get(game_pk, {}),
            line=1.5,
        )
        data_quality, quality_notes = hrr_data_quality(summary, stat, confirmed, game_context.get(game_pk, {}), config)
        base_rows.append(
            {
                "game_pk": game_pk,
                "game_date": game["game_date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "team": team,
                "opponent": opponent,
                "player_id": int(player.player_id),
                "player_name": player.player_name,
                "market": HRR_MARKET,
                "odds_market": HRR_ODDS_MARKET,
                "prop": "Hits + Runs + RBIs over",
                "line": 1.5,
                "model_probability": probability,
                "projection": projection,
                "projected_pa": projected_pa(int(getattr(player, "batting_order", 9) or 9)),
                "lineup_confirmed": float(confirmed),
                "data_quality": data_quality,
                "quality_notes": "; ".join(quality_notes),
                "model_version": QUALIFIER_MODEL_VERSION,
                "fallback_calculation": bool(not stat),
            }
        )
    return attach_best_prop_price(pd.DataFrame(base_rows), prop_lines, config=config)


def build_pitcher_k_candidates(
    slate: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    statcast_pitchers: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    lineups: pd.DataFrame,
    player_stats: pd.DataFrame,
    prop_lines: pd.DataFrame | None,
    *,
    config: PropQualificationConfig,
) -> pd.DataFrame:
    if slate.empty:
        return pd.DataFrame()
    from .features import build_prediction_frame, build_training_frame

    dated_team_stats = add_game_dates_to_team_stats(team_stats, pd.concat([history_games, slate], ignore_index=True))
    frame = build_prediction_frame(slate, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    props = pitcher_props(frame, "strikeouts", player_stats, dated_team_stats)
    training_frame = build_training_frame(history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    props = apply_pitcher_k_regression(props, training_frame, dated_team_stats, player_stats)
    if props.empty:
        return pd.DataFrame()
    rows = []
    for row in props.to_dict("records"):
        projection = float(row.get("projection") or 0.0)
        model_prob = float(row.get("over_probability") or 0.0)
        data_quality = pitcher_k_data_quality(row, config)
        player_name = row.get("player_name")
        if player_name is None or pd.isna(player_name) or str(player_name).strip() == "":
            player_name = f"{row.get('team', 'Unknown team')} probable starter"
        rows.append(
            {
                **row,
                "player_name": player_name,
                "prop": "Pitcher strikeouts over",
                "model_probability": model_prob,
                "data_quality": data_quality,
                "quality_notes": pitcher_k_quality_notes(row),
                "model_version": row.get("model_version", "pitcher-k-baseline-v1"),
                "fallback_calculation": str(row.get("ml_guardrail", "")).startswith("disabled"),
                "strikeout_distribution_mean": projection,
                "strikeout_probability_line": row.get("line"),
            }
        )
    return attach_best_prop_price(pd.DataFrame(rows), prop_lines, config=config)


def attach_best_prop_price(candidates: pd.DataFrame, prop_lines: pd.DataFrame | None, *, config: PropQualificationConfig) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    rows = []
    lines = prop_lines.copy() if prop_lines is not None else pd.DataFrame()
    if not lines.empty:
        lines["player_key"] = lines["player_name"].map(player_key)
        lines["fetched_at"] = pd.to_datetime(lines["fetched_at"], errors="coerce")
    for candidate in candidates.to_dict("records"):
        row = candidate.copy()
        matching = find_matching_lines(lines, row)
        best = None
        best_ev = -999.0
        evaluated = []
        for line in matching.to_dict("records") if not matching.empty else []:
            if pd.isna(line.get("over_odds")) or pd.isna(line.get("under_odds")):
                continue
            model_p = model_probability_for_line(row, line)
            market_over, _ = devig_two_way(line["over_odds"], line["under_odds"])
            edge = model_p - market_over
            ev = expected_value_per_dollar(model_p, line["over_odds"])
            age_hours = odds_age_hours(line.get("fetched_at"))
            evaluated.append((ev, edge, line, market_over, age_hours, model_p))
            if ev > best_ev:
                best_ev = ev
                best = (line, market_over, edge, ev, age_hours, model_p)
        if best is None:
            reasons = ["missing_prop_odds"]
            if row.get("market") == HRR_MARKET and config.require_confirmed_lineup and float(row.get("lineup_confirmed", 0.0)) < 1.0:
                reasons.append("lineup unconfirmed")
            if float(row.get("data_quality", 0.0)) < config.min_data_quality:
                reasons.append("data quality below threshold")
            if bool(row.get("missing_starter", False)):
                reasons.append("missing probable starter")
            if bool(row.get("short_workload_role", False)):
                reasons.append("expected pitch restriction")
            if bool(row.get("insufficient_data", False)):
                reasons.append("insufficient Statcast/model data")
            row.update(empty_price_columns("; ".join(dict.fromkeys(reasons))))
            rows.append(row)
            continue
        line, market_over, edge, ev, age_hours, model_p = best
        fair_odds = implied_prob_to_moneyline(float(np.clip(model_p, 0.001, 0.999)))
        reasons = rejection_reasons(row, edge=edge, ev=ev, odds_age_hours=age_hours, config=config)
        row.update(
            {
                "sportsbook": line.get("bookmaker"),
                "bookmaker": line.get("bookmaker"),
                "line": float(line.get("line")),
                "model_probability": model_p,
                "over_odds": int(line.get("over_odds")),
                "under_odds": int(line.get("under_odds")),
                "bet_odds": int(line.get("over_odds")),
                "market_no_vig_probability": market_over,
                "edge": edge,
                "ev_per_dollar": ev,
                "fair_odds": fair_odds,
                "odds_fetched_at": line.get("fetched_at"),
                "odds_age_hours": age_hours,
                "decision": "over" if not reasons else "no bet",
                "qualified": not reasons,
                "rejection_reason": "; ".join(reasons),
                "confidence": "qualified" if not reasons else "rejected",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def find_matching_lines(lines: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    if lines.empty:
        return lines
    subset = lines[
        lines["market"].eq(candidate["odds_market"])
        & lines["home_team"].eq(candidate["home_team"])
        & lines["away_team"].eq(candidate["away_team"])
        & lines["player_key"].eq(player_key(candidate["player_name"]))
    ].copy()
    if subset.empty:
        return subset
    if candidate["market"] == HRR_MARKET:
        subset = subset[subset["line"].astype(float).eq(1.5)]
    if subset.empty:
        return subset
    return subset.sort_values(["bookmaker", "fetched_at"]).drop_duplicates(["bookmaker", "line"], keep="last")


def model_probability_for_line(candidate: dict, line: dict) -> float:
    market = candidate.get("market")
    sportsbook_line = float(line.get("line"))
    if market == "strikeouts":
        mean = float(candidate.get("strikeout_distribution_mean", candidate.get("projection", 0.0)) or 0.0)
        variance = candidate.get("strikeout_distribution_variance")
        probability = normal_over_probability(mean, sportsbook_line, variance=variance)
        return conservative_prop_probability(probability)
    if market == HRR_MARKET:
        projection = float(candidate.get("projection", 0.0) or 0.0)
        return float(np.clip(poisson_over_probability(projection, sportsbook_line), 0.08, 0.82))
    return float(candidate.get("model_probability") or 0.0)


def rejection_reasons(row: dict, *, edge: float, ev: float, odds_age_hours: float, config: PropQualificationConfig) -> list[str]:
    reasons = []
    if row.get("market") == HRR_MARKET and config.require_confirmed_lineup and float(row.get("lineup_confirmed", 0.0)) < 1.0:
        reasons.append("lineup unconfirmed")
    if float(row.get("data_quality", 0.0)) < config.min_data_quality:
        reasons.append("data quality below threshold")
    if odds_age_hours > config.max_odds_age_hours:
        reasons.append("stale odds")
    if edge < config.min_edge:
        reasons.append("no price edge")
    if ev < config.min_ev:
        reasons.append("EV below threshold")
    if bool(row.get("missing_starter", False)):
        reasons.append("missing probable starter")
    if bool(row.get("short_workload_role", False)):
        reasons.append("expected pitch restriction")
    if bool(row.get("insufficient_data", False)):
        reasons.append("insufficient Statcast/model data")
    return reasons


def empty_price_columns(reason: str) -> dict[str, object]:
    return {
        "sportsbook": None,
        "bookmaker": None,
        "bet_odds": np.nan,
        "market_no_vig_probability": np.nan,
        "edge": np.nan,
        "ev_per_dollar": np.nan,
        "fair_odds": np.nan,
        "odds_fetched_at": pd.NaT,
        "odds_age_hours": np.nan,
        "decision": "no bet",
        "qualified": False,
        "rejection_reason": reason,
        "confidence": "rejected",
    }


def hrr_projection_probability(
    *,
    batting_order: int,
    player_summary: dict,
    player_statcast: dict,
    team_context: dict,
    opponent_context: dict,
    game_context: dict,
    line: float,
) -> tuple[float, float]:
    pa = projected_pa(batting_order)
    hrr_per_pa = shrink_rate(
        player_summary.get("hrr", 0.0),
        player_summary.get("pa", 0.0),
        baseline=0.46,
        prior_pa=160,
    )
    xwoba = player_statcast.get("xwoba")
    xba = player_statcast.get("xba")
    hard_hit = player_statcast.get("hard_hit_rate")
    barrel = player_statcast.get("barrel_rate")
    statcast_multiplier = 1.0
    if xwoba is not None:
        statcast_multiplier += np.clip((float(xwoba) - 0.320) * 1.15, -0.14, 0.18)
    if xba is not None:
        statcast_multiplier += np.clip((float(xba) - 0.245) * 0.55, -0.06, 0.08)
    if hard_hit is not None:
        statcast_multiplier += np.clip((float(hard_hit) - 0.38) * 0.18, -0.04, 0.05)
    if barrel is not None:
        statcast_multiplier += np.clip((float(barrel) - 0.075) * 0.35, -0.03, 0.05)
    team_multiplier = 1.0 + np.clip(float(team_context.get("runs_per_game", 4.4) or 4.4) - 4.4, -1.2, 1.4) * 0.035
    pitcher_multiplier = 1.0 + np.clip(float(opponent_context.get("starter_era", 4.4) or 4.4) - 4.4, -1.8, 2.0) * 0.025
    park_multiplier = float(game_context.get("park_run_factor", 1.0) or 1.0)
    projection = float(np.clip(pa * hrr_per_pa * statcast_multiplier * team_multiplier * pitcher_multiplier * park_multiplier, 0.35, 4.2))
    probability = poisson_over_probability(projection, line)
    return projection, float(np.clip(probability, 0.08, 0.82))


def hrr_data_quality(summary: dict, stat: dict, confirmed: bool, game_context: dict, config: PropQualificationConfig) -> tuple[float, list[str]]:
    quality = 1.0
    notes = []
    if not confirmed:
        quality -= 0.25
        notes.append("lineup unconfirmed")
    if float(summary.get("pa", 0.0)) < 120:
        quality -= 0.18
        notes.append("thin hitter PA sample")
    if not stat:
        quality -= 0.10
        notes.append("fallback Statcast estimate")
    if game_context.get("weather_uncertain"):
        quality -= 0.08
        notes.append("weather risk")
    return float(np.clip(quality, 0.0, 1.0)), notes


def pitcher_k_data_quality(row: dict, config: PropQualificationConfig) -> float:
    quality = 1.0
    quality -= min(float(row.get("feature_missing_rate", 0.0) or 0.0), 0.5)
    if bool(row.get("thin_history", False)):
        quality -= 0.18
    if bool(row.get("short_workload_role", False)):
        quality -= 0.30
    if bool(row.get("weather_uncertain", False)):
        quality -= 0.08
    return float(np.clip(quality, 0.0, 1.0))


def pitcher_k_quality_notes(row: dict) -> str:
    notes = []
    if bool(row.get("thin_history", False)):
        notes.append("thin starter history")
    if bool(row.get("short_workload_role", False)):
        notes.append("short workload role")
    if bool(row.get("insufficient_data", False)):
        notes.append("insufficient feature data")
    if row.get("ml_guardrail"):
        notes.append(str(row.get("ml_guardrail")))
    return "; ".join(notes)


def player_hrr_summaries(player_stats: pd.DataFrame, before_date) -> dict[int, dict]:
    if player_stats is None or player_stats.empty:
        return {}
    df = player_stats.copy()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df[df["game_date"] < pd.Timestamp(before_date)]
    grouped = df.groupby("player_id", dropna=True).agg(
        hits=("hits", "sum"),
        runs=("runs", "sum"),
        rbi=("rbi", "sum"),
        at_bats=("at_bats", "sum"),
        walks=("walks", "sum"),
        strikeouts=("strikeouts", "sum"),
    )
    out = {}
    for player_id, row in grouped.iterrows():
        pa = max(float(row["at_bats"] or 0) + float(row["walks"] or 0), 0.0)
        if pa <= 0:
            continue
        out[int(player_id)] = {
            "pa": pa,
            "hrr": (float(row["hits"] or 0) + float(row["runs"] or 0) + float(row["rbi"] or 0)) / pa,
            "k_rate": float(row["strikeouts"] or 0) / pa,
        }
    return out


def player_statcast_summaries(statcast: pd.DataFrame | None, before_date) -> dict[int, dict]:
    if statcast is None or statcast.empty:
        return {}
    df = statcast.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_date"] < pd.Timestamp(before_date)]
    if df.empty:
        return {}
    grouped = df.groupby("player_id", dropna=True).tail(30).groupby("player_id", dropna=True).agg(
        batted_balls=("batted_balls", "sum"),
        xwoba=("xwoba", "mean"),
        xba=("xba", "mean"),
        xslg=("xslg", "mean"),
        hard_hit_rate=("hard_hit_rate", "mean"),
        barrel_rate=("barrel_rate", "mean"),
        k_rate=("k_rate", "mean"),
    )
    return {int(pid): row.dropna().to_dict() for pid, row in grouped.iterrows() if int(row.get("batted_balls", 0) or 0) >= 25}


def team_run_context(team_stats: pd.DataFrame, before_date) -> dict[str, dict]:
    if team_stats is None or team_stats.empty:
        return {}
    df = team_stats.copy()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df[df["game_date"] < pd.Timestamp(before_date)]
    out = {}
    for team, group in df.groupby("team_name", dropna=True):
        recent = group.tail(20)
        out[str(team)] = {
            "runs_per_game": float(pd.to_numeric(recent["runs"], errors="coerce").mean()) if "runs" in recent else 4.4,
            "starter_era": float(pd.to_numeric(recent.get("starter_er", pd.Series(dtype=float)), errors="coerce").sum() * 9.0 / max(pd.to_numeric(recent.get("starter_ip", pd.Series(dtype=float)), errors="coerce").sum(), 1.0)),
        }
    return out


def game_context_lookup(slate: pd.DataFrame, weather: pd.DataFrame | None) -> dict[int, dict]:
    out = {int(row.game_pk): {"park_run_factor": float(getattr(row, "park_run_factor", 1.0) or 1.0)} for row in slate.itertuples(index=False)}
    if weather is not None and not weather.empty:
        for row in weather.itertuples(index=False):
            if not hasattr(row, "game_pk") or int(row.game_pk) not in out:
                continue
            wind = float(getattr(row, "wind_speed_mph", 0.0) or 0.0)
            precip = float(getattr(row, "precipitation_in", 0.0) or 0.0)
            out[int(row.game_pk)]["weather_uncertain"] = precip > 0.12 or wind > 22
    return out


def parlay_outputs(qualified: pd.DataFrame, config: PropQualificationConfig) -> dict[str, pd.DataFrame]:
    return {
        "two_leg": build_qualified_parlays(qualified, 2, config),
        "three_leg": build_qualified_parlays(qualified, 3, config),
        "four_leg": build_qualified_parlays(qualified, 4, config),
    }


def build_qualified_parlays(qualified: pd.DataFrame, legs: int, config: PropQualificationConfig) -> pd.DataFrame:
    if qualified.empty or len(qualified) < legs or legs < 2:
        return pd.DataFrame()
    rows = []
    for combo in combinations(qualified.to_dict("records"), legs):
        same_game_counts = pd.Series([leg["game_pk"] for leg in combo]).value_counts()
        if same_game_counts.max() > config.max_same_game_legs:
            continue
        joint = float(np.prod([float(leg["model_probability"]) for leg in combo]))
        correlation_note, adjustment = parlay_correlation_adjustment(combo)
        joint = float(np.clip(joint * adjustment, 0.0001, 0.98))
        decimal_price = float(np.prod([1.0 + american_profit_per_dollar(leg["bet_odds"]) for leg in combo]))
        break_even = 1.0 / decimal_price
        ev = joint * (decimal_price - 1.0) - (1.0 - joint)
        if ev < config.min_ev:
            continue
        rows.append(
            {
                "legs": " | ".join(f"{leg['player_name']} {leg['prop']} {leg['line']}" for leg in combo),
                "sportsbook_price": decimal_to_american(decimal_price),
                "model_joint_probability": joint,
                "fair_price": implied_prob_to_moneyline(joint),
                "break_even_probability": break_even,
                "ev_per_dollar": ev,
                "correlation_note": correlation_note,
                "game_pks": ",".join(str(leg["game_pk"]) for leg in combo),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ev_per_dollar", ascending=False).head(20).reset_index(drop=True)


def parlay_correlation_adjustment(combo: tuple[dict, ...]) -> tuple[str, float]:
    adjustment = 1.0
    notes = []
    for a, b in combinations(combo, 2):
        if a["game_pk"] != b["game_pk"]:
            continue
        if a["market"] == "strikeouts" and b["market"] == HRR_MARKET and a["team"] == b["opponent"]:
            adjustment *= 0.88
            notes.append("pitcher K over vs opposing batter HRR over")
        elif a["market"] == HRR_MARKET and b["market"] == HRR_MARKET and a["team"] == b["team"]:
            adjustment *= 0.96
            notes.append("same-team hitter legs share run environment")
        else:
            adjustment *= 0.94
            notes.append("same-game correlation haircut")
    return ("; ".join(sorted(set(notes))) if notes else "different-game legs treated independent"), adjustment


def format_qualified_props_report(qualified: pd.DataFrame, rejected: pd.DataFrame, parlays: dict[str, pd.DataFrame], report_date) -> str:
    lines = [f"MLB qualified player props for {report_date}"]
    lines.append("")
    lines.append("QUALIFIED SINGLES")
    if qualified.empty:
        lines.append("No qualified straight props.")
    else:
        cols = ["player_name", "prop", "line", "sportsbook", "bet_odds", "model_probability", "market_no_vig_probability", "edge", "ev_per_dollar", "data_quality"]
        lines.append(markdown_table(qualified[cols].head(30)))
    lines.append("")
    lines.append("QUALIFIED PARLAYS")
    any_parlay = False
    for name, frame in parlays.items():
        if frame.empty:
            continue
        any_parlay = True
        lines.append(name.replace("_", "-").upper())
        lines.append(markdown_table(frame[["legs", "sportsbook_price", "model_joint_probability", "fair_price", "ev_per_dollar", "correlation_note"]].head(10)))
    if not any_parlay:
        lines.append("No qualified parlay.")
    lines.append("")
    lines.append("REJECTED")
    if rejected.empty:
        lines.append("No rejected candidates.")
    else:
        lines.append(markdown_table(rejected[["player_name", "prop", "rejection_reason"]].head(40)))
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(format_cell)
    header = "| " + " | ".join(clean.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in clean.to_numpy()]
    return "\n".join([header, sep, *rows])


def format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_qualified_prop_outputs(qualified: pd.DataFrame, rejected: pd.DataFrame, parlays: dict[str, pd.DataFrame], report_dir: Path, report_date) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "qualified": report_dir / f"qualified_player_props_{report_date}.csv",
        "rejected": report_dir / f"rejected_player_props_{report_date}.csv",
        "two_leg": report_dir / f"qualified_player_prop_parlays_2leg_{report_date}.csv",
        "three_leg": report_dir / f"qualified_player_prop_parlays_3leg_{report_date}.csv",
        "four_leg": report_dir / f"qualified_player_prop_parlays_4leg_{report_date}.csv",
    }
    qualified.to_csv(paths["qualified"], index=False)
    rejected.to_csv(paths["rejected"], index=False)
    for key in ["two_leg", "three_leg", "four_leg"]:
        parlays[key].to_csv(paths[key], index=False)
    return paths


def poisson_over_probability(mean: float, line: float) -> float:
    mean = max(float(mean), 0.001)
    threshold = int(np.floor(float(line)))
    cdf = sum(exp(-mean + k * np.log(mean) - lgamma(k + 1)) for k in range(threshold + 1))
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def shrink_rate(rate: float, sample: float, *, baseline: float, prior_pa: float) -> float:
    sample = max(float(sample or 0.0), 0.0)
    return float((float(rate or 0.0) * sample + baseline * prior_pa) / max(sample + prior_pa, 1.0))


def odds_age_hours(fetched_at) -> float:
    if fetched_at is None or pd.isna(fetched_at):
        return 999.0
    ts = pd.Timestamp(fetched_at)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return float((pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0)


def player_key(name: str | None) -> str:
    return "".join(ch.lower() for ch in str(name or "") if ch.isalnum())


def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds <= 1:
        raise ValueError("decimal odds must be greater than 1")
    profit = decimal_odds - 1.0
    if decimal_odds >= 2.0:
        return int(round(profit * 100))
    return int(round(-100 / profit))
