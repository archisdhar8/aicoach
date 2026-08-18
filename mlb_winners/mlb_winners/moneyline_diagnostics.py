from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .db import upsert_df, write_csv
from .ensemble import build_elo_ratings, blend_probabilities, elo_probs_for_frame, model_disagreement
from .features import FEATURE_COLUMNS, build_training_frame
from .modeling import evaluate_predictions, predict_home_prob, train_baseline, train_model
from .odds import american_profit_per_dollar, devig_two_way


ALPHAS = [0.15, 0.30, 0.50, 1.00]


@dataclass(frozen=True)
class MoneylineEvaluationResult:
    summary: pd.DataFrame
    segments: pd.DataFrame
    market_variants: pd.DataFrame
    feature_subsets: pd.DataFrame


@dataclass(frozen=True)
class WeeklyMoneylineStats:
    summary: pd.DataFrame
    segments: pd.DataFrame
    candidates: pd.DataFrame


def market_adjusted_probability(model_prob: float, market_prob: float, alpha: float) -> float:
    return float(np.clip(float(market_prob) + alpha * (float(model_prob) - float(market_prob)), 0.001, 0.999))


def bucket_edge(edge: float | None) -> str:
    if edge is None or pd.isna(edge):
        return "no_edge"
    value = abs(float(edge))
    if value < 0.03:
        return "0-3%"
    if value < 0.05:
        return "3-5%"
    if value < 0.07:
        return "5-7%"
    if value < 0.10:
        return "7-10%"
    return "10%+"


def bucket_ev(ev: float | None) -> str:
    if ev is None or pd.isna(ev):
        return "no_ev"
    value = float(ev)
    if value <= 0:
        return "<=0"
    if value < 0.05:
        return "0-5%"
    if value < 0.10:
        return "5-10%"
    if value < 0.20:
        return "10-20%"
    return "20%+"


def bucket_disagreement(disagreement: float | None) -> str:
    if disagreement is None or pd.isna(disagreement):
        return "unknown"
    value = float(disagreement)
    if value < 0.03:
        return "low"
    if value < 0.07:
        return "medium"
    return "high"


def feature_subset_from_importance(importance: pd.DataFrame, top_n: int, importance_type: str = "permutation_log_loss") -> list[str]:
    if importance.empty:
        return FEATURE_COLUMNS[:top_n]
    ranked = importance[importance["importance_type"].eq(importance_type)].copy()
    if ranked.empty:
        ranked = importance.copy()
    ranked = ranked[ranked["feature_name"].isin(FEATURE_COLUMNS)].sort_values(
        ["importance", "feature_name"], ascending=[False, True]
    )
    selected = ranked["feature_name"].drop_duplicates().head(top_n).tolist()
    if len(selected) < top_n:
        selected.extend([feature for feature in FEATURE_COLUMNS if feature not in selected][: top_n - len(selected)])
    return selected


def write_moneyline_candidates(con, predictions: pd.DataFrame) -> int:
    if predictions.empty:
        return 0
    rows = []
    for row in predictions.to_dict("records"):
        bet_side = row.get("bet_side")
        home_line = row.get("home_moneyline")
        away_line = row.get("away_moneyline")
        fair_home = row.get("fair_home_prob")
        fair_away = row.get("fair_away_prob")
        if pd.isna(fair_home) and not pd.isna(home_line) and not pd.isna(away_line):
            fair_home, fair_away = devig_two_way(int(home_line), int(away_line))
        pick_side = "home" if row.get("model_pick") == row.get("home_team") else "away"
        price_side = _price_side(row, bet_side)
        candidate_id = hashlib.sha1(
            f"{row.get('game_pk')}:moneyline:{row.get('game_date')}:{bet_side}:{row.get('bookmaker')}:{row.get('confidence')}".encode()
        ).hexdigest()
        rows.append(
            {
                "candidate_id": candidate_id,
                "game_pk": row.get("game_pk"),
                "game_date": pd.to_datetime(row.get("game_date")).date() if row.get("game_date") is not None else None,
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "model_pick": row.get("model_pick"),
                "bet_side": bet_side,
                "sportsbook": row.get("bookmaker"),
                "home_moneyline": _nullable_int(home_line),
                "away_moneyline": _nullable_int(away_line),
                "model_home_prob": _nullable_float(row.get("model_home_prob")),
                "model_away_prob": _nullable_float(row.get("model_away_prob")),
                "market_home_prob": _nullable_float(fair_home),
                "market_away_prob": _nullable_float(fair_away),
                "bet_probability": _nullable_float(row.get("bet_probability")),
                "edge": _nullable_float(row.get("edge")),
                "ev_per_unit": _nullable_float(row.get("ev_per_dollar")),
                "confidence": row.get("confidence"),
                "skip_reason": row.get("skip_reason"),
                "tier": row.get("tier") or ("official" if row.get("confidence") in {"strong", "medium"} else row.get("confidence")),
                "risk_flags": row.get("risk_flags"),
                "stake_units": _nullable_float(row.get("stake_units")) if row.get("stake_units") is not None else (1.0 if row.get("confidence") in {"strong", "medium"} else 0.0),
                "pick_side": pick_side,
                "price_side": price_side,
                "model_disagreement": _nullable_float(row.get("model_disagreement")),
                "uncertainty_score": _nullable_float(row.get("uncertainty_score")),
                "lineups_confirmed": bool(min(float(row.get("home_lineup_confirmed", 0.0) or 0.0), float(row.get("away_lineup_confirmed", 0.0) or 0.0)) >= 1.0),
                "official_play": bool(row.get("confidence") in {"strong", "medium"} and bet_side not in {None, "no bet", "no odds"}),
                "raw_payload": json.dumps(row, default=str),
            }
        )
    return upsert_df(con, "moneyline_candidate_snapshots", pd.DataFrame(rows))


def weekly_moneyline_stats(con, start_date, end_date, report_dir: Path | None = None) -> WeeklyMoneylineStats:
    candidates = con.execute(
        """
        SELECT *
        FROM moneyline_candidate_snapshots
        WHERE game_date BETWEEN ? AND ?
        """,
        [start_date, end_date],
    ).df()
    if candidates.empty:
        empty = pd.DataFrame()
        return WeeklyMoneylineStats(empty, empty, empty)
    games = con.execute(
        """
        SELECT game_pk, home_score, away_score, home_won, status
        FROM games
        WHERE game_date BETWEEN ? AND ?
        """,
        [start_date, end_date],
    ).df()
    scored = candidates.merge(games, on="game_pk", how="left")
    final_status = scored["status"].fillna("").astype(str).str.lower().isin({"final", "game over"})
    scored["is_final"] = final_status & scored["home_score"].notna() & scored["away_score"].notna()
    scored["pick_won"] = scored.apply(_candidate_won, axis=1)
    scored["units"] = scored.apply(_candidate_units, axis=1)
    scored["edge_bucket"] = scored["edge"].map(bucket_edge)
    scored["ev_bucket"] = scored["ev_per_unit"].map(bucket_ev)
    scored["disagreement_bucket"] = scored["model_disagreement"].map(bucket_disagreement)
    scored["confidence"] = scored["confidence"].fillna("unknown")
    scored["skip_reason"] = scored["skip_reason"].fillna("")
    if "risk_flags" not in scored:
        scored["risk_flags"] = ""
    scored["risk_flags"] = scored["risk_flags"].fillna("")
    if "tier" not in scored:
        scored["tier"] = scored["confidence"].map(lambda value: "official" if value in {"strong", "medium"} else value)
    scored["tier"] = scored["tier"].fillna("unknown")
    summary_rows = [
        _weekly_segment(scored, "all_candidates", "all"),
        _weekly_segment(scored[scored["official_play"].astype(bool)], "official", "strong_medium"),
        _weekly_segment(scored[scored["confidence"].eq("strong")], "confidence", "strong"),
        _weekly_segment(scored[scored["confidence"].eq("medium")], "confidence", "medium"),
        _weekly_segment(scored[scored["confidence"].eq("watchlist")], "confidence", "watchlist"),
    ]
    segment_rows = []
    for segment_name in ["edge_bucket", "ev_bucket", "pick_side", "price_side", "confidence", "tier", "risk_flags", "skip_reason", "disagreement_bucket"]:
        for value, group in scored.groupby(segment_name, dropna=False):
            segment_rows.append(_weekly_segment(group, segment_name, str(value) if value != "" else "none"))
    summary = pd.DataFrame(summary_rows)
    segments = pd.DataFrame(segment_rows).sort_values(["segment_type", "segment"], kind="stable") if segment_rows else pd.DataFrame()
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{pd.Timestamp(start_date).date()}_to_{pd.Timestamp(end_date).date()}"
        write_csv(scored, report_dir / f"moneyline_weekly_candidates_{suffix}.csv")
        write_csv(summary, report_dir / f"moneyline_weekly_summary_{suffix}.csv")
        write_csv(segments, report_dir / f"moneyline_weekly_segments_{suffix}.csv")
    return WeeklyMoneylineStats(summary, segments, scored)


def _candidate_won(row) -> bool | None:
    if not bool(row.get("is_final")) or row.get("bet_side") in {None, "no bet", "no odds"} or pd.isna(row.get("bet_side")):
        return None
    if row.get("bet_side") == row.get("home_team"):
        return bool(row.get("home_won"))
    if row.get("bet_side") == row.get("away_team"):
        return not bool(row.get("home_won"))
    return None


def _candidate_units(row) -> float | None:
    won = row.get("pick_won")
    if won is None or pd.isna(won):
        return None
    stake = row.get("stake_units")
    stake = float(stake) if stake is not None and not pd.isna(stake) else 1.0
    if won:
        if row.get("bet_side") == row.get("home_team"):
            odds = row.get("home_moneyline")
        else:
            odds = row.get("away_moneyline")
        if odds is None or pd.isna(odds):
            return None
        return float(american_profit_per_dollar(int(odds))) * stake
    return -stake


def _weekly_segment(group: pd.DataFrame, segment_type: str, segment: str) -> dict:
    final = group[group["pick_won"].notna()].copy()
    wins = int(final["pick_won"].sum()) if not final.empty else 0
    losses = int((~final["pick_won"].astype(bool)).sum()) if not final.empty else 0
    units = float(final["units"].dropna().sum()) if "units" in final else 0.0
    return {
        "segment_type": segment_type,
        "segment": segment,
        "candidates": int(len(group)),
        "official_candidates": int(group["official_play"].astype(bool).sum()) if "official_play" in group else 0,
        "final_graded": int(len(final)),
        "pending": int(len(group) - len(final)),
        "wins": wins,
        "losses": losses,
        "record": f"{wins}-{losses}",
        "win_pct": float(wins / max(wins + losses, 1)),
        "units": units,
        "avg_edge": float(group["edge"].dropna().mean()) if "edge" in group and group["edge"].notna().any() else np.nan,
        "avg_ev": float(group["ev_per_unit"].dropna().mean()) if "ev_per_unit" in group and group["ev_per_unit"].notna().any() else np.nan,
    }


def evaluate_moneyline(
    con,
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    years: list[int],
    report_dir: Path,
    statcast_team: pd.DataFrame | None = None,
    statcast_pitchers: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    player_stats: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
) -> MoneylineEvaluationResult:
    frame = build_training_frame(games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    summary_rows: list[dict] = []
    segment_rows: list[dict] = []
    market_rows: list[dict] = []
    subset_rows: list[dict] = []
    for year in years:
        train = frame[(frame["season"] < year) & frame["target_home_win"].notna()].copy()
        test = frame[(frame["season"] == year) & frame["target_home_win"].notna()].copy()
        if train.empty and not test.empty:
            warmup = max(200, int(len(test) * 0.25))
            train = test.iloc[:warmup].copy()
            test = test.iloc[warmup:].copy()
        if train.empty or test.empty:
            continue
        bundle = train_model(train)
        baseline = train_baseline(train)
        xgb_probs = predict_home_prob(bundle, test)
        logistic_probs = baseline.predict_proba(test[bundle.feature_columns])[:, 1]
        elo_probs = elo_probs_for_frame(test, build_elo_ratings(games[games["game_date"] < pd.Timestamp(f"{year}-01-01")]))
        ensemble_probs = np.array(
            [
                blend_probabilities({"xgb": x, "logistic": l, "elo": e}, {"xgb": 0.5, "logistic": 0.2, "elo": 0.2})
                for x, l, e in zip(xgb_probs, logistic_probs, elo_probs)
            ]
        )
        summary_rows.extend(
            [
                _metric_row("model_quality", year, "xgb_all_98", evaluate_predictions(test["target_home_win"], xgb_probs)),
                _metric_row("model_quality", year, "logistic_all_98", evaluate_predictions(test["target_home_win"], logistic_probs)),
                _metric_row("model_quality", year, "ensemble_current", evaluate_predictions(test["target_home_win"], ensemble_probs)),
            ]
        )
        scored = test[["game_pk", "game_date", "home_team", "away_team", "target_home_win"]].copy()
        scored["prob"] = ensemble_probs
        scored["model_pick_side"] = np.where(scored["prob"] >= 0.5, "home", "away")
        scored["model_disagreement"] = [model_disagreement({"xgb": x, "logistic": l, "elo": e}) for x, l, e in zip(xgb_probs, logistic_probs, elo_probs)]
        segment_rows.extend(_segment_rows(scored, year, "historical_model"))
        subset_rows.extend(_feature_subset_rows(train, test, year))
    if odds is not None and not odds.empty:
        market_rows = _market_variant_rows(con, games, odds)
    summary = pd.DataFrame(summary_rows)
    segments = pd.DataFrame(segment_rows)
    market_variants = pd.DataFrame(market_rows)
    feature_subsets = pd.DataFrame(subset_rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary, report_dir / "moneyline_evaluation_summary.csv")
    write_csv(segments, report_dir / "moneyline_segment_performance.csv")
    write_csv(market_variants, report_dir / "moneyline_market_prior_variants.csv")
    write_csv(feature_subsets, report_dir / "moneyline_feature_subset_experiments.csv")
    persist_evaluation_rows(con, "model_quality", summary)
    persist_evaluation_rows(con, "segments", segments)
    persist_evaluation_rows(con, "market_prior", market_variants)
    persist_evaluation_rows(con, "feature_subsets", feature_subsets)
    return MoneylineEvaluationResult(summary, segments, market_variants, feature_subsets)


def _feature_subset_rows(train: pd.DataFrame, test: pd.DataFrame, year: int) -> list[dict]:
    rows = []
    for size in [15, 25, 50, len(FEATURE_COLUMNS)]:
        features = FEATURE_COLUMNS[:size] if size < len(FEATURE_COLUMNS) else FEATURE_COLUMNS.copy()
        label = f"first_{size}" if size < len(FEATURE_COLUMNS) else "all_98"
        bundle = train_model(train, features)
        probs = predict_home_prob(bundle, test)
        metrics = evaluate_predictions(test["target_home_win"], probs)
        rows.append({"year": year, "feature_set": label, "features": ",".join(features), **metrics})
    return rows


def feature_subset_experiment_rows(train: pd.DataFrame, test: pd.DataFrame, importance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for size in [15, 25, 50, len(FEATURE_COLUMNS)]:
        features = feature_subset_from_importance(importance, size) if size < len(FEATURE_COLUMNS) else FEATURE_COLUMNS.copy()
        label = f"top_{size}" if size < len(FEATURE_COLUMNS) else "all_98"
        bundle = train_model(train, features)
        baseline = train_baseline(train, features)
        model_probs = predict_home_prob(bundle, test)
        baseline_probs = baseline.predict_proba(test[features])[:, 1]
        model_metrics = evaluate_predictions(test["target_home_win"], model_probs)
        baseline_metrics = evaluate_predictions(test["target_home_win"], baseline_probs)
        rows.append({"feature_set": label, "model_type": "xgb", "features": ",".join(features), **model_metrics})
        rows.append({"feature_set": label, "model_type": "logistic", "features": ",".join(features), **baseline_metrics})
    return pd.DataFrame(rows)


def persist_evaluation_rows(con, report_type: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = []
    for record in df.to_dict("records"):
        year = _nullable_int(record.get("year"))
        segment = str(record.get("segment") or record.get("model") or record.get("feature_set") or record.get("variant") or "all")
        games = _nullable_int(record.get("games"))
        for metric_name, metric_value in record.items():
            if metric_name in {"year", "segment", "model", "feature_set", "variant", "features"}:
                continue
            if not isinstance(metric_value, (int, float, np.integer, np.floating)) or pd.isna(metric_value):
                continue
            report_id = hashlib.sha1(f"{report_type}:{year}:{segment}:{metric_name}".encode()).hexdigest()
            rows.append(
                {
                    "report_id": report_id,
                    "report_type": report_type,
                    "year": year,
                    "segment": segment,
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "games": games,
                    "raw_payload": json.dumps(record, default=str),
                }
            )
    return upsert_df(con, "moneyline_evaluation_reports", pd.DataFrame(rows))


def _market_variant_rows(con, games: pd.DataFrame, odds: pd.DataFrame) -> list[dict]:
    if games.empty or odds.empty:
        return []
    candidates = con.execute("SELECT * FROM moneyline_candidate_snapshots").df()
    merged = _join_forward_odds(games, odds, candidates)
    if merged.empty:
        return []
    rows = []
    for year, group in merged.groupby("season"):
        y = group["home_won"].astype(int)
        variants = {"market_only": group["fair_home_prob"].to_numpy(), "model_only": group["model_home_prob"].to_numpy()}
        for alpha in ALPHAS:
            variants[f"market_adjusted_alpha_{alpha:.2f}"] = np.array(
                [market_adjusted_probability(m, p, alpha) for m, p in zip(group["model_home_prob"], group["fair_home_prob"])]
            )
        for variant, probs in variants.items():
            rows.append({"year": int(year), "variant": variant, **evaluate_predictions(y, probs)})
    return rows


def _join_forward_odds(games: pd.DataFrame, odds: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    odds = odds.copy()
    odds["fetched_at"] = pd.to_datetime(odds["fetched_at"])
    odds["commence_time"] = pd.to_datetime(odds["commence_time"])
    odds = odds[odds["fetched_at"] < odds["commence_time"]].copy()
    if odds.empty:
        return pd.DataFrame()
    odds = odds.sort_values("fetched_at").drop_duplicates(["home_team", "away_team", "snapshot_date"], keep="last")
    candidates["game_date"] = pd.to_datetime(candidates["game_date"])
    joined = candidates.merge(
        odds[["snapshot_date", "home_team", "away_team"]],
        left_on=["game_date", "home_team", "away_team"],
        right_on=["snapshot_date", "home_team", "away_team"],
        how="inner",
    )
    joined = joined.merge(games[["game_pk", "season", "home_won"]], on="game_pk", how="inner")
    joined = joined[joined["home_won"].notna()].copy()
    if joined.empty:
        return joined
    joined["fair_home_prob"] = joined["market_home_prob"]
    joined = joined[joined["fair_home_prob"].notna() & joined["model_home_prob"].notna()].copy()
    return joined


def _segment_rows(scored: pd.DataFrame, year: int, label: str) -> list[dict]:
    rows = []
    rows.append(_segment_metric(scored, year, f"{label}:all"))
    for segment, group in scored.groupby("model_pick_side"):
        rows.append(_segment_metric(group, year, f"{label}:pick_{segment}"))
    scored = scored.copy()
    scored["disagreement_bucket"] = scored["model_disagreement"].map(bucket_disagreement)
    for segment, group in scored.groupby("disagreement_bucket"):
        rows.append(_segment_metric(group, year, f"{label}:disagreement_{segment}"))
    return rows


def _segment_metric(group: pd.DataFrame, year: int, segment: str) -> dict:
    if group.empty:
        return {"year": year, "segment": segment, "games": 0, "accuracy": np.nan, "actual_win_rate": np.nan, "avg_prob": np.nan}
    pick_win = np.where(group["model_pick_side"].eq("home"), group["target_home_win"].astype(bool), ~group["target_home_win"].astype(bool))
    return {
        "year": year,
        "segment": segment,
        "games": len(group),
        "accuracy": float(np.mean(pick_win)),
        "actual_home_win_rate": float(group["target_home_win"].astype(int).mean()),
        "avg_home_prob": float(group["prob"].mean()),
    }


def _metric_row(report_type: str, year: int, model: str, metrics: dict[str, float]) -> dict:
    return {"year": year, "model": model, **metrics}


def _price_side(row: dict, bet_side: str | None) -> str:
    if bet_side in {None, "no bet", "no odds"}:
        bet_side = row.get("model_pick")
    if bet_side == row.get("home_team"):
        line = row.get("home_moneyline")
    elif bet_side == row.get("away_team"):
        line = row.get("away_moneyline")
    else:
        return "unknown"
    if line is None or pd.isna(line):
        return "unknown"
    return "favorite" if float(line) < 0 else "underdog"


def _nullable_float(value):
    return None if value is None or pd.isna(value) else float(value)


def _nullable_int(value):
    return None if value is None or pd.isna(value) else int(value)
