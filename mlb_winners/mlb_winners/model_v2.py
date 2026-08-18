from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .db import upsert_df, write_csv
from .ensemble import build_elo_ratings, blend_probabilities, elo_probs_for_frame
from .features import FEATURE_COLUMNS, build_prediction_frame, build_training_frame
from .market import apply_moneyline_tier_filter, line_move_for_side, latest_odds_by_game, market_adjusted_probability
from .modeling import ModelBundle, predict_home_prob, train_baseline, train_model
from .odds import devig_two_way, expected_value_per_dollar
from .predict import EXPLANATION_FEATURE_COLUMNS, elite_opposing_starter_flag


TOP15_FEATURES = [
    "run_diff_per_game_diff",
    "win_pct_diff",
    "last30_run_diff_diff",
    "last14_run_diff_diff",
    "starter_fip_advantage",
    "starter_kbb_advantage",
    "starter_era_advantage",
    "starter_last5_fip_advantage",
    "ewma_runs_for_diff",
    "ewma_run_prevention_diff",
    "bullpen_quality_advantage",
    "bullpen_fatigue_advantage",
    "venue_split_win_pct_diff",
    "lineup_offense_advantage",
    "park_run_factor",
]

TOP25_FEATURES = TOP15_FEATURES + [
    "last7_run_diff_diff",
    "last14_runs_for_diff",
    "last30_runs_for_diff",
    "starter_whip_advantage",
    "starter_rest_advantage",
    "starter_workload_advantage",
    "bullpen_ip_last3_diff",
    "bullpen_er_last3_diff",
    "bullpen_kbb_advantage",
    "both_lineups_confirmed",
]

ENSEMBLE_WEIGHTS = {
    "current": {"xgb": 0.5, "logistic": 0.2, "elo": 0.2},
    "balanced": {"xgb": 1.0, "logistic": 1.0, "elo": 1.0},
    "xgb_logistic": {"xgb": 0.45, "logistic": 0.35, "elo": 0.20},
    "logistic_heavy": {"xgb": 0.30, "logistic": 0.50, "elo": 0.20},
}

HOME_THRESHOLDS = {"strong": 0.08, "medium": 0.055}
AWAY_THRESHOLDS = {"strong": 0.10, "medium": 0.07}

CALIBRATION_BINS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
CALIBRATION_LABELS = ["50-55", "55-60", "60-65", "65-70", "70+"]


@dataclass(frozen=True)
class V2Config:
    feature_set: str
    feature_columns: list[str]
    include_bullpen: bool
    weights_name: str
    weights: dict[str, float]


@dataclass
class V2Bundle:
    xgb: ModelBundle
    logistic: object
    feature_columns: list[str]
    weights_name: str
    weights: dict[str, float]
    feature_set: str
    include_bullpen: bool
    train_rows: int


def feature_columns_for_v2(feature_set: str, include_bullpen: bool = True) -> list[str]:
    if feature_set == "top15":
        columns = TOP15_FEATURES.copy()
    elif feature_set == "top25":
        columns = TOP25_FEATURES.copy()
    elif feature_set == "all98":
        columns = FEATURE_COLUMNS.copy()
    else:
        raise ValueError(f"Unknown V2 feature set: {feature_set}")
    if not include_bullpen:
        columns = [column for column in columns if "bullpen" not in column]
    return columns


def train_v2_bundle(train: pd.DataFrame, config: V2Config) -> V2Bundle:
    xgb = train_model(train, config.feature_columns)
    logistic = train_baseline(train, config.feature_columns)
    return V2Bundle(
        xgb=xgb,
        logistic=logistic,
        feature_columns=config.feature_columns,
        weights_name=config.weights_name,
        weights=config.weights,
        feature_set=config.feature_set,
        include_bullpen=config.include_bullpen,
        train_rows=xgb.train_rows,
    )


def predict_v2_probs(bundle: V2Bundle, frame: pd.DataFrame, elo_ratings: dict[int, float]) -> pd.DataFrame:
    out = frame.copy()
    out["xgb_home_prob"] = predict_home_prob(bundle.xgb, out)
    out["logistic_home_prob"] = bundle.logistic.predict_proba(out[bundle.feature_columns])[:, 1]
    out["elo_home_prob"] = elo_probs_for_frame(out, elo_ratings)
    out["model_home_prob"] = [
        blend_probabilities({"xgb": x, "logistic": l, "elo": e}, bundle.weights)
        for x, l, e in zip(out["xgb_home_prob"], out["logistic_home_prob"], out["elo_home_prob"])
    ]
    out["model_away_prob"] = 1.0 - out["model_home_prob"]
    out["model_disagreement_score"] = [
        disagreement_score(x, l, e) for x, l, e in zip(out["xgb_home_prob"], out["logistic_home_prob"], out["elo_home_prob"])
    ]
    out["disagreement_bucket"] = out["model_disagreement_score"].map(disagreement_bucket)
    return out


def disagreement_score(xgb_prob: float, logistic_prob: float, elo_prob: float) -> float:
    values = [float(xgb_prob), float(logistic_prob), float(elo_prob)]
    return float(np.mean([abs(values[0] - values[1]), abs(values[0] - values[2]), abs(values[1] - values[2])]))


def disagreement_bucket(score: float) -> str:
    if score < 0.05:
        return "low"
    if score < 0.12:
        return "medium"
    return "high"


def ece_score(y_true: pd.Series, probs: np.ndarray, bins: int = 10) -> float:
    df = pd.DataFrame({"actual": y_true.astype(int), "prob": np.clip(probs, 0.001, 0.999)})
    df["bucket"] = pd.cut(df["prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    total = len(df)
    ece = 0.0
    for _, group in df.groupby("bucket", observed=False):
        if group.empty:
            continue
        ece += len(group) / total * abs(group["prob"].mean() - group["actual"].mean())
    return float(ece)


def metrics_row(y_true: pd.Series, probs: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    p = np.clip(np.asarray(probs, dtype=float), 0.001, 0.999)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece_score(y_true, p),
    }


def calibration_bucket_report(scored: pd.DataFrame, year: int, model_version: str) -> pd.DataFrame:
    rows = []
    df = scored.copy()
    df["pick_prob"] = np.where(df["model_home_prob"] >= 0.5, df["model_home_prob"], df["model_away_prob"])
    df["pick_won"] = np.where(df["model_home_prob"] >= 0.5, df["target_home_win"].astype(bool), ~df["target_home_win"].astype(bool))
    for label, (low, high) in zip(CALIBRATION_LABELS, CALIBRATION_BINS):
        group = df[(df["pick_prob"] >= low) & (df["pick_prob"] < high)]
        rows.append(
            {
                "year": year,
                "model_version": model_version,
                "bucket": label,
                "games": len(group),
                "predicted_pct": float(group["pick_prob"].mean()) if not group.empty else np.nan,
                "actual_pct": float(group["pick_won"].mean()) if not group.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def betting_segment_report(scored: pd.DataFrame, year: int, model_version: str) -> pd.DataFrame:
    df = scored.copy()
    df["pick_side"] = np.where(df["model_home_prob"] >= 0.5, "home", "away")
    df["pick_won"] = np.where(df["pick_side"].eq("home"), df["target_home_win"].astype(bool), ~df["target_home_win"].astype(bool))
    rows = []
    for column, prefix in [
        ("pick_side", "pick"),
        ("disagreement_bucket", "disagreement"),
    ]:
        for value, group in df.groupby(column):
            rows.append(_segment_row(group, year, model_version, f"{prefix}_{value}"))
    return pd.DataFrame(rows)


def _segment_row(group: pd.DataFrame, year: int, model_version: str, segment: str) -> dict:
    return {
        "year": year,
        "model_version": model_version,
        "segment": segment,
        "record": f"{int(group['pick_won'].sum())}-{int((~group['pick_won']).sum())}",
        "games": len(group),
        "win_pct": float(group["pick_won"].mean()) if not group.empty else np.nan,
        "average_edge": float(group["edge"].mean()) if "edge" in group and group["edge"].notna().any() else np.nan,
        "average_ev": float(group["ev_per_dollar"].mean()) if "ev_per_dollar" in group and group["ev_per_dollar"].notna().any() else np.nan,
    }


def statcast_audit(frame: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    rows = []
    statcast_columns = [
        column
        for column in FEATURE_COLUMNS
        if any(token in column for token in ["xwoba", "xba", "hard_hit", "barrel", "velocity", "spin"])
    ]
    defaults = {
        "xwoba": 0.320,
        "xba": 0.245,
        "hard_hit": 0.39,
        "barrel": 0.075,
        "velocity": 92.5,
        "spin": 2250.0,
    }
    for column in statcast_columns:
        series = frame[column] if column in frame else pd.Series(dtype=float)
        default_value = next((value for key, value in defaults.items() if key in column), None)
        rows.append(
            {
                "feature_name": column,
                "missing_rate": float(series.isna().mean()) if len(series) else np.nan,
                "default_filled_rate": float(series.eq(default_value).mean()) if default_value is not None and len(series) else np.nan,
                "recommendation": "review_or_exclude_if_importance_zero",
            }
        )
    report = pd.DataFrame(rows)
    write_csv(report, report_dir / "model_v2_statcast_audit.csv")
    return report


def run_v2_backtests(
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
) -> dict[str, pd.DataFrame | dict]:
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = build_training_frame(games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    statcast = statcast_audit(frame, report_dir)
    experiment_rows = []
    calibration_rows = []
    segment_rows = []
    prediction_rows = []
    for year in years:
        train = frame[(frame["season"] < year) & frame["target_home_win"].notna()].copy()
        test = frame[(frame["season"] == year) & frame["target_home_win"].notna()].copy()
        if train.empty and not test.empty:
            warmup = max(200, int(len(test) * 0.25))
            train = test.iloc[:warmup].copy()
            test = test.iloc[warmup:].copy()
        if train.empty or test.empty:
            continue
        elo_ratings = build_elo_ratings(games[games["game_date"] < pd.Timestamp(f"{year}-01-01")])
        for feature_set in ["top15", "top25", "all98"]:
            for include_bullpen in [True, False]:
                columns = feature_columns_for_v2(feature_set, include_bullpen)
                base_config = V2Config(feature_set, columns, include_bullpen, "current", ENSEMBLE_WEIGHTS["current"])
                base_bundle = train_v2_bundle(train, base_config)
                for weights_name, weights in ENSEMBLE_WEIGHTS.items():
                    bundle = V2Bundle(
                        xgb=base_bundle.xgb,
                        logistic=base_bundle.logistic,
                        feature_columns=base_bundle.feature_columns,
                        weights_name=weights_name,
                        weights=weights,
                        feature_set=feature_set,
                        include_bullpen=include_bullpen,
                        train_rows=base_bundle.train_rows,
                    )
                    config = V2Config(feature_set, columns, include_bullpen, weights_name, weights)
                    scored = predict_v2_probs(bundle, test, elo_ratings)
                    version = model_version_name(config)
                    metrics = metrics_row(scored["target_home_win"], scored["model_home_prob"])
                    experiment_rows.append(
                        {
                            "model_version": version,
                            "feature_set": feature_set,
                            "include_bullpen": include_bullpen,
                            "weights": weights_name,
                            "year": year,
                            "games": len(scored),
                            **metrics,
                        }
                    )
                    calibration_rows.extend(calibration_bucket_report(scored, year, version).to_dict("records"))
                    segment_rows.extend(betting_segment_report(scored, year, version).to_dict("records"))
                    prediction_rows.extend(
                        scored[["game_pk", "game_date", "season", "home_team", "away_team", "target_home_win", "model_home_prob", "model_away_prob", "model_disagreement_score", "disagreement_bucket"]]
                        .assign(model_version=version)
                        .to_dict("records")
                    )
    experiments = pd.DataFrame(experiment_rows)
    calibration = pd.DataFrame(calibration_rows)
    segments = pd.DataFrame(segment_rows)
    predictions = pd.DataFrame(prediction_rows)
    best = select_best_config(experiments)
    write_csv(experiments, report_dir / "model_v2_experiments.csv")
    write_csv(calibration, report_dir / "model_v2_calibration.csv")
    write_csv(segments, report_dir / "model_v2_betting_segments.csv")
    write_csv(predictions, report_dir / "model_v2_predictions.csv")
    persist_model_experiments(con, experiments)
    write_v2_markdown_report(report_dir / "MODEL_V2_RESULTS.md", experiments, calibration, segments, statcast, best)
    return {"experiments": experiments, "calibration": calibration, "segments": segments, "predictions": predictions, "statcast": statcast, "best": best}


def select_best_config(experiments: pd.DataFrame) -> dict:
    if experiments.empty:
        return {}
    grouped = (
        experiments.groupby(["model_version", "feature_set", "include_bullpen", "weights"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), auc=("auc", "mean"), brier=("brier", "mean"), log_loss=("log_loss", "mean"), ece=("ece", "mean"))
        .sort_values(["brier", "ece", "log_loss", "accuracy"], ascending=[True, True, True, False])
    )
    return grouped.iloc[0].to_dict()


def persist_model_experiments(con, experiments: pd.DataFrame) -> int:
    if experiments.empty:
        return 0
    rows = []
    for row in experiments.to_dict("records"):
        experiment_id = hashlib.sha1(f"{row['model_version']}:{row['year']}".encode()).hexdigest()
        rows.append(
            {
                "experiment_id": experiment_id,
                "model_version": row["model_version"],
                "feature_set": row["feature_set"],
                "year": int(row["year"]),
                "accuracy": float(row["accuracy"]),
                "auc": float(row["auc"]),
                "brier": float(row["brier"]),
                "log_loss": float(row["log_loss"]),
                "ece": float(row["ece"]),
            }
        )
    return upsert_df(con, "model_experiments", pd.DataFrame(rows))


def model_version_name(config: V2Config) -> str:
    bullpen = "with_bullpen" if config.include_bullpen else "without_bullpen"
    return f"v2_{config.feature_set}_{bullpen}_{config.weights_name}"


def save_v2_bundle(bundle: V2Bundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_v2_bundle(path: Path) -> V2Bundle:
    return joblib.load(path)


def train_best_v2(
    con,
    frame: pd.DataFrame,
    train_through: int,
    report_dir: Path,
    model_dir: Path,
) -> tuple[V2Bundle, dict]:
    experiments_path = report_dir / "model_v2_experiments.csv"
    if experiments_path.exists():
        best = select_best_config(pd.read_csv(experiments_path))
    else:
        best = {"feature_set": "top15", "include_bullpen": True, "weights": "current"}
    feature_set = str(best.get("feature_set", "top15"))
    include_bullpen = bool(best.get("include_bullpen", True))
    weights_name = str(best.get("weights", "current"))
    columns = feature_columns_for_v2(feature_set, include_bullpen)
    config = V2Config(feature_set, columns, include_bullpen, weights_name, ENSEMBLE_WEIGHTS[weights_name])
    train = frame[(frame["season"] <= train_through) & frame["target_home_win"].notna()].copy()
    bundle = train_v2_bundle(train, config)
    save_v2_bundle(bundle, model_dir / f"{model_version_name(config)}_through_{train_through}.joblib")
    return bundle, best


def make_daily_predictions_v2(
    today_games: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    statcast_pitchers: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    odds: pd.DataFrame | None,
    bundle: V2Bundle,
    lineups: pd.DataFrame | None = None,
    player_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = build_prediction_frame(today_games, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    if frame.empty:
        return pd.DataFrame()
    scored = predict_v2_probs(bundle, frame, build_elo_ratings(history_games))
    columns = [
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
        "model_disagreement_score",
        "disagreement_bucket",
    ]
    columns.extend([column for column in EXPLANATION_FEATURE_COLUMNS if column in scored.columns and column not in columns])
    output = scored[columns].copy()
    output["model_pick"] = np.where(output["model_home_prob"] >= 0.5, output["home_team"], output["away_team"])
    output["model_version"] = f"v2_{bundle.feature_set}_{'with_bullpen' if bundle.include_bullpen else 'without_bullpen'}_{bundle.weights_name}"
    if odds is None or odds.empty:
        output["bet_side"] = "no odds"
        output["bet_moneyline"] = np.nan
        output["edge"] = np.nan
        output["ev_per_dollar"] = np.nan
        output["confidence"] = "no odds"
        output["skip_reason"] = "missing_odds"
        output["tier"] = "no odds"
        output["risk_flags"] = ""
        output["stake_units"] = 0.0
        return output
    return attach_odds_v2(output, odds)


def attach_odds_v2(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
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
    latest = latest_odds_by_game(all_odds).sort_values(["home_team", "away_team", "bookmaker"]).drop_duplicates(["home_team", "away_team"])
    odds_lookup = {(row.home_team, row.away_team): row for row in latest.itertuples(index=False)}
    rows = []
    for row in predictions.itertuples(index=False):
        result = row._asdict()
        market = odds_lookup.get((row.home_team, row.away_team))
        if market is None:
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
        market_adjusted_home = market_adjusted_probability(row.model_home_prob, fair_home)
        market_adjusted_away = 1.0 - market_adjusted_home
        home_ev = expected_value_per_dollar(market_adjusted_home, market.home_moneyline)
        away_ev = expected_value_per_dollar(market_adjusted_away, market.away_moneyline)
        home_edge = market_adjusted_home - fair_home
        away_edge = market_adjusted_away - fair_away
        if home_ev >= away_ev:
            side, line, side_prob, edge, ev = row.home_team, market.home_moneyline, market_adjusted_home, home_edge, home_ev
        else:
            side, line, side_prob, edge, ev = row.away_team, market.away_moneyline, market_adjusted_away, away_edge, away_ev
        line_move = line_move_for_side(all_odds, row.home_team, row.away_team, side)
        missing_starter = bool(pd.isna(row.home_probable_pitcher_id) or pd.isna(row.away_probable_pitcher_id))
        missing_lineup = bool(min(row.home_lineup_confirmed, row.away_lineup_confirmed) < 1.0)
        weather_uncertain = bool((row.precipitation_in or 0.0) > 0.12 or (row.wind_speed_mph or 0.0) > 22)
        thin_starter_history = bool(min(row.home_starter_games_prior, row.away_starter_games_prior) < 2)
        bullpen_uncertain = bool(max(row.home_bullpen_fatigue_score, row.away_bullpen_fatigue_score) > 0.85)
        filter_result = apply_moneyline_tier_filter(
            edge=edge,
            ev=ev,
            moneyline=line,
            uncertainty_score=float(getattr(row, "uncertainty_score", 0.25) or 0.25),
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
        if confidence == "no bet":
            side = "no bet"
        result.update(
            {
                "bookmaker": market.bookmaker,
                "home_moneyline": market.home_moneyline,
                "away_moneyline": market.away_moneyline,
                "fair_home_prob": fair_home,
                "fair_away_prob": fair_away,
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
    return pd.DataFrame(rows).sort_values(["confidence", "ev_per_dollar"], ascending=[True, False])


def classify_v2_play(
    *,
    side: str,
    home_team: str,
    edge: float,
    ev: float,
    missing_starter: bool,
    missing_lineup: bool,
    weather_uncertain: bool,
    adverse_line_move: float,
    thin_starter_history: bool,
) -> tuple[str, str]:
    if missing_starter:
        return "no bet", "missing_starter"
    if missing_lineup and edge < 0.07:
        return "no bet", "missing_lineup"
    if weather_uncertain:
        return "no bet", "weather_uncertainty"
    if thin_starter_history and edge < 0.12:
        return "no bet", "thin_starter_history"
    if adverse_line_move <= -0.025:
        return "no bet", "market_moving_against_pick"
    if ev <= 0:
        return "no bet", "negative_ev"
    thresholds = HOME_THRESHOLDS if side == home_team else AWAY_THRESHOLDS
    if edge >= thresholds["strong"] and ev > 0:
        return "strong", ""
    if edge >= thresholds["medium"] and ev > 0:
        return "medium", ""
    return "no bet", f"edge_below_v2_threshold_{thresholds['medium']:.3f}"


def write_v2_markdown_report(
    path: Path,
    experiments: pd.DataFrame,
    calibration: pd.DataFrame,
    segments: pd.DataFrame,
    statcast: pd.DataFrame,
    best: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_version = best.get("model_version", "unknown")
    avg = (
        experiments.groupby("model_version", as_index=False)
        .agg(accuracy=("accuracy", "mean"), auc=("auc", "mean"), brier=("brier", "mean"), log_loss=("log_loss", "mean"), ece=("ece", "mean"))
        .sort_values(["brier", "ece", "log_loss"])
        .head(10)
    )
    best_cal = calibration[calibration["model_version"].eq(best_version)] if not calibration.empty else pd.DataFrame()
    best_seg = segments[segments["model_version"].eq(best_version)] if not segments.empty else pd.DataFrame()
    noisy_statcast = statcast.sort_values(["default_filled_rate", "missing_rate"], ascending=False).head(10) if not statcast.empty else pd.DataFrame()
    content = [
        "# Model V2 Results",
        "",
        "## Best Configuration",
        f"- Best model version: `{best_version}`",
        f"- Feature set: `{best.get('feature_set', 'unknown')}`",
        f"- Include bullpen features: `{best.get('include_bullpen', 'unknown')}`",
        f"- Ensemble weights: `{best.get('weights', 'unknown')}`",
        "",
        "## Average Metrics By Configuration",
        markdown_table(avg) if not avg.empty else "No experiment rows.",
        "",
        "## Calibration Buckets For Best Model",
        markdown_table(best_cal) if not best_cal.empty else "No calibration rows.",
        "",
        "## Home/Away And Disagreement Performance",
        markdown_table(best_seg) if not best_seg.empty else "No segment rows.",
        "",
        "## Statcast Audit",
        markdown_table(noisy_statcast) if not noisy_statcast.empty else "No Statcast audit rows.",
        "",
        "## Production Recommendation",
        "- Use V2 only after reviewing the daily candidate behavior for several slates.",
        "- Prefer the best Brier/ECE configuration over raw accuracy.",
        "- Keep stricter away-pick thresholds: strong >= 10%, medium >= 7%.",
        "- Do not auto-skip high disagreement; use edge and EV to decide.",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.fillna("").to_numpy().tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
