from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from .db import write_csv
from .features import FEATURE_COLUMNS, build_training_frame
from .market import BetFilterConfig, apply_bet_filter, latest_odds_by_game
from .odds import devig_two_way, expected_value_per_dollar


@dataclass
class TotalsModelBundle:
    model: object
    feature_columns: list[str]
    train_rows: int
    residual_std: float


TOTALS_BET_FILTER = BetFilterConfig(
    base_edge_threshold=0.03,
    favorite_edge_threshold=0.05,
    underdog_edge_threshold=0.025,
    max_official_underdog_moneyline=10000,
    max_official_favorite_moneyline=10000,
    max_reasonable_edge=1.0,
    max_reasonable_ev=10.0,
)


def classify_total_result(total_runs: float, line: float) -> str:
    if total_runs > line:
        return "over"
    if total_runs < line:
        return "under"
    return "push"


def make_totals_model() -> object:
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    except Exception:
        model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, random_state=42)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def clean_totals_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "target_total_runs" not in frame.columns:
        raise ValueError("frame is missing target_total_runs")
    return frame[frame["target_total_runs"].notna()].copy()


def train_totals_model(frame: pd.DataFrame) -> TotalsModelBundle:
    train = clean_totals_frame(frame)
    if len(train) < 50:
        raise ValueError("Need at least 50 labeled games to train a totals model.")
    model = make_totals_model()
    model.fit(train[FEATURE_COLUMNS], train["target_total_runs"].astype(float))
    fitted = model.predict(train[FEATURE_COLUMNS]).astype(float)
    residual_std = float(np.std(train["target_total_runs"].astype(float).to_numpy() - fitted))
    return TotalsModelBundle(model=model, feature_columns=FEATURE_COLUMNS.copy(), train_rows=len(train), residual_std=max(residual_std, 1.5))


def predict_total_runs(bundle: TotalsModelBundle, frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.array([])
    predictions = bundle.model.predict(frame[bundle.feature_columns])
    return np.clip(predictions.astype(float), 1.0, 20.0)


def evaluate_total_predictions(y_true: pd.Series, predicted_total_runs: np.ndarray) -> dict[str, float]:
    y = y_true.astype(float).to_numpy()
    preds = predicted_total_runs.astype(float)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    metrics = {
        "games": float(len(y)),
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": rmse,
        "avg_actual_total": float(np.mean(y)),
        "avg_predicted_total": float(np.mean(preds)),
    }
    for line in [7.5, 8.5, 9.5]:
        actual = y > line
        predicted = preds > line
        metrics[f"synthetic_over_{line}_accuracy"] = float(np.mean(actual == predicted))
    return metrics


def normal_over_probability(mean: float, line: float, std: float) -> float:
    std = max(float(std), 0.25)
    z = (float(line) - float(mean)) / std
    cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def make_daily_totals_predictions(
    today_games: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame,
    statcast_team: pd.DataFrame | None,
    statcast_pitchers: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    odds: pd.DataFrame | None,
    model_bundle: TotalsModelBundle,
    lineups: pd.DataFrame | None = None,
    player_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from .features import build_prediction_frame

    frame = build_prediction_frame(today_games, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    if frame.empty:
        return pd.DataFrame()
    output = frame[
        [
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
        ]
    ].copy()
    output["predicted_total_runs"] = predict_total_runs(model_bundle, frame)
    output["total_prediction_std"] = float(model_bundle.residual_std)
    if odds is not None and not odds.empty:
        output = attach_totals_odds(output, odds)
    else:
        output["decision"] = "no odds"
        output["confidence"] = "no odds"
        output["skip_reason"] = "missing_totals_odds"
    return output.sort_values(["confidence", "ev_per_dollar"], ascending=[True, False], na_position="last")


def attach_totals_odds(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    totals_odds = latest_odds_by_game(odds)
    totals_odds = totals_odds[
        totals_odds["total_points"].notna()
        & totals_odds["over_price"].notna()
        & totals_odds["under_price"].notna()
    ].sort_values(["home_team", "away_team", "bookmaker"])
    totals_odds = totals_odds.drop_duplicates(["home_team", "away_team"], keep="first")
    odds_lookup = {(r.home_team, r.away_team): r for r in totals_odds.itertuples(index=False)}
    for row in predictions.itertuples(index=False):
        result = row._asdict()
        market = odds_lookup.get((row.home_team, row.away_team))
        if not market:
            result.update(
                {
                    "bookmaker": None,
                    "sportsbook_total_line": np.nan,
                    "over_odds": np.nan,
                    "under_odds": np.nan,
                    "projected_over_probability": np.nan,
                    "projected_under_probability": np.nan,
                    "market_over_probability": np.nan,
                    "market_under_probability": np.nan,
                    "edge": np.nan,
                    "ev_per_dollar": np.nan,
                    "decision": "no odds",
                    "confidence": "no odds",
                    "skip_reason": "missing_totals_odds",
                }
            )
            rows.append(result)
            continue
        line = float(market.total_points)
        over_prob = normal_over_probability(row.predicted_total_runs, line, row.total_prediction_std)
        under_prob = 1.0 - over_prob
        market_over, market_under = devig_two_way(market.over_price, market.under_price)
        over_ev = expected_value_per_dollar(over_prob, market.over_price)
        under_ev = expected_value_per_dollar(under_prob, market.under_price)
        if over_ev >= under_ev:
            decision = "over"
            selected_prob = over_prob
            market_prob = market_over
            odds_price = market.over_price
            ev = over_ev
        else:
            decision = "under"
            selected_prob = under_prob
            market_prob = market_under
            odds_price = market.under_price
            ev = under_ev
        edge = selected_prob - market_prob
        uncertainty = totals_uncertainty_score(row, line)
        confidence, skip_reason = apply_bet_filter(
            edge=edge,
            ev=ev,
            moneyline=odds_price,
            uncertainty_score=uncertainty,
            missing_starter=bool(pd.isna(row.home_probable_pitcher_id) or pd.isna(row.away_probable_pitcher_id)),
            missing_lineup=bool(min(row.home_lineup_confirmed, row.away_lineup_confirmed) < 1.0),
            weather_uncertain=bool((row.precipitation_in or 0.0) > 0.12 or (row.wind_speed_mph or 0.0) > 22),
            adverse_line_move=totals_line_move_for_side(odds, row.home_team, row.away_team, decision),
            bullpen_uncertainty=bool(max(row.home_bullpen_fatigue_score, row.away_bullpen_fatigue_score) > 0.85),
            thin_starter_history=bool(min(row.home_starter_games_prior, row.away_starter_games_prior) < 2),
            config=TOTALS_BET_FILTER,
        )
        if confidence == "no bet":
            decision = "no bet"
        result.update(
            {
                "bookmaker": market.bookmaker,
                "sportsbook_total_line": line,
                "over_odds": int(market.over_price),
                "under_odds": int(market.under_price),
                "projected_over_probability": over_prob,
                "projected_under_probability": under_prob,
                "market_over_probability": market_over,
                "market_under_probability": market_under,
                "bet_probability": selected_prob if decision != "no bet" else np.nan,
                "bet_odds": int(odds_price) if decision != "no bet" else np.nan,
                "edge": edge,
                "ev_per_dollar": ev,
                "uncertainty_score": uncertainty,
                "decision": decision,
                "confidence": confidence,
                "skip_reason": skip_reason,
            }
        )
        rows.append(result)
    return pd.DataFrame(rows)


def totals_uncertainty_score(row, line: float) -> float:
    distribution_width = min(max(float(row.total_prediction_std) / 5.0, 0.0), 0.6)
    near_line = 0.15 if abs(float(row.predicted_total_runs) - line) < 0.75 else 0.0
    lineup_penalty = (1.0 - min(float(row.home_lineup_confirmed), float(row.away_lineup_confirmed))) * 0.10
    starter_penalty = 0.10 if min(float(row.home_starter_games_prior), float(row.away_starter_games_prior)) < 2 else 0.0
    return float(np.clip(distribution_width + near_line + lineup_penalty + starter_penalty, 0.0, 1.0))


def totals_line_move_for_side(market_rows: pd.DataFrame, home_team: str, away_team: str, decision: str) -> float:
    group = market_rows[(market_rows["home_team"] == home_team) & (market_rows["away_team"] == away_team)].copy()
    group = group[group["total_points"].notna()]
    if group.empty or decision not in {"over", "under"}:
        return 0.0
    group["fetched_at"] = pd.to_datetime(group["fetched_at"])
    group = group.sort_values("fetched_at")
    move = float(group.iloc[-1].total_points) - float(group.iloc[0].total_points)
    if decision == "over":
        return move / 10.0
    return -move / 10.0


def total_bucket_table(y_true: pd.Series, predicted_total_runs: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"actual_total_runs": y_true.astype(float), "predicted_total_runs": predicted_total_runs})
    df["abs_error"] = (df["predicted_total_runs"] - df["actual_total_runs"]).abs()
    bins = [0, 6, 8, 10, 12, 30]
    labels = ["<=6", "6-8", "8-10", "10-12", "12+"]
    df["bucket"] = pd.cut(df["predicted_total_runs"], bins=bins, labels=labels, include_lowest=True, right=True)
    return (
        df.groupby("bucket", observed=False)
        .agg(
            games=("actual_total_runs", "size"),
            avg_predicted_total=("predicted_total_runs", "mean"),
            avg_actual_total=("actual_total_runs", "mean"),
            mae=("abs_error", "mean"),
        )
        .reset_index()
    )


def run_totals_year_backtest(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    test_year: int,
    statcast_team: pd.DataFrame | None = None,
    statcast_pitchers: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    frame = build_training_frame(games, team_stats, statcast_team, statcast_pitchers, weather)
    train = frame[(frame["season"] < test_year) & frame["target_total_runs"].notna()]
    test = frame[(frame["season"] == test_year) & frame["target_total_runs"].notna()].copy()
    training_note = "trained_on_prior_seasons"
    if train.empty and not test.empty:
        warmup_games = max(200, int(len(test) * 0.25))
        train = test.iloc[:warmup_games].copy()
        test = test.iloc[warmup_games:].copy()
        training_note = "first_year_in_season_warmup"
    if train.empty or test.empty:
        raise ValueError(f"Not enough totals data for {test_year}: train={len(train)}, test={len(test)}")

    bundle = train_totals_model(train)
    preds = predict_total_runs(bundle, test)
    metrics = evaluate_total_predictions(test["target_total_runs"], preds)
    baseline_preds = np.full(len(test), float(train["target_total_runs"].astype(float).mean()))
    baseline_metrics = evaluate_total_predictions(test["target_total_runs"], baseline_preds)
    metrics.update(
        {
            "year": test_year,
            "train_games": len(train),
            "test_games": len(test),
            "baseline_mae": baseline_metrics["mae"],
            "baseline_rmse": baseline_metrics["rmse"],
            "mae_improvement_vs_baseline": baseline_metrics["mae"] - metrics["mae"],
            "rmse_improvement_vs_baseline": baseline_metrics["rmse"] - metrics["rmse"],
            "training_note": training_note,
            "roi_note": "No totals ROI calculated unless historical totals lines or pregame snapshots are supplied.",
        }
    )
    predictions = test[
        ["game_pk", "game_date", "season", "home_team", "away_team", "target_total_runs"]
    ].copy()
    predictions["predicted_total_runs"] = preds
    predictions["error"] = predictions["predicted_total_runs"] - predictions["target_total_runs"]
    for line in [7.5, 8.5, 9.5]:
        predictions[f"synthetic_{line}_pick"] = np.where(predictions["predicted_total_runs"] > line, "over", "under")
        predictions[f"synthetic_{line}_result"] = predictions["target_total_runs"].map(lambda total: classify_total_result(total, line))
    buckets = total_bucket_table(test["target_total_runs"], preds)
    return metrics, predictions, buckets


def run_totals_backtests(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    years: list[int],
    report_dir: Path,
    statcast_team: pd.DataFrame | None = None,
    statcast_pitchers: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for year in years:
        metrics, predictions, buckets = run_totals_year_backtest(
            games, team_stats, year, statcast_team, statcast_pitchers, weather
        )
        metrics_rows.append(metrics)
        write_csv(predictions, report_dir / f"totals_predictions_{year}.csv")
        write_csv(buckets, report_dir / f"totals_calibration_{year}.csv")
    summary = pd.DataFrame(metrics_rows)
    write_csv(summary, report_dir / "totals_backtest_summary.csv")
    return summary


def save_totals_bundle(bundle: TotalsModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_totals_bundle(path: Path) -> TotalsModelBundle:
    return joblib.load(path)
