from __future__ import annotations

from pathlib import Path

import pandas as pd

from .db import write_csv
from .features import build_training_frame
from .modeling import calibration_table, evaluate_predictions, predict_home_prob, train_baseline, train_model


def run_year_backtest(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    test_year: int,
    statcast_team: pd.DataFrame | None = None,
    statcast_pitchers: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    frame = build_training_frame(games, team_stats, statcast_team, statcast_pitchers, weather)
    train = frame[(frame["season"] < test_year) & frame["target_home_win"].notna()]
    test = frame[(frame["season"] == test_year) & frame["target_home_win"].notna()].copy()
    training_note = "trained_on_prior_seasons"
    if train.empty and not test.empty:
        warmup_games = max(200, int(len(test) * 0.25))
        train = test.iloc[:warmup_games].copy()
        test = test.iloc[warmup_games:].copy()
        training_note = "first_year_in_season_warmup"
    if train.empty or test.empty:
        raise ValueError(f"Not enough data for {test_year}: train={len(train)}, test={len(test)}")

    bundle = train_model(train)
    baseline = train_baseline(train)
    probs = predict_home_prob(bundle, test)
    baseline_probs = baseline.predict_proba(test[baseline.feature_names_in_ if hasattr(baseline, "feature_names_in_") else bundle.feature_columns])[:, 1]
    metrics = evaluate_predictions(test["target_home_win"], probs)
    baseline_metrics = evaluate_predictions(test["target_home_win"], baseline_probs)
    metrics.update(
        {
            "year": test_year,
            "train_games": len(train),
            "test_games": len(test),
            "baseline_log_loss": baseline_metrics["log_loss"],
            "baseline_brier": baseline_metrics["brier"],
            "baseline_accuracy": baseline_metrics["accuracy"],
            "training_note": training_note,
            "roi_note": "No ROI calculated unless historical moneyline odds CSVs are supplied.",
        }
    )
    predictions = test[
        ["game_pk", "game_date", "season", "home_team", "away_team", "target_home_win"]
    ].copy()
    predictions["model_home_prob"] = probs
    predictions["model_pick"] = predictions["model_home_prob"].ge(0.5).map({True: "home", False: "away"})
    calibration = calibration_table(test["target_home_win"], probs)
    return metrics, predictions, calibration


def run_backtests(
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
        metrics, predictions, calibration = run_year_backtest(
            games, team_stats, year, statcast_team, statcast_pitchers, weather
        )
        metrics_rows.append(metrics)
        write_csv(predictions, report_dir / f"predictions_{year}.csv")
        write_csv(calibration, report_dir / f"calibration_{year}.csv")
    summary = pd.DataFrame(metrics_rows)
    write_csv(summary, report_dir / "backtest_summary.csv")
    return summary
