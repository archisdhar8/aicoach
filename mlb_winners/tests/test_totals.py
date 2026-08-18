import numpy as np
import pandas as pd
import pytest

from mlb_winners.totals import attach_totals_odds, classify_total_result, evaluate_total_predictions, normal_over_probability, total_bucket_table


def test_classify_total_result_handles_pushes():
    assert classify_total_result(9, 8.5) == "over"
    assert classify_total_result(7, 8.5) == "under"
    assert classify_total_result(8, 8) == "push"


def test_evaluate_total_predictions_reports_error_and_synthetic_direction():
    metrics = evaluate_total_predictions(pd.Series([8, 10, 5]), np.array([8.5, 9.0, 6.0]))

    assert metrics["games"] == 3
    assert metrics["mae"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.25 + 1.0 + 1.0) / 3))
    assert metrics["synthetic_over_8.5_accuracy"] == pytest.approx(1.0)


def test_total_bucket_table_has_expected_columns():
    table = total_bucket_table(pd.Series([5, 8, 11]), np.array([5.5, 8.2, 11.5]))

    assert {"bucket", "games", "avg_predicted_total", "avg_actual_total", "mae"}.issubset(table.columns)
    assert table["games"].sum() == 3


def test_normal_over_probability_increases_with_projection():
    low = normal_over_probability(mean=7.0, line=8.5, std=3.0)
    high = normal_over_probability(mean=10.0, line=8.5, std=3.0)

    assert low < 0.5
    assert high > 0.5
    assert high > low


def test_attach_totals_odds_calculates_value_side():
    predictions = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2026-05-21",
                "home_team": "Home",
                "away_team": "Away",
                "home_probable_pitcher_id": 100,
                "away_probable_pitcher_id": 200,
                "home_starter_games_prior": 5,
                "away_starter_games_prior": 5,
                "home_lineup_confirmed": 1.0,
                "away_lineup_confirmed": 1.0,
                "home_bullpen_fatigue_score": 0.2,
                "away_bullpen_fatigue_score": 0.2,
                "precipitation_in": 0.0,
                "wind_speed_mph": 5.0,
                "predicted_total_runs": 11.2,
                "total_prediction_std": 2.5,
            }
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-05-21",
                "fetched_at": "2026-05-21T17:00:00Z",
                "home_team": "Home",
                "away_team": "Away",
                "bookmaker": "book",
                "total_points": 8.5,
                "over_price": -110,
                "under_price": -110,
            }
        ]
    )

    out = attach_totals_odds(predictions, odds)

    assert out.loc[0, "decision"] == "over"
    assert out.loc[0, "projected_over_probability"] > out.loc[0, "market_over_probability"]
    assert out.loc[0, "ev_per_dollar"] > 0
