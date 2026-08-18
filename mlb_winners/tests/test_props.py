import numpy as np
import pandas as pd

from mlb_winners.props import (
    attach_prop_lines,
    build_pitcher_k_training_frame,
    conservative_prop_probability,
    expected_batters_faced,
    pitcher_k_ml_gate,
    normal_over_probability,
    PITCHER_K_FEATURES,
    pitcher_k_feature_values,
    pitcher_strikeout_probability,
    poisson_over_probability,
    projected_starter_ip_from_context,
)


def test_pitcher_strikeout_components_are_bounded_and_directional():
    low = pitcher_strikeout_probability(pitcher_kbb=1.0, opponent_k_rate=0.18, starter_velocity=90.0, starter_spin=2100)
    high = pitcher_strikeout_probability(pitcher_kbb=5.0, opponent_k_rate=0.28, starter_velocity=96.0, starter_spin=2450)

    assert 0.12 <= low <= 0.40
    assert 0.12 <= high <= 0.40
    assert high > low
    assert expected_batters_faced(6.0, workload_trend=20) > expected_batters_faced(5.0, workload_trend=-20)


def test_poisson_over_probability_handles_half_lines():
    assert poisson_over_probability(6.0, 4.5) > poisson_over_probability(3.0, 4.5)


def test_normal_over_probability_handles_half_lines():
    assert normal_over_probability(6.0, 4.5) > normal_over_probability(3.0, 4.5)


def test_conservative_prop_probability_caps_formula_extremes():
    assert conservative_prop_probability(0.99) == 0.88
    assert conservative_prop_probability(0.01) == 0.12
    assert conservative_prop_probability(0.55) == 0.55


def test_attach_prop_lines_calculates_over_value():
    props = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2026-05-22",
                "home_team": "Home",
                "away_team": "Away",
                "team": "Home",
                "opponent": "Away",
                "market": "strikeouts",
                "odds_market": "pitcher_strikeouts",
                "player_id": 100,
                "player_name": "Jane Pitcher",
                "projection": 7.0,
                "line": 4.5,
                "over_probability": 0.0,
                "under_probability": 0.0,
                "lineup_required": False,
                "lineup_confirmed": 1.0,
                "missing_starter": False,
                "thin_history": False,
                "weather_uncertain": False,
                "bullpen_uncertain": False,
                "uncertainty_score": 0.25,
            }
        ]
    )
    lines = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-05-22",
                "fetched_at": "2026-05-22T17:00:00Z",
                "home_team": "Home",
                "away_team": "Away",
                "bookmaker": "book",
                "player_name": "Jane Pitcher",
                "market": "pitcher_strikeouts",
                "line": 4.5,
                "over_odds": -110,
                "under_odds": -110,
            }
        ]
    )

    out = attach_prop_lines(props, lines, "strikeouts")

    assert out.loc[0, "decision"] == "over"
    assert out.loc[0, "over_probability"] > out.loc[0, "market_over_probability"]
    assert out.loc[0, "ev_per_dollar"] > 0


def test_attach_prop_lines_skips_short_workload_role():
    props = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2026-05-22",
                "home_team": "Home",
                "away_team": "Away",
                "team": "Home",
                "opponent": "Away",
                "market": "strikeouts",
                "odds_market": "pitcher_strikeouts",
                "player_id": 100,
                "player_name": "Opener Pitcher",
                "projection": 5.2,
                "line": 4.5,
                "over_probability": 0.0,
                "under_probability": 0.0,
                "lineup_required": False,
                "lineup_confirmed": 1.0,
                "missing_starter": False,
                "thin_history": False,
                "weather_uncertain": False,
                "bullpen_uncertain": False,
                "uncertainty_score": 0.25,
                "short_workload_role": True,
            }
        ]
    )
    lines = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-05-22",
                "fetched_at": "2026-05-22T17:00:00Z",
                "home_team": "Home",
                "away_team": "Away",
                "bookmaker": "book",
                "player_name": "Opener Pitcher",
                "market": "pitcher_strikeouts",
                "line": 4.5,
                "over_odds": -110,
                "under_odds": -110,
            }
        ]
    )

    out = attach_prop_lines(props, lines, "strikeouts")

    assert out.loc[0, "decision"] == "no bet"
    assert out.loc[0, "skip_reason"] == "short_workload_role"


def test_projected_starter_ip_uses_role_and_quality():
    ace = projected_starter_ip_from_context(fip=2.8, workload_trend=10, season_starts=12, season_era=2.4, opponent_xwoba=0.300)
    spot = projected_starter_ip_from_context(fip=5.4, workload_trend=-20, season_starts=1, season_era=5.8, opponent_xwoba=0.360)

    assert ace > spot
    assert 1.4 <= spot < 3.0
    assert 3.8 <= ace <= 7.1


def test_pitcher_k_feature_values_are_leakage_safe_shape():
    row = type(
        "Row",
        (),
        {
            "game_date": pd.Timestamp("2026-06-10"),
            "home_probable_pitcher_id": 1,
            "home_starter_kbb_prior": 4.0,
            "away_team_k_rate_last14": 0.26,
            "home_starter_velocity": 95.0,
            "home_starter_spin_rate": 2400,
            "home_starter_fip_proxy": 3.2,
            "home_starter_last5_fip": 2.9,
            "home_starter_season_era": 2.7,
            "home_starter_season_whip": 1.05,
            "home_starter_season_fip_proxy": 3.0,
            "home_starter_season_starts": 10,
            "home_starter_rest_days": 5,
            "home_starter_workload_trend": 8,
            "away_team_xwoba_last14": 0.310,
            "park_run_factor": 1.0,
            "temperature_f": 72.0,
            "wind_speed_mph": 3.0,
        },
    )()

    values = pitcher_k_feature_values(row, "home", "away", pd.DataFrame())

    assert values["expected_batters_faced"] > 0
    assert 0.12 <= values["projected_k_rate"] <= 0.38
    assert values["starter_season_starts"] == 10


def test_pitcher_k_training_target_uses_numeric_starter_so():
    feature_frame = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": pd.Timestamp("2025-06-01"),
                "target_home_win": 1,
                "home_probable_pitcher_id": 10,
                "home_probable_pitcher": "Starter A",
                "away_probable_pitcher_id": 20,
                "away_probable_pitcher": "Starter B",
                "home_starter_kbb_prior": 3.5,
                "away_starter_kbb_prior": 2.5,
                "away_team_k_rate_last14": 0.24,
                "home_team_k_rate_last14": 0.21,
                "home_starter_fip_proxy": 3.6,
                "away_starter_fip_proxy": 4.1,
                "home_starter_games_prior": 8,
                "away_starter_games_prior": 8,
            }
        ]
    )
    team_stats = pd.DataFrame(
        [
            {"game_pk": 1, "game_date": "2025-06-01", "side": "home", "starter_id": 10, "starter_ip": 6.0, "starter_so": 8},
            {"game_pk": 1, "game_date": "2025-06-01", "side": "away", "starter_id": 20, "starter_ip": 5.0, "starter_so": 3},
        ]
    )

    train = build_pitcher_k_training_frame(feature_frame, team_stats, pd.DataFrame())

    assert train["target_strikeouts"].dtype.kind in {"f", "i"}
    assert sorted(train["target_strikeouts"].tolist()) == [3.0, 8.0]
    assert train["target_strikeouts"].max() > 5


def test_pitcher_k_feature_column_order_is_stable():
    assert PITCHER_K_FEATURES[:4] == [
        "expected_batters_faced",
        "projected_k_rate",
        "projected_starter_ip",
        "pitcher_kbb",
    ]
    assert len(PITCHER_K_FEATURES) == len(dict.fromkeys(PITCHER_K_FEATURES))


def test_pitcher_k_constant_prediction_guard_disables_xgb():
    enabled, reason = pitcher_k_ml_gate(
        xgb_mae=2.0,
        baseline_mae=3.0,
        prediction_std=0.01,
        predictions=np.array([0.5, 0.5, 0.5]),
    )

    assert not enabled
    assert "near-zero" in reason


def test_baseline_pitcher_k_projection_is_realistic_nonzero():
    row = type(
        "Row",
        (),
        {
            "game_date": pd.Timestamp("2026-06-10"),
            "home_probable_pitcher_id": 1,
            "home_starter_kbb_prior": 4.0,
            "away_team_k_rate_last14": 0.26,
            "home_starter_fip_proxy": 3.2,
            "home_starter_season_era": 2.7,
            "home_starter_season_starts": 10,
            "away_team_xwoba_last14": 0.310,
        },
    )()

    values = pitcher_k_feature_values(row, "home", "away", pd.DataFrame(), pd.DataFrame())
    projection = values["expected_batters_faced"] * values["projected_k_rate"]

    assert 2.0 <= projection <= 9.5
    assert projection != 0.5


def test_missing_pitcher_k_features_trip_guard():
    row = type("Row", (), {"game_date": pd.Timestamp("2026-06-10"), "home_probable_pitcher_id": 1})()

    values = pitcher_k_feature_values(row, "home", "away", pd.DataFrame(), pd.DataFrame())

    assert values["feature_missing_rate"] > 0.25
