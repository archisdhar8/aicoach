from datetime import date

import pandas as pd
import pytest

from mlb_winners.market import (
    apply_bet_filter,
    apply_moneyline_tier_filter,
    dynamic_edge_threshold,
    market_adjusted_probability,
    market_report,
)


def test_dynamic_threshold_higher_for_favorites_and_uncertainty():
    assert dynamic_edge_threshold(-150, 0.2) > dynamic_edge_threshold(130, 0.2)
    assert dynamic_edge_threshold(130, 0.6) > dynamic_edge_threshold(130, 0.2)


def test_bet_filter_skip_reasons():
    confidence, reason = apply_bet_filter(
        edge=0.08,
        ev=0.12,
        moneyline=120,
        uncertainty_score=0.2,
        missing_starter=True,
        missing_lineup=False,
        weather_uncertain=False,
        adverse_line_move=0.0,
        bullpen_uncertainty=False,
    )
    assert confidence == "no bet"
    assert reason == "missing_starter"


def test_bet_filter_blocks_thin_starter_history():
    confidence, reason = apply_bet_filter(
        edge=0.10,
        ev=0.13,
        moneyline=-150,
        uncertainty_score=0.2,
        missing_starter=False,
        missing_lineup=False,
        weather_uncertain=False,
        adverse_line_move=0.0,
        bullpen_uncertainty=False,
        thin_starter_history=True,
    )
    assert confidence == "no bet"
    assert reason == "thin_starter_history"


def test_market_adjusted_probability_anchors_to_market():
    adjusted = market_adjusted_probability(0.70, 0.50)

    assert adjusted == pytest.approx(0.56)


def test_bet_filter_blocks_longshot_and_dislocation_edges():
    confidence, reason = apply_bet_filter(
        edge=0.20,
        ev=1.5,
        moneyline=800,
        uncertainty_score=0.2,
        missing_starter=False,
        missing_lineup=False,
        weather_uncertain=False,
        adverse_line_move=0.0,
        bullpen_uncertainty=False,
    )
    assert confidence == "no bet"
    assert reason == "longshot_market_outlier"

    confidence, reason = apply_bet_filter(
        edge=0.19,
        ev=0.40,
        moneyline=120,
        uncertainty_score=0.2,
        missing_starter=False,
        missing_lineup=False,
        weather_uncertain=False,
        adverse_line_move=0.0,
        bullpen_uncertainty=False,
    )
    assert confidence == "no bet"
    assert reason == "market_model_dislocation"


def test_bet_filter_allows_normal_priced_medium_edge():
    confidence, reason = apply_bet_filter(
        edge=0.061,
        ev=0.08,
        moneyline=110,
        uncertainty_score=0.2,
        missing_starter=False,
        missing_lineup=False,
        weather_uncertain=False,
        adverse_line_move=0.0,
        bullpen_uncertainty=False,
    )

    assert confidence == "medium"
    assert reason == ""


def test_moneyline_tier_filter_assigns_official_medium_stake():
    result = apply_moneyline_tier_filter(
        edge=0.056,
        ev=0.05,
        moneyline=110,
        uncertainty_score=0.2,
        missing_starter=False,
        missing_lineup=False,
        weather_uncertain=False,
        adverse_line_move=0.0,
        bullpen_uncertainty=False,
    )

    assert result.confidence == "medium"
    assert result.tier == "official"
    assert result.stake_units == pytest.approx(1.0)
    assert result.risk_flags == ()


def test_moneyline_tier_filter_downgrades_missing_lineup_to_watchlist():
    result = apply_moneyline_tier_filter(
        edge=0.06,
        ev=0.05,
        moneyline=120,
        uncertainty_score=0.2,
        missing_starter=False,
        missing_lineup=True,
        weather_uncertain=False,
        adverse_line_move=0.0,
        bullpen_uncertainty=False,
    )

    assert result.confidence == "watchlist"
    assert result.tier == "watchlist"
    assert result.stake_units == pytest.approx(0.25)
    assert "missing_lineup" in result.risk_flags


def test_moneyline_tier_filter_downgrades_bullpen_uncertainty_to_watchlist():
    result = apply_moneyline_tier_filter(
        edge=0.07,
        ev=0.06,
        moneyline=-125,
        uncertainty_score=0.2,
        missing_starter=False,
        missing_lineup=False,
        weather_uncertain=False,
        adverse_line_move=0.0,
        bullpen_uncertainty=True,
    )

    assert result.confidence == "watchlist"
    assert "bullpen_uncertainty" in result.risk_flags


def test_moneyline_tier_filter_keeps_longshot_and_dislocation_no_bet():
    longshot = apply_moneyline_tier_filter(
        edge=0.08,
        ev=0.12,
        moneyline=300,
        uncertainty_score=0.2,
        missing_starter=False,
        missing_lineup=False,
        weather_uncertain=False,
        adverse_line_move=0.0,
        bullpen_uncertainty=False,
    )
    dislocation = apply_moneyline_tier_filter(
        edge=0.19,
        ev=0.12,
        moneyline=120,
        uncertainty_score=0.2,
        missing_starter=False,
        missing_lineup=False,
        weather_uncertain=False,
        adverse_line_move=0.0,
        bullpen_uncertainty=False,
    )

    assert longshot.confidence == "no bet"
    assert longshot.skip_reason == "longshot_market_outlier"
    assert dislocation.confidence == "no bet"
    assert dislocation.skip_reason == "market_model_dislocation"


def test_market_report_tracks_line_movement():
    odds = pd.DataFrame(
        [
            {
                "snapshot_date": date(2026, 5, 19),
                "fetched_at": "2026-05-19T12:00:00Z",
                "home_team": "A",
                "away_team": "B",
                "bookmaker": "book",
                "home_moneyline": -110,
                "away_moneyline": 100,
            },
            {
                "snapshot_date": date(2026, 5, 19),
                "fetched_at": "2026-05-19T18:00:00Z",
                "home_team": "A",
                "away_team": "B",
                "bookmaker": "book",
                "home_moneyline": -140,
                "away_moneyline": 120,
            },
        ]
    )
    report = market_report(odds, date(2026, 5, 19))
    assert report.iloc[0]["snapshots"] == 2
    assert report.iloc[0]["home_line_move_prob"] > 0
    assert report.iloc[0]["line_velocity_per_hour"] == pytest.approx(report.iloc[0]["home_line_move_prob"] / 6)
