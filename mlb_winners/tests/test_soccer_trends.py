from __future__ import annotations

from datetime import date

import pandas as pd

from soccerworldcup.bet_card import must_play_candidates
from soccerworldcup.trends import (
    confidence_label,
    grade_historical_markets,
    hard_audit_ban,
    summarize_trends,
    trend_supported_candidates,
)


def matches_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": "1", "match_date": date(2026, 6, 11), "kickoff_utc": "2026-06-11T18:00:00Z", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 1, "status": "Final"},
            {"match_id": "2", "match_date": date(2026, 6, 12), "kickoff_utc": "2026-06-12T18:00:00Z", "home_team": "C", "away_team": "D", "home_score": 3, "away_score": 0, "status": "Final"},
            {"match_id": "3", "match_date": date(2026, 6, 13), "kickoff_utc": "2026-06-13T18:00:00Z", "home_team": "E", "away_team": "F", "home_score": 0, "away_score": 1, "status": "Final"},
        ]
    )


def odds_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"snapshot_date": date(2026, 6, 11), "home_team": "A", "away_team": "B", "home_price": 2.1, "draw_price": 3.2, "away_price": 3.4, "over_2_5_price": 2.0, "under_2_5_price": 1.9, "over_3_5_price": 3.0, "under_3_5_price": 1.4, "btts_yes_price": 1.8, "btts_no_price": 2.0},
            {"snapshot_date": date(2026, 6, 12), "home_team": "C", "away_team": "D", "home_price": 1.5, "draw_price": 4.0, "away_price": 6.0, "over_2_5_price": 2.2, "under_2_5_price": 1.7, "over_3_5_price": 2.8, "under_3_5_price": 1.5, "btts_yes_price": 1.9, "btts_no_price": 1.9},
            {"snapshot_date": date(2026, 6, 13), "home_team": "E", "away_team": "F", "home_price": 1.7, "draw_price": 3.8, "away_price": 5.5, "over_2_5_price": 2.1, "under_2_5_price": 1.8, "over_3_5_price": 3.1, "under_3_5_price": 1.35, "btts_yes_price": 1.7, "btts_no_price": 2.1},
        ]
    )


def test_trend_grading_for_draws_works():
    grades = grade_historical_markets(matches_fixture(), odds_fixture())
    draws = grades[grades["bucket"] == "all_draws"]

    assert len(draws) == 3
    assert draws["won"].sum() == 1


def test_over_under_grading_works():
    grades = grade_historical_markets(matches_fixture(), odds_fixture())

    assert grades[grades["bucket"].eq("over_2_5")]["won"].tolist() == [False, True, False]
    assert grades[grades["bucket"].eq("under_2_5")]["won"].tolist() == [True, False, True]


def test_btts_grading_works():
    grades = grade_historical_markets(matches_fixture(), odds_fixture())

    assert grades[grades["bucket"].eq("btts_yes")]["won"].tolist() == [True, False, False]
    assert grades[grades["bucket"].eq("btts_no")]["won"].tolist() == [False, True, True]


def test_units_and_roi_calculate_from_odds():
    grades = grade_historical_markets(matches_fixture(), odds_fixture())
    summary = summarize_trends(grades)
    under = summary[summary["bucket"] == "under_2_5"].iloc[0]

    assert under["wins"] == 2
    assert round(float(under["units_profit_loss"]), 2) == 0.70
    assert round(float(under["roi"]), 3) == 0.233


def test_small_sample_trends_are_not_actionable():
    assert confidence_label(sample=3, hit_rate=0.80, roi=0.50, break_even=0.50, longshot_share=0.0) == "watchlist"


def test_longshot_driven_trends_are_flagged_risky_or_noise():
    label = confidence_label(sample=10, hit_rate=0.50, roi=0.30, break_even=0.20, longshot_share=0.60)

    assert label == "strong but risky"


def test_trend_support_can_upgrade_no_bet_lean_to_action_lean():
    predictions = pd.DataFrame(
        [
            {"home_team": "Belgium", "away_team": "Iran", "candidate_market": "total_2_5", "candidate_selection": "Under 2.5", "candidate_odds": 2.05, "final_recommendation": "no_bet", "edge": 0.044, "confidence_score": 0.59}
        ]
    )
    trends = pd.DataFrame([{"bucket": "under_2_5", "confidence_label": "actionable support"}])

    candidates = trend_supported_candidates(predictions, trends, [])

    assert candidates.iloc[0]["trend_adjusted_verdict"] == "action lean"


def test_trend_support_cannot_override_hard_audit_bans():
    predictions = pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "candidate_market": "total_2_5", "candidate_selection": "Over 2.5", "candidate_odds": 2.2, "final_recommendation": "no_bet", "edge": 0.08, "confidence_score": 0.70}
        ]
    )
    trends = pd.DataFrame([{"bucket": "over_2_5", "confidence_label": "actionable support"}])

    candidates = trend_supported_candidates(predictions, trends, ["Over rate among total recommendations is high: 50.0%"])

    assert candidates.iloc[0]["trend_adjusted_verdict"] == "no bet"
    assert candidates.iloc[0]["hard_ban"] == "over warning"


def test_trend_supported_must_play_avoids_extreme_longshots():
    rows = pd.DataFrame(
        [
            {"home_team": "Uruguay", "away_team": "Cabo Verde", "candidate_market": "1x2", "candidate_selection": "Cabo Verde", "candidate_odds": 8.25, "final_recommendation": "no_bet", "edge": 0.08, "confidence_score": 0.70, "trend_support_score": 3.0}
        ]
    )

    candidates, rejected = must_play_candidates(rows, [])

    assert candidates.empty
    assert any("extreme longshot" in item for item in rejected)


def test_hard_audit_ban_lottery_draw():
    assert hard_audit_ban({"market": "1x2", "pick": "Draw", "odds": 9.25}, []) == "lottery draw"
