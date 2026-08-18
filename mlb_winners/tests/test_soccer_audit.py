from __future__ import annotations

from datetime import date

import pandas as pd

from soccerworldcup.audit import audit_predictions
from soccerworldcup.predict import apply_final_recommendation_gates


def fake_prediction_rows(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "created_at": pd.Timestamp("2026-06-20T12:00:00Z"),
        "match_date": date(2026, 6, 20),
        "market": "1x2",
        "selection": "no bet",
        "final_recommendation": "no bet",
        "odds": pd.NA,
        "confidence": "no bet",
        "guardrails_triggered": "",
        "skip_reason": "",
        "raw_model_prob": 0.50,
        "market_prob": 0.48,
        "calibrated_prob": 0.49,
        "draw_prob": 0.25,
        "raw_expected_total_goals": 2.45,
        "calibrated_expected_total_goals": 2.45,
        "edge": 0.01,
        "home_team": "Home",
        "away_team": "Away",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_audit_counts_final_bets_and_no_bets():
    result = audit_predictions(
        fake_prediction_rows(
            [
                {"match_id": "1", "selection": "Home", "final_recommendation": "Home", "odds": 1.80, "confidence": "medium"},
                {"match_id": "2", "selection": "no bet", "final_recommendation": "no bet"},
                {"match_id": "3", "market": "total_2_5", "selection": "Under 2.5", "final_recommendation": "Under 2.5", "odds": 2.10, "confidence": "thin"},
            ]
        ),
        date(2026, 6, 20),
    )

    assert result.run_summary["total_matches"] == 3
    assert result.run_summary["final_bets"] == 2
    assert result.run_summary["no_bets"] == 1
    assert result.pick_distribution["favorite_ml_count"] == 1
    assert result.pick_distribution["under_count"] == 1
    assert result.pick_distribution["no_bet_count"] == 1


def test_underdog_ml_warning_triggers_when_rate_too_high():
    result = audit_predictions(
        fake_prediction_rows(
            [
                {"match_id": "1", "selection": "Away", "final_recommendation": "Away", "odds": 4.00, "confidence": "medium"},
                {"match_id": "2", "selection": "Home", "final_recommendation": "Home", "odds": 1.70, "confidence": "medium"},
            ]
        ),
        date(2026, 6, 20),
    )

    assert any("Underdog ML rate" in warning for warning in result.warnings)


def test_over_warning_triggers_when_over_rate_too_high():
    result = audit_predictions(
        fake_prediction_rows(
            [
                {"match_id": "1", "market": "total_2_5", "selection": "Over 2.5", "final_recommendation": "Over 2.5", "odds": 2.10, "confidence": "medium", "edge": 0.08},
                {"match_id": "2", "market": "total_2_5", "selection": "Under 2.5", "final_recommendation": "Under 2.5", "odds": 2.05, "confidence": "medium"},
            ]
        ),
        date(2026, 6, 20),
    )

    assert any("Over rate" in warning for warning in result.warnings)


def test_draw_probability_warning_triggers_when_average_too_low():
    result = audit_predictions(
        fake_prediction_rows(
            [
                {"match_id": "1", "draw_prob": 0.18},
                {"match_id": "2", "draw_prob": 0.19},
            ]
        ),
        date(2026, 6, 20),
    )

    assert any("Average draw probability is low" in warning for warning in result.warnings)


def test_no_bet_warning_triggers_when_too_high_or_too_low():
    high = audit_predictions(
        fake_prediction_rows(
                [
                    {"match_id": "1"},
                    {"match_id": "2"},
                    {"match_id": "3"},
                    {"match_id": "4"},
                    {"match_id": "5"},
                    {"match_id": "6"},
                    {"match_id": "7"},
                    {"match_id": "8", "selection": "Home", "final_recommendation": "Home", "odds": 1.70, "confidence": "medium"},
                ]
            ),
        date(2026, 6, 20),
    )
    low = audit_predictions(
        fake_prediction_rows(
            [
                {"match_id": "1", "selection": "Home", "final_recommendation": "Home", "odds": 1.70, "confidence": "medium"},
                {"match_id": "2", "selection": "Away", "final_recommendation": "Away", "odds": 2.10, "confidence": "medium"},
                {"match_id": "3"},
            ]
        ),
        date(2026, 6, 20),
    )

    assert any("No-bet rate is very high" in warning for warning in high.warnings)
    assert any("No-bet rate is too low" in warning for warning in low.warnings)


def test_guardrail_counts_parse_and_normalize_names():
    result = audit_predictions(
        fake_prediction_rows(
            [
                {
                    "match_id": "1",
                    "guardrails_triggered": "early_group_market_anchor,missing_weather_total,big_underdog_ml,over_total_not_clear_market",
                    "skip_reason": "edge_below_threshold_0.065",
                }
            ]
        ),
        date(2026, 6, 20),
    )

    assert result.guardrail_counts["early_group_market_anchor"] == 1
    assert result.guardrail_counts["early_wc_market_anchor"] == 1
    assert result.guardrail_counts["missing_weather_total_penalty"] == 1
    assert result.guardrail_counts["underdog_threshold"] == 1
    assert result.guardrail_counts["over_requires_extra_edge"] == 1
    assert result.guardrail_counts["low_edge"] == 1


def gating_rows(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "match_id": "m",
        "home_team": "Home",
        "away_team": "Away",
        "market": "1x2",
        "selection": "Home",
        "candidate_selection": "Home",
        "pre_gate_recommendation": "Home",
        "final_recommendation": "Home",
        "odds": 1.80,
        "candidate_odds": 1.80,
        "confidence": "medium",
        "edge": 0.08,
        "calibrated_prob": 0.55,
        "market_prob": 0.47,
        "guardrails_triggered": "",
        "skip_reason": "",
        "uncertainty_score": 0.10,
        "elo_diff": 50.0,
        "wc_matches_used_home": 2,
        "wc_matches_used_away": 2,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_final_gate_demotes_underdog_below_confidence_threshold():
    rows = gating_rows(
        [
            {
                "match_id": "dog",
                "selection": "Away",
                "pre_gate_recommendation": "Away",
                "final_recommendation": "Away",
                "odds": 3.20,
                "candidate_odds": 3.20,
                "confidence": "thin",
                "edge": 0.09,
            }
        ]
    )

    gated = apply_final_recommendation_gates(rows)

    assert gated.iloc[0]["selection"] == "no bet"
    assert "final_confidence_threshold" in gated.iloc[0]["guardrails_triggered"]


def test_final_gate_demotes_big_underdog_below_big_dog_threshold():
    rows = gating_rows(
        [
            {
                "match_id": "bigdog",
                "selection": "Away",
                "pre_gate_recommendation": "Away",
                "final_recommendation": "Away",
                "odds": 5.50,
                "candidate_odds": 5.50,
                "confidence": "medium",
                "edge": 0.12,
            }
        ]
    )

    gated = apply_final_recommendation_gates(rows)

    assert gated.iloc[0]["selection"] == "no bet"
    assert "final_confidence_threshold" in gated.iloc[0]["guardrails_triggered"]


def test_slate_recommendation_cap_demotes_extra_picks():
    rows = gating_rows(
        [
            {"match_id": "1", "selection": "Home", "pre_gate_recommendation": "Home", "final_recommendation": "Home", "edge": 0.12},
            {"match_id": "2", "selection": "Home", "pre_gate_recommendation": "Home", "final_recommendation": "Home", "edge": 0.10},
            {"match_id": "3", "selection": "Home", "pre_gate_recommendation": "Home", "final_recommendation": "Home", "edge": 0.08},
            {"match_id": "4", "selection": "Home", "pre_gate_recommendation": "Home", "final_recommendation": "Home", "edge": 0.06},
            {"match_id": "5", "selection": "no bet", "pre_gate_recommendation": "no_bet", "final_recommendation": "no_bet", "odds": pd.NA},
        ]
    )

    gated = apply_final_recommendation_gates(rows)

    assert (gated["selection"] != "no bet").sum() == 2
    assert gated["guardrails_triggered"].str.contains("slate_recommendation_cap", na=False).sum() >= 1


def test_slate_underdog_rate_cap_demotes_lowest_ranked_underdogs():
    rows = gating_rows(
        [
            {"match_id": "fav1", "selection": "Home", "pre_gate_recommendation": "Home", "final_recommendation": "Home", "odds": 1.70, "candidate_odds": 1.70, "edge": 0.12},
            {"match_id": "fav2", "selection": "Home", "pre_gate_recommendation": "Home", "final_recommendation": "Home", "odds": 1.80, "candidate_odds": 1.80, "edge": 0.11},
            {"match_id": "fav3", "selection": "Home", "pre_gate_recommendation": "Home", "final_recommendation": "Home", "odds": 1.90, "candidate_odds": 1.90, "edge": 0.10},
            {"match_id": "fav4", "selection": "Home", "pre_gate_recommendation": "Home", "final_recommendation": "Home", "odds": 2.00, "candidate_odds": 2.00, "edge": 0.09},
            {"match_id": "dog1", "selection": "Away", "pre_gate_recommendation": "Away", "final_recommendation": "Away", "odds": 3.00, "candidate_odds": 3.00, "edge": 0.15, "confidence": "strong"},
            {"match_id": "dog2", "selection": "Away", "pre_gate_recommendation": "Away", "final_recommendation": "Away", "odds": 3.20, "candidate_odds": 3.20, "edge": 0.08, "confidence": "strong"},
        ]
    )

    gated = apply_final_recommendation_gates(rows)
    final = gated[gated["selection"] != "no bet"]
    dogs = final[(final["market"] == "1x2") & (final["odds"] > 2.25)]

    assert len(dogs) / len(final) <= 0.20
    assert gated["guardrails_triggered"].str.contains("slate_underdog_rate_cap", na=False).sum() >= 1


def test_slate_no_bet_floor_enforces_half_no_bets():
    rows = gating_rows(
        [
            {"match_id": "1", "selection": "Home", "pre_gate_recommendation": "Home", "final_recommendation": "Home", "edge": 0.15},
        ]
    )

    gated = apply_final_recommendation_gates(rows)
    no_bet_rate = (gated["selection"] == "no bet").mean()

    assert no_bet_rate >= 0.50
    assert gated["guardrails_triggered"].str.contains("slate_no_bet_floor", na=False).sum() >= 1
