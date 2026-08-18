import pandas as pd
import pytest

from mlb_winners.player_prop_qualifier import (
    HRR_MARKET,
    PropQualificationConfig,
    attach_best_prop_price,
    build_qualified_parlays,
    hrr_projection_probability,
    model_probability_for_line,
    parlay_correlation_adjustment,
)


def test_hrr_projection_uses_lineup_slot_and_quality_inputs():
    weak_projection, weak_probability = hrr_projection_probability(
        batting_order=8,
        player_summary={"pa": 400, "hrr": 0.34},
        player_statcast={"xwoba": 0.280, "xba": 0.210, "hard_hit_rate": 0.30, "barrel_rate": 0.04},
        team_context={"runs_per_game": 3.7},
        opponent_context={"starter_era": 3.2},
        game_context={"park_run_factor": 0.96},
        line=1.5,
    )
    strong_projection, strong_probability = hrr_projection_probability(
        batting_order=1,
        player_summary={"pa": 400, "hrr": 0.62},
        player_statcast={"xwoba": 0.390, "xba": 0.295, "hard_hit_rate": 0.48, "barrel_rate": 0.12},
        team_context={"runs_per_game": 5.3},
        opponent_context={"starter_era": 5.4},
        game_context={"park_run_factor": 1.08},
        line=1.5,
    )

    assert strong_projection > weak_projection
    assert strong_probability > weak_probability
    assert 0.08 <= weak_probability <= 0.82
    assert 0.08 <= strong_probability <= 0.82


def test_attach_best_prop_price_qualifies_best_available_hrr_price():
    candidates = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2026-08-12",
                "home_team": "Home",
                "away_team": "Away",
                "team": "Home",
                "opponent": "Away",
                "player_id": 7,
                "player_name": "Good Hitter",
                "market": HRR_MARKET,
                "odds_market": "batter_hits_runs_rbis",
                "prop": "Hits + Runs + RBIs over",
                "line": 1.5,
                "projection": 2.05,
                "model_probability": 0.60,
                "lineup_confirmed": 1.0,
                "data_quality": 0.92,
            }
        ]
    )
    lines = pd.DataFrame(
        [
            {
                "fetched_at": pd.Timestamp.now("UTC"),
                "home_team": "Home",
                "away_team": "Away",
                "bookmaker": "worsebook",
                "player_name": "Good Hitter",
                "market": "batter_hits_runs_rbis",
                "line": 1.5,
                "over_odds": -140,
                "under_odds": 120,
            },
            {
                "fetched_at": pd.Timestamp.now("UTC"),
                "home_team": "Home",
                "away_team": "Away",
                "bookmaker": "bestbook",
                "player_name": "Good Hitter",
                "market": "batter_hits_runs_rbis",
                "line": 1.5,
                "over_odds": 125,
                "under_odds": -145,
            },
        ]
    )

    out = attach_best_prop_price(candidates, lines, config=PropQualificationConfig(max_odds_age_hours=9999))

    assert bool(out.loc[0, "qualified"]) is True
    assert out.loc[0, "sportsbook"] == "bestbook"
    assert out.loc[0, "edge"] > 0.04
    assert out.loc[0, "ev_per_dollar"] > 0.05


def test_attach_best_prop_price_rejects_stale_or_no_edge():
    candidates = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2026-08-12",
                "home_team": "Home",
                "away_team": "Away",
                "team": "Home",
                "opponent": "Away",
                "player_id": 7,
                "player_name": "Fair Hitter",
                "market": HRR_MARKET,
                "odds_market": "batter_hits_runs_rbis",
                "prop": "Hits + Runs + RBIs over",
                "line": 1.5,
                "projection": 1.1,
                "model_probability": 0.40,
                "lineup_confirmed": 1.0,
                "data_quality": 0.95,
            }
        ]
    )
    lines = pd.DataFrame(
        [
            {
                "fetched_at": "2026-01-01T00:00:00Z",
                "home_team": "Home",
                "away_team": "Away",
                "bookmaker": "book",
                "player_name": "Fair Hitter",
                "market": "batter_hits_runs_rbis",
                "line": 1.5,
                "over_odds": -150,
                "under_odds": 130,
            }
        ]
    )

    out = attach_best_prop_price(candidates, lines, config=PropQualificationConfig(max_odds_age_hours=1))

    assert bool(out.loc[0, "qualified"]) is False
    assert "stale odds" in out.loc[0, "rejection_reason"]
    assert "no price edge" in out.loc[0, "rejection_reason"]


def test_pitcher_k_probability_recomputed_for_each_sportsbook_line():
    candidate = {"market": "strikeouts", "projection": 6.0, "strikeout_distribution_mean": 6.0}

    lower = model_probability_for_line(candidate, {"line": 4.5})
    higher = model_probability_for_line(candidate, {"line": 6.5})

    assert lower > higher
    assert 0.12 <= higher <= 0.88


def test_parlay_haircuts_negative_same_game_pitcher_hitter_relationship():
    pitcher = {
        "game_pk": 10,
        "market": "strikeouts",
        "team": "Home",
        "opponent": "Away",
        "player_name": "Starter",
        "prop": "Pitcher strikeouts over",
        "line": 5.5,
        "model_probability": 0.58,
        "bet_odds": 120,
    }
    hitter = {
        "game_pk": 10,
        "market": HRR_MARKET,
        "team": "Away",
        "opponent": "Home",
        "player_name": "Opposing Hitter",
        "prop": "Hits + Runs + RBIs over",
        "line": 1.5,
        "model_probability": 0.57,
        "bet_odds": 120,
    }

    note, adjustment = parlay_correlation_adjustment((pitcher, hitter))
    parlays = build_qualified_parlays(pd.DataFrame([pitcher, hitter]), 2, PropQualificationConfig(min_ev=-1.0))

    assert "pitcher K over vs opposing batter HRR over" in note
    assert adjustment < 1.0
    assert parlays.loc[0, "model_joint_probability"] == pytest.approx(0.58 * 0.57 * adjustment)
