from datetime import date

import duckdb
import pandas as pd
import pytest

from mlb_winners.db import init_db
from mlb_winners.moneyline_diagnostics import (
    bucket_edge,
    bucket_ev,
    feature_subset_from_importance,
    market_adjusted_probability,
    weekly_moneyline_stats,
    write_moneyline_candidates,
)


def test_market_adjusted_probability_shrinks_model_edge_to_market():
    assert market_adjusted_probability(0.60, 0.50, 0.30) == pytest.approx(0.53)
    assert market_adjusted_probability(0.40, 0.55, 0.50) == pytest.approx(0.475)


def test_edge_and_ev_buckets_are_stable():
    assert bucket_edge(0.025) == "0-3%"
    assert bucket_edge(0.081) == "7-10%"
    assert bucket_edge(0.11) == "10%+"
    assert bucket_ev(-0.01) == "<=0"
    assert bucket_ev(0.08) == "5-10%"
    assert bucket_ev(0.21) == "20%+"


def test_feature_subset_from_importance_is_deterministic():
    importance = pd.DataFrame(
        [
            {"feature_name": "starter_fip_advantage", "importance_type": "permutation_log_loss", "importance": 0.2},
            {"feature_name": "win_pct_diff", "importance_type": "permutation_log_loss", "importance": 0.2},
            {"feature_name": "run_diff_per_game_diff", "importance_type": "permutation_log_loss", "importance": 0.1},
        ]
    )

    assert feature_subset_from_importance(importance, 2) == ["starter_fip_advantage", "win_pct_diff"]


def test_candidate_tracking_writes_no_bets_without_official_record():
    con = duckdb.connect(":memory:")
    init_db(con)
    predictions = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": date(2026, 6, 1),
                "home_team": "Home",
                "away_team": "Away",
                "model_pick": "Home",
                "bet_side": "no bet",
                "bookmaker": "book",
                "home_moneyline": -120,
                "away_moneyline": 110,
                "model_home_prob": 0.56,
                "model_away_prob": 0.44,
                "fair_home_prob": 0.53,
                "fair_away_prob": 0.47,
                "bet_probability": float("nan"),
                "edge": 0.03,
                "ev_per_dollar": 0.02,
                "confidence": "no bet",
                "skip_reason": "edge_below_dynamic_threshold_0.050",
                "model_disagreement": 0.02,
                "uncertainty_score": 0.10,
                "home_lineup_confirmed": 1.0,
                "away_lineup_confirmed": 1.0,
            }
        ]
    )

    assert write_moneyline_candidates(con, predictions) == 1
    row = con.execute("SELECT confidence, official_play FROM moneyline_candidate_snapshots").fetchone()
    assert row == ("no bet", False)
    assert con.execute("SELECT count(*) FROM bet_recommendations").fetchone()[0] == 0


def test_candidate_tracking_writes_watchlist_without_official_record():
    con = duckdb.connect(":memory:")
    init_db(con)
    predictions = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": date(2026, 6, 1),
                "home_team": "Home",
                "away_team": "Away",
                "model_pick": "Home",
                "bet_side": "Home",
                "bookmaker": "book",
                "home_moneyline": -120,
                "away_moneyline": 110,
                "model_home_prob": 0.56,
                "model_away_prob": 0.44,
                "fair_home_prob": 0.53,
                "fair_away_prob": 0.47,
                "bet_probability": 0.56,
                "edge": 0.045,
                "ev_per_dollar": 0.03,
                "confidence": "watchlist",
                "skip_reason": "soft_filter_watchlist",
                "tier": "watchlist",
                "risk_flags": "missing_lineup",
                "stake_units": 0.25,
                "model_disagreement": 0.02,
                "uncertainty_score": 0.10,
                "home_lineup_confirmed": 0.0,
                "away_lineup_confirmed": 1.0,
            }
        ]
    )

    assert write_moneyline_candidates(con, predictions) == 1
    row = con.execute(
        "SELECT confidence, tier, risk_flags, stake_units, official_play FROM moneyline_candidate_snapshots"
    ).fetchone()
    assert row == ("watchlist", "watchlist", "missing_lineup", 0.25, False)


def test_weekly_moneyline_stats_bucketizes_official_and_pending():
    con = duckdb.connect(":memory:")
    init_db(con)
    con.execute(
        """
        INSERT INTO games (game_pk, game_date, home_team, away_team, home_score, away_score, home_won, status)
        VALUES
        (1, '2026-06-01', 'Home', 'Away', 5, 3, true, 'Final'),
        (2, '2026-06-01', 'Home2', 'Away2', NULL, NULL, NULL, 'Scheduled')
        """
    )
    predictions = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": date(2026, 6, 1),
                "home_team": "Home",
                "away_team": "Away",
                "model_pick": "Home",
                "bet_side": "Home",
                "bookmaker": "book",
                "home_moneyline": -120,
                "away_moneyline": 110,
                "model_home_prob": 0.60,
                "model_away_prob": 0.40,
                "fair_home_prob": 0.52,
                "fair_away_prob": 0.48,
                "bet_probability": 0.60,
                "edge": 0.08,
                "ev_per_dollar": 0.10,
                "confidence": "strong",
                "skip_reason": None,
                "model_disagreement": 0.02,
                "uncertainty_score": 0.10,
                "home_lineup_confirmed": 1.0,
                "away_lineup_confirmed": 1.0,
            },
            {
                "game_pk": 2,
                "game_date": date(2026, 6, 1),
                "home_team": "Home2",
                "away_team": "Away2",
                "model_pick": "Away2",
                "bet_side": "Away2",
                "bookmaker": "book",
                "home_moneyline": -110,
                "away_moneyline": -110,
                "model_home_prob": 0.45,
                "model_away_prob": 0.55,
                "fair_home_prob": 0.50,
                "fair_away_prob": 0.50,
                "bet_probability": 0.55,
                "edge": 0.05,
                "ev_per_dollar": 0.02,
                "confidence": "watchlist",
                "skip_reason": "soft_filter_watchlist",
                "tier": "watchlist",
                "risk_flags": "missing_lineup",
                "stake_units": 0.25,
                "model_disagreement": 0.08,
                "uncertainty_score": 0.20,
                "home_lineup_confirmed": 0.0,
                "away_lineup_confirmed": 1.0,
            },
        ]
    )
    write_moneyline_candidates(con, predictions)

    result = weekly_moneyline_stats(con, date(2026, 6, 1), date(2026, 6, 7))

    official = result.summary[result.summary["segment"].eq("strong_medium")].iloc[0]
    assert official["record"] == "1-0"
    assert official["official_candidates"] == 1
    watchlist = result.summary[result.summary["segment"].eq("watchlist")].iloc[0]
    assert watchlist["official_candidates"] == 0
    assert watchlist["pending"] == 1
    missing_lineup = result.segments[result.segments["segment"].eq("missing_lineup")].iloc[0]
    assert missing_lineup["pending"] == 1


def test_weekly_moneyline_stats_requires_final_status():
    con = duckdb.connect(":memory:")
    init_db(con)
    con.execute(
        """
        INSERT INTO games (game_pk, game_date, home_team, away_team, home_score, away_score, home_won, status)
        VALUES (1, '2026-06-01', 'Home', 'Away', 0, 0, false, 'Scheduled')
        """
    )
    predictions = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": date(2026, 6, 1),
                "home_team": "Home",
                "away_team": "Away",
                "model_pick": "Away",
                "bet_side": "Away",
                "bookmaker": "book",
                "home_moneyline": -120,
                "away_moneyline": 110,
                "model_home_prob": 0.40,
                "model_away_prob": 0.60,
                "fair_home_prob": 0.52,
                "fair_away_prob": 0.48,
                "bet_probability": 0.60,
                "edge": 0.08,
                "ev_per_dollar": 0.10,
                "confidence": "strong",
                "skip_reason": None,
                "model_disagreement": 0.02,
                "uncertainty_score": 0.10,
                "home_lineup_confirmed": 1.0,
                "away_lineup_confirmed": 1.0,
            }
        ]
    )
    write_moneyline_candidates(con, predictions)

    result = weekly_moneyline_stats(con, date(2026, 6, 1), date(2026, 6, 1))

    official = result.summary[result.summary["segment"].eq("strong_medium")].iloc[0]
    assert official["record"] == "0-0"
    assert official["pending"] == 1
