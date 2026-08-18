from __future__ import annotations

import sys
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from mlb_winners import cli
from mlb_winners.db import init_db
from soccerworldcup.data import SoccerFetchResult, normalize_eloratings_payload
from soccerworldcup.predict import double_chance_probabilities, draw_no_bet_probabilities, make_soccer_predictions
from soccerworldcup.ratings import completed_matches_before, rate_team
from soccerworldcup.risk import apply_risk_filter


def soccer_matches_fixture() -> pd.DataFrame:
    rows = []
    teams = ["USA", "Germany", "Brazil", "France"]
    for idx in range(8):
        home = teams[idx % len(teams)]
        away = teams[(idx + 1) % len(teams)]
        rows.append(
            {
                "match_id": f"hist-{idx}",
                "match_date": pd.Timestamp(f"2026-05-{idx + 1:02d}").date(),
                "kickoff_utc": pd.Timestamp(f"2026-05-{idx + 1:02d}T20:00:00Z"),
                "status": "Match Finished",
                "home_team": home,
                "away_team": away,
                "home_score": 2 if idx % 2 == 0 else 1,
                "away_score": 1 if idx % 2 == 0 else 1,
                "venue_name": "Test Stadium",
                "venue_city": "Test City",
                "temperature_f": 78.0,
                "wind_speed_mph": 6.0,
            }
        )
    rows.append(
        {
            "match_id": "target",
            "match_date": date(2026, 6, 14),
            "kickoff_utc": pd.Timestamp("2026-06-14T20:00:00Z"),
            "status": "Not Started",
            "home_team": "USA",
            "away_team": "Germany",
            "home_score": None,
            "away_score": None,
            "venue_name": "World Cup Stadium",
            "venue_city": "Los Angeles",
            "temperature_f": 82.0,
            "wind_speed_mph": 8.0,
        }
    )
    rows.append(
        {
            "match_id": "future",
            "match_date": date(2026, 6, 15),
            "kickoff_utc": pd.Timestamp("2026-06-15T20:00:00Z"),
            "status": "Match Finished",
            "home_team": "USA",
            "away_team": "Germany",
            "home_score": 9,
            "away_score": 0,
            "venue_name": "Future Stadium",
            "venue_city": "Future City",
            "temperature_f": 82.0,
            "wind_speed_mph": 8.0,
        }
    )
    return pd.DataFrame(rows)


def test_soccer_ratings_exclude_target_and_future_matches():
    matches = soccer_matches_fixture()
    prior = completed_matches_before(matches, pd.Timestamp("2026-06-14T20:00:00Z"))

    assert "target" not in set(prior["match_id"])
    assert "future" not in set(prior["match_id"])

    rating = rate_team("USA", matches, pd.Timestamp("2026-06-14T20:00:00Z"))
    assert rating.matches_used > 0
    assert rating.goals_for_per_match < 9.0


def test_soccer_risk_filters_missing_odds_ratings_and_small_edge():
    assert apply_risk_filter(
        edge=0.10,
        ev_per_dollar=0.12,
        has_odds=False,
        has_team_ratings=True,
        odds_stale=False,
        squad_uncertainty=False,
        weather_uncertainty=False,
        uncertainty_score=0.1,
    ) == ("no bet", "missing_odds")
    assert apply_risk_filter(
        edge=0.10,
        ev_per_dollar=0.12,
        has_odds=True,
        has_team_ratings=False,
        odds_stale=False,
        squad_uncertainty=False,
        weather_uncertainty=False,
        uncertainty_score=0.1,
    ) == ("no bet", "missing_team_ratings")
    confidence, reason = apply_risk_filter(
        edge=0.01,
        ev_per_dollar=0.02,
        has_odds=True,
        has_team_ratings=True,
        odds_stale=False,
        squad_uncertainty=False,
        weather_uncertainty=False,
        uncertainty_score=0.1,
    )
    assert confidence == "no bet"
    assert reason.startswith("edge_below_threshold")


def test_soccer_prediction_output_schema():
    matches = soccer_matches_fixture()
    slate = matches[matches["match_id"] == "target"]
    odds = pd.DataFrame(
        [
            {
                "event_id": "event-target",
                "snapshot_date": date(2026, 6, 14),
                "commence_time": pd.Timestamp("2026-06-14T20:00:00Z"),
                "bookmaker": "testbook",
                "home_team": "USA",
                "away_team": "Germany",
                "home_price": 2.25,
                "draw_price": 3.20,
                "away_price": 3.10,
                "over_2_5_price": 1.95,
                "under_2_5_price": 1.90,
                "btts_yes_price": None,
                "btts_no_price": None,
                "fetched_at": pd.Timestamp.now(tz="UTC"),
            }
        ]
    )

    predictions = make_soccer_predictions(slate, matches, odds=odds)

    required = {
        "match_id",
        "kickoff_utc",
        "home_team",
        "away_team",
        "projected_score",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "over_2_5_probability",
        "under_2_5_probability",
        "btts_probability",
        "top_correct_scores",
        "edge",
        "ev_per_dollar",
        "confidence",
        "explanation",
        "elo_diff",
        "wc_matches_used_home",
        "wc_matches_used_away",
        "wc_form_weight_home",
        "wc_form_weight_away",
        "raw_model_prob",
        "market_prob",
        "calibrated_prob",
        "draw_prob",
        "raw_expected_total_goals",
        "calibrated_expected_total_goals",
        "guardrails_triggered",
        "final_recommendation",
    }
    assert required.issubset(predictions.columns)
    assert len(predictions) == 1


def test_one_world_cup_upset_does_not_create_huge_ml_jump():
    matches = pd.DataFrame(
        [
            {
                "match_id": "upset",
                "match_date": date(2026, 6, 11),
                "kickoff_utc": pd.Timestamp("2026-06-11T20:00:00Z"),
                "status": "Final",
                "home_team": "Underdog",
                "away_team": "Power",
                "home_score": 3,
                "away_score": 0,
            },
            {
                "match_id": "target",
                "match_date": date(2026, 6, 15),
                "kickoff_utc": pd.Timestamp("2026-06-15T20:00:00Z"),
                "status": "Scheduled",
                "home_team": "Underdog",
                "away_team": "Power",
                "home_score": None,
                "away_score": None,
            },
        ]
    )
    priors = pd.DataFrame(
        [
            {"team": "Underdog", "as_of_date": date(2026, 6, 1), "elo": 1400, "attack_rating": 0.9, "defense_rating": 0.9, "recent_form_rating": 0.45, "goals_for_per_match": 1.0, "goals_against_per_match": 1.4, "matches_used": 100, "source": "eloratings_net"},
            {"team": "Power", "as_of_date": date(2026, 6, 1), "elo": 1900, "attack_rating": 1.35, "defense_rating": 1.25, "recent_form_rating": 0.65, "goals_for_per_match": 1.9, "goals_against_per_match": 0.8, "matches_used": 100, "source": "eloratings_net"},
        ]
    )

    predictions = make_soccer_predictions(matches[matches["match_id"] == "target"], matches, fifa_rankings=priors)

    assert predictions.iloc[0]["home_win_probability"] < 0.30
    assert predictions.iloc[0]["wc_form_weight_home"] <= 0.13


def test_one_high_scoring_match_does_not_automatically_create_over_pick():
    matches = pd.DataFrame(
        [
            {
                "match_id": "wild",
                "match_date": date(2026, 6, 11),
                "kickoff_utc": pd.Timestamp("2026-06-11T20:00:00Z"),
                "status": "Final",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 5,
                "away_score": 4,
                "temperature_f": None,
            },
            {
                "match_id": "target",
                "match_date": date(2026, 6, 15),
                "kickoff_utc": pd.Timestamp("2026-06-15T20:00:00Z"),
                "status": "Scheduled",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": None,
                "away_score": None,
                "temperature_f": None,
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "event_id": "event-target",
                "snapshot_date": date(2026, 6, 15),
                "commence_time": pd.Timestamp("2026-06-15T20:00:00Z"),
                "bookmaker": "testbook",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_price": None,
                "draw_price": None,
                "away_price": None,
                "over_2_5_price": 2.05,
                "under_2_5_price": 1.80,
                "btts_yes_price": None,
                "btts_no_price": None,
                "fetched_at": pd.Timestamp.now(tz="UTC"),
            }
        ]
    )

    predictions = make_soccer_predictions(matches[matches["match_id"] == "target"], matches, odds=odds)

    assert predictions.iloc[0]["selection"] == "no bet"
    assert "over_total_not_clear_market" in str(predictions.iloc[0]["guardrails_triggered"])


def test_underdog_moneyline_requires_higher_edge_than_favorite():
    matches = soccer_matches_fixture()
    slate = matches[matches["match_id"] == "target"]
    odds = pd.DataFrame(
        [
            {
                "event_id": "event-target",
                "snapshot_date": date(2026, 6, 14),
                "commence_time": pd.Timestamp("2026-06-14T20:00:00Z"),
                "bookmaker": "testbook",
                "home_team": "USA",
                "away_team": "Germany",
                "home_price": 1.75,
                "draw_price": 3.5,
                "away_price": 3.0,
                "over_2_5_price": None,
                "under_2_5_price": None,
                "btts_yes_price": None,
                "btts_no_price": None,
                "fetched_at": pd.Timestamp.now(tz="UTC"),
            }
        ]
    )

    predictions = make_soccer_predictions(slate, matches, odds=odds)

    if predictions.iloc[0]["candidate_selection"] == "Germany":
        assert predictions.iloc[0]["edge"] < 0.055 or predictions.iloc[0]["selection"] == "no bet"


def test_draw_probability_is_clamped_to_realistic_range():
    matches = soccer_matches_fixture()
    predictions = make_soccer_predictions(matches[matches["match_id"] == "target"], matches)

    assert 0.18 <= predictions.iloc[0]["draw_probability"] <= 0.36


def test_missing_odds_still_predicts_but_lowers_confidence():
    matches = soccer_matches_fixture()
    predictions = make_soccer_predictions(matches[matches["match_id"] == "target"], matches)

    assert predictions.iloc[0]["home_win_probability"] > 0
    assert predictions.iloc[0]["confidence"] == "no odds"
    assert predictions.iloc[0]["skip_reason"] == "missing_odds"


def test_missing_weather_reduces_totals_confidence():
    matches = soccer_matches_fixture()
    slate = matches[matches["match_id"] == "target"].copy()
    slate.loc[:, "temperature_f"] = np.nan
    all_matches = pd.concat([matches[matches["match_id"] != "target"], slate], ignore_index=True)
    odds = pd.DataFrame(
        [
            {
                "event_id": "event-target",
                "snapshot_date": date(2026, 6, 14),
                "commence_time": pd.Timestamp("2026-06-14T20:00:00Z"),
                "bookmaker": "testbook",
                "home_team": "USA",
                "away_team": "Germany",
                "home_price": None,
                "draw_price": None,
                "away_price": None,
                "over_2_5_price": 2.20,
                "under_2_5_price": 1.70,
                "btts_yes_price": None,
                "btts_no_price": None,
                "fetched_at": pd.Timestamp.now(tz="UTC"),
            }
        ]
    )

    predictions = make_soccer_predictions(slate, all_matches, odds=odds)

    assert "missing_weather_total" in str(predictions.iloc[0]["guardrails_triggered"])


def test_double_chance_probability_calculation_works():
    home_or_draw, away_or_draw, home_or_away = double_chance_probabilities(0.45, 0.25, 0.30)

    assert home_or_draw == pytest.approx(0.70)
    assert away_or_draw == pytest.approx(0.55)
    assert home_or_away == pytest.approx(0.75)


def test_draw_no_bet_probability_calculation_works():
    home_dnb, away_dnb = draw_no_bet_probabilities(0.45, 0.30)

    assert home_dnb == pytest.approx(0.60)
    assert away_dnb == pytest.approx(0.40)


def test_prediction_uses_derivative_markets_when_odds_available():
    matches = soccer_matches_fixture()
    slate = matches[matches["match_id"] == "target"]
    odds = pd.DataFrame(
        [
            {
                "event_id": "event-target",
                "snapshot_date": date(2026, 6, 14),
                "commence_time": pd.Timestamp("2026-06-14T20:00:00Z"),
                "bookmaker": "testbook",
                "home_team": "USA",
                "away_team": "Germany",
                "home_price": None,
                "draw_price": None,
                "away_price": None,
                "home_or_draw_price": 1.80,
                "away_or_draw_price": 1.10,
                "home_or_away_price": 1.25,
                "home_dnb_price": 2.50,
                "away_dnb_price": 1.20,
                "over_2_5_price": None,
                "under_2_5_price": None,
                "btts_yes_price": None,
                "btts_no_price": None,
                "fetched_at": pd.Timestamp.now(tz="UTC"),
            }
        ]
    )

    predictions = make_soccer_predictions(slate, matches, odds=odds)

    assert predictions.iloc[0]["candidate_market"] in {"draw_no_bet", "double_chance"}
    assert "home_win" in str(predictions.iloc[0]["derivative_source_probabilities"])


def test_soccer_prediction_matches_market_team_aliases():
    matches = soccer_matches_fixture()
    slate = matches[matches["match_id"] == "target"].copy()
    slate.loc[:, "away_team"] = "Czechia"
    odds = pd.DataFrame(
        [
            {
                "event_id": "event-target",
                "snapshot_date": date(2026, 6, 14),
                "commence_time": pd.Timestamp("2026-06-14T20:00:00Z"),
                "bookmaker": "testbook",
                "home_team": "USA",
                "away_team": "Czech Republic",
                "home_price": 2.25,
                "draw_price": 3.20,
                "away_price": 3.10,
                "over_2_5_price": None,
                "under_2_5_price": None,
                "btts_yes_price": None,
                "btts_no_price": None,
                "fetched_at": pd.Timestamp.now(tz="UTC"),
            }
        ]
    )

    predictions = make_soccer_predictions(slate, matches, odds=odds)

    assert predictions.iloc[0]["bookmaker"] == "testbook"
    assert predictions.iloc[0]["candidate_market"] == "1x2"


def test_fetch_soccer_schedule_cli_smoke(monkeypatch, capsys):
    con = duckdb.connect(":memory:")
    init_db(con)

    monkeypatch.setattr(cli, "connect", lambda settings: con)
    monkeypatch.setattr(cli, "ensure_dirs", lambda settings: None)
    monkeypatch.setattr(
        cli,
        "fetch_world_cup_schedule",
        lambda con, start_date, end_date, force=False: SoccerFetchResult(rows=0, source="mock", from_cache=True, message=None),
    )
    monkeypatch.setattr(
        cli,
        "fetch_espn_world_cup_scores",
        lambda con, score_date, force=False: SoccerFetchResult(rows=0, source="mock_scores", from_cache=True, message=None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "fetch-soccer-schedule",
            "--start-date",
            "2026-06-11",
            "--end-date",
            "2026-06-12",
        ],
    )

    cli.main()
    output = capsys.readouterr().out
    assert "soccer schedule rows: 0" in output


def test_eloratings_payload_normalizes_world_cup_rows():
    payload = {
        "ratings_tsv": "1\t18\tMX\t1875\t1\t1900\t1\t1800\t1\t1700\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1030\t0\t0\t0\t500\t250\t280\t1800\t900\n",
        "teams_tsv": "MX\tMexico\nZA\tSouth Africa\nBA\tBosnia and Herzegovina\tBosnia & Herzegovina\n",
        "successors_tsv": "",
    }

    rows = normalize_eloratings_payload(payload, date(2026, 6, 11))

    assert rows.iloc[0]["team"] == "Mexico"
    assert rows.iloc[0]["elo"] == 1875
    assert rows.iloc[0]["matches_used"] == 1030
    assert rows.iloc[0]["source"] == "eloratings_net"
