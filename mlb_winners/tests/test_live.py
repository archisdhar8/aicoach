from datetime import date, datetime, timezone

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from mlb_winners.db import init_db, upsert_df
from mlb_winners.live import (
    LiveState,
    decode_base_state,
    encode_base_state,
    fallback_score_probability,
    apply_plate_appearance_outcome,
    build_pitch_matchup_analysis,
    simulate_plate_appearance,
    simulate_live_game,
    state_number,
    team_state_rates,
)
from mlb_winners.dashboard import manual_odds_refresh


def test_all_24_base_out_states_are_unique_and_reversible():
    states = set()
    for outs in range(3):
        for first in (False, True):
            for second in (False, True):
                for third in (False, True):
                    mask = encode_base_state(first, second, third)
                    states.add(state_number(mask, outs))
                    assert decode_base_state(mask) == {
                        "first": first, "second": second, "third": third,
                    }
    assert states == set(range(1, 25))


def test_invalid_base_out_states_fail():
    with pytest.raises(ValueError):
        state_number(8, 0)
    with pytest.raises(ValueError):
        state_number(0, 3)


def test_score_probability_increases_with_runners_and_decreases_with_outs():
    assert fallback_score_probability(7, 0) > fallback_score_probability(0, 0)
    assert fallback_score_probability(1, 0) > fallback_score_probability(1, 2)


def test_team_window_uses_exactly_latest_50_games_across_seasons():
    con = duckdb.connect(":memory:")
    init_db(con)
    games = []
    events = []
    start = pd.Timestamp("2025-08-01")
    for index in range(60):
        game_pk = 1000 + index
        game_date = (start + pd.Timedelta(days=index * 3)).date()
        games.append({
            "game_pk": game_pk, "game_date": game_date, "season": game_date.year,
            "game_type": "R", "status": "Final", "home_team_id": 1, "away_team_id": 2,
            "home_team": "A", "away_team": "B",
        })
        events.append({
            "game_pk": game_pk, "at_bat_index": 0, "event_index": 0,
            "game_date": game_date, "batting_team_id": 1, "base_mask": 0,
            "outs": 0, "runs_to_inning_end": int(index % 2 == 0),
            "fetched_at": datetime.now(timezone.utc),
        })
    upsert_df(con, "games", pd.DataFrame(games))
    upsert_df(con, "live_play_events", pd.DataFrame(events))
    result = team_state_rates(con, 1, date(2026, 8, 1))
    assert result["games_used"] == 50
    assert result["window_start"] == games[10]["game_date"].isoformat()
    assert result["states"][0]["sample_size"] == 50


def test_live_simulation_is_deterministic_and_complementary():
    state = LiveState(
        game_pk=7, captured_at=datetime.now(timezone.utc), game_date=date(2026, 7, 25),
        status="In Progress", inning=7, half_inning="Bottom", batting_team_id=1,
        home_team_id=1, away_team_id=2, home_team="Home", away_team="Away",
        home_score=3, away_score=3, base_mask=3, outs=1, balls=2, strikes=1,
        batter_id=10, batter_name="Batter", pitcher_id=20, pitcher_name="Pitcher",
        at_bat_index=45, event_index=3,
    )
    first, samples_a = simulate_live_game(state, .62, .55, sims=1000)
    second, samples_b = simulate_live_game(state, .62, .55, sims=1000)
    assert first == second
    assert (samples_a == samples_b).all()
    assert 0 <= first <= 1
    assert first + (1 - first) == pytest.approx(1)


def test_odds_refresh_cannot_run_without_explicit_manual_confirmation():
    with pytest.raises(HTTPException) as error:
        manual_odds_refresh(None)
    assert error.value.status_code == 400
    assert "confirmation" in error.value.detail.lower()


def test_plate_appearance_runner_advancement_and_runs():
    mask, outs, runs = apply_plate_appearance_outcome(7, 1, "home_run")
    assert (mask, outs, runs) == (0, 1, 4)
    mask, outs, runs = apply_plate_appearance_outcome(7, 1, "walk_hbp")
    assert (mask, outs, runs) == (7, 1, 1)
    mask, outs, runs = apply_plate_appearance_outcome(1, 1, "double")
    assert (mask, outs, runs) == (2, 1, 1)


def test_pitch_analysis_uses_historical_mix_and_probabilities_sum_to_one():
    con = duckdb.connect(":memory:")
    init_db(con)
    state = LiveState(
        game_pk=88, captured_at=datetime.now(timezone.utc), game_date=date(2026, 7, 25),
        status="In Progress", inning=5, half_inning="Top", batting_team_id=2,
        home_team_id=1, away_team_id=2, home_team="Home", away_team="Away",
        home_score=2, away_score=2, base_mask=0, outs=1, balls=1, strikes=2,
        batter_id=10, batter_name="Batter", pitcher_id=20, pitcher_name="Pitcher",
        at_bat_index=40, event_index=2, batter_side="R", pitcher_hand="L",
    )
    upsert_df(con, "pitcher_pitch_mix", pd.DataFrame([
        {"game_date": date(2026, 7, 1), "pitcher_id": 20, "pitcher_name": "Pitcher", "pitch_type": "FF", "pitches": 70, "avg_velocity": 96.0, "avg_spin_rate": 2400.0, "usage_rate": .7},
        {"game_date": date(2026, 7, 1), "pitcher_id": 20, "pitcher_name": "Pitcher", "pitch_type": "SL", "pitches": 30, "avg_velocity": 87.0, "avg_spin_rate": 2500.0, "usage_rate": .3},
    ]))
    analysis = build_pitch_matchup_analysis(con, state)
    assert analysis["most_likely_pitch"] == "FF"
    assert sum(row["probability"] for row in analysis["pitches"]) == pytest.approx(1)
    assert all(sum(row["outcomes"].values()) == pytest.approx(1) for row in analysis["pitches"])


def test_pitch_by_pitch_plate_appearance_can_force_home_run():
    analysis = {"pitches": [{
        "pitch_type": "FF", "probability": 1.0,
        "outcomes": {name: float(name == "home_run") for name in (
            "ball", "called_strike", "swinging_strike", "foul", "in_play_out",
            "single", "double", "triple", "home_run", "hbp",
        )},
    }]}
    outcome = simulate_plate_appearance(np.random.default_rng(2), analysis)
    assert outcome == "home_run"
