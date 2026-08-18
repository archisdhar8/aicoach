import pandas as pd

from mlb_winners.ensemble import build_elo_ratings, elo_probability, elo_probs_for_frame


def test_elo_updates_after_win():
    games = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2026-04-01",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 6,
                "away_score": 2,
                "home_won": True,
            }
        ]
    )
    ratings = build_elo_ratings(games)
    assert ratings[1] > 1500
    assert ratings[2] < 1500


def test_elo_probability_uses_home_advantage():
    assert elo_probability(1500, 1500) > 0.5


def test_elo_probs_for_frame_applies_starter_adjustment():
    frame = pd.DataFrame(
        [
            {
                "home_team_id": 1,
                "away_team_id": 2,
                "home_starter_fip_proxy": 3.0,
                "away_starter_fip_proxy": 5.0,
            }
        ]
    )
    prob = elo_probs_for_frame(frame, {1: 1500, 2: 1500})[0]
    assert prob > elo_probability(1500, 1500)
