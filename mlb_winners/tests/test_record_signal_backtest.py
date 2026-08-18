import pandas as pd

from mlb_winners.record_signal_backtest import attach_season_to_date_records, record_gap_bucket


def test_attach_season_to_date_records_uses_only_prior_games():
    frame = pd.DataFrame(
        [
            {
                "game_date": pd.Timestamp("2026-04-01"),
                "game_pk": 1,
                "home_team_id": 10,
                "away_team_id": 20,
                "target_home_win": True,
            },
            {
                "game_date": pd.Timestamp("2026-04-02"),
                "game_pk": 2,
                "home_team_id": 10,
                "away_team_id": 20,
                "target_home_win": False,
            },
        ]
    )

    out = attach_season_to_date_records(frame)

    assert out.loc[0, "home_wins_prior"] == 0
    assert out.loc[0, "away_losses_prior"] == 0
    assert out.loc[1, "home_wins_prior"] == 1
    assert out.loc[1, "away_losses_prior"] == 1


def test_record_gap_bucket_handles_half_game_gaps():
    assert record_gap_bucket(0.5) == "0"
    assert record_gap_bucket(3.5) == "1-3"
    assert record_gap_bucket(8.5) == "4-8"
    assert record_gap_bucket(9.0) == "9+"
