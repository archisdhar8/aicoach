import pytest
import pandas as pd

from mlb_winners.features import FEATURE_COLUMNS, TeamState, build_training_frame, bullpen_fatigue


def test_features_use_only_prior_games():
    games = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2024-04-01",
                "season": 2024,
                "home_team_id": 10,
                "away_team_id": 20,
                "home_team": "Home",
                "away_team": "Away",
                "home_score": 10,
                "away_score": 1,
                "home_won": True,
                "home_probable_pitcher_id": 100,
                "away_probable_pitcher_id": 200,
                "day_night": "day",
                "doubleheader": "N",
            },
            {
                "game_pk": 2,
                "game_date": "2024-04-02",
                "season": 2024,
                "home_team_id": 10,
                "away_team_id": 20,
                "home_team": "Home",
                "away_team": "Away",
                "home_score": 1,
                "away_score": 2,
                "home_won": False,
                "home_probable_pitcher_id": 100,
                "away_probable_pitcher_id": 200,
                "day_night": "night",
                "doubleheader": "N",
            },
        ]
    )
    stats = pd.DataFrame(
        [
            {"game_pk": 1, "team_id": 10, "starter_id": 100, "starter_ip": 6.0, "starter_er": 1, "starter_hits": 4, "starter_bb": 1, "starter_so": 8, "bullpen_ip": 3.0, "bullpen_er": 0},
            {"game_pk": 1, "team_id": 20, "starter_id": 200, "starter_ip": 5.0, "starter_er": 5, "starter_hits": 8, "starter_bb": 3, "starter_so": 2, "bullpen_ip": 3.0, "bullpen_er": 5},
        ]
    )

    frame = build_training_frame(games, stats)
    first = frame.iloc[0]
    second = frame.iloc[1]

    assert first["home_win_pct"] == 0.5
    assert first["home_starter_era_prior"] == 4.5
    assert second["home_win_pct"] == 1.0
    assert second["away_win_pct"] == 0.0
    assert second["home_last7_run_diff"] == 9.0
    assert second["home_starter_era_prior"] == 1.5
    assert second["starter_era_advantage"] > 0
    assert second["run_diff_per_game_diff"] == 18.0
    assert first["target_total_runs"] == 11
    assert second["target_total_runs"] == 3


def test_bullpen_fatigue_uses_graduated_workload_thresholds():
    state = TeamState()
    for day in range(7):
        state.update(
            pd.Timestamp(f"2024-04-{day + 1:02d}"),
            "home",
            runs_for=4,
            runs_against=3,
            bullpen_ip=4.0,
            bullpen_er=1,
        )

    assert bullpen_fatigue(state) == pytest.approx(0.866, abs=0.01)


def test_feature_columns_prefer_matchup_diffs_and_exclude_duplicate_lineup_penalties():
    assert "starter_fip_advantage" in FEATURE_COLUMNS
    assert "bullpen_quality_advantage" in FEATURE_COLUMNS
    assert "lineup_offense_advantage" in FEATURE_COLUMNS
    assert "home_missing_lineup_penalty" not in FEATURE_COLUMNS
    assert "away_missing_lineup_penalty" not in FEATURE_COLUMNS
    assert "home_starter_xwoba_allowed" not in FEATURE_COLUMNS
