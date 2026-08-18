import pandas as pd

from mlb_winners.statcast import aggregate_pitch_type_matchups, aggregate_statcast


def test_statcast_aggregation_builds_team_and_pitcher_quality_metrics():
    raw = pd.DataFrame(
        [
            {
                "game_date": "2025-04-01",
                "inning_topbot": "Top",
                "away_team": "NYY",
                "home_team": "BOS",
                "launch_speed": 101.0,
                "launch_angle": 28.0,
                "estimated_woba_using_speedangle": 0.650,
                "estimated_ba_using_speedangle": 0.510,
                "events": "single",
                "release_speed": 96.0,
                "release_spin_rate": 2400.0,
                "pitcher": 123,
                "player_name": "Starter A",
            },
            {
                "game_date": "2025-04-01",
                "inning_topbot": "Top",
                "away_team": "NYY",
                "home_team": "BOS",
                "launch_speed": None,
                "launch_angle": None,
                "estimated_woba_using_speedangle": None,
                "estimated_ba_using_speedangle": None,
                "events": "strikeout",
                "release_speed": 97.0,
                "release_spin_rate": 2450.0,
                "pitcher": 123,
                "player_name": "Starter A",
            },
        ]
    )

    team_daily, pitcher_daily = aggregate_statcast(raw)
    yankees = team_daily[team_daily["team_name"] == "New York Yankees"].iloc[0]
    pitcher = pitcher_daily.iloc[0]

    assert yankees["xwoba"] == 0.65
    assert yankees["hard_hit_rate"] == 1.0
    assert yankees["barrel_rate"] == 1.0
    assert pitcher["xwoba_allowed"] == 0.65
    assert pitcher["k_rate"] == 0.5


def test_pitch_type_matchup_aggregation_builds_count_specific_outcomes():
    raw = pd.DataFrame([
        {
            "game_date": "2026-04-01", "batter": 10, "pitcher": 20,
            "pitch_type": "FF", "stand": "R", "p_throws": "L", "balls": 1,
            "strikes": 1, "description": "called_strike", "events": None,
            "release_speed": 96.0, "release_spin_rate": 2400.0,
            "estimated_woba_using_speedangle": None, "woba_value": None,
            "delta_run_exp": -0.03,
        },
        {
            "game_date": "2026-04-01", "batter": 10, "pitcher": 20,
            "pitch_type": "FF", "stand": "R", "p_throws": "L", "balls": 1,
            "strikes": 1, "description": "hit_into_play", "events": "home_run",
            "release_speed": 95.0, "release_spin_rate": 2350.0,
            "estimated_woba_using_speedangle": 1.8, "woba_value": 2.0,
            "delta_run_exp": 1.1,
        },
    ])
    result = aggregate_pitch_type_matchups(raw)
    row = result.iloc[0]
    assert row["pitch_count"] == 2
    assert row["called_strike_rate"] == 0.5
    assert row["home_run_rate"] == 0.5
    assert row["in_play_rate"] == 0.5
    assert row["avg_velocity"] == 95.5
