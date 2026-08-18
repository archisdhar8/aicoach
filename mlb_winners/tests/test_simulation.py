import pandas as pd

from mlb_winners.simulation import SimulationConfig, add_simulation_columns, simulate_game_from_row


def test_simulation_reproducible_with_fixed_seed():
    row = {
        "game_pk": 123,
        "home_team": "Home",
        "away_team": "Away",
        "park_run_factor": 1.0,
        "temperature_f": 72,
        "wind_out_proxy": 0,
        "home_team_xwoba_last14": 0.330,
        "away_team_xwoba_last14": 0.300,
        "home_team_barrel_last14": 0.080,
        "away_team_barrel_last14": 0.060,
        "home_team_k_rate_last14": 0.20,
        "away_team_k_rate_last14": 0.24,
        "home_team_bb_rate_last14": 0.09,
        "away_team_bb_rate_last14": 0.07,
        "home_run_diff_per_game": 0.5,
        "away_run_diff_per_game": -0.2,
        "home_starter_fip_proxy": 3.8,
        "away_starter_fip_proxy": 4.6,
        "home_starter_xwoba_allowed": 0.300,
        "away_starter_xwoba_allowed": 0.340,
        "home_bullpen_er_last7": 2,
        "away_bullpen_er_last7": 5,
        "home_bullpen_kbb_last7": 3.0,
        "away_bullpen_kbb_last7": 1.8,
    }
    a, _ = simulate_game_from_row(row, SimulationConfig(sims=500, seed=7))
    b, _ = simulate_game_from_row(row, SimulationConfig(sims=500, seed=7))

    assert a["sim_home_win_prob"].iloc[0] == b["sim_home_win_prob"].iloc[0]
    assert 0 <= a["sim_home_win_prob"].iloc[0] <= 1
    assert a["mean_total_runs"].iloc[0] > 4


def test_simulation_adds_decision_support_columns():
    predictions = pd.DataFrame(
        [
            {
                "game_pk": 321,
                "home_team": "Home",
                "away_team": "Away",
                "model_pick": "Home",
                "model_home_prob": 0.58,
                "model_away_prob": 0.42,
                "park_run_factor": 1.0,
                "temperature_f": 72,
                "home_team_xwoba_last14": 0.340,
                "away_team_xwoba_last14": 0.300,
                "home_team_barrel_last14": 0.085,
                "away_team_barrel_last14": 0.060,
                "home_team_k_rate_last14": 0.19,
                "away_team_k_rate_last14": 0.25,
                "home_team_bb_rate_last14": 0.095,
                "away_team_bb_rate_last14": 0.070,
                "home_lineup_confirmed": 1.0,
                "away_lineup_confirmed": 1.0,
                "home_lineup_offense_rating": 0.75,
                "away_lineup_offense_rating": 0.65,
                "home_starter_fip_proxy": 3.6,
                "away_starter_fip_proxy": 4.8,
                "home_starter_xwoba_allowed": 0.295,
                "away_starter_xwoba_allowed": 0.350,
                "home_starter_games_prior": 12,
                "away_starter_games_prior": 8,
                "home_bullpen_kbb_last7": 3.1,
                "away_bullpen_kbb_last7": 1.8,
            }
        ]
    )

    enriched = add_simulation_columns(predictions, SimulationConfig(sims=600, seed=9))

    assert "sim_home_win_prob" in enriched.columns
    assert "expected_home_runs" in enriched.columns
    assert "model_sim_agree" in enriched.columns
    assert 0 <= enriched["sim_home_win_prob"].iloc[0] <= 1
    assert enriched["model_sim_disagreement"].iloc[0] >= 0
