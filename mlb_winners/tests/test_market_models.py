from types import SimpleNamespace

from mlb_winners.market_models import f5_runs, first_inning_score_probability


def test_nrfi_component_probability_is_bounded_and_directional():
    weak_offense_vs_ace = SimpleNamespace(
        away_lineup_offense_rating=0.5,
        away_team_xwoba_last14=0.290,
        home_starter_fip_proxy=2.8,
        home_starter_kbb_prior=5.0,
        home_starter_barrel_allowed=0.04,
        park_run_factor=0.95,
        wind_speed_mph=2.0,
    )
    strong_offense_vs_poor_starter = SimpleNamespace(
        away_lineup_offense_rating=0.9,
        away_team_xwoba_last14=0.360,
        home_starter_fip_proxy=5.4,
        home_starter_kbb_prior=1.4,
        home_starter_barrel_allowed=0.12,
        park_run_factor=1.08,
        wind_speed_mph=12.0,
    )

    low = first_inning_score_probability(weak_offense_vs_ace, "away", "home")
    high = first_inning_score_probability(strong_offense_vs_poor_starter, "away", "home")

    assert 0.08 <= low <= 0.48
    assert 0.08 <= high <= 0.48
    assert high > low


def test_f5_runs_are_starter_and_lineup_sensitive():
    strong = SimpleNamespace(
        home_last14_runs_for=5.8,
        home_team_xwoba_last14=0.355,
        home_lineup_offense_rating=0.9,
        away_starter_fip_proxy=5.2,
        away_starter_xwoba_allowed=0.360,
        park_run_factor=1.08,
    )
    weak = SimpleNamespace(
        home_last14_runs_for=3.2,
        home_team_xwoba_last14=0.290,
        home_lineup_offense_rating=0.5,
        away_starter_fip_proxy=2.9,
        away_starter_xwoba_allowed=0.285,
        park_run_factor=0.94,
    )

    assert f5_runs(strong, "home", "away") > f5_runs(weak, "home", "away")
