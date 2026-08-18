import pandas as pd

from mlb_winners.portfolio import PortfolioConfig, kelly_stake_units, optimize_portfolio


def test_kelly_and_exposure_caps():
    assert kelly_stake_units(0.55, 110, PortfolioConfig(max_bet_units=1.0)) > 0
    candidates = pd.DataFrame(
        [
            {"game_pk": 1, "market": "moneyline", "selection": "A", "odds": 110, "probability": 0.60, "ev_per_unit": 0.26},
            {"game_pk": 1, "market": "moneyline", "selection": "B", "odds": 120, "probability": 0.58, "ev_per_unit": 0.27},
            {"game_pk": 2, "market": "moneyline", "selection": "C", "odds": 100, "probability": 0.55, "ev_per_unit": 0.10},
        ]
    )
    out = optimize_portfolio(candidates, config=PortfolioConfig(max_game_exposure=1.0, max_daily_exposure=2.0))

    assert len(out[out["portfolio_status"] == "accepted"]) == 2
    assert "same_game_correlation_cap" in set(out["skip_reason"].dropna())
