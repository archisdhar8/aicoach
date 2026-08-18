import pandas as pd
import pytest

from mlb_winners.parlay import american_to_decimal, build_lotto_parlay, decimal_to_american


def test_american_decimal_conversion():
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert decimal_to_american(2.5) == 150
    assert decimal_to_american(1.5) == -200


def test_build_lotto_parlay_requires_four_legs():
    predictions = pd.DataFrame(
        [
            {
                "game_pk": i,
                "confidence": "medium",
                "bet_side": f"Team {i}",
                "bet_moneyline": 100 + i,
                "bet_probability": 0.55,
                "edge": 0.06,
                "ev_per_dollar": 0.10,
                "skip_reason": None,
            }
            for i in range(1, 5)
        ]
    )
    parlay = build_lotto_parlay(predictions, min_legs=4, max_legs=4, stake_units=0.1)
    assert parlay is not None
    assert len(parlay.legs) == 4
    assert parlay.stake_units == pytest.approx(0.1)
    assert parlay.hit_probability == pytest.approx(0.55**4)
    assert parlay.combined_american_odds > 1000


def test_build_lotto_parlay_skips_if_not_enough_legs():
    predictions = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "confidence": "medium",
                "bet_side": "A",
                "bet_moneyline": 120,
                "bet_probability": 0.55,
                "edge": 0.06,
                "ev_per_dollar": 0.10,
            }
        ]
    )
    assert build_lotto_parlay(predictions, min_legs=4) is None


def test_lotto_parlay_uses_model_pick_fallbacks():
    predictions = pd.DataFrame(
        [
            {
                "game_pk": i,
                "confidence": "no bet",
                "home_team": f"Home {i}",
                "away_team": f"Away {i}",
                "model_pick": f"Away {i}",
                "home_moneyline": -130,
                "away_moneyline": 120,
                "model_home_prob": 0.47,
                "model_away_prob": 0.53,
                "market_adjusted_home_prob": 0.47,
                "market_adjusted_away_prob": 0.53,
                "fair_home_prob": 0.55,
                "fair_away_prob": 0.45,
                "bet_side": "no bet",
                "bet_moneyline": None,
                "bet_probability": None,
                "edge": 0.08,
                "ev_per_dollar": 0.10,
                "skip_reason": "bullpen_uncertainty",
            }
            for i in range(1, 6)
        ]
    )
    parlay = build_lotto_parlay(predictions, min_legs=4, max_legs=5, stake_units=0.1)
    assert parlay is not None
    assert len(parlay.legs) == 5
    assert set(parlay.legs["lotto_source"]) == {"model_pick"}
    assert all(parlay.legs["bet_side"].str.startswith("Away "))
