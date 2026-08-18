import pytest

from mlb_winners.odds import (
    devig_two_way,
    expected_value_per_dollar,
    implied_prob_to_moneyline,
    moneyline_to_implied_prob,
)


def test_moneyline_to_implied_probability():
    assert moneyline_to_implied_prob(-150) == pytest.approx(0.6)
    assert moneyline_to_implied_prob(200) == pytest.approx(1 / 3)


def test_implied_probability_to_moneyline():
    assert implied_prob_to_moneyline(0.6) == -150
    assert implied_prob_to_moneyline(1 / 3) == 200


def test_devig_two_way_removes_overround():
    home, away = devig_two_way(-120, 110)
    assert home + away == pytest.approx(1.0)
    assert home > away


def test_expected_value_per_dollar():
    assert expected_value_per_dollar(0.55, 110) == pytest.approx(0.155)
    assert expected_value_per_dollar(0.55, -110) == pytest.approx(0.05, abs=0.001)
