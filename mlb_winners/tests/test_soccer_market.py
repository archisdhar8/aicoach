import pytest

from soccerworldcup.market import devig_three_way, devig_two_way, expected_value_per_dollar


def test_soccer_three_way_devig_removes_overround():
    home, draw, away = devig_three_way(1.90, 3.40, 4.20)

    assert home + draw + away == pytest.approx(1.0)
    assert home > draw > 0
    assert away > 0


def test_soccer_two_way_devig_and_ev():
    over, under = devig_two_way(1.91, 1.91)

    assert over + under == pytest.approx(1.0)
    assert expected_value_per_dollar(0.55, 1.91) == pytest.approx(0.0505)
