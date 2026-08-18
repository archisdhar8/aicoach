import pytest

from soccerworldcup.poisson import simulate_scorelines


def test_poisson_scoreline_probabilities_sum_correctly():
    result = simulate_scorelines(1.45, 1.05, max_goals=10)

    assert result.home_win_probability + result.draw_probability + result.away_win_probability == pytest.approx(1.0)
    assert result.over_2_5_probability + result.under_2_5_probability == pytest.approx(1.0)
    assert 0.0 <= result.btts_probability <= 1.0
    assert result.top_scores == sorted(result.top_scores, key=lambda score: score.probability, reverse=True)
