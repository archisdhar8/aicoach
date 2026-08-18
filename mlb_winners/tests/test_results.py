import duckdb
import pandas as pd

from mlb_winners.db import init_db
from mlb_winners.results import settle_bet_recommendations
from mlb_winners.cli import format_moneyline_recap


def test_settle_bet_recommendations_skips_pregame_placeholders():
    con = duckdb.connect(":memory:")
    init_db(con)
    games = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "status": "Pre-Game",
                "home_team": "Home",
                "away_team": "Away",
                "home_score": 0,
                "away_score": 0,
                "home_won": False,
            }
        ]
    )
    recommendations = pd.DataFrame(
        [
            {
                "recommendation_id": "rec-1",
                "game_pk": 1,
                "selection": "Away",
                "odds": 150,
                "stake_units": 1.0,
            }
        ]
    )

    settled = settle_bet_recommendations(con, games, recommendations)

    assert settled.empty


def test_format_moneyline_recap_includes_zero_record_when_empty():
    con = duckdb.connect(":memory:")
    init_db(con)

    message = format_moneyline_recap(con, pd.Timestamp("2026-05-29").date())

    assert "MLB moneyline recap for 2026-05-29: 0-0, +0.00u" in message
    assert "No settled recommendations for this date." in message
    assert "Winners: (none)" in message
    assert "Losers: (none)" in message


def test_format_moneyline_recap_ignores_undelivered_recommendations():
    con = duckdb.connect(":memory:")
    init_db(con)
    con.execute(
        """
        INSERT INTO bet_recommendations
        (recommendation_id, game_pk, game_date, market, selection, odds, stake_units, confidence)
        VALUES ('rec-1', 1, DATE '2026-06-01', 'moneyline', 'Away', 120, 1.0, 'medium')
        """
    )
    con.execute(
        """
        INSERT INTO bet_results (recommendation_id, result, units_profit)
        VALUES ('rec-1', 'win', 1.2)
        """
    )

    message = format_moneyline_recap(con, pd.Timestamp("2026-06-01").date())

    assert "MLB moneyline recap for 2026-06-01: 0-0, +0.00u" in message
    assert "Away" not in message
