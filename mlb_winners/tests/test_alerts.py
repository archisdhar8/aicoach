from datetime import date

import duckdb
import pandas as pd

from mlb_winners.cli import filter_unsent_alerts


def test_filter_unsent_alerts_ignores_null_delivery_rows():
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE alert_deliveries (
            game_pk BIGINT,
            selection VARCHAR,
            game_date DATE
        )
        """
    )
    alert_date = date(2026, 5, 25)
    con.execute(
        "INSERT INTO alert_deliveries VALUES (NULL, 'Some Team', ?), (123, NULL, ?), (123, 'A', ?)",
        [alert_date, alert_date, alert_date],
    )

    predictions = pd.DataFrame(
        [
            {"game_pk": 123, "confidence": "strong", "bet_side": "A"},
            {"game_pk": 456, "confidence": "medium", "bet_side": "B"},
            {"game_pk": 789, "confidence": "no bet", "bet_side": None},
        ]
    )
    filtered = filter_unsent_alerts(con, predictions, alert_date)
    assert set(filtered["game_pk"].tolist()) == {456, 789}
