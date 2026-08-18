import pandas as pd

from mlb_winners.notifications import format_value_alert


def test_value_alert_includes_official_and_watchlist_sections():
    predictions = pd.DataFrame(
        [
            {
                "confidence": "medium",
                "bet_side": "Astros",
                "bet_moneyline": -115,
                "bet_probability": 0.57,
                "edge": 0.061,
                "ev_per_dollar": 0.055,
                "stake_units": 1.0,
            },
            {
                "confidence": "watchlist",
                "bet_side": "Rays",
                "bet_moneyline": 125,
                "bet_probability": 0.49,
                "edge": 0.045,
                "ev_per_dollar": 0.035,
                "stake_units": 0.25,
                "risk_flags": "missing_lineup",
                "skip_reason": "soft_filter_watchlist",
            },
        ]
    )

    message = format_value_alert(predictions, "2026-07-20")

    assert "MEDIUM: Astros -115" in message
    assert "stake=1.00u" in message
    assert "WATCHLIST / LEANS - not official" in message
    assert "LEAN: Rays 125" in message
    assert "stake=0.25u" in message
    assert "flags: missing_lineup" in message


def test_value_alert_empty_mentions_watchlist():
    message = format_value_alert(pd.DataFrame([{"confidence": "no bet"}]), "2026-07-20")

    assert "no strong/medium/watchlist" in message
