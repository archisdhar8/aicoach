import pandas as pd

from mlb_winners.reasoning import add_ollama_reasons, fallback_play_reason


def test_fallback_play_reason_uses_actual_market_values():
    row = {
        "away_team": "Away",
        "home_team": "Home",
        "bet_side": "Away",
        "bet_probability": 0.55,
        "fair_away_prob": 0.50,
        "edge": 0.05,
        "ev_per_dollar": 0.12,
        "home_lineup_confirmed": 1.0,
        "away_lineup_confirmed": 1.0,
        "uncertainty_score": 0.1,
    }

    reason = fallback_play_reason(row)

    assert "Away" in reason
    assert "55.0%" in reason
    assert "50.0%" in reason
    assert "EV 0.120" in reason


def test_add_ollama_reasons_falls_back_when_ollama_unavailable():
    predictions = pd.DataFrame(
        [
            {
                "away_team": "Away",
                "home_team": "Home",
                "bet_side": "Away",
                "confidence": "medium",
                "bet_probability": 0.55,
                "fair_away_prob": 0.50,
                "edge": 0.05,
                "ev_per_dollar": 0.12,
                "home_lineup_confirmed": 1.0,
                "away_lineup_confirmed": 1.0,
            }
        ]
    )

    enriched = add_ollama_reasons(predictions, url="http://127.0.0.1:1/api/generate", timeout=0.01)

    assert enriched.loc[0, "reason"].startswith("Reason:")
