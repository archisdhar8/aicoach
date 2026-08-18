from datetime import date

from mlb_winners.odds import normalize_odds_payload


def test_normalize_odds_payload_accepts_totals_only_market():
    payload = [
        {
            "id": "evt1",
            "commence_time": "2026-05-21T23:05:00Z",
            "home_team": "New York Yankees",
            "away_team": "Toronto Blue Jays",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -110, "point": 8.5},
                                {"name": "Under", "price": -110, "point": 8.5},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    rows = normalize_odds_payload(payload, date(2026, 5, 21), fetched_at="2026-05-21T17:00:00Z")

    assert len(rows) == 1
    assert rows.loc[0, "total_points"] == 8.5
    assert rows.loc[0, "over_price"] == -110
    assert rows.loc[0, "under_price"] == -110
    assert rows.loc[0, "home_moneyline"] is None
    assert rows.loc[0, "away_moneyline"] is None
