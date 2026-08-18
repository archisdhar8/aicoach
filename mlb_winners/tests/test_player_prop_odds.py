from datetime import date

from mlb_winners.odds import normalize_player_prop_payload


def test_normalize_player_prop_payload_pairs_over_under_by_player_and_line():
    payload = {
        "id": "evt1",
        "commence_time": "2026-05-22T23:05:00Z",
        "home_team": "New York Yankees",
        "away_team": "Tampa Bay Rays",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "pitcher_strikeouts",
                        "outcomes": [
                            {"name": "Over", "description": "Jane Pitcher", "price": -115, "point": 5.5},
                            {"name": "Under", "description": "Jane Pitcher", "price": -105, "point": 5.5},
                        ],
                    }
                ],
            }
        ],
    }

    rows = normalize_player_prop_payload(payload, date(2026, 5, 22), fetched_at="2026-05-22T17:00:00Z")

    assert len(rows) == 1
    assert rows.loc[0, "market"] == "pitcher_strikeouts"
    assert rows.loc[0, "player_name"] == "Jane Pitcher"
    assert rows.loc[0, "line"] == 5.5
    assert rows.loc[0, "over_odds"] == -115
    assert rows.loc[0, "under_odds"] == -105
