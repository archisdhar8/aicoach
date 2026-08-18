from mlb_winners.team_map import normalize_team_name


def test_team_aliases_match_common_odds_names():
    assert normalize_team_name("NY Yankees") == "New York Yankees"
    assert normalize_team_name("LA Dodgers") == "Los Angeles Dodgers"
    assert normalize_team_name("Oakland Athletics") == "Athletics"
    assert normalize_team_name("St Louis Cardinals") == "St. Louis Cardinals"
