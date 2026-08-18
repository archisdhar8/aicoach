import pandas as pd

from mlb_winners.integrity import run_data_integrity_checks


class DummyCon:
    def register(self, *args, **kwargs):
        pass

    def unregister(self, *args, **kwargs):
        pass

    def execute(self, *args, **kwargs):
        return self


def test_integrity_flags_missing_core_inputs(monkeypatch):
    import mlb_winners.integrity as integrity

    monkeypatch.setattr(integrity, "upsert_df", lambda con, table, df: len(df))
    games = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2026-05-18",
                "home_team": "Home",
                "away_team": "Away",
                "home_probable_pitcher": None,
                "away_probable_pitcher": "Starter",
            }
        ]
    )
    checks = run_data_integrity_checks(DummyCon(), games, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2026-05-18").date())

    assert {"missing_starter", "missing_weather", "missing_lineup", "missing_odds"}.issubset(set(checks["check_name"]))
