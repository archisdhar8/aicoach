import pandas as pd

from mlb_winners.predict import make_daily_predictions


def test_make_daily_predictions_empty_slate_has_alert_columns():
    predictions = make_daily_predictions(
        today_games=pd.DataFrame(),
        history_games=pd.DataFrame({"home_won": []}),
        team_stats=pd.DataFrame(),
        statcast_team=None,
        statcast_pitchers=None,
        weather=None,
        odds=pd.DataFrame(),
        model_bundle=object(),
    )

    assert predictions.empty
    for column in ["bet_side", "bet_moneyline", "edge", "ev_per_dollar", "confidence"]:
        assert column in predictions.columns
