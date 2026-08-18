from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from .backtest import run_backtests
from .config import Settings, ensure_dirs, load_env_file
from .db import connect, read_table, upsert_df, write_csv
from .feature_analysis import analyze_features
from .feature_store import write_engineered_features
from .features import build_training_frame
from .integrity import run_data_integrity_checks
from .live import fetch_historical_play_by_play
from .market import market_report
from .market_models import predict_f5, predict_hitter_total_bases, predict_nrfi, write_market_predictions
from .mlb_api import fetch_boxscore, fetch_history, fetch_schedule
from .modeling import evaluate_predictions, predict_home_prob, save_bundle, train_baseline, train_model
from .moneyline_diagnostics import evaluate_moneyline, weekly_moneyline_stats, write_moneyline_candidates
from .model_v2 import make_daily_predictions_v2, run_v2_backtests, train_best_v2
from .notifications import filter_upcoming_predictions, format_value_alert, send_sms, send_telegram
from .parlay import build_lotto_parlay, format_lotto_parlay_alert
from .odds import american_profit_per_dollar, fetch_current_odds, fetch_historical_odds, fetch_player_prop_lines, import_historical_odds_csv, implied_prob_to_moneyline
from .portfolio import optimize_portfolio
from .predict import make_daily_predictions
from .player_prop_qualifier import (
    HRR_ODDS_MARKET,
    PropQualificationConfig,
    format_qualified_props_report,
    qualify_player_props,
    write_qualified_prop_outputs,
)
from .props import ODDS_MARKET_BY_PROP, predict_props, run_pitcher_k_backtest, write_props
from .reasoning import add_ollama_reasons
from .record_signal_backtest import backtest_record_signal_strategy
from .results import settle_bet_recommendations
from .simulation import SimulationConfig, add_simulation_columns, simulate_game_from_row, simulate_slate
from .statcast import backfill_pitch_matchups, fetch_statcast_range
from .pitch_prediction import diagnose_pitcher_counts, evaluate_next_pitch_model
from .totals import make_daily_totals_predictions, run_totals_backtests, save_totals_bundle, train_totals_model
from .value_backtest import run_value_backtest
from .weather import fetch_weather_for_games
from soccerworldcup.audit import audit_predictions, format_audit_report, save_audit_report
from soccerworldcup.backtest import run_soccer_backtest
from soccerworldcup.bet_card import format_bet_card, load_bet_card, save_bet_card
from soccerworldcup.challenge_ledger import (
    format_challenge_status,
    format_ledger_summary,
    load_challenge_status,
    write_challenge_ledger,
)
from soccerworldcup.data import (
    fetch_eloratings_ratings,
    fetch_espn_world_cup_scores,
    fetch_soccer_odds,
    fetch_soccer_rankings,
    fetch_soccer_weather,
    fetch_world_cup_schedule,
    load_root_worldcup_schedule,
)
from soccerworldcup.predict import (
    load_soccer_matches_for_date,
    load_soccer_odds_for_date,
    make_soccer_predictions,
    write_soccer_prediction_rows,
)
from soccerworldcup.ratings import build_ratings_for_matches
from soccerworldcup.report import write_daily_report
from soccerworldcup.trends import format_trend_report, load_trend_report


@contextlib.contextmanager
def acquire_cli_lock(settings: Settings, command: str) -> contextlib.AbstractContextManager[None]:
    locks_dir = settings.data_dir / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = locks_dir / "mlb_winners_cli.lock"
    lock_file = lock_path.open("a+")
    try:
        import fcntl

        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another mlb_winners CLI process appears to be running (lock held at {lock_path})."
            ) from exc

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "command": command,
                    "started_at_local": datetime.now().isoformat(timespec="seconds"),
                }
            )
            + "\n"
        )
        lock_file.flush()
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB daily winner and value-bet model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch-history", help="Fetch MLB schedules and boxscores")
    fetch_parser.add_argument("--start-year", type=int, required=True)
    fetch_parser.add_argument("--end-year", type=int, required=True)
    fetch_parser.add_argument("--skip-boxscores", action="store_true")

    statcast_parser = subparsers.add_parser("fetch-statcast", help="Fetch Baseball Savant Statcast aggregates")
    statcast_parser.add_argument("--start-date", type=parse_date, required=True)
    statcast_parser.add_argument("--end-date", type=parse_date, required=True)
    statcast_parser.add_argument("--chunk-days", type=int, default=7)

    pitch_backfill_parser = subparsers.add_parser("backfill-pitch-matchups", help="Resumable Statcast pitch-event and matchup backfill")
    pitch_backfill_parser.add_argument("--start-date", type=parse_date, default=date(2024, 3, 20))
    pitch_backfill_parser.add_argument("--end-date", type=parse_date, default=date.today() - timedelta(days=1))
    pitch_backfill_parser.add_argument("--chunk-days", type=int, default=7)
    pitch_backfill_parser.add_argument("--force", action="store_true")

    pitch_eval_parser = subparsers.add_parser("evaluate-next-pitch", help="Chronologically evaluate next-pitch probabilities")
    pitch_eval_parser.add_argument("--start-date", type=parse_date, required=True)
    pitch_eval_parser.add_argument("--end-date", type=parse_date, required=True)
    pitch_eval_parser.add_argument("--max-pitches", type=int)
    pitch_eval_parser.add_argument("--league-strength", type=float, default=20.0)
    pitch_eval_parser.add_argument("--count-strength", type=float, default=12.0)
    pitch_eval_parser.add_argument("--matchup-strength", type=float, default=40.0)
    pitch_eval_parser.add_argument("--same-game-strength", type=float, default=36.0)

    pitch_diag_parser = subparsers.add_parser("pitch-diagnostic", help="Show count-specific posterior evidence for a pitcher")
    pitch_diag_parser.add_argument("--pitcher-id", type=int, required=True)
    pitch_diag_parser.add_argument("--as-of-date", type=parse_date, default=date.today())
    pitch_diag_parser.add_argument("--batter-side", choices=["L", "R", "S"], default="R")

    weather_parser = subparsers.add_parser("fetch-weather", help="Fetch park weather for cached games")
    weather_parser.add_argument("--start-date", type=parse_date)
    weather_parser.add_argument("--end-date", type=parse_date)
    weather_parser.add_argument("--force", action="store_true")

    refresh_parser = subparsers.add_parser(
        "refresh-recent",
        help="Force-refresh recent schedule/results and cache boxscores for rolling features",
    )
    refresh_parser.add_argument("--date", type=parse_date, default=date.today())
    refresh_parser.add_argument("--days", type=int, default=5)
    refresh_parser.add_argument("--force", action="store_true", help="Force API refresh even if data exists")
    refresh_parser.add_argument("--refresh-statcast", action="store_true", help="Refresh Statcast aggregates if stale")
    refresh_parser.add_argument("--refresh-weather", action="store_true", help="Refresh weather rows if stale")

    integrity_parser = subparsers.add_parser("data-integrity", help="Check slate data quality")
    integrity_parser.add_argument("--date", type=parse_date, required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--train-through", type=int, required=True)
    train_parser.add_argument("--test-year", type=int)

    train_v2_parser = subparsers.add_parser("train-v2", help="Train the selected Model V2 moneyline ensemble")
    train_v2_parser.add_argument("--train-through", type=int, required=True)

    analyze_parser = subparsers.add_parser("analyze-features", help="Analyze feature importance and leakage risk")
    analyze_parser.add_argument("--train-through", type=int, required=True)
    analyze_parser.add_argument("--test-year", type=int, required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Run annual backtests")
    backtest_parser.add_argument("--years", required=True, help="Comma-separated years, e.g. 2021,2022,2023")

    record_signal_parser = subparsers.add_parser(
        "backtest-record-signals",
        help="Backtest record differential + starter FIP proxy + bullpen availability + lineup offense signals",
    )
    record_signal_parser.add_argument("--year", type=int, default=date.today().year)

    eval_moneyline_parser = subparsers.add_parser("evaluate-moneyline", help="Evaluate moneyline model segments and market-prior variants")
    eval_moneyline_parser.add_argument("--years", required=True, help="Comma-separated years, e.g. 2021,2022,2023")

    weekly_moneyline_parser = subparsers.add_parser("moneyline-weekly-stats", help="Report weekly moneyline candidate and official-play stats")
    weekly_moneyline_parser.add_argument("--start-date", type=parse_date, required=True)
    weekly_moneyline_parser.add_argument("--end-date", type=parse_date, required=True)

    backtest_v2_parser = subparsers.add_parser("backtest-v2", help="Run Model V2 moneyline backtests and diagnostics")
    backtest_v2_parser.add_argument("--years", required=True, help="Comma-separated years, e.g. 2021,2022,2023")

    totals_backtest_parser = subparsers.add_parser("backtest-totals", help="Run annual model-only totals backtests")
    totals_backtest_parser.add_argument("--years", required=True, help="Comma-separated years, e.g. 2021,2022,2023")

    totals_odds_parser = subparsers.add_parser("fetch-totals-odds", help="Fetch current/future totals odds")
    totals_odds_parser.add_argument("--date", type=parse_date, required=True)
    totals_odds_parser.add_argument("--api-key")
    totals_odds_parser.add_argument("--bookmaker")
    totals_odds_parser.add_argument("--markets", default="totals")
    totals_odds_parser.add_argument("--force", action="store_true")

    totals_predict_parser = subparsers.add_parser("predict-totals-today", help="Predict daily MLB totals and value")
    totals_predict_parser.add_argument("--date", type=parse_date, default=date.today())
    totals_predict_parser.add_argument("--fetch-odds", action="store_true")
    totals_predict_parser.add_argument("--api-key")
    totals_predict_parser.add_argument("--bookmaker")
    totals_predict_parser.add_argument("--odds-markets", default="totals")
    totals_predict_parser.add_argument("--force-odds", action="store_true")

    odds_parser = subparsers.add_parser("fetch-odds", help="Fetch current/future moneyline odds")
    odds_parser.add_argument("--date", type=parse_date, required=True)
    odds_parser.add_argument("--api-key")
    odds_parser.add_argument("--bookmaker")
    odds_parser.add_argument("--markets", default="h2h")
    odds_parser.add_argument("--force", action="store_true")

    market_parser = subparsers.add_parser("market-report", help="Summarize odds movement and disagreement")
    market_parser.add_argument("--date", type=parse_date, required=True)

    import_odds_parser = subparsers.add_parser("import-odds-csv", help="Import historical moneyline odds CSV")
    import_odds_parser.add_argument("--path", required=True)
    import_odds_parser.add_argument("--source", default="csv_import")

    hist_odds_parser = subparsers.add_parser("fetch-historical-odds", help="Fetch historical moneyline odds")
    hist_odds_parser.add_argument("--start-date", type=parse_date, required=True)
    hist_odds_parser.add_argument("--end-date", type=parse_date, required=True)
    hist_odds_parser.add_argument("--api-key")
    hist_odds_parser.add_argument("--bookmaker")
    hist_odds_parser.add_argument("--snapshot-time-utc", default="16:00:00")
    hist_odds_parser.add_argument("--max-days", type=int, default=49)
    hist_odds_parser.add_argument("--force", action="store_true")

    value_parser = subparsers.add_parser("backtest-value", help="Backtest top value bets with historical odds")
    value_parser.add_argument("--year", type=int, required=True)
    value_parser.add_argument("--top-n", type=int, default=3)
    value_parser.add_argument("--confidence", default="strong")

    sim_game_parser = subparsers.add_parser("simulate-game", help="Run deterministic game simulations")
    sim_game_parser.add_argument("--date", type=parse_date, required=True)
    sim_game_parser.add_argument("--game-pk", type=int, required=True)
    sim_game_parser.add_argument("--sims", type=int, default=20000)
    sim_game_parser.add_argument("--seed", type=int, default=42)

    sim_slate_parser = subparsers.add_parser("simulate-slate", help="Run deterministic simulations for a slate")
    sim_slate_parser.add_argument("--date", type=parse_date, required=True)
    sim_slate_parser.add_argument("--sims", type=int, default=20000)
    sim_slate_parser.add_argument("--seed", type=int, default=42)

    props_parser = subparsers.add_parser("predict-props", help="Predict starter and batter props")
    props_parser.add_argument("--date", type=parse_date, required=True)
    props_parser.add_argument("--market", choices=["strikeouts", "hits_allowed", "total_bases", "hr", "hits", "rbi", "runs", "earned_runs", "outs_recorded"], required=True)

    player_logs_parser = subparsers.add_parser("fetch-player-logs", help="Fetch player game logs from MLB boxscores")
    player_logs_parser.add_argument("--start-year", type=int, required=True)
    player_logs_parser.add_argument("--end-year", type=int, required=True)

    player_props_parser = subparsers.add_parser("fetch-player-props", help="Fetch current player prop odds")
    player_props_parser.add_argument("--date", type=parse_date, required=True)
    player_props_parser.add_argument("--market", choices=["strikeouts", "hits_allowed", "total_bases", "hr", "hits", "rbi", "runs", "earned_runs", "outs_recorded"], required=True)
    player_props_parser.add_argument("--api-key")
    player_props_parser.add_argument("--bookmaker")
    player_props_parser.add_argument("--max-events", type=int)
    player_props_parser.add_argument("--force", action="store_true")

    props_today_parser = subparsers.add_parser("predict-props-today", help="Predict daily player props and value")
    props_today_parser.add_argument("--date", type=parse_date, required=True)
    props_today_parser.add_argument("--market", choices=["strikeouts", "hits_allowed", "total_bases", "hr", "hits", "rbi", "runs", "earned_runs", "outs_recorded"], required=True)
    props_today_parser.add_argument("--fetch-odds", action="store_true")
    props_today_parser.add_argument("--api-key")
    props_today_parser.add_argument("--bookmaker")
    props_today_parser.add_argument("--max-events", type=int)
    props_today_parser.add_argument("--force-odds", action="store_true")

    qualified_props_parser = subparsers.add_parser(
        "predict-qualified-player-props",
        help="Qualify HRR and pitcher strikeout player props using model probability, odds edge, and data quality",
    )
    qualified_props_parser.add_argument("--date", type=parse_date, required=True)
    qualified_props_parser.add_argument("--fetch-odds", action="store_true")
    qualified_props_parser.add_argument("--api-key")
    qualified_props_parser.add_argument("--bookmaker")
    qualified_props_parser.add_argument("--max-events", type=int)
    qualified_props_parser.add_argument("--force-odds", action="store_true")
    qualified_props_parser.add_argument("--min-edge", type=float, default=0.04)
    qualified_props_parser.add_argument("--min-ev", type=float, default=0.05)
    qualified_props_parser.add_argument("--min-data-quality", type=float, default=0.75)
    qualified_props_parser.add_argument("--max-odds-age-hours", type=float, default=8.0)
    qualified_props_parser.add_argument("--allow-unconfirmed-lineups", action="store_true")
    qualified_props_parser.add_argument("--telegram", action="store_true")

    qualified_props_backtest_parser = subparsers.add_parser(
        "backtest-qualified-player-props",
        help="Grade and summarize stored qualified player prop recommendations over a date range",
    )
    qualified_props_backtest_parser.add_argument("--start-date", type=parse_date, required=True)
    qualified_props_backtest_parser.add_argument("--end-date", type=parse_date, required=True)
    qualified_props_backtest_parser.add_argument("--grade", action="store_true")

    pitcher_k_parser = subparsers.add_parser("predict-pitcher-k-today", help="Predict today's pitcher strikeout props")
    pitcher_k_parser.add_argument("--date", type=parse_date, required=True)
    pitcher_k_parser.add_argument("--fetch-odds", action="store_true")
    pitcher_k_parser.add_argument("--api-key")
    pitcher_k_parser.add_argument("--bookmaker")
    pitcher_k_parser.add_argument("--max-events", type=int)
    pitcher_k_parser.add_argument("--force-odds", action="store_true")

    backtest_pitcher_k_parser = subparsers.add_parser("backtest-pitcher-k", help="Backtest and audit pitcher strikeout projections")
    backtest_pitcher_k_parser.add_argument("--start-year", type=int, default=2021)
    backtest_pitcher_k_parser.add_argument("--end-year", type=int, default=2025)

    nrfi_parser = subparsers.add_parser("predict-nrfi-today", help="Scaffold daily NRFI predictions")
    nrfi_parser.add_argument("--date", type=parse_date, required=True)

    f5_parser = subparsers.add_parser("predict-f5-today", help="Scaffold daily first-five predictions")
    f5_parser.add_argument("--date", type=parse_date, required=True)

    hitter_tb_parser = subparsers.add_parser("predict-hitter-tb-today", help="Predict hitter total bases props")
    hitter_tb_parser.add_argument("--date", type=parse_date, required=True)
    hitter_tb_parser.add_argument("--fetch-odds", action="store_true")
    hitter_tb_parser.add_argument("--api-key")
    hitter_tb_parser.add_argument("--bookmaker")
    hitter_tb_parser.add_argument("--max-events", type=int)
    hitter_tb_parser.add_argument("--force-odds", action="store_true")

    grade_props_parser = subparsers.add_parser("grade-props", help="Grade stored player prop recommendations")
    grade_props_parser.add_argument("--date", type=parse_date, required=True)

    portfolio_parser = subparsers.add_parser("backtest-portfolio", help="Backtest value bets with staking controls")
    portfolio_parser.add_argument("--year", type=int, required=True)
    portfolio_parser.add_argument("--markets", default="moneyline")
    portfolio_parser.add_argument("--staking", choices=["flat", "kelly"], default="flat")
    portfolio_parser.add_argument("--top-n", type=int, default=3)

    results_parser = subparsers.add_parser("record-results", help="Settle stored recommendations from final scores")
    results_parser.add_argument("--year", type=int, required=True)

    recap_parser = subparsers.add_parser("send-daily-recap", help="Send settled moneyline record and units recap")
    recap_parser.add_argument("--date", type=parse_date, default=date.today(), help="Run date; defaults to today")
    recap_parser.add_argument("--results-date", type=parse_date, help="Date to recap; defaults to prior day")
    recap_parser.add_argument("--settle", action="store_true", help="Settle stored recommendations before reporting")
    recap_parser.add_argument("--sms", action="store_true")
    recap_parser.add_argument("--telegram", action="store_true")

    alert_parser = subparsers.add_parser("send-alerts", help="Send strong/medium plays from a prediction report")
    alert_parser.add_argument("--date", type=parse_date, required=True)
    alert_parser.add_argument("--sms", action="store_true")
    alert_parser.add_argument("--telegram", action="store_true")
    alert_parser.add_argument("--include-started", action="store_true")
    alert_parser.add_argument("--window-minutes", type=int, default=60)
    alert_parser.add_argument("--force-resend", action="store_true")
    alert_parser.add_argument("--send-empty", action="store_true")
    alert_parser.add_argument("--no-ollama-explain", action="store_true", help="Disable local Ollama reasons in alerts")
    alert_parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3"))
    alert_parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"))

    lotto_parser = subparsers.add_parser("send-lotto-parlay", help="Send an upcoming 4+ leg lotto parlay")
    lotto_parser.add_argument("--date", type=parse_date, required=True)
    lotto_parser.add_argument("--min-legs", type=int, default=4)
    lotto_parser.add_argument("--max-legs", type=int, default=8)
    lotto_parser.add_argument("--stake-units", type=float, default=0.1)
    lotto_parser.add_argument("--window-minutes", type=int, default=360)
    lotto_parser.add_argument("--telegram", action="store_true")
    lotto_parser.add_argument("--sms", action="store_true")
    lotto_parser.add_argument("--send-empty", action="store_true")
    lotto_parser.add_argument("--force-resend", action="store_true")

    predict_parser = subparsers.add_parser("predict-today", help="Predict a date's MLB games")
    predict_parser.add_argument("--date", type=parse_date, default=date.today())
    predict_parser.add_argument("--fetch-odds", action="store_true")
    predict_parser.add_argument("--api-key")
    predict_parser.add_argument("--bookmaker")
    predict_parser.add_argument("--odds-markets", default="h2h")
    predict_parser.add_argument("--force-odds", action="store_true")

    predict_v2_parser = subparsers.add_parser("predict-today-v2", help="Predict a date's MLB games with Model V2")
    predict_v2_parser.add_argument("--date", type=parse_date, default=date.today())
    predict_v2_parser.add_argument("--fetch-odds", action="store_true")
    predict_v2_parser.add_argument("--api-key")
    predict_v2_parser.add_argument("--bookmaker")
    predict_v2_parser.add_argument("--odds-markets", default="h2h")
    predict_v2_parser.add_argument("--force-odds", action="store_true")
    predict_v2_parser.add_argument("--include-simulation", action="store_true", help="Add Monte Carlo run-distribution columns to the V2 report")
    predict_v2_parser.add_argument("--sims", type=int, default=10000, help="Number of Monte Carlo simulations when --include-simulation is used")
    predict_v2_parser.add_argument("--seed", type=int, default=42, help="Simulation seed when --include-simulation is used")

    soccer_schedule_parser = subparsers.add_parser("fetch-soccer-schedule", help="Fetch FIFA World Cup 2026 schedule and results")
    soccer_schedule_parser.add_argument("--start-date", type=parse_date, required=True)
    soccer_schedule_parser.add_argument("--end-date", type=parse_date, required=True)
    soccer_schedule_parser.add_argument("--force", action="store_true")

    soccer_scores_parser = subparsers.add_parser("fetch-soccer-scores", help="Fetch FIFA World Cup final scores")
    soccer_scores_parser.add_argument("--date", type=parse_date, required=True)
    soccer_scores_parser.add_argument("--force", action="store_true")

    soccer_ratings_parser = subparsers.add_parser("fetch-soccer-ratings", help="Fetch optional FIFA rankings and build local soccer ratings")
    soccer_ratings_parser.add_argument("--as-of-date", type=parse_date, required=True)
    soccer_ratings_parser.add_argument("--force", action="store_true")

    soccer_odds_parser = subparsers.add_parser("fetch-soccer-odds", help="Fetch current FIFA World Cup odds")
    soccer_odds_parser.add_argument("--date", type=parse_date, required=True)
    soccer_odds_parser.add_argument("--api-key")
    soccer_odds_parser.add_argument("--bookmaker")
    soccer_odds_parser.add_argument("--markets", default="h2h,totals,spreads")
    soccer_odds_parser.add_argument("--force", action="store_true")

    soccer_predict_parser = subparsers.add_parser("predict-soccer-today", help="Predict a date's FIFA World Cup matches")
    soccer_predict_parser.add_argument("--date", type=parse_date, default=date.today())
    soccer_predict_parser.add_argument("--fetch-odds", action="store_true")
    soccer_predict_parser.add_argument("--force-odds", action="store_true")

    world_cup_predict_parser = subparsers.add_parser("predict-world-cup", help="Alias for predict-soccer-today")
    world_cup_predict_parser.add_argument("--date", type=parse_date, default=date.today())
    world_cup_predict_parser.add_argument("--fetch-odds", action="store_true")
    world_cup_predict_parser.add_argument("--force-odds", action="store_true")

    soccer_backtest_parser = subparsers.add_parser("backtest-soccer", help="Run model-only soccer backtests")
    soccer_backtest_parser.add_argument("--start-date", type=parse_date, required=True)
    soccer_backtest_parser.add_argument("--end-date", type=parse_date, required=True)

    soccer_audit_parser = subparsers.add_parser("audit-soccer-predictions", help="Audit FIFA World Cup prediction behavior")
    soccer_audit_parser.add_argument("--date", type=parse_date, required=True)
    soccer_audit_parser.add_argument("--no-save", action="store_true")

    soccer_bet_card_parser = subparsers.add_parser("soccer-bet-card", help="Print approved FIFA World Cup betting card")
    soccer_bet_card_parser.add_argument("--date", type=parse_date, required=True)
    soccer_bet_card_parser.add_argument("--no-save", action="store_true")
    soccer_bet_card_parser.add_argument("--mode", choices=["strict", "challenge", "must-play"], default="strict")
    soccer_bet_card_parser.add_argument("--bankroll", type=float, default=300.0)
    soccer_bet_card_parser.add_argument("--target", type=float, default=1500.0)
    soccer_bet_card_parser.add_argument("--allow-lottery", action="store_true")

    soccer_challenge_ledger_parser = subparsers.add_parser("soccer-challenge-ledger", help="Write World Cup challenge ledger for a date")
    soccer_challenge_ledger_parser.add_argument("--date", type=parse_date, required=True)
    soccer_challenge_ledger_parser.add_argument("--bankroll", type=float, default=300.0)
    soccer_challenge_ledger_parser.add_argument("--target", type=float, default=1500.0)
    soccer_challenge_ledger_parser.add_argument("--allow-lottery", action="store_true")
    soccer_challenge_ledger_parser.add_argument("--no-grade", action="store_true")

    soccer_challenge_status_parser = subparsers.add_parser("soccer-challenge-status", help="Show World Cup challenge performance")
    soccer_challenge_status_parser.add_argument("--target", type=float, default=1500.0)

    soccer_trends_parser = subparsers.add_parser("soccer-trends", help="Analyze World Cup trend support for soccer bets")
    soccer_trends_parser.add_argument("--date", type=parse_date, required=True)

    live_history_parser = subparsers.add_parser(
        "fetch-play-by-play", help="Backfill MLB pitch/play events for the live probability model"
    )
    live_history_parser.add_argument("--start-date", type=parse_date, required=True)
    live_history_parser.add_argument("--end-date", type=parse_date, required=True)
    live_history_parser.add_argument("--max-games", type=int)
    live_history_parser.add_argument("--force", action="store_true")

    live_dashboard_parser = subparsers.add_parser("serve-live-dashboard", help="Run the local MLB live dashboard")
    live_dashboard_parser.add_argument("--host", default="127.0.0.1")
    live_dashboard_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    load_env_file()
    settings = Settings()
    ensure_dirs(settings)
    con = connect(settings)

    lock_commands = {
        "predict-today",
        "predict-today-v2",
        "send-alerts",
        "send-daily-recap",
        "predict-totals-today",
        "predict-props-today",
        "predict-pitcher-k-today",
        "predict-nrfi-today",
        "predict-f5-today",
        "predict-hitter-tb-today",
        "grade-props",
        "refresh-recent",
        "fetch-odds",
        "fetch-totals-odds",
        "fetch-player-props",
        "fetch-soccer-odds",
        "fetch-soccer-scores",
        "fetch-soccer-schedule",
        "predict-soccer-today",
        "predict-world-cup",
    }
    if args.command in lock_commands:
        # Intentionally not released: when the CLI exits, the OS closes the file descriptor
        # and releases the lock. This avoids refactoring the command-dispatch below.
        lock_guard = acquire_cli_lock(settings, args.command)
        lock_guard.__enter__()

    if args.command == "fetch-history":
        counts = fetch_history(con, args.start_year, args.end_year, include_boxscores=not args.skip_boxscores)
        for season, count in counts.items():
            print(f"{season}: cached {count} final regular-season games")
        return

    if args.command == "fetch-play-by-play":
        if args.start_date < date(2024, 1, 1):
            raise RuntimeError("Pitch/play history for this model begins in 2024.")
        slate = con.execute(
            """
            SELECT game_pk, game_date FROM games
            WHERE game_date BETWEEN ? AND ? AND game_type='R' AND status ILIKE '%final%'
            ORDER BY game_date, game_pk
            """, [args.start_date, args.end_date],
        ).fetchall()
        if args.max_games:
            slate = slate[: args.max_games]
        events = failures = 0
        for game_pk, game_date in slate:
            try:
                events += fetch_historical_play_by_play(con, int(game_pk), game_date, force=args.force)
            except Exception as exc:
                failures += 1
                print(f"warning: play-by-play failed for game_pk={game_pk} ({exc})")
        print(f"play-by-play games={len(slate) - failures} failures={failures} event_rows={events}")
        return

    if args.command == "serve-live-dashboard":
        con.close()
        import uvicorn

        uvicorn.run("mlb_winners.dashboard:app", host=args.host, port=args.port, reload=False)
        return

    if args.command == "fetch-statcast":
        counts = fetch_statcast_range(con, args.start_date, args.end_date, chunk_days=args.chunk_days)
        print(
            f"statcast team rows: {counts['team_daily_rows']} "
            f"pitcher rows: {counts['pitcher_daily_rows']} "
            f"pitch matchup rows: {counts['pitch_matchup_rows']}"
        )
        return

    if args.command == "backfill-pitch-matchups":
        report = backfill_pitch_matchups(con, args.start_date, args.end_date, chunk_days=args.chunk_days, force=args.force)
        print(json.dumps(report, indent=2, default=str))
        return

    if args.command == "evaluate-next-pitch":
        report = evaluate_next_pitch_model(
            con, args.start_date, args.end_date, max_pitches=args.max_pitches,
            posterior_kwargs={"league_strength": args.league_strength,
                              "count_prior_strength": args.count_strength,
                              "matchup_prior_strength": args.matchup_strength,
                              "same_game_prior_strength": args.same_game_strength},
        )
        print(json.dumps(report, indent=2, default=str))
        return

    if args.command == "pitch-diagnostic":
        print(json.dumps(diagnose_pitcher_counts(con, args.pitcher_id, args.as_of_date, args.batter_side), indent=2, default=str))
        return

    if args.command == "fetch-weather":
        games = read_table(con, "games")
        if games.empty:
            raise RuntimeError("No games found. Run fetch-history first.")
        games["game_date"] = pd.to_datetime(games["game_date"])
        if args.start_date:
            games = games[games["game_date"] >= pd.Timestamp(args.start_date)]
        if args.end_date:
            games = games[games["game_date"] <= pd.Timestamp(args.end_date)]
        rows = fetch_weather_for_games(con, games, force=args.force)
        print(f"weather rows cached: {rows}")
        return

    if args.command == "refresh-recent":
        if args.days < 1:
            raise RuntimeError("--days must be >= 1")
        start_date = args.date - timedelta(days=args.days - 1)
        end_date = args.date
        slate = fetch_schedule(con, start_date, end_date, end_date.year, force=True)
        print(f"schedule rows upserted: {len(slate)} ({start_date.isoformat()} to {end_date.isoformat()})")

        # Cache final-game boxscores for the window (rolling features).
        final_mask = slate["status"].astype(str).str.lower().str.contains("final", na=False)
        final_game_pks = slate.loc[final_mask, "game_pk"].dropna().astype(int).tolist()
        final_failures = 0
        for game_pk in final_game_pks:
            try:
                fetch_boxscore(con, int(game_pk), force=args.force)
            except RuntimeError as exc:
                final_failures += 1
                print(f"warning: boxscore refresh failed for game_pk={int(game_pk)} ({exc})")
        print(f"final boxscores refreshed: {len(final_game_pks) - final_failures} failures={final_failures}")

        # Refresh today's boxscore/lineup snapshots for all scheduled games.
        today_mask = slate["game_date"].astype(str) == end_date.isoformat()
        today_game_pks = slate.loc[today_mask, "game_pk"].dropna().astype(int).tolist()
        today_failures = 0
        for game_pk in today_game_pks:
            try:
                fetch_boxscore(con, int(game_pk), force=True)
            except RuntimeError as exc:
                today_failures += 1
                print(f"warning: today boxscore refresh failed for game_pk={int(game_pk)} ({exc})")
        print(f"today boxscore/lineup snapshots refreshed: {len(today_game_pks) - today_failures} failures={today_failures}")

        if args.refresh_weather:
            weather_rows = fetch_weather_for_games(con, slate, force=args.force)
            print(f"weather rows cached: {weather_rows}")

        if args.refresh_statcast:
            try:
                statcast_counts = fetch_statcast_range(con, start_date, end_date)
                print(
                    f"statcast team rows: {statcast_counts['team_daily_rows']} "
                    f"pitcher rows: {statcast_counts['pitcher_daily_rows']} "
                    f"pitch matchup rows: {statcast_counts['pitch_matchup_rows']}"
                )
            except Exception as exc:
                print(f"warning: statcast refresh failed ({exc})")
        return

    if args.command == "data-integrity":
        games, _, _, _, weather = load_dataset(con)
        odds = read_table(con, "odds_snapshots")
        lineups = read_table(con, "lineups")
        checks = run_data_integrity_checks(con, games, odds, weather, lineups, args.date)
        print(checks[["severity", "check_name", "game_pk", "message"]].to_string(index=False))
        return

    if args.command == "train":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        frame = build_training_frame(games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
        write_engineered_features(con, frame)
        train = frame[(frame["season"] <= args.train_through) & frame["target_home_win"].notna()]
        bundle = train_model(train)
        model_path = settings.model_dir / f"xgb_calibrated_through_{args.train_through}.joblib"
        save_bundle(bundle, model_path)
        print(f"saved model: {model_path}")
        print(f"training games: {bundle.train_rows}")
        if args.test_year:
            test = frame[(frame["season"] == args.test_year) & frame["target_home_win"].notna()]
            if not test.empty:
                probs = predict_home_prob(bundle, test)
                print(pd.Series(evaluate_predictions(test["target_home_win"], probs)).to_string())
        return

    if args.command == "train-v2":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        frame = build_training_frame(games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
        bundle, best = train_best_v2(con, frame, args.train_through, settings.report_dir, settings.model_dir)
        print(f"saved V2 model through {args.train_through}")
        print(f"feature_set={bundle.feature_set} include_bullpen={bundle.include_bullpen} weights={bundle.weights_name}")
        print(f"selected_from={best}")
        return

    if args.command == "analyze-features":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        frame = build_training_frame(games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
        write_engineered_features(con, frame)
        result = analyze_features(con, frame, args.train_through, args.test_year, settings.report_dir)
        print(pd.Series(result.metrics).to_string())
        print(result.summary.head(40).to_string(index=False))
        print(f"feature reports written to {settings.report_dir}")
        return

    if args.command == "backtest":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
        summary = run_backtests(games, team_stats, years, settings.report_dir, statcast_team, statcast_pitchers, weather)
        print(summary.to_string(index=False))
        print(f"reports written to {settings.report_dir}")
        return

    if args.command == "backtest-record-signals":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        frame = build_training_frame(games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
        reports = backtest_record_signal_strategy(frame, args.year, settings.report_dir)
        print("Best weight combinations:")
        print(reports["summary"].head(10).to_string(index=False))
        best_label = str(reports["summary"].iloc[0]["weight_combo"])
        print(f"\nBest by record-gap bucket ({best_label}):")
        print(reports["by_bucket"][reports["by_bucket"]["weight_combo"].eq(best_label)].to_string(index=False))
        print("\nBaselines:")
        print(reports["baselines"].head(25).to_string(index=False))
        print(f"reports written to {settings.report_dir}")
        return

    if args.command == "evaluate-moneyline":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        odds = read_table(con, "odds_snapshots")
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
        result = evaluate_moneyline(
            con,
            games,
            team_stats,
            years,
            settings.report_dir,
            statcast_team=statcast_team,
            statcast_pitchers=statcast_pitchers,
            weather=weather,
            lineups=lineups,
            player_stats=player_stats,
            odds=odds,
        )
        print(result.summary.to_string(index=False) if not result.summary.empty else "no moneyline summary rows")
        if not result.market_variants.empty:
            print(result.market_variants.to_string(index=False))
        print(f"moneyline evaluation reports written to {settings.report_dir}")
        return

    if args.command == "moneyline-weekly-stats":
        result = weekly_moneyline_stats(con, args.start_date, args.end_date, settings.report_dir)
        if result.summary.empty:
            print(f"No moneyline candidates found from {args.start_date} to {args.end_date}.")
            return
        print("SUMMARY")
        print(result.summary.to_string(index=False))
        print("\nSEGMENTS")
        show_cols = ["segment_type", "segment", "candidates", "final_graded", "pending", "record", "win_pct", "units", "avg_edge", "avg_ev"]
        print(result.segments[show_cols].to_string(index=False))
        print(f"weekly moneyline stats written to {settings.report_dir}")
        return

    if args.command == "backtest-v2":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
        result = run_v2_backtests(
            con,
            games,
            team_stats,
            years,
            settings.report_dir,
            statcast_team=statcast_team,
            statcast_pitchers=statcast_pitchers,
            weather=weather,
            lineups=lineups,
            player_stats=player_stats,
        )
        experiments = result["experiments"]
        best = result["best"]
        print(
            experiments.groupby("model_version")[["accuracy", "auc", "brier", "log_loss", "ece"]]
            .mean()
            .sort_values(["brier", "ece"])
            .head(10)
            .to_string()
        )
        print(f"best={best}")
        print(f"V2 reports written to {settings.report_dir}; markdown: {settings.report_dir / 'MODEL_V2_RESULTS.md'}")
        return

    if args.command == "backtest-totals":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
        summary = run_totals_backtests(games, team_stats, years, settings.report_dir, statcast_team, statcast_pitchers, weather)
        print(summary.to_string(index=False))
        print(f"totals reports written to {settings.report_dir}")
        print("ROI note: totals value ROI requires historical totals lines or pregame odds snapshots.")
        return

    if args.command == "fetch-totals-odds":
        result = fetch_current_odds(
            con,
            snapshot_date=args.date,
            api_key=args.api_key,
            force=args.force,
            bookmaker=args.bookmaker,
            markets=args.markets,
        )
        cache_note = "cache" if result.from_cache else "api"
        print(
            f"totals odds rows: {result.rows} source={cache_note} "
            f"remaining={result.remaining} used={result.used} last={result.last}"
        )
        return

    if args.command == "fetch-odds":
        result = fetch_current_odds(
            con,
            snapshot_date=args.date,
            api_key=args.api_key,
            force=args.force,
            bookmaker=args.bookmaker,
            markets=args.markets,
        )
        cache_note = "cache" if result.from_cache else "api"
        print(
            f"odds rows: {result.rows} source={cache_note} "
            f"remaining={result.remaining} used={result.used} last={result.last}"
        )
        return

    if args.command == "fetch-soccer-schedule":
        result = fetch_world_cup_schedule(con, args.start_date, args.end_date, force=args.force)
        local_rows = 0
        if result.rows == 0:
            local_schedule = load_root_worldcup_schedule(args.start_date, args.end_date)
            local_rows = upsert_df(con, "soccer_matches", local_schedule) if not local_schedule.empty else 0
        score_rows = 0
        current = args.start_date
        while current <= args.end_date:
            scores_result = fetch_espn_world_cup_scores(con, current, force=args.force)
            score_rows += scores_result.rows
            current += timedelta(days=1)
        print(
            f"soccer schedule rows: {result.rows} source={result.source} "
            f"cache={result.from_cache} message={result.message}"
        )
        if local_rows:
            print(f"root soccerworldcup schedule rows loaded: {local_rows}")
        print(f"espn soccer score rows synced: {score_rows}")
        return

    if args.command == "fetch-soccer-scores":
        result = fetch_espn_world_cup_scores(con, args.date, force=args.force)
        print(
            f"soccer score rows synced: {result.rows} source={result.source} "
            f"cache={result.from_cache} message={result.message}"
        )
        return

    if args.command == "fetch-soccer-ratings":
        elo_result = fetch_eloratings_ratings(con, args.as_of_date, force=args.force)
        ranking_result = fetch_soccer_rankings(con, args.as_of_date, force=args.force)
        matches = read_table(con, "soccer_matches")
        if matches.empty:
            print(
                f"elo rating rows: {elo_result.rows} source={elo_result.source} "
                f"cache={elo_result.from_cache} message={elo_result.message}"
            )
            print(
                f"fifa ranking rows: {ranking_result.rows} source={ranking_result.source} "
                f"cache={ranking_result.from_cache} message={ranking_result.message}"
            )
            print("no soccer matches cached; run fetch-soccer-schedule before building local ratings")
            return
        matches["kickoff_utc"] = pd.to_datetime(matches["kickoff_utc"], utc=True, errors="coerce")
        teams = pd.concat([matches["home_team"], matches["away_team"]]).dropna().drop_duplicates().tolist()
        pseudo_matches = pd.DataFrame(
            [
                {
                    "match_id": f"rating-{team}",
                    "kickoff_utc": pd.Timestamp(args.as_of_date).tz_localize("UTC"),
                    "home_team": team,
                    "away_team": team,
                }
                for team in teams
            ]
        )
        fifa_rankings = read_table(con, "soccer_team_ratings")
        if not fifa_rankings.empty:
            fifa_rankings = fifa_rankings[fifa_rankings["source"] == "footballdata_io"]
        ratings = build_ratings_for_matches(matches, pseudo_matches, fifa_rankings=fifa_rankings)
        if not ratings.empty:
            ratings = ratings.drop(columns=["match_id"], errors="ignore").drop_duplicates(["team", "as_of_date"], keep="last")
            ratings["source"] = "local_world_cup_v1"
            ratings["created_at"] = pd.Timestamp.now(tz="UTC")
            upsert_df(con, "soccer_team_ratings", ratings)
        print(
            f"elo rating rows: {elo_result.rows} source={elo_result.source} "
            f"cache={elo_result.from_cache} message={elo_result.message}"
        )
        print(
            f"fifa ranking rows: {ranking_result.rows} source={ranking_result.source} "
            f"cache={ranking_result.from_cache} message={ranking_result.message}"
        )
        print(f"local soccer rating rows: {len(ratings)}")
        return

    if args.command == "fetch-soccer-odds":
        result = fetch_soccer_odds(
            con,
            snapshot_date=args.date,
            api_key=args.api_key,
            force=args.force,
            bookmaker=args.bookmaker,
            markets=args.markets,
        )
        print(
            f"soccer odds rows: {result.rows} source={result.source} "
            f"cache={result.from_cache} message={result.message}"
        )
        return

    if args.command in {"predict-soccer-today", "predict-world-cup"}:
        if args.fetch_odds:
            odds_result = fetch_soccer_odds(con, snapshot_date=args.date, force=args.force_odds)
            print(
                f"soccer odds rows: {odds_result.rows} source={odds_result.source} "
                f"cache={odds_result.from_cache} message={odds_result.message}"
            )
        schedule_result = fetch_world_cup_schedule(con, args.date, args.date)
        if schedule_result.message:
            print(f"schedule note: {schedule_result.message}")
        if schedule_result.rows == 0:
            local_schedule = load_root_worldcup_schedule(args.date, args.date)
            local_rows = upsert_df(con, "soccer_matches", local_schedule) if not local_schedule.empty else 0
            if local_rows:
                print(f"root soccerworldcup schedule rows loaded: {local_rows}")
        fetch_soccer_weather(con, args.date)
        matches = load_soccer_matches_for_date(con, args.date)
        if matches.empty:
            print(f"no soccer matches found for {args.date.isoformat()}")
            return
        all_matches = read_table(con, "soccer_matches")
        fifa_rankings = read_table(con, "soccer_team_ratings")
        if not fifa_rankings.empty:
            fifa_rankings = fifa_rankings[fifa_rankings["source"].isin(["eloratings_net", "footballdata_io", "local_world_cup_v1"])]
        odds = load_soccer_odds_for_date(con, args.date)
        predictions = make_soccer_predictions(matches, all_matches, odds=odds, fifa_rankings=fifa_rankings)
        stored = write_soccer_prediction_rows(con, predictions)
        output_path = settings.report_dir / f"soccer_predictions_{args.date.isoformat()}.csv"
        report_path = settings.report_dir / f"soccer_world_cup_report_{args.date.isoformat()}.md"
        write_csv(predictions, output_path)
        write_daily_report(predictions, report_path)
        print(predictions.to_string(index=False))
        print(f"soccer prediction rows stored: {stored}")
        print(f"soccer predictions written to {output_path}")
        print(f"soccer report written to {report_path}")
        return

    if args.command == "backtest-soccer":
        matches = read_table(con, "soccer_matches")
        if matches.empty:
            raise RuntimeError("No soccer matches found. Run fetch-soccer-schedule first.")
        result = run_soccer_backtest(matches, args.start_date, args.end_date)
        output_path = settings.report_dir / f"soccer_backtest_{args.start_date.isoformat()}_{args.end_date.isoformat()}.csv"
        write_csv(result, output_path)
        print(result.to_string(index=False) if not result.empty else "no completed soccer matches found")
        print("ROI note: soccer value ROI requires valid historical odds or pregame snapshots.")
        print(f"soccer backtest written to {output_path}")
        return

    if args.command == "audit-soccer-predictions":
        rows = con.execute(
            """
            SELECT *
            FROM soccer_predictions
            WHERE match_date = ?
            ORDER BY created_at DESC
            """,
            [args.date],
        ).df()
        result = audit_predictions(rows, args.date)
        print(format_audit_report(result))
        if not args.no_save:
            path = save_audit_report(result, settings.report_dir)
            print(f"audit report written to {path}")
        return

    if args.command == "soccer-bet-card":
        card = load_bet_card(settings.db_path, args.date)
        print(
            format_bet_card(
                card,
                mode=args.mode,
                bankroll=args.bankroll,
                target=args.target,
                allow_lottery=args.allow_lottery,
            )
        )
        if not args.no_save:
            path = save_bet_card(
                card,
                settings.report_dir,
                mode=args.mode,
                bankroll=args.bankroll,
                target=args.target,
                allow_lottery=args.allow_lottery,
            )
            print(f"bet card written to {path}")
        return

    if args.command == "soccer-challenge-ledger":
        card = load_bet_card(settings.db_path, args.date)
        result = write_challenge_ledger(
            con,
            card,
            bankroll=args.bankroll,
            target=args.target,
            allow_lottery=args.allow_lottery,
            grade=not args.no_grade,
        )
        print(format_ledger_summary(result))
        return

    if args.command == "soccer-challenge-status":
        print(format_challenge_status(load_challenge_status(con, target=args.target)))
        return

    if args.command == "soccer-trends":
        print(format_trend_report(load_trend_report(settings.db_path, args.date)))
        return

    if args.command == "market-report":
        odds = read_table(con, "odds_snapshots")
        report = market_report(odds, args.date)
        output_path = settings.report_dir / f"market_report_{args.date.isoformat()}.csv"
        write_csv(report, output_path)
        print(report.to_string(index=False) if not report.empty else "no odds snapshots found")
        print(f"market report written to {output_path}")
        return

    if args.command == "import-odds-csv":
        rows = import_historical_odds_csv(con, args.path, source=args.source)
        print(f"imported odds rows: {rows}")
        return

    if args.command == "fetch-historical-odds":
        games = read_table(con, "games")
        if games.empty:
            raise RuntimeError("No games found. Run fetch-history first.")
        games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
        dates = (
            games[
                (games["game_date"] >= args.start_date)
                & (games["game_date"] <= args.end_date)
                & (games["status"].isin(["Final", "Completed Early"]))
            ]["game_date"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if args.max_days:
            dates = dates[-args.max_days :]
        for snapshot_date in dates:
            result = fetch_historical_odds(
                con,
                snapshot_date=snapshot_date,
                api_key=args.api_key,
                snapshot_time_utc=args.snapshot_time_utc,
                force=args.force,
                bookmaker=args.bookmaker,
            )
            cache_note = "cache" if result.from_cache else "api"
            print(
                f"{snapshot_date}: odds rows={result.rows} source={cache_note} "
                f"remaining={result.remaining} used={result.used} last={result.last}"
            )
        return

    if args.command == "backtest-value":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        odds = read_table(con, "odds_snapshots")
        summary, bets = run_value_backtest(
            games,
            team_stats,
            odds,
            test_year=args.year,
            report_dir=settings.report_dir,
            statcast_team=statcast_team,
            statcast_pitchers=statcast_pitchers,
            weather=weather,
            top_n=args.top_n,
            min_confidence=args.confidence,
        )
        print(summary.to_string(index=False))
        if not bets.empty:
            print(bets.tail(20).to_string(index=False))
        print(f"value backtest written to {settings.report_dir}")
        return

    if args.command == "simulate-game":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        slate = fetch_schedule(con, args.date, args.date, args.date.year)
        game = slate[slate["game_pk"] == args.game_pk]
        if game.empty:
            game = games[(games["game_pk"] == args.game_pk)]
        if game.empty:
            raise RuntimeError(f"game_pk {args.game_pk} not found")
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        frame = build_prediction_frame_for_cli(game, history_games, team_stats, statcast_team, statcast_pitchers, weather)
        summary, sims = simulate_game_from_row(frame.iloc[0], SimulationConfig(sims=args.sims, seed=args.seed))
        output_path = settings.report_dir / f"simulation_{args.game_pk}.csv"
        write_csv(summary, output_path)
        write_prediction_snapshots(con, summary, market_prefix="simulation", model_version="sim-v1", odds_source="none")
        print(summary.to_string(index=False))
        print(f"simulation summary written to {output_path}")
        return

    if args.command == "simulate-slate":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        slate = fetch_schedule(con, args.date, args.date, args.date.year)
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        summary, _ = simulate_slate(
            slate,
            history_games,
            team_stats,
            statcast_team,
            statcast_pitchers,
            weather,
            SimulationConfig(sims=args.sims, seed=args.seed),
        )
        output_path = settings.report_dir / f"simulation_slate_{args.date.isoformat()}.csv"
        write_csv(summary, output_path)
        write_prediction_snapshots(con, summary, market_prefix="simulation", model_version="sim-v1", odds_source="none")
        print(summary.to_string(index=False))
        print(f"simulation slate written to {output_path}")
        return

    if args.command == "predict-props":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        slate = fetch_schedule(con, args.date, args.date, args.date.year)
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        prop_lines = read_player_prop_lines_for_date(con, args.date)
        props = predict_props(slate, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats, args.market, prop_lines)
        output_path = settings.report_dir / f"props_{args.market}_{args.date.isoformat()}.csv"
        write_props(props, output_path)
        print(props.head(50).to_string(index=False))
        print(f"props written to {output_path}")
        return

    if args.command == "fetch-player-logs":
        counts = fetch_history(con, args.start_year, args.end_year, include_boxscores=True)
        for season, count in counts.items():
            print(f"{season}: cached player logs from {count} final regular-season games")
        return

    if args.command == "fetch-player-props":
        odds_market = ODDS_MARKET_BY_PROP[args.market]
        result = fetch_player_prop_lines(
            con,
            snapshot_date=args.date,
            markets=odds_market,
            api_key=args.api_key,
            force=args.force,
            bookmaker=args.bookmaker,
            max_events=args.max_events,
        )
        cache_note = "cache" if result.from_cache else "api"
        print(
            f"player prop rows: {result.rows} market={odds_market} source={cache_note} "
            f"remaining={result.remaining} used={result.used} last={result.last}"
        )
        return

    if args.command == "predict-props-today":
        if args.fetch_odds:
            odds_market = ODDS_MARKET_BY_PROP[args.market]
            result = fetch_player_prop_lines(
                con,
                snapshot_date=args.date,
                markets=odds_market,
                api_key=args.api_key,
                force=args.force_odds,
                bookmaker=args.bookmaker,
                max_events=args.max_events,
            )
            print(
                f"player prop rows: {result.rows} market={odds_market} "
                f"remaining={result.remaining} used={result.used} last={result.last}"
            )
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        slate = fetch_schedule_with_fallback(con, args.date)
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        prop_lines = read_player_prop_lines_for_date(con, args.date)
        props = predict_props(slate, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats, args.market, prop_lines)
        write_props_prediction_rows(con, props, model_version=f"props-v1-{args.market}", odds_source="the_odds_api")
        write_prop_recommendations(con, props)
        output_path = settings.report_dir / f"props_{args.market}_{args.date.isoformat()}.csv"
        write_props(props, output_path)
        print(props.head(80).to_string(index=False))
        print(f"props written to {output_path}")
        return

    if args.command == "predict-qualified-player-props":
        if args.fetch_odds:
            odds_markets = ",".join([HRR_ODDS_MARKET, ODDS_MARKET_BY_PROP["strikeouts"]])
            result = fetch_player_prop_lines(
                con,
                snapshot_date=args.date,
                markets=odds_markets,
                api_key=args.api_key,
                force=args.force_odds,
                bookmaker=args.bookmaker,
                max_events=args.max_events,
            )
            print(
                f"player prop rows: {result.rows} markets={odds_markets} "
                f"remaining={result.remaining} used={result.used} last={result.last}"
            )
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        slate = fetch_schedule_with_fallback(con, args.date)
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        prop_lines = read_player_prop_lines_for_date(con, args.date)
        config = PropQualificationConfig(
            min_edge=args.min_edge,
            min_ev=args.min_ev,
            min_data_quality=args.min_data_quality,
            max_odds_age_hours=args.max_odds_age_hours,
            require_confirmed_lineup=not args.allow_unconfirmed_lineups,
        )
        qualified, rejected, parlays = qualify_player_props(
            slate,
            history_games,
            team_stats,
            statcast_team,
            statcast_pitchers,
            weather,
            lineups,
            player_stats,
            prop_lines,
            config=config,
        )
        write_qualified_prop_snapshots(con, qualified, rejected)
        write_prop_recommendations(con, qualified)
        paths = write_qualified_prop_outputs(qualified, rejected, parlays, settings.report_dir, args.date.isoformat())
        report = format_qualified_props_report(qualified, rejected, parlays, args.date)
        print(report)
        print("outputs written:")
        for label, path in paths.items():
            print(f"  {label}: {path}")
        if args.telegram:
            result = send_telegram(report)
            print(f"telegram sent: {result}")
        return

    if args.command == "backtest-qualified-player-props":
        if args.grade:
            current = args.start_date
            while current <= args.end_date:
                grade_prop_recommendations(con, current)
                current += timedelta(days=1)
        summary, calibration, by_book = summarize_qualified_prop_results(con, args.start_date, args.end_date)
        print("QUALIFIED PLAYER PROP RESULTS")
        print(summary.to_string(index=False) if not summary.empty else "No graded qualified prop recommendations in range.")
        if not calibration.empty:
            print("")
            print("CALIBRATION")
            print(calibration.to_string(index=False))
        if not by_book.empty:
            print("")
            print("BY SPORTSBOOK")
            print(by_book.to_string(index=False))
        return

    if args.command == "predict-pitcher-k-today":
        args.market = "strikeouts"
        if args.fetch_odds:
            odds_market = ODDS_MARKET_BY_PROP[args.market]
            result = fetch_player_prop_lines(
                con,
                snapshot_date=args.date,
                markets=odds_market,
                api_key=args.api_key,
                force=args.force_odds,
                bookmaker=args.bookmaker,
                max_events=args.max_events,
            )
            print(
                f"player prop rows: {result.rows} market={odds_market} "
                f"remaining={result.remaining} used={result.used} last={result.last}"
            )
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        slate = fetch_schedule_with_fallback(con, args.date)
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        prop_lines = read_player_prop_lines_for_date(con, args.date)
        props = predict_props(slate, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats, "strikeouts", prop_lines)
        write_props_prediction_rows(con, props, model_version="pitcher-k-xgb-blend-v1", odds_source="the_odds_api")
        write_prop_recommendations(con, props)
        output_path = settings.report_dir / f"pitcher_k_{args.date.isoformat()}.csv"
        write_props(props, output_path)
        print(props.head(80).to_string(index=False))
        print(f"pitcher strikeout props written to {output_path}")
        return

    if args.command == "backtest-pitcher-k":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        result = run_pitcher_k_backtest(
            games,
            team_stats,
            statcast_team,
            statcast_pitchers,
            weather,
            lineups,
            player_stats,
            settings.report_dir,
            settings.data_dir.parent / "docs",
            start_year=args.start_year,
            end_year=args.end_year,
        )
        print(pd.DataFrame([result.metrics]).to_string(index=False))
        if not result.sample_predictions.empty:
            print(result.sample_predictions.head(20).to_string(index=False))
        print(f"pitcher-K diagnostic report written to {settings.data_dir.parent / 'docs' / 'PITCHER_K_DIAGNOSTIC_REPORT.md'}")
        return

    if args.command == "predict-nrfi-today":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        slate = fetch_schedule_with_fallback(con, args.date)
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        nrfi = predict_nrfi(slate, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
        write_nrfi_predictions(con, nrfi)
        output_path = settings.report_dir / f"nrfi_{args.date.isoformat()}.csv"
        write_market_predictions(nrfi, output_path)
        print(nrfi.head(80).to_string(index=False))
        print(f"NRFI scaffold predictions written to {output_path}")
        return

    if args.command == "predict-f5-today":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        slate = fetch_schedule_with_fallback(con, args.date)
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        f5 = predict_f5(slate, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
        write_f5_predictions(con, f5)
        output_path = settings.report_dir / f"f5_{args.date.isoformat()}.csv"
        write_market_predictions(f5, output_path)
        print(f5.head(80).to_string(index=False))
        print(f"F5 scaffold predictions written to {output_path}")
        return

    if args.command == "predict-hitter-tb-today":
        if args.fetch_odds:
            odds_market = ODDS_MARKET_BY_PROP["total_bases"]
            result = fetch_player_prop_lines(
                con,
                snapshot_date=args.date,
                markets=odds_market,
                api_key=args.api_key,
                force=args.force_odds,
                bookmaker=args.bookmaker,
                max_events=args.max_events,
            )
            print(
                f"player prop rows: {result.rows} market={odds_market} "
                f"remaining={result.remaining} used={result.used} last={result.last}"
            )
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        slate = fetch_schedule_with_fallback(con, args.date)
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        prop_lines = read_player_prop_lines_for_date(con, args.date)
        props = predict_hitter_total_bases(slate, history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats, prop_lines)
        write_props_prediction_rows(con, props, model_version="hitter-tb-scaffold-v1", odds_source="the_odds_api")
        write_prop_recommendations(con, props)
        output_path = settings.report_dir / f"hitter_total_bases_{args.date.isoformat()}.csv"
        write_props(props, output_path)
        print(props.head(80).to_string(index=False))
        print(f"hitter total bases props written to {output_path}")
        return

    if args.command == "grade-props":
        graded = grade_prop_recommendations(con, args.date)
        print(graded.to_string(index=False) if not graded.empty else f"No prop recommendations to grade for {args.date.isoformat()}")
        return

    if args.command == "backtest-portfolio":
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        odds = read_table(con, "odds_snapshots")
        summary, bets = run_value_backtest(
            games,
            team_stats,
            odds,
            test_year=args.year,
            report_dir=settings.report_dir,
            statcast_team=statcast_team,
            statcast_pitchers=statcast_pitchers,
            weather=weather,
            top_n=args.top_n,
            min_confidence="strong",
        )
        if bets.empty:
            print(summary.to_string(index=False))
            return
        candidates = bets.rename(columns={"bet_moneyline": "odds", "ev_per_dollar": "ev_per_unit"})
        candidates["selection"] = candidates["bet_side"]
        candidates["probability"] = candidates.apply(
            lambda r: r["model_home_prob"] if r["selection"] == r["home_team"] else r["model_away_prob"],
            axis=1,
        )
        portfolio = optimize_portfolio(candidates, staking=args.staking)
        output_path = settings.report_dir / f"portfolio_{args.year}_{args.staking}.csv"
        write_csv(portfolio, output_path)
        print(portfolio.to_string(index=False))
        print(f"portfolio backtest written to {output_path}")
        return

    if args.command == "predict-totals-today":
        if args.fetch_odds:
            result = fetch_current_odds(
                con,
                snapshot_date=args.date,
                api_key=args.api_key,
                force=args.force_odds,
                bookmaker=args.bookmaker,
                markets=args.odds_markets,
            )
            print(
                f"totals odds rows: {result.rows} remaining={result.remaining} "
                f"used={result.used} last={result.last}"
            )
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        today_games = fetch_schedule_with_fallback(con, args.date)
        today_weather_rows = fetch_weather_for_games(con, today_games)
        if today_weather_rows:
            weather = read_table(con, "game_weather")
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        model_year = min(args.date.year - 1, int(history_games["season"].max()))
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        frame = build_training_frame(history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
        bundle = train_totals_model(frame[frame["season"] <= model_year])
        save_totals_bundle(bundle, settings.model_dir / f"totals_regression_through_{model_year}.joblib")
        odds = load_odds_for_date(con, args.date)
        predictions = make_daily_totals_predictions(
            today_games=today_games,
            history_games=history_games,
            team_stats=team_stats,
            statcast_team=statcast_team,
            statcast_pitchers=statcast_pitchers,
            weather=weather,
            odds=odds,
            model_bundle=bundle,
            lineups=lineups,
            player_stats=player_stats,
        )
        write_totals_prediction_rows(con, predictions, model_version=f"totals-regression-through-{model_year}", odds_source="the_odds_api")
        write_totals_recommendations(con, predictions)
        output_path = settings.report_dir / f"totals_predictions_{args.date.isoformat()}.csv"
        write_csv(predictions, output_path)
        print(predictions.to_string(index=False))
        print(f"totals predictions written to {output_path}")
        return

    if args.command == "record-results":
        games = read_table(con, "games")
        recs = read_table(con, "bet_recommendations")
        games = games[games["season"] == args.year]
        settled = settle_bet_recommendations(con, games, recs)
        print(settled.to_string(index=False) if not settled.empty else "no recommendations settled")
        return

    if args.command == "send-daily-recap":
        results_date = args.results_date or (args.date - timedelta(days=1))
        if args.settle:
            games = read_table(con, "games")
            recs = read_table(con, "bet_recommendations")
            settle_bet_recommendations(con, games[games["season"] == args.date.year], recs)
        message = format_moneyline_recap(con, results_date)
        print(message)
        if args.sms:
            result = send_sms(message)
            print(f"sms sent sid={result.get('sid')}")
        if args.telegram:
            result = send_telegram(message)
            if not result.get("ok", False):
                print(f"telegram send failed: {result.get('error')}")
                return
            message_id = result.get("result", {}).get("message_id")
            print(f"telegram sent message_id={message_id}")
        return

    if args.command == "send-alerts":
        path = settings.report_dir / f"predictions_{args.date.isoformat()}.csv"
        if not path.exists():
            raise RuntimeError(f"No prediction report found at {path}. Run predict-today first.")
        predictions = pd.read_csv(path)
        if not args.include_started:
            schedule = fetch_schedule_with_fallback(con, args.date, force_api=True)
            predictions = filter_upcoming_predictions(predictions, schedule, window_minutes=args.window_minutes)
        predictions = filter_unsent_alerts(con, predictions, args.date, force_resend=args.force_resend)
        plays = predictions[predictions["confidence"].isin(["strong", "medium", "watchlist"])] if not predictions.empty else predictions
        if plays.empty and not args.send_empty:
            print(f"MLB {args.date.isoformat()}: no new strong/medium/watchlist alerts to send.")
            return
        if (args.telegram or args.sms) and not args.no_ollama_explain and not plays.empty:
            predictions = add_ollama_reasons(predictions, model=args.ollama_model, url=args.ollama_url)
        message = format_value_alert(
            predictions,
            args.date.isoformat(),
            empty_text=f"MLB {args.date.isoformat()}: no new strong/medium/watchlist alerts right now.",
        )
        print(message)
        if args.sms:
            result = send_sms(message)
            print(f"sms sent sid={result.get('sid')}")
            write_recommendations_from_daily(con, predictions)
            write_alert_deliveries(con, predictions, args.date, "sms", result.get("sid"))
        if args.telegram:
            result = send_telegram(message)
            if not result.get("ok", False):
                print(f"telegram send failed: {result.get('error')}")
                return
            message_id = result.get("result", {}).get("message_id")
            if result.get("queued"):
                print(f"telegram queued but not delivered message_id={message_id} error={result.get('error')}")
                return
            print(f"telegram sent message_id={message_id}")
            write_recommendations_from_daily(con, predictions)
            write_alert_deliveries(con, predictions, args.date, "telegram", str(message_id) if message_id else None)
        return

    if args.command == "send-lotto-parlay":
        path = settings.report_dir / f"predictions_{args.date.isoformat()}.csv"
        if not path.exists():
            raise RuntimeError(f"No prediction report found at {path}. Run predict-today first.")
        predictions = pd.read_csv(path)
        schedule = fetch_schedule_with_fallback(con, args.date, force_api=True)
        predictions = filter_upcoming_predictions(predictions, schedule, window_minutes=args.window_minutes)
        parlay = build_lotto_parlay(
            predictions,
            min_legs=args.min_legs,
            max_legs=args.max_legs,
            stake_units=args.stake_units,
        )
        if parlay and lotto_parlay_was_sent(con, args.date, parlay.parlay_id) and not args.force_resend:
            print(f"MLB {args.date.isoformat()} lotto parlay: no new lotto parlay to send.")
            return
        if parlay is None and not args.send_empty:
            print(f"MLB {args.date.isoformat()} lotto parlay: no 4+ leg upcoming parlay qualifies right now.")
            return
        message = format_lotto_parlay_alert(parlay, args.date.isoformat())
        print(message)
        message_id = None
        if args.sms:
            result = send_sms(message)
            message_id = result.get("sid")
            print(f"sms sent sid={message_id}")
        if args.telegram:
            result = send_telegram(message)
            if not result.get("ok", False):
                print(f"telegram send failed: {result.get('error')}")
                return
            message_id = result.get("result", {}).get("message_id")
            print(f"telegram sent message_id={message_id}")
        if parlay is not None and (args.sms or args.telegram):
            write_lotto_parlay_delivery(con, args.date, parlay, "telegram" if args.telegram else "sms", str(message_id) if message_id else None)
        return

    if args.command == "predict-today":
        if args.fetch_odds:
            result = fetch_current_odds(
                con,
                snapshot_date=args.date,
                api_key=args.api_key,
                force=args.force_odds,
                bookmaker=args.bookmaker,
                markets=args.odds_markets,
            )
            print(
                f"odds rows: {result.rows} remaining={result.remaining} "
                f"used={result.used} last={result.last}"
            )
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        today_games = fetch_schedule_with_fallback(con, args.date)
        today_weather_rows = fetch_weather_for_games(con, today_games)
        if today_weather_rows:
            weather = read_table(con, "game_weather")
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        model_year = min(args.date.year - 1, int(history_games["season"].max()))
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        frame = build_training_frame(history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
        write_engineered_features(con, frame)
        bundle = train_model(frame[frame["season"] <= model_year])
        baseline = train_baseline(frame[frame["season"] <= model_year])
        save_bundle(bundle, settings.model_dir / f"xgb_calibrated_through_{model_year}.joblib")
        odds = load_odds_for_date(con, args.date)
        predictions = make_daily_predictions(
            today_games=today_games,
            history_games=history_games,
            team_stats=team_stats,
            statcast_team=statcast_team,
            statcast_pitchers=statcast_pitchers,
            weather=weather,
            odds=odds,
            model_bundle=bundle,
            lineups=lineups,
            player_stats=player_stats,
            baseline_model=baseline,
            edge_threshold=settings.edge_threshold,
        )
        write_prediction_snapshots_from_daily(con, predictions, model_version=f"xgb-through-{model_year}", odds_source="the_odds_api")
        candidate_rows = write_moneyline_candidates(con, predictions)
        output_path = settings.report_dir / f"predictions_{args.date.isoformat()}.csv"
        write_csv(predictions, output_path)
        print(predictions.to_string(index=False))
        print(f"moneyline candidate rows stored: {candidate_rows}")
        print(f"predictions written to {output_path}")
        return

    if args.command == "predict-today-v2":
        if args.fetch_odds:
            result = fetch_current_odds(
                con,
                snapshot_date=args.date,
                api_key=args.api_key,
                force=args.force_odds,
                bookmaker=args.bookmaker,
                markets=args.odds_markets,
            )
            print(
                f"odds rows: {result.rows} remaining={result.remaining} "
                f"used={result.used} last={result.last}"
            )
        games, team_stats, statcast_team, statcast_pitchers, weather = load_dataset(con)
        today_games = fetch_schedule_with_fallback(con, args.date)
        today_weather_rows = fetch_weather_for_games(con, today_games)
        if today_weather_rows:
            weather = read_table(con, "game_weather")
        history_games = games[games["game_date"] < pd.Timestamp(args.date)]
        model_year = min(args.date.year - 1, int(history_games["season"].max()))
        lineups = read_table(con, "lineups")
        player_stats = read_table(con, "player_game_stats")
        frame = build_training_frame(history_games, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
        bundle, best = train_best_v2(con, frame, model_year, settings.report_dir, settings.model_dir)
        odds = load_odds_for_date(con, args.date)
        predictions = make_daily_predictions_v2(
            today_games=today_games,
            history_games=history_games,
            team_stats=team_stats,
            statcast_team=statcast_team,
            statcast_pitchers=statcast_pitchers,
            weather=weather,
            odds=odds,
            bundle=bundle,
            lineups=lineups,
            player_stats=player_stats,
        )
        if args.include_simulation:
            predictions = add_simulation_columns(predictions, SimulationConfig(sims=args.sims, seed=args.seed))
        if "model_disagreement_score" in predictions.columns:
            predictions["model_disagreement"] = predictions["model_disagreement_score"]
        candidate_rows = write_moneyline_candidates(con, predictions)
        output_path = settings.report_dir / f"predictions_v2_{args.date.isoformat()}.csv"
        write_csv(predictions, output_path)
        print(predictions.to_string(index=False))
        print(f"V2 selected config: {best}")
        print(f"moneyline candidate rows stored: {candidate_rows}")
        print(f"V2 predictions written to {output_path}")
        return


def load_games_and_stats(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    games, stats, *_ = load_dataset(con)
    return games, stats


def load_dataset(con) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = read_table(con, "games")
    stats = read_table(con, "team_game_stats")
    statcast_team = read_table(con, "statcast_team_daily")
    statcast_pitchers = read_table(con, "statcast_pitcher_daily")
    weather = read_table(con, "game_weather")
    if games.empty:
        raise RuntimeError("No games found. Run fetch-history first.")
    games["game_date"] = pd.to_datetime(games["game_date"])
    return games, stats, statcast_team, statcast_pitchers, weather


def load_schedule_from_db(con, slate_date: date) -> pd.DataFrame:
    games = read_table(con, "games")
    if games.empty:
        return games
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    return games[games["game_date"] == slate_date].copy()


def fetch_schedule_with_fallback(con, slate_date: date, *, force_api: bool = False) -> pd.DataFrame:
    try:
        return fetch_schedule(con, slate_date, slate_date, slate_date.year, force=force_api)
    except RuntimeError as exc:
        slate = load_schedule_from_db(con, slate_date)
        if slate.empty:
            raise
        print(f"warning: schedule API unavailable; using cached DB schedule for {slate_date.isoformat()} ({exc})")
        return slate


def build_prediction_frame_for_cli(games, history_games, team_stats, statcast_team, statcast_pitchers, weather):
    from .features import build_prediction_frame

    return build_prediction_frame(games, history_games, team_stats, statcast_team, statcast_pitchers, weather)


def write_prediction_snapshots(con, summary: pd.DataFrame, market_prefix: str, model_version: str, odds_source: str) -> None:
    rows = []
    for row in summary.to_dict("records"):
        for market, prob_col, selection in [
            (f"{market_prefix}_moneyline", "sim_home_win_prob", row.get("home_team")),
            (f"{market_prefix}_moneyline", "sim_away_win_prob", row.get("away_team")),
        ]:
            probability = float(row.get(prob_col, 0.5))
            prediction_id = hashlib.sha1(f"{row.get('game_pk')}:{market}:{selection}:{model_version}".encode()).hexdigest()
            rows.append(
                {
                    "prediction_id": prediction_id,
                    "game_pk": row.get("game_pk"),
                    "game_date": None,
                    "market": market,
                    "selection": selection,
                    "probability": probability,
                    "fair_line": implied_prob_to_moneyline(min(max(probability, 0.001), 0.999)),
                    "model_version": model_version,
                    "data_version": "duckdb-local",
                    "odds_source": odds_source,
                    "uncertainty_score": None,
                    "skip_reason": None,
                    "raw_payload": row,
                }
            )
    upsert_df(con, "prediction_snapshots", pd.DataFrame(rows))


def write_prediction_snapshots_from_daily(con, predictions: pd.DataFrame, model_version: str, odds_source: str) -> None:
    rows = []
    for row in predictions.to_dict("records"):
        for side, prob_col in [("home", "model_home_prob"), ("away", "model_away_prob")]:
            selection = row[f"{side}_team"]
            probability = float(row[prob_col])
            prediction_id = hashlib.sha1(f"{row['game_pk']}:moneyline:{selection}:{model_version}".encode()).hexdigest()
            rows.append(
                {
                    "prediction_id": prediction_id,
                    "game_pk": row["game_pk"],
                    "game_date": pd.to_datetime(row["game_date"]).date(),
                    "market": "moneyline",
                    "selection": selection,
                    "probability": probability,
                    "fair_line": implied_prob_to_moneyline(min(max(probability, 0.001), 0.999)),
                    "model_version": model_version,
                    "data_version": "duckdb-local",
                    "odds_source": odds_source,
                    "uncertainty_score": row.get("uncertainty_score"),
                    "skip_reason": row.get("skip_reason") if row.get("bet_side") == "no bet" else None,
                    "raw_payload": row,
                }
            )
    upsert_df(con, "prediction_snapshots", pd.DataFrame(rows))


def write_recommendations_from_daily(con, predictions: pd.DataFrame) -> None:
    rows = []
    for row in predictions.to_dict("records"):
        if row.get("confidence") not in {"strong", "medium"}:
            continue
        if row.get("bet_side") in [None, "no bet", "no odds"] or pd.isna(row.get("bet_moneyline")):
            continue
        rec_id = hashlib.sha1(f"{row['game_pk']}:moneyline:{row['bet_side']}:{row['bet_moneyline']}".encode()).hexdigest()
        rows.append(
            {
                "recommendation_id": rec_id,
                "prediction_id": hashlib.sha1(f"{row['game_pk']}:moneyline:{row['bet_side']}".encode()).hexdigest(),
                "game_pk": row["game_pk"],
                "game_date": pd.to_datetime(row["game_date"]).date(),
                "market": "moneyline",
                "selection": row["bet_side"],
                "sportsbook": row.get("bookmaker"),
                "odds": int(row["bet_moneyline"]),
                "stake_units": 1.0,
                "edge": row.get("edge"),
                "ev_per_unit": row.get("ev_per_dollar"),
                "confidence": row.get("confidence"),
                "status": "open",
                "skip_reason": None,
            }
        )
    if rows:
        upsert_df(con, "bet_recommendations", pd.DataFrame(rows))


def write_totals_prediction_rows(con, predictions: pd.DataFrame, model_version: str, odds_source: str) -> None:
    rows = []
    for row in predictions.to_dict("records"):
        prediction_id = hashlib.sha1(f"{row['game_pk']}:totals:{model_version}".encode()).hexdigest()
        rows.append(
            {
                "prediction_id": prediction_id,
                "game_pk": row["game_pk"],
                "game_date": pd.to_datetime(row["game_date"]).date(),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "predicted_total_runs": row.get("predicted_total_runs"),
                "total_prediction_std": row.get("total_prediction_std"),
                "sportsbook": row.get("bookmaker"),
                "total_line": row.get("sportsbook_total_line"),
                "over_odds": row.get("over_odds"),
                "under_odds": row.get("under_odds"),
                "over_probability": row.get("projected_over_probability"),
                "under_probability": row.get("projected_under_probability"),
                "market_over_probability": row.get("market_over_probability"),
                "market_under_probability": row.get("market_under_probability"),
                "decision": row.get("decision"),
                "edge": row.get("edge"),
                "ev_per_unit": row.get("ev_per_dollar"),
                "confidence": row.get("confidence"),
                "uncertainty_score": row.get("uncertainty_score"),
                "skip_reason": row.get("skip_reason"),
                "raw_payload": {**row, "model_version": model_version, "odds_source": odds_source},
            }
        )
    if rows:
        upsert_df(con, "totals_predictions", pd.DataFrame(rows))


def write_totals_recommendations(con, predictions: pd.DataFrame) -> None:
    rows = []
    for row in predictions.to_dict("records"):
        decision = row.get("decision")
        if decision in [None, "no bet", "no odds"] or pd.isna(row.get("bet_odds")):
            continue
        rec_id = hashlib.sha1(f"{row['game_pk']}:totals:{decision}:{row['sportsbook_total_line']}:{row['bet_odds']}".encode()).hexdigest()
        prediction_id = hashlib.sha1(f"{row['game_pk']}:totals".encode()).hexdigest()
        rows.append(
            {
                "recommendation_id": rec_id,
                "prediction_id": prediction_id,
                "game_pk": row["game_pk"],
                "game_date": pd.to_datetime(row["game_date"]).date(),
                "selection": decision,
                "sportsbook": row.get("bookmaker"),
                "total_line": row.get("sportsbook_total_line"),
                "odds": int(row["bet_odds"]),
                "stake_units": 1.0,
                "edge": row.get("edge"),
                "ev_per_unit": row.get("ev_per_dollar"),
                "confidence": row.get("confidence"),
                "status": "open",
                "skip_reason": None,
            }
        )
    if rows:
        upsert_df(con, "totals_recommendations", pd.DataFrame(rows))


def write_props_prediction_rows(con, props: pd.DataFrame, model_version: str, odds_source: str) -> None:
    if props.empty:
        return
    rows = []
    for row in props.to_dict("records"):
        prediction_id = hashlib.sha1(f"{row.get('game_pk')}:{row.get('market')}:{row.get('player_name')}:{model_version}".encode()).hexdigest()
        rows.append(
            {
                "prediction_id": prediction_id,
                "game_pk": row.get("game_pk"),
                "game_date": pd.to_datetime(row.get("game_date")).date(),
                "player_id": row.get("player_id"),
                "player_name": row.get("player_name"),
                "team": row.get("team"),
                "opponent": row.get("opponent"),
                "market": row.get("market"),
                "projection": row.get("projection"),
                "line": row.get("line"),
                "sportsbook": row.get("bookmaker"),
                "over_odds": row.get("over_odds"),
                "under_odds": row.get("under_odds"),
                "over_probability": row.get("over_probability"),
                "under_probability": row.get("under_probability"),
                "market_over_probability": row.get("market_over_probability"),
                "market_under_probability": row.get("market_under_probability"),
                "decision": row.get("decision"),
                "edge": row.get("edge"),
                "ev_per_unit": row.get("ev_per_dollar"),
                "confidence": row.get("confidence"),
                "uncertainty_score": row.get("uncertainty_score"),
                "skip_reason": row.get("skip_reason"),
                "raw_payload": {**row, "model_version": model_version, "odds_source": odds_source},
            }
        )
    upsert_df(con, "props_predictions", pd.DataFrame(rows))
    upsert_df(con, "player_prop_predictions", pd.DataFrame(rows))


def write_prop_recommendations(con, props: pd.DataFrame) -> None:
    if props.empty:
        return
    rows = []
    for row in props.to_dict("records"):
        decision = row.get("decision")
        if decision in [None, "no bet", "no odds"] or pd.isna(row.get("bet_odds")):
            continue
        rec_id = hashlib.sha1(f"{row.get('game_pk')}:{row.get('market')}:{row.get('player_name')}:{decision}:{row.get('line')}:{row.get('bet_odds')}".encode()).hexdigest()
        prediction_id = hashlib.sha1(f"{row.get('game_pk')}:{row.get('market')}:{row.get('player_name')}".encode()).hexdigest()
        rows.append(
            {
                "recommendation_id": rec_id,
                "prediction_id": prediction_id,
                "game_pk": row.get("game_pk"),
                "game_date": pd.to_datetime(row.get("game_date")).date(),
                "player_id": row.get("player_id"),
                "player_name": row.get("player_name"),
                "market": row.get("market"),
                "selection": decision,
                "sportsbook": row.get("bookmaker"),
                "line": row.get("line"),
                "odds": int(row.get("bet_odds")),
                "stake_units": 1.0,
                "edge": row.get("edge"),
                "ev_per_unit": row.get("ev_per_dollar"),
                "confidence": row.get("confidence"),
                "status": "open",
                "skip_reason": None,
            }
        )
    if rows:
        upsert_df(con, "prop_recommendations", pd.DataFrame(rows))
        upsert_df(con, "player_prop_recommendations", pd.DataFrame(rows))


def write_qualified_prop_snapshots(con, qualified: pd.DataFrame, rejected: pd.DataFrame) -> None:
    frames = [frame for frame in [qualified, rejected] if frame is not None and not frame.empty]
    if not frames:
        return
    props = pd.concat(frames, ignore_index=True, sort=False)
    rows = []
    for row in props.to_dict("records"):
        snapshot_id = hashlib.sha1(
            f"{row.get('game_pk')}:{row.get('market')}:{row.get('player_name')}:{row.get('line')}:{row.get('sportsbook')}:{row.get('model_version')}".encode()
        ).hexdigest()
        odds_value = row.get("bet_odds")
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "game_pk": row.get("game_pk"),
                "game_date": pd.to_datetime(row.get("game_date")).date() if not pd.isna(row.get("game_date")) else None,
                "player_id": row.get("player_id"),
                "player_name": row.get("player_name"),
                "team": row.get("team"),
                "opponent": row.get("opponent"),
                "market": row.get("market"),
                "prop": row.get("prop"),
                "line": row.get("line"),
                "sportsbook": row.get("sportsbook"),
                "odds": int(odds_value) if not pd.isna(odds_value) else None,
                "model_probability": row.get("model_probability"),
                "market_no_vig_probability": row.get("market_no_vig_probability"),
                "fair_odds": row.get("fair_odds"),
                "edge": row.get("edge"),
                "ev_per_dollar": row.get("ev_per_dollar"),
                "data_quality": row.get("data_quality"),
                "qualified": bool(row.get("qualified", False)),
                "rejection_reason": row.get("rejection_reason"),
                "model_version": row.get("model_version"),
                "odds_fetched_at": row.get("odds_fetched_at"),
                "raw_payload": row,
            }
        )
    upsert_df(con, "qualified_player_prop_snapshots", pd.DataFrame(rows))


def write_nrfi_predictions(con, nrfi: pd.DataFrame) -> None:
    if nrfi.empty:
        return
    rows = []
    for row in nrfi.to_dict("records"):
        prediction_id = hashlib.sha1(f"{row.get('game_pk')}:nrfi:{row.get('model_version')}".encode()).hexdigest()
        rows.append(
            {
                "prediction_id": prediction_id,
                "game_pk": row.get("game_pk"),
                "game_date": pd.to_datetime(row.get("game_date")).date(),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "away_top1_score_probability": row.get("away_top1_score_probability"),
                "home_bottom1_score_probability": row.get("home_bottom1_score_probability"),
                "nrfi_probability": row.get("nrfi_probability"),
                "model_version": row.get("model_version"),
                "confidence": row.get("confidence"),
                "uncertainty_score": row.get("uncertainty_score"),
                "skip_reason": row.get("skip_reason"),
                "raw_payload": row,
            }
        )
    upsert_df(con, "nrfi_predictions", pd.DataFrame(rows))


def write_f5_predictions(con, f5: pd.DataFrame) -> None:
    if f5.empty:
        return
    rows = []
    for row in f5.to_dict("records"):
        prediction_id = hashlib.sha1(f"{row.get('game_pk')}:f5:{row.get('model_version')}".encode()).hexdigest()
        rows.append(
            {
                "prediction_id": prediction_id,
                "game_pk": row.get("game_pk"),
                "game_date": pd.to_datetime(row.get("game_date")).date(),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "projected_home_f5_runs": row.get("projected_home_f5_runs"),
                "projected_away_f5_runs": row.get("projected_away_f5_runs"),
                "projected_f5_total": row.get("projected_f5_total"),
                "projected_f5_run_diff": row.get("projected_f5_run_diff"),
                "model_version": row.get("model_version"),
                "confidence": row.get("confidence"),
                "uncertainty_score": row.get("uncertainty_score"),
                "skip_reason": row.get("skip_reason"),
                "raw_payload": row,
            }
        )
    upsert_df(con, "f5_predictions", pd.DataFrame(rows))


def grade_prop_recommendations(con, grade_date: date) -> pd.DataFrame:
    recs = con.execute(
        """
        SELECT *
        FROM player_prop_recommendations
        WHERE game_date = ?
          AND status = 'open'
        """,
        [grade_date],
    ).df()
    if recs.empty:
        return recs
    games = read_table(con, "games")
    if games.empty:
        return pd.DataFrame({"message": [f"No games cached for {grade_date.isoformat()}"]})
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    finals = games[
        games["game_date"].eq(grade_date)
        & games["home_score"].notna()
        & games["away_score"].notna()
    ]
    final_game_pks = set(int(game_pk) for game_pk in finals["game_pk"].dropna().tolist())
    recs = recs[recs["game_pk"].isin(final_game_pks)].copy()
    if recs.empty:
        return pd.DataFrame({"message": [f"No final games with open prop recommendations for {grade_date.isoformat()}"]})
    stats = read_table(con, "player_game_stats")
    team_stats = read_table(con, "team_game_stats")
    if stats.empty:
        return pd.DataFrame({"message": [f"No player game stats cached for {grade_date.isoformat()}"]})
    stats["game_date"] = pd.to_datetime(stats["game_date"]).dt.date
    stat_lookup = {(int(row.game_pk), int(row.player_id)): row for row in stats.itertuples(index=False) if pd.notna(row.player_id)}
    pitcher_stat_lookup = {}
    if not team_stats.empty:
        pitcher_stat_lookup = {
            (int(row.game_pk), int(row.starter_id)): row
            for row in team_stats.itertuples(index=False)
            if pd.notna(getattr(row, "starter_id", None))
        }
    rows = []
    for rec in recs.to_dict("records"):
        if rec["market"] in {"strikeouts", "hits_allowed", "earned_runs", "outs_recorded"}:
            stat = pitcher_stat_lookup.get((int(rec["game_pk"]), int(rec["player_id"])))
        else:
            stat = stat_lookup.get((int(rec["game_pk"]), int(rec["player_id"])))
        if stat is None:
            continue
        if not has_prop_boxscore_activity(stat, rec["market"]):
            continue
        actual = actual_prop_value(stat, rec["market"])
        line = float(rec["line"])
        selection = rec["selection"]
        if actual == line:
            status = "push"
        elif selection == "over":
            status = "won" if actual > line else "lost"
        elif selection == "under":
            status = "won" if actual < line else "lost"
        else:
            status = "graded"
        con.execute(
            "UPDATE player_prop_recommendations SET status = ? WHERE recommendation_id = ?",
            [status, rec["recommendation_id"]],
        )
        con.execute(
            "UPDATE prop_recommendations SET status = ? WHERE recommendation_id = ?",
            [status, rec["recommendation_id"]],
        )
        rec.update({"actual": actual, "graded_status": status})
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_qualified_prop_results(con, start_date: date, end_date: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recs = con.execute(
        """
        SELECT *
        FROM player_prop_recommendations
        WHERE game_date BETWEEN ? AND ?
          AND status IN ('won', 'lost', 'push')
          AND market IN ('hrr', 'strikeouts')
        """,
        [start_date, end_date],
    ).df()
    if recs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    snaps = con.execute(
        """
        SELECT game_pk, game_date, player_name, market, line, sportsbook,
               model_probability, ev_per_dollar, model_version
        FROM qualified_player_prop_snapshots
        WHERE game_date BETWEEN ? AND ?
          AND qualified = true
        """,
        [start_date, end_date],
    ).df()
    recs["game_date"] = pd.to_datetime(recs["game_date"]).dt.date
    if not snaps.empty:
        snaps["game_date"] = pd.to_datetime(snaps["game_date"]).dt.date
        recs = recs.merge(
            snaps,
            on=["game_pk", "game_date", "player_name", "market", "line", "sportsbook"],
            how="left",
            suffixes=("", "_snapshot"),
        )
        if "ev_per_dollar_snapshot" in recs:
            recs["ev_per_unit"] = recs["ev_per_unit"].fillna(recs["ev_per_dollar_snapshot"])
    else:
        recs["model_probability"] = np.nan
        recs["model_version"] = "unknown"
    recs["win"] = recs["status"].eq("won").astype(float)
    recs.loc[recs["status"].eq("push"), "win"] = np.nan
    recs["profit"] = recs.apply(prop_profit_per_unit, axis=1)
    recs = recs.sort_values(["game_date", "game_pk", "player_name"]).reset_index(drop=True)
    recs["equity"] = recs["profit"].cumsum()
    recs["drawdown"] = recs["equity"] - recs["equity"].cummax()
    groups = ["overall", "market", "sportsbook", "model_version"]
    summary_rows = []
    for group in groups:
        if group == "overall":
            iterable = [("overall", recs)]
        else:
            iterable = recs.groupby(group, dropna=False)
        for label, frame in iterable:
            decisions = frame[frame["status"].isin(["won", "lost"])]
            n = len(frame)
            resolved = len(decisions)
            win_rate = float(decisions["win"].mean()) if resolved else np.nan
            ci_low, ci_high = wilson_interval(int(decisions["win"].sum()), resolved) if resolved else (np.nan, np.nan)
            brier = float(((decisions["model_probability"] - decisions["win"]) ** 2).mean()) if resolved and decisions["model_probability"].notna().any() else np.nan
            summary_rows.append(
                {
                    "segment": group,
                    "value": str(label),
                    "bets": n,
                    "resolved": resolved,
                    "win_rate": win_rate,
                    "win_rate_ci_low": ci_low,
                    "win_rate_ci_high": ci_high,
                    "avg_odds": float(pd.to_numeric(frame["odds"], errors="coerce").mean()),
                    "expected_roi": float(pd.to_numeric(frame.get("ev_per_unit"), errors="coerce").mean()),
                    "actual_roi": float(frame["profit"].sum() / max(n, 1)),
                    "max_drawdown_units": float(frame["drawdown"].min()) if "drawdown" in frame else np.nan,
                    "brier": brier,
                }
            )
    calibration = calibration_table(recs)
    by_book = pd.DataFrame(summary_rows)
    by_book = by_book[by_book["segment"].eq("sportsbook")].reset_index(drop=True)
    return pd.DataFrame(summary_rows), calibration, by_book


def prop_profit_per_unit(row: pd.Series) -> float:
    if row.get("status") == "push":
        return 0.0
    if row.get("status") == "won":
        return american_profit_per_dollar(row.get("odds"))
    if row.get("status") == "lost":
        return -1.0
    return 0.0


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    phat = wins / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def calibration_table(recs: pd.DataFrame) -> pd.DataFrame:
    resolved = recs[recs["status"].isin(["won", "lost"]) & recs["model_probability"].notna()].copy()
    if resolved.empty:
        return pd.DataFrame()
    resolved["prob_bucket"] = pd.cut(
        resolved["model_probability"],
        bins=[0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0],
        include_lowest=True,
    ).astype(str)
    return (
        resolved.groupby("prob_bucket", dropna=False)
        .agg(
            bets=("win", "size"),
            avg_model_probability=("model_probability", "mean"),
            actual_win_rate=("win", "mean"),
            actual_roi=("profit", "mean"),
        )
        .reset_index()
    )


def actual_prop_value(stat, market: str) -> float:
    if market == "strikeouts":
        return float(getattr(stat, "starter_so", getattr(stat, "strikeouts", 0.0)) or 0.0)
    if market == "hrr":
        return float(getattr(stat, "hits", 0.0) or 0.0) + float(getattr(stat, "runs", 0.0) or 0.0) + float(getattr(stat, "rbi", 0.0) or 0.0)
    if market == "total_bases":
        return float(getattr(stat, "total_bases", 0.0) or 0.0)
    if market == "hits_allowed":
        return float(getattr(stat, "starter_hits", getattr(stat, "hits", 0.0)) or 0.0)
    if market == "earned_runs":
        return float(getattr(stat, "starter_er", getattr(stat, "earned_runs", 0.0)) or 0.0)
    if market == "outs_recorded":
        return float(getattr(stat, "starter_ip", getattr(stat, "innings_pitched", 0.0)) or 0.0) * 3.0
    return float(getattr(stat, "total_bases", 0.0) or 0.0)


def has_prop_boxscore_activity(stat, market: str) -> bool:
    if market in {"strikeouts", "hits_allowed", "earned_runs", "outs_recorded"}:
        return (
            float(getattr(stat, "starter_ip", getattr(stat, "innings_pitched", 0.0)) or 0.0) > 0
            or float(getattr(stat, "starter_so", 0.0) or 0.0) > 0
            or float(getattr(stat, "starter_hits", 0.0) or 0.0) > 0
        )
    return (
        float(getattr(stat, "at_bats", 0.0) or 0.0) > 0
        or float(getattr(stat, "walks", 0.0) or 0.0) > 0
        or float(getattr(stat, "hits", 0.0) or 0.0) > 0
        or float(getattr(stat, "total_bases", 0.0) or 0.0) > 0
    )


def load_odds_for_date(con, snapshot_date: date) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM odds_snapshots WHERE snapshot_date = ?",
        [snapshot_date],
    ).df()


def read_player_prop_lines_for_date(con, snapshot_date: date) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM player_prop_lines WHERE snapshot_date = ?",
        [snapshot_date],
    ).df()


def filter_unsent_alerts(con, predictions: pd.DataFrame, alert_date: date, force_resend: bool = False) -> pd.DataFrame:
    if predictions.empty or force_resend:
        return predictions
    alert_confidences = {"strong", "medium", "watchlist"}
    plays = predictions[predictions["confidence"].isin(alert_confidences)].copy()
    if plays.empty:
        return predictions
    sent = con.execute(
        "SELECT game_pk, selection FROM alert_deliveries WHERE game_date = ?",
        [alert_date],
    ).df()
    if sent.empty:
        return predictions
    sent = sent.dropna(subset=["game_pk", "selection"])
    if sent.empty:
        return predictions
    sent_keys = {(int(row.game_pk), row.selection) for row in sent.itertuples(index=False)}
    keep = predictions.apply(
        lambda row: (row.get("confidence") not in alert_confidences)
        or pd.isna(row.get("game_pk"))
        or pd.isna(row.get("bet_side"))
        or (int(row["game_pk"]), row.get("bet_side")) not in sent_keys,
        axis=1,
    )
    return predictions[keep].reset_index(drop=True)


def write_alert_deliveries(con, predictions: pd.DataFrame, alert_date: date, channel: str, message_id: str | None) -> None:
    plays = predictions[predictions["confidence"].isin(["strong", "medium", "watchlist"])].copy()
    if plays.empty:
        return
    rows = []
    for row in plays.to_dict("records"):
        alert_id = hashlib.sha1(f"{alert_date}:{channel}:{row['game_pk']}:{row.get('bet_side')}".encode()).hexdigest()
        rows.append(
            {
                "alert_id": alert_id,
                "game_pk": row["game_pk"],
                "game_date": alert_date,
                "channel": channel,
                "selection": row.get("bet_side"),
                "confidence": row.get("confidence"),
                "message_id": message_id,
                "raw_payload": json.dumps(row, default=str),
            }
        )
    upsert_df(con, "alert_deliveries", pd.DataFrame(rows))


def lotto_parlay_was_sent(con, alert_date: date, parlay_id: str) -> bool:
    rows = con.execute(
        "SELECT 1 FROM alert_deliveries WHERE game_date = ? AND selection = ? LIMIT 1",
        [alert_date, f"LOTTO_PARLAY:{parlay_id}"],
    ).fetchall()
    return bool(rows)


def write_lotto_parlay_delivery(con, alert_date: date, parlay, channel: str, message_id: str | None) -> None:
    alert_id = hashlib.sha1(f"{alert_date}:{channel}:LOTTO_PARLAY:{parlay.parlay_id}".encode()).hexdigest()
    row = {
        "alert_id": alert_id,
        "game_pk": None,
        "game_date": alert_date,
        "channel": channel,
        "selection": f"LOTTO_PARLAY:{parlay.parlay_id}",
        "confidence": "lotto",
        "message_id": message_id,
        "raw_payload": json.dumps(
            {
                "stake_units": parlay.stake_units,
                "hit_probability": parlay.hit_probability,
                "combined_american_odds": parlay.combined_american_odds,
                "ev_per_unit": parlay.ev_per_unit,
                "legs": parlay.legs.to_dict("records"),
            },
            default=str,
        ),
    }
    upsert_df(con, "alert_deliveries", pd.DataFrame([row]))


def format_moneyline_recap(con, results_date: date) -> str:
    daily = con.execute(
        """
        WITH latest_recs AS (
            SELECT *
            FROM bet_recommendations
            WHERE market = 'moneyline'
            QUALIFY row_number() OVER (
                PARTITION BY game_pk, selection
                ORDER BY created_at DESC
            ) = 1
        ),
        alerted_recs AS (
            SELECT DISTINCT br.*
            FROM latest_recs br
            JOIN alert_deliveries ad
              ON br.game_date = ad.game_date
             AND br.game_pk = ad.game_pk
             AND br.selection = ad.selection
            WHERE ad.confidence IN ('strong', 'medium')
              AND ad.channel IN ('telegram', 'sms')
              AND COALESCE(ad.message_id, '') NOT IN ('', 'queued')
        )
        SELECT
            br.game_date,
            br.selection,
            br.odds,
            br.confidence,
            r.result,
            r.units_profit,
            g.away_team,
            g.home_team,
            g.away_score,
            g.home_score
        FROM alerted_recs br
        JOIN bet_results r USING (recommendation_id)
        LEFT JOIN games g ON br.game_pk = g.game_pk
        WHERE br.game_date = ?
        ORDER BY br.created_at, br.selection
        """,
        [results_date],
    ).df()
    overall = con.execute(
        """
        WITH latest_recs AS (
            SELECT *
            FROM bet_recommendations
            WHERE market = 'moneyline'
            QUALIFY row_number() OVER (
                PARTITION BY game_pk, selection
                ORDER BY created_at DESC
            ) = 1
        ),
        alerted_recs AS (
            SELECT DISTINCT br.*
            FROM latest_recs br
            JOIN alert_deliveries ad
              ON br.game_date = ad.game_date
             AND br.game_pk = ad.game_pk
             AND br.selection = ad.selection
            WHERE ad.confidence IN ('strong', 'medium')
              AND ad.channel IN ('telegram', 'sms')
              AND COALESCE(ad.message_id, '') NOT IN ('', 'queued')
        )
        SELECT
            count(*) AS bets,
            sum(CASE WHEN r.result = 'win' THEN 1 ELSE 0 END) AS wins,
            sum(CASE WHEN r.result = 'loss' THEN 1 ELSE 0 END) AS losses,
            sum(r.units_profit) AS units
        FROM alerted_recs br
        JOIN bet_results r USING (recommendation_id)
        """
    ).df()
    if daily.empty:
        lines = [f"MLB moneyline recap for {results_date.isoformat()}: 0-0, +0.00u"]
        lines.append("No settled recommendations for this date.")
        lines.append("Winners: (none)")
        lines.append("Losers: (none)")
        if not overall.empty and int(overall.loc[0, "bets"] or 0) > 0:
            row = overall.loc[0]
            lines.append(
                f"Overall settled: {int(row['wins'])}-{int(row['losses'])}, {float(row['units']):+.2f}u"
            )
        return "\n".join(lines)

    wins = int((daily["result"] == "win").sum())
    losses = int((daily["result"] == "loss").sum())
    units = float(daily["units_profit"].sum())
    lines = [f"MLB moneyline recap for {results_date.isoformat()}: {wins}-{losses}, {units:+.2f}u"]
    if not overall.empty and int(overall.loc[0, "bets"] or 0) > 0:
        row = overall.loc[0]
        lines.append(f"Overall settled: {int(row['wins'])}-{int(row['losses'])}, {float(row['units']):+.2f}u")
    for row in daily.to_dict("records"):
        marker = "W" if row["result"] == "win" else "L"
        odds = int(row["odds"])
        units_profit = float(row["units_profit"])
        score = ""
        if not pd.isna(row.get("away_score")) and not pd.isna(row.get("home_score")):
            score = f" ({row['away_team']} {int(row['away_score'])}, {row['home_team']} {int(row['home_score'])})"
        lines.append(f"{marker}: {row['selection']} {odds} {units_profit:+.2f}u{score}")
    return "\n".join(lines)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


if __name__ == "__main__":
    main()
