from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import Settings, ensure_dirs


def connect(settings: Settings = Settings()) -> duckdb.DuckDBPyConnection:
    ensure_dirs(settings)
    con = duckdb.connect(str(settings.db_path))
    init_db(con)
    return con


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_api_cache (
            source TEXT,
            cache_key TEXT,
            fetched_at TIMESTAMP DEFAULT now(),
            payload JSON,
            PRIMARY KEY (source, cache_key)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS games (
            game_pk BIGINT PRIMARY KEY,
            game_date DATE,
            game_datetime TIMESTAMP,
            season INTEGER,
            game_type TEXT,
            status TEXT,
            doubleheader TEXT,
            game_number INTEGER,
            venue_name TEXT,
            day_night TEXT,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            home_probable_pitcher_id BIGINT,
            away_probable_pitcher_id BIGINT,
            home_probable_pitcher TEXT,
            away_probable_pitcher TEXT,
            home_won BOOLEAN
        )
        """
    )
    ensure_column(con, "games", "game_datetime", "TIMESTAMP")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS team_game_stats (
            game_pk BIGINT,
            team_id INTEGER,
            team_name TEXT,
            side TEXT,
            starter_id BIGINT,
            starter_name TEXT,
            runs INTEGER,
            hits INTEGER,
            errors INTEGER,
            at_bats INTEGER,
            doubles INTEGER,
            triples INTEGER,
            home_runs INTEGER,
            walks INTEGER,
            strikeouts INTEGER,
            left_on_base INTEGER,
            starter_ip DOUBLE,
            starter_er INTEGER,
            starter_so INTEGER,
            starter_bb INTEGER,
            starter_hits INTEGER,
            starter_home_runs INTEGER,
            bullpen_ip DOUBLE,
            bullpen_er INTEGER,
            bullpen_so INTEGER,
            bullpen_bb INTEGER,
            PRIMARY KEY (game_pk, team_id)
        )
        """
    )
    ensure_column(con, "team_game_stats", "starter_home_runs", "INTEGER")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            event_id TEXT,
            snapshot_date DATE,
            commence_time TIMESTAMP,
            bookmaker TEXT,
            home_team TEXT,
            away_team TEXT,
            home_moneyline INTEGER,
            away_moneyline INTEGER,
            home_spread DOUBLE,
            away_spread DOUBLE,
            home_spread_price INTEGER,
            away_spread_price INTEGER,
            total_points DOUBLE,
            over_price INTEGER,
            under_price INTEGER,
            raw_payload JSON,
            fetched_at TIMESTAMP DEFAULT now(),
            PRIMARY KEY (event_id, snapshot_date, bookmaker)
        )
        """
    )
    ensure_column(con, "odds_snapshots", "home_spread", "DOUBLE")
    ensure_column(con, "odds_snapshots", "away_spread", "DOUBLE")
    ensure_column(con, "odds_snapshots", "home_spread_price", "INTEGER")
    ensure_column(con, "odds_snapshots", "away_spread_price", "INTEGER")
    ensure_column(con, "odds_snapshots", "total_points", "DOUBLE")
    ensure_column(con, "odds_snapshots", "over_price", "INTEGER")
    ensure_column(con, "odds_snapshots", "under_price", "INTEGER")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS statcast_team_daily (
            game_date DATE,
            team_name TEXT,
            team_id INTEGER,
            batted_balls INTEGER,
            xwoba DOUBLE,
            xba DOUBLE,
            hard_hit_rate DOUBLE,
            barrel_rate DOUBLE,
            avg_exit_velocity DOUBLE,
            avg_launch_angle DOUBLE,
            k_rate DOUBLE,
            bb_rate DOUBLE,
            pitches_seen INTEGER,
            xwoba_allowed DOUBLE,
            hard_hit_allowed DOUBLE,
            barrel_allowed DOUBLE,
            avg_pitch_velocity DOUBLE,
            avg_spin_rate DOUBLE,
            pitches_thrown INTEGER,
            PRIMARY KEY (game_date, team_name)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS statcast_pitcher_daily (
            game_date DATE,
            pitcher_id BIGINT,
            pitcher_name TEXT,
            team_name TEXT,
            batters_faced INTEGER,
            batted_balls_allowed INTEGER,
            xwoba_allowed DOUBLE,
            hard_hit_allowed DOUBLE,
            barrel_allowed DOUBLE,
            avg_exit_velocity_allowed DOUBLE,
            avg_pitch_velocity DOUBLE,
            avg_spin_rate DOUBLE,
            k_rate DOUBLE,
            bb_rate DOUBLE,
            PRIMARY KEY (game_date, pitcher_id)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS game_weather (
            game_pk BIGINT PRIMARY KEY,
            game_date DATE,
            venue_name TEXT,
            temperature_f DOUBLE,
            relative_humidity DOUBLE,
            wind_speed_mph DOUBLE,
            wind_direction_degrees DOUBLE,
            precipitation_in DOUBLE,
            source TEXT,
            fetched_at TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS players (
            player_id BIGINT PRIMARY KEY,
            player_name TEXT,
            primary_position TEXT,
            bats TEXT,
            throws TEXT,
            updated_at TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS lineups (
            game_pk BIGINT,
            team_id INTEGER,
            team_name TEXT,
            side TEXT,
            player_id BIGINT,
            player_name TEXT,
            batting_order INTEGER,
            position TEXT,
            confirmed BOOLEAN DEFAULT false,
            source TEXT,
            captured_at TIMESTAMP DEFAULT now(),
            PRIMARY KEY (game_pk, team_id, player_id)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS player_game_stats (
            game_pk BIGINT,
            game_date DATE,
            team_id INTEGER,
            team_name TEXT,
            player_id BIGINT,
            player_name TEXT,
            side TEXT,
            batting_order INTEGER,
            position TEXT,
            at_bats INTEGER,
            hits INTEGER,
            doubles INTEGER,
            triples INTEGER,
            home_runs INTEGER,
            walks INTEGER,
            strikeouts INTEGER,
            total_bases INTEGER,
            rbi INTEGER,
            runs INTEGER,
            innings_pitched DOUBLE,
            earned_runs INTEGER,
            batters_faced INTEGER,
            pitches_thrown INTEGER,
            PRIMARY KEY (game_pk, player_id)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS pitcher_pitch_mix (
            game_date DATE,
            pitcher_id BIGINT,
            pitcher_name TEXT,
            pitch_type TEXT,
            pitches INTEGER,
            avg_velocity DOUBLE,
            avg_spin_rate DOUBLE,
            usage_rate DOUBLE,
            PRIMARY KEY (game_date, pitcher_id, pitch_type)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS pitch_type_matchup_daily (
            game_date DATE,
            batter_id BIGINT,
            pitcher_id BIGINT,
            pitch_type TEXT,
            batter_side TEXT,
            pitcher_hand TEXT,
            balls INTEGER,
            strikes INTEGER,
            pitch_count INTEGER,
            ball_rate DOUBLE,
            called_strike_rate DOUBLE,
            swinging_strike_rate DOUBLE,
            foul_rate DOUBLE,
            in_play_rate DOUBLE,
            in_play_out_rate DOUBLE,
            single_rate DOUBLE,
            double_rate DOUBLE,
            triple_rate DOUBLE,
            extra_base_hit_rate DOUBLE,
            home_run_rate DOUBLE,
            hbp_rate DOUBLE,
            expected_woba DOUBLE,
            run_value DOUBLE,
            avg_velocity DOUBLE,
            avg_spin_rate DOUBLE,
            PRIMARY KEY (game_date, batter_id, pitcher_id, pitch_type, batter_side, pitcher_hand, balls, strikes)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS statcast_pitch_events (
            game_pk BIGINT,
            game_date DATE,
            at_bat_number INTEGER,
            pitch_number INTEGER,
            batter_id BIGINT,
            pitcher_id BIGINT,
            pitch_type TEXT,
            batter_side TEXT,
            pitcher_hand TEXT,
            balls INTEGER,
            strikes INTEGER,
            description TEXT,
            events TEXT,
            release_speed DOUBLE,
            release_spin_rate DOUBLE,
            estimated_woba DOUBLE,
            run_value DOUBLE,
            PRIMARY KEY (game_pk, at_bat_number, pitch_number)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS statcast_backfill_dates (
            game_date DATE PRIMARY KEY,
            status TEXT,
            pitch_rows INTEGER,
            matchup_rows INTEGER,
            fetched_at TIMESTAMP DEFAULT now(),
            error TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS pitch_prediction_evaluations (
            evaluation_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            start_date DATE,
            end_date DATE,
            pitches INTEGER,
            report_json JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS player_statcast_daily (
            game_date DATE,
            player_id BIGINT,
            player_name TEXT,
            team_name TEXT,
            batted_balls INTEGER,
            xwoba DOUBLE,
            xba DOUBLE,
            xslg DOUBLE,
            hard_hit_rate DOUBLE,
            barrel_rate DOUBLE,
            avg_exit_velocity DOUBLE,
            avg_launch_angle DOUBLE,
            k_rate DOUBLE,
            bb_rate DOUBLE,
            PRIMARY KEY (game_date, player_id)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS player_prop_lines (
            prop_line_id TEXT PRIMARY KEY,
            event_id TEXT,
            snapshot_date DATE,
            commence_time TIMESTAMP,
            bookmaker TEXT,
            home_team TEXT,
            away_team TEXT,
            player_id BIGINT,
            player_name TEXT,
            market TEXT,
            line DOUBLE,
            over_odds INTEGER,
            under_odds INTEGER,
            raw_payload JSON,
            fetched_at TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_snapshots (
            prediction_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            market TEXT,
            selection TEXT,
            probability DOUBLE,
            fair_line INTEGER,
            model_version TEXT,
            data_version TEXT,
            odds_source TEXT,
            uncertainty_score DOUBLE,
            skip_reason TEXT,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS engineered_game_features (
            feature_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            feature_set_version TEXT,
            data_version TEXT,
            target_home_win BOOLEAN,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_importance_reports (
            report_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            train_through INTEGER,
            test_year INTEGER,
            feature_name TEXT,
            importance_type TEXT,
            importance DOUBLE,
            rank INTEGER,
            risk_flag TEXT,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS moneyline_evaluation_reports (
            report_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            report_type TEXT,
            year INTEGER,
            segment TEXT,
            metric_name TEXT,
            metric_value DOUBLE,
            games INTEGER,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS moneyline_candidate_snapshots (
            candidate_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            home_team TEXT,
            away_team TEXT,
            model_pick TEXT,
            bet_side TEXT,
            sportsbook TEXT,
            home_moneyline INTEGER,
            away_moneyline INTEGER,
            model_home_prob DOUBLE,
            model_away_prob DOUBLE,
            market_home_prob DOUBLE,
            market_away_prob DOUBLE,
            bet_probability DOUBLE,
            edge DOUBLE,
            ev_per_unit DOUBLE,
            confidence TEXT,
            skip_reason TEXT,
            tier TEXT,
            risk_flags TEXT,
            stake_units DOUBLE,
            pick_side TEXT,
            price_side TEXT,
            model_disagreement DOUBLE,
            uncertainty_score DOUBLE,
            lineups_confirmed BOOLEAN,
            official_play BOOLEAN,
            raw_payload JSON
        )
        """
    )
    ensure_column(con, "moneyline_candidate_snapshots", "tier", "TEXT")
    ensure_column(con, "moneyline_candidate_snapshots", "risk_flags", "TEXT")
    ensure_column(con, "moneyline_candidate_snapshots", "stake_units", "DOUBLE")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS model_experiments (
            experiment_id TEXT PRIMARY KEY,
            model_version TEXT,
            feature_set TEXT,
            year INTEGER,
            accuracy DOUBLE,
            auc DOUBLE,
            brier DOUBLE,
            log_loss DOUBLE,
            ece DOUBLE,
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS totals_predictions (
            prediction_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            home_team TEXT,
            away_team TEXT,
            predicted_total_runs DOUBLE,
            total_prediction_std DOUBLE,
            sportsbook TEXT,
            total_line DOUBLE,
            over_odds INTEGER,
            under_odds INTEGER,
            over_probability DOUBLE,
            under_probability DOUBLE,
            market_over_probability DOUBLE,
            market_under_probability DOUBLE,
            decision TEXT,
            edge DOUBLE,
            ev_per_unit DOUBLE,
            confidence TEXT,
            uncertainty_score DOUBLE,
            skip_reason TEXT,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS totals_recommendations (
            recommendation_id TEXT PRIMARY KEY,
            prediction_id TEXT,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            selection TEXT,
            sportsbook TEXT,
            total_line DOUBLE,
            odds INTEGER,
            stake_units DOUBLE,
            edge DOUBLE,
            ev_per_unit DOUBLE,
            confidence TEXT,
            status TEXT DEFAULT 'open',
            skip_reason TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS props_predictions (
            prediction_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            player_id BIGINT,
            player_name TEXT,
            team TEXT,
            opponent TEXT,
            market TEXT,
            projection DOUBLE,
            line DOUBLE,
            sportsbook TEXT,
            over_odds INTEGER,
            under_odds INTEGER,
            over_probability DOUBLE,
            under_probability DOUBLE,
            market_over_probability DOUBLE,
            market_under_probability DOUBLE,
            decision TEXT,
            edge DOUBLE,
            ev_per_unit DOUBLE,
            confidence TEXT,
            uncertainty_score DOUBLE,
            skip_reason TEXT,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS player_prop_predictions (
            prediction_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            player_id BIGINT,
            player_name TEXT,
            team TEXT,
            opponent TEXT,
            market TEXT,
            projection DOUBLE,
            line DOUBLE,
            sportsbook TEXT,
            over_odds INTEGER,
            under_odds INTEGER,
            over_probability DOUBLE,
            under_probability DOUBLE,
            market_over_probability DOUBLE,
            market_under_probability DOUBLE,
            decision TEXT,
            edge DOUBLE,
            ev_per_unit DOUBLE,
            confidence TEXT,
            uncertainty_score DOUBLE,
            skip_reason TEXT,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prop_recommendations (
            recommendation_id TEXT PRIMARY KEY,
            prediction_id TEXT,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            player_id BIGINT,
            player_name TEXT,
            market TEXT,
            selection TEXT,
            sportsbook TEXT,
            line DOUBLE,
            odds INTEGER,
            stake_units DOUBLE,
            edge DOUBLE,
            ev_per_unit DOUBLE,
            confidence TEXT,
            status TEXT DEFAULT 'open',
            skip_reason TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS player_prop_recommendations (
            recommendation_id TEXT PRIMARY KEY,
            prediction_id TEXT,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            player_id BIGINT,
            player_name TEXT,
            market TEXT,
            selection TEXT,
            sportsbook TEXT,
            line DOUBLE,
            odds INTEGER,
            stake_units DOUBLE,
            edge DOUBLE,
            ev_per_unit DOUBLE,
            confidence TEXT,
            status TEXT DEFAULT 'open',
            skip_reason TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS qualified_player_prop_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            player_id BIGINT,
            player_name TEXT,
            team TEXT,
            opponent TEXT,
            market TEXT,
            prop TEXT,
            line DOUBLE,
            sportsbook TEXT,
            odds INTEGER,
            model_probability DOUBLE,
            market_no_vig_probability DOUBLE,
            fair_odds INTEGER,
            edge DOUBLE,
            ev_per_dollar DOUBLE,
            data_quality DOUBLE,
            qualified BOOLEAN,
            rejection_reason TEXT,
            model_version TEXT,
            odds_fetched_at TIMESTAMP,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS nrfi_predictions (
            prediction_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            home_team TEXT,
            away_team TEXT,
            away_top1_score_probability DOUBLE,
            home_bottom1_score_probability DOUBLE,
            nrfi_probability DOUBLE,
            model_version TEXT,
            confidence TEXT,
            uncertainty_score DOUBLE,
            skip_reason TEXT,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS f5_predictions (
            prediction_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            home_team TEXT,
            away_team TEXT,
            projected_home_f5_runs DOUBLE,
            projected_away_f5_runs DOUBLE,
            projected_f5_total DOUBLE,
            projected_f5_run_diff DOUBLE,
            model_version TEXT,
            confidence TEXT,
            uncertainty_score DOUBLE,
            skip_reason TEXT,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bet_recommendations (
            recommendation_id TEXT PRIMARY KEY,
            prediction_id TEXT,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            market TEXT,
            selection TEXT,
            sportsbook TEXT,
            odds INTEGER,
            stake_units DOUBLE,
            edge DOUBLE,
            ev_per_unit DOUBLE,
            confidence TEXT,
            status TEXT DEFAULT 'open',
            skip_reason TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bet_results (
            recommendation_id TEXT PRIMARY KEY,
            settled_at TIMESTAMP DEFAULT now(),
            result TEXT,
            units_profit DOUBLE,
            closing_odds INTEGER,
            clv DOUBLE,
            notes TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS data_quality_checks (
            check_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            severity TEXT,
            check_name TEXT,
            message TEXT,
            status TEXT,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_deliveries (
            alert_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            game_date DATE,
            channel TEXT,
            selection TEXT,
            confidence TEXT,
            message_id TEXT,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS news_notes (
            note_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            game_pk BIGINT,
            team_name TEXT,
            player_id BIGINT,
            source_url TEXT,
            note_type TEXT,
            summary TEXT,
            raw_payload JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS soccer_matches (
            match_id TEXT PRIMARY KEY,
            competition TEXT,
            season INTEGER,
            match_date DATE,
            kickoff_utc TIMESTAMP,
            status TEXT,
            stage TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            venue_name TEXT,
            venue_city TEXT,
            venue_country TEXT,
            venue_latitude DOUBLE,
            venue_longitude DOUBLE,
            temperature_f DOUBLE,
            wind_speed_mph DOUBLE,
            precipitation_probability DOUBLE,
            weather_fetched_at TIMESTAMP,
            source TEXT,
            fetched_at TIMESTAMP DEFAULT now(),
            raw_payload JSON
        )
        """
    )
    ensure_column(con, "soccer_matches", "temperature_f", "DOUBLE")
    ensure_column(con, "soccer_matches", "wind_speed_mph", "DOUBLE")
    ensure_column(con, "soccer_matches", "precipitation_probability", "DOUBLE")
    ensure_column(con, "soccer_matches", "weather_fetched_at", "TIMESTAMP")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS soccer_team_ratings (
            team TEXT,
            as_of_date DATE,
            elo DOUBLE,
            attack_rating DOUBLE,
            defense_rating DOUBLE,
            overall_rating DOUBLE,
            recent_form_rating DOUBLE,
            goals_for_per_match DOUBLE,
            goals_against_per_match DOUBLE,
            rest_days DOUBLE,
            matches_used INTEGER,
            fifa_rank DOUBLE,
            data_quality TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT now(),
            PRIMARY KEY (team, as_of_date, source)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS soccer_odds_snapshots (
            event_id TEXT,
            snapshot_date DATE,
            commence_time TIMESTAMP,
            bookmaker TEXT,
            home_team TEXT,
            away_team TEXT,
            home_price DOUBLE,
            draw_price DOUBLE,
            away_price DOUBLE,
            over_2_5_price DOUBLE,
            under_2_5_price DOUBLE,
            btts_yes_price DOUBLE,
            btts_no_price DOUBLE,
            home_or_draw_price DOUBLE,
            away_or_draw_price DOUBLE,
            home_or_away_price DOUBLE,
            home_dnb_price DOUBLE,
            away_dnb_price DOUBLE,
            over_1_5_price DOUBLE,
            under_1_5_price DOUBLE,
            over_3_0_price DOUBLE,
            under_3_0_price DOUBLE,
            over_3_5_price DOUBLE,
            under_3_5_price DOUBLE,
            home_spread_0_price DOUBLE,
            away_spread_0_price DOUBLE,
            home_spread_minus_0_25_price DOUBLE,
            away_spread_minus_0_25_price DOUBLE,
            home_spread_plus_0_25_price DOUBLE,
            away_spread_plus_0_25_price DOUBLE,
            raw_payload JSON,
            fetched_at TIMESTAMP DEFAULT now(),
            PRIMARY KEY (event_id, snapshot_date, bookmaker)
        )
        """
    )
    for column in [
        "home_or_draw_price",
        "away_or_draw_price",
        "home_or_away_price",
        "home_dnb_price",
        "away_dnb_price",
        "over_1_5_price",
        "under_1_5_price",
        "over_3_0_price",
        "under_3_0_price",
        "over_3_5_price",
        "under_3_5_price",
        "home_spread_0_price",
        "away_spread_0_price",
        "home_spread_minus_0_25_price",
        "away_spread_minus_0_25_price",
        "home_spread_plus_0_25_price",
        "away_spread_plus_0_25_price",
    ]:
        ensure_column(con, "soccer_odds_snapshots", column, "DOUBLE")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS soccer_predictions (
            prediction_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            match_id TEXT,
            match_date DATE,
            kickoff_utc TIMESTAMP,
            home_team TEXT,
            away_team TEXT,
            model_version TEXT,
            home_xg DOUBLE,
            away_xg DOUBLE,
            projected_score TEXT,
            home_win_probability DOUBLE,
            draw_probability DOUBLE,
            away_win_probability DOUBLE,
            over_2_5_probability DOUBLE,
            under_2_5_probability DOUBLE,
            btts_probability DOUBLE,
            top_correct_scores JSON,
            sportsbook TEXT,
            market TEXT,
            selection TEXT,
            odds DOUBLE,
            market_probability DOUBLE,
            edge DOUBLE,
            ev_per_unit DOUBLE,
            confidence TEXT,
            uncertainty_score DOUBLE,
            skip_reason TEXT,
            odds_source TEXT,
            raw_payload JSON
        )
        """
    )
    ensure_column(con, "soccer_predictions", "elo_diff", "DOUBLE")
    ensure_column(con, "soccer_predictions", "wc_matches_used_home", "INTEGER")
    ensure_column(con, "soccer_predictions", "wc_matches_used_away", "INTEGER")
    ensure_column(con, "soccer_predictions", "wc_form_weight_home", "DOUBLE")
    ensure_column(con, "soccer_predictions", "wc_form_weight_away", "DOUBLE")
    ensure_column(con, "soccer_predictions", "raw_model_prob", "DOUBLE")
    ensure_column(con, "soccer_predictions", "market_prob", "DOUBLE")
    ensure_column(con, "soccer_predictions", "calibrated_prob", "DOUBLE")
    ensure_column(con, "soccer_predictions", "draw_prob", "DOUBLE")
    ensure_column(con, "soccer_predictions", "raw_expected_total_goals", "DOUBLE")
    ensure_column(con, "soccer_predictions", "calibrated_expected_total_goals", "DOUBLE")
    ensure_column(con, "soccer_predictions", "guardrails_triggered", "TEXT")
    ensure_column(con, "soccer_predictions", "pre_gate_recommendation", "TEXT")
    ensure_column(con, "soccer_predictions", "final_recommendation", "TEXT")
    ensure_column(con, "soccer_predictions", "derivative_source_probabilities", "TEXT")
    ensure_column(con, "soccer_predictions", "candidate_market", "TEXT")
    ensure_column(con, "soccer_predictions", "candidate_selection", "TEXT")
    ensure_column(con, "soccer_predictions", "candidate_odds", "DOUBLE")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS soccer_recommendations (
            recommendation_id TEXT PRIMARY KEY,
            prediction_id TEXT,
            created_at TIMESTAMP DEFAULT now(),
            match_id TEXT,
            match_date DATE,
            market TEXT,
            selection TEXT,
            sportsbook TEXT,
            odds DOUBLE,
            stake_units DOUBLE,
            edge DOUBLE,
            ev_per_unit DOUBLE,
            confidence TEXT,
            status TEXT DEFAULT 'open',
            skip_reason TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS soccer_challenge_ledger (
            id TEXT PRIMARY KEY,
            date DATE,
            mode TEXT,
            starting_bankroll DOUBLE,
            ending_bankroll DOUBLE,
            target_bankroll DOUBLE,
            official_bets_count INTEGER,
            lottery_leans_count INTEGER,
            skipped_day BOOLEAN,
            skip_reason TEXT,
            total_staked DOUBLE,
            profit_loss DOUBLE,
            progress_to_target_pct DOUBLE,
            notes TEXT,
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS soccer_challenge_bets (
            id TEXT PRIMARY KEY,
            date DATE,
            match_id TEXT,
            home_team TEXT,
            away_team TEXT,
            market TEXT,
            pick TEXT,
            odds DOUBLE,
            stake DOUBLE,
            stake_units DOUBLE,
            confidence TEXT,
            edge DOUBLE,
            status TEXT,
            bet_type TEXT,
            block_reason TEXT,
            profit_loss DOUBLE,
            created_at TIMESTAMP DEFAULT now(),
            graded_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS live_play_events (
            game_pk BIGINT,
            at_bat_index INTEGER,
            event_index INTEGER,
            event_id TEXT,
            game_date DATE,
            inning INTEGER,
            half_inning TEXT,
            batting_team_id INTEGER,
            fielding_team_id INTEGER,
            batter_id BIGINT,
            batter_name TEXT,
            pitcher_id BIGINT,
            pitcher_name TEXT,
            base_mask INTEGER,
            outs INTEGER,
            balls INTEGER,
            strikes INTEGER,
            home_score INTEGER,
            away_score INTEGER,
            event_type TEXT,
            event_description TEXT,
            pitch_type TEXT,
            start_speed DOUBLE,
            spin_rate DOUBLE,
            plate_x DOUBLE,
            plate_z DOUBLE,
            is_pitch BOOLEAN,
            is_in_play BOOLEAN,
            runs_to_inning_end INTEGER,
            fetched_at TIMESTAMP,
            raw_payload JSON,
            PRIMARY KEY (game_pk, at_bat_index, event_index)
        )
        """
    )
    ensure_column(con, "live_play_events", "batter_side", "TEXT")
    ensure_column(con, "live_play_events", "pitcher_hand", "TEXT")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS live_game_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            game_pk BIGINT,
            captured_at TIMESTAMP,
            game_date DATE,
            status TEXT,
            inning INTEGER,
            half_inning TEXT,
            batting_team_id INTEGER,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            base_mask INTEGER,
            outs INTEGER,
            balls INTEGER,
            strikes INTEGER,
            batter_id BIGINT,
            batter_name TEXT,
            pitcher_id BIGINT,
            pitcher_name TEXT,
            score_this_inning_prob DOUBLE,
            home_win_prob DOUBLE,
            away_win_prob DOUBLE,
            uncertainty_low DOUBLE,
            uncertainty_high DOUBLE,
            model_version TEXT,
            state_sample_size INTEGER,
            quality_flags TEXT
        )
        """
    )
    ensure_column(con, "live_game_snapshots", "batter_side", "TEXT")
    ensure_column(con, "live_game_snapshots", "pitcher_hand", "TEXT")
    ensure_column(con, "live_game_snapshots", "pitcher_pitch_count", "INTEGER")
    ensure_column(con, "live_game_snapshots", "times_through_order", "INTEGER")
    ensure_column(con, "live_game_snapshots", "most_likely_pitch", "TEXT")
    ensure_column(con, "live_game_snapshots", "most_likely_pitch_prob", "DOUBLE")
    ensure_column(con, "live_game_snapshots", "top_three_pitches_json", "TEXT")
    ensure_column(con, "live_game_snapshots", "pitch_analysis_json", "TEXT")
    ensure_column(con, "live_game_snapshots", "pitch_sensitivities_json", "TEXT")
    ensure_column(con, "live_game_snapshots", "pa_outcomes_json", "TEXT")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS live_odds_snapshots (
            snapshot_id TEXT,
            event_id TEXT,
            game_pk BIGINT,
            fetched_at TIMESTAMP,
            bookmaker TEXT,
            home_team TEXT,
            away_team TEXT,
            home_moneyline INTEGER,
            away_moneyline INTEGER,
            home_no_vig DOUBLE,
            away_no_vig DOUBLE,
            api_remaining TEXT,
            api_used TEXT,
            raw_payload JSON,
            PRIMARY KEY (snapshot_id, bookmaker)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS live_odds_refreshes (
            refresh_id TEXT PRIMARY KEY,
            requested_at TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT,
            rows_written INTEGER,
            api_remaining TEXT,
            api_used TEXT,
            error TEXT
        )
        """
    )


def ensure_column(con: duckdb.DuckDBPyConnection, table: str, column: str, dtype: str) -> None:
    cols = con.execute(f"PRAGMA table_info('{table}')").df()
    if column not in set(cols["name"].tolist()):
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")


def cache_get(con: duckdb.DuckDBPyConnection, source: str, cache_key: str) -> Any | None:
    rows = con.execute(
        "SELECT payload FROM raw_api_cache WHERE source = ? AND cache_key = ?",
        [source, cache_key],
    ).fetchall()
    if not rows:
        return None
    value = rows[0][0]
    return json.loads(value) if isinstance(value, str) else value


def cache_put(
    con: duckdb.DuckDBPyConnection, source: str, cache_key: str, payload: Any
) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO raw_api_cache (source, cache_key, payload, fetched_at)
        VALUES (?, ?, ?, now())
        """,
        [source, cache_key, json.dumps(payload)],
    )


def upsert_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    temp = f"tmp_{table}"
    con.register(temp, df)
    columns = ", ".join(df.columns)
    con.execute(f"INSERT OR REPLACE INTO {table} ({columns}) SELECT {columns} FROM {temp}")
    con.unregister(temp)
    return len(df)


def read_table(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    return con.execute(f"SELECT * FROM {table}").df()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
