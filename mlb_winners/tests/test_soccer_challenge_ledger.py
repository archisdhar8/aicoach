from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from mlb_winners.db import init_db, upsert_df
from soccerworldcup.audit import audit_predictions
from soccerworldcup.bet_card import BetCard, approved_bets, no_bet_leans
from soccerworldcup.challenge_ledger import progress_to_target, write_challenge_ledger


def card_rows(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "created_at": pd.Timestamp("2026-06-20T12:00:00Z"),
        "match_id": "match-1",
        "match_date": date(2026, 6, 20),
        "kickoff_utc": pd.Timestamp("2026-06-20T18:00:00Z"),
        "home_team": "Home",
        "away_team": "Away",
        "market": "1x2",
        "selection": "no bet",
        "candidate_selection": "Home",
        "final_recommendation": "no bet",
        "pre_gate_recommendation": "Home",
        "odds": pd.NA,
        "candidate_odds": 1.90,
        "confidence": "no bet",
        "confidence_score": 0.0,
        "guardrails_triggered": "early_group_market_anchor",
        "skip_reason": "",
        "raw_model_prob": 0.55,
        "market_prob": 0.50,
        "calibrated_prob": 0.54,
        "draw_prob": 0.25,
        "raw_expected_total_goals": 2.50,
        "calibrated_expected_total_goals": 2.45,
        "edge": 0.04,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def make_card(rows: pd.DataFrame, warnings: list[str] | None = None) -> BetCard:
    audit = audit_predictions(rows, date(2026, 6, 20))
    if warnings is not None:
        audit = audit.__class__(
            report_date=audit.report_date,
            run_summary=audit.run_summary,
            pre_gate_distribution=audit.pre_gate_distribution,
            pick_distribution=audit.pick_distribution,
            guardrail_counts=audit.guardrail_counts,
            probability_diagnostics=audit.probability_diagnostics,
            warnings=warnings,
            rows=audit.rows,
        )
    return BetCard(
        report_date=date(2026, 6, 20),
        audit=audit,
        approved_bets=approved_bets(audit.rows),
        no_bet_leans=no_bet_leans(audit.rows),
        score_sync_freshness="fresh",
        odds_freshness="fresh",
    )


def memory_db():
    con = duckdb.connect(":memory:")
    init_db(con)
    return con


def test_skipped_day_preserves_bankroll():
    con = memory_db()
    card = make_card(card_rows([]), warnings=["No soccer prediction rows found for this date."])

    result = write_challenge_ledger(con, card, bankroll=300, target=1500, grade=False)

    assert result.skipped_day is True
    assert result.ending_bankroll == 300
    assert result.progress_to_target_pct == 0
    con.close()


def test_official_bets_update_pending_ledger_and_staked_amount():
    con = memory_db()
    rows = card_rows(
        [
            {
                "match_id": "official-1",
                "selection": "Home",
                "final_recommendation": "Home",
                "odds": 2.0,
                "confidence": "strong",
                "confidence_score": 0.80,
                "edge": 0.09,
            }
        ]
    )

    result = write_challenge_ledger(con, make_card(rows, warnings=[]), bankroll=300, target=1500, grade=False)
    bets = con.execute("SELECT * FROM soccer_challenge_bets").df()

    assert result.official_bets_count == 1
    assert result.total_staked == 22.5
    assert result.ending_bankroll == 300
    assert bets.iloc[0]["status"] == "pending"
    con.close()


def test_lottery_leans_are_recorded_but_not_counted_as_official():
    con = memory_db()
    rows = card_rows(
        [
            {
                "match_id": "lottery-1",
                "selection": "Draw",
                "final_recommendation": "Draw",
                "odds": 9.25,
                "confidence": "strong",
                "confidence_score": 0.75,
                "edge": 0.11,
            }
        ]
    )

    result = write_challenge_ledger(con, make_card(rows, warnings=[]), bankroll=300, target=1500, grade=False)
    bets = con.execute("SELECT bet_type FROM soccer_challenge_bets").df()

    assert result.official_bets_count == 0
    assert result.lottery_leans_count == 1
    assert set(bets["bet_type"]) == {"lottery_lean"}
    con.close()


def test_blocked_lottery_draw_records_block_reason():
    con = memory_db()
    rows = card_rows(
        [
            {
                "match_id": "lottery-1",
                "selection": "Draw",
                "final_recommendation": "Draw",
                "odds": 9.25,
                "confidence": "strong",
                "confidence_score": 0.75,
                "edge": 0.11,
            }
        ]
    )

    write_challenge_ledger(
        con,
        make_card(rows, warnings=["Average draw probability is low: 20.0%"]),
        bankroll=300,
        target=1500,
        grade=False,
    )
    bet = con.execute("SELECT status, block_reason FROM soccer_challenge_bets").fetchone()

    assert bet[0] == "blocked"
    assert "Blocked because draw probability audit warning is active." in bet[1]
    con.close()


def test_grading_updates_bankroll_correctly():
    con = memory_db()
    rows = card_rows(
        [
            {
                "match_id": "official-1",
                "selection": "Home",
                "final_recommendation": "Home",
                "odds": 2.0,
                "confidence": "strong",
                "confidence_score": 0.80,
                "edge": 0.09,
            }
        ]
    )
    upsert_df(
        con,
        "soccer_matches",
        pd.DataFrame(
            [
                {
                    "match_id": "official-1",
                    "match_date": date(2026, 6, 20),
                    "home_team": "Home",
                    "away_team": "Away",
                    "home_score": 2,
                    "away_score": 0,
                    "status": "Final",
                }
            ]
        ),
    )

    result = write_challenge_ledger(con, make_card(rows, warnings=[]), bankroll=300, target=1500, grade=True)
    bet = con.execute("SELECT status, profit_loss FROM soccer_challenge_bets").fetchone()

    assert bet[0] == "win"
    assert bet[1] == 22.5
    assert result.ending_bankroll == 322.5
    con.close()


def test_progress_to_target_pct_is_correct():
    assert progress_to_target(300, 1500) == 0
    assert progress_to_target(450, 1500) == 12.5
    assert progress_to_target(1500, 1500) == 100
