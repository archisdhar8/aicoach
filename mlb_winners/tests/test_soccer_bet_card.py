from __future__ import annotations

from datetime import date

import pandas as pd

from soccerworldcup.audit import audit_predictions
from soccerworldcup.bet_card import (
    BetCard,
    apply_challenge_stakes,
    approve_lottery_bets,
    approved_bets,
    challenge_candidates,
    format_bet_card,
    lottery_leans,
    must_play_candidates,
    select_must_play_pick,
    no_bet_leans,
    suggested_stake_units,
)


def card_rows(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "created_at": pd.Timestamp("2026-06-20T12:00:00Z"),
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
        score_sync_freshness="2026-06-20T12:00:00+00:00",
        odds_freshness="predictions created 2026-06-20T12:00:00+00:00",
    )


def test_approved_bets_exclude_no_bet_rows():
    rows = card_rows(
        [
            {"match_id": "1", "selection": "Home", "final_recommendation": "Home", "odds": 1.9, "confidence": "medium", "edge": 0.07},
            {"match_id": "2", "selection": "no bet", "final_recommendation": "no bet", "edge": 0.10},
        ]
    )

    bets = approved_bets(audit_predictions(rows, date(2026, 6, 20)).rows)

    assert len(bets) == 1
    assert bets.iloc[0]["match_id"] == "1"


def test_no_bet_leans_are_separated_and_labeled():
    rows = card_rows(
        [
            {"match_id": "1", "selection": "Home", "final_recommendation": "Home", "odds": 1.9, "confidence": "medium", "edge": 0.07},
            {"match_id": "2", "selection": "no bet", "final_recommendation": "no bet", "edge": 0.10, "skip_reason": "slate_cap"},
        ]
    )

    report = format_bet_card(make_card(rows, warnings=[]))

    assert "## Approved Bets" in report
    assert "## No-Bet Leans" in report
    assert "NO BET" in report


def test_audit_warnings_block_betting_output():
    rows = card_rows(
        [
            {"match_id": "1", "selection": "Home", "final_recommendation": "Home", "odds": 1.9, "confidence": "medium", "edge": 0.07},
        ]
    )

    report = format_bet_card(make_card(rows, warnings=["test warning"]))

    assert "DO NOT BET: audit warnings present" in report
    assert "Blocked by audit warnings" in report


def test_challenge_mode_does_not_block_entire_slate_on_warning():
    rows = card_rows(
        [
            {"match_id": "1", "selection": "Home", "final_recommendation": "Home", "odds": 1.9, "confidence": "medium", "edge": 0.07},
        ]
    )

    report = format_bet_card(make_card(rows, warnings=["No-bet rate is too low: 0.0%"]), mode="challenge", bankroll=300, target=1500)

    assert "CHALLENGE MODE" in report
    assert "DO NOT BET" not in report
    assert "Approved Challenge Bets" in report


def test_challenge_underdog_warning_bans_underdog_mls():
    rows = card_rows(
        [
            {"match_id": "dog", "selection": "Away", "final_recommendation": "Away", "odds": 3.5, "confidence": "strong", "edge": 0.12},
            {"match_id": "fav", "selection": "Home", "final_recommendation": "Home", "odds": 1.8, "confidence": "strong", "edge": 0.08},
        ]
    )
    audit = make_card(rows, warnings=["Underdog ML rate among recommended bets is high: 50.0%"]).audit

    candidates = challenge_candidates(audit.rows, audit.warnings)

    assert "Away" not in set(candidates["final_recommendation"])
    assert "Home" in set(candidates["final_recommendation"])


def test_challenge_over_warning_bans_overs():
    rows = card_rows(
        [
            {"match_id": "over", "market": "total_2_5", "selection": "Over 2.5", "final_recommendation": "Over 2.5", "odds": 2.1, "confidence": "strong", "edge": 0.10},
            {"match_id": "under", "market": "total_2_5", "selection": "Under 2.5", "final_recommendation": "Under 2.5", "odds": 2.0, "confidence": "strong", "edge": 0.08},
        ]
    )
    audit = make_card(rows, warnings=["Over rate among total recommendations is high: 50.0%"]).audit

    candidates = challenge_candidates(audit.rows, audit.warnings)

    assert "Over 2.5" not in set(candidates["final_recommendation"])
    assert "Under 2.5" in set(candidates["final_recommendation"])


def test_challenge_no_bet_rate_warning_reduces_max_bets():
    rows = card_rows(
        [
            {"match_id": "1", "selection": "Home", "final_recommendation": "Home", "odds": 1.8, "confidence": "strong", "edge": 0.12},
            {"match_id": "2", "selection": "Away", "final_recommendation": "Away", "odds": 2.1, "confidence": "strong", "edge": 0.11},
        ]
    )

    report = format_bet_card(make_card(rows, warnings=["No-bet rate is too low: 0.0%"]), mode="challenge", bankroll=300, target=1500)

    assert report.count("Market:") == 1


def test_challenge_missing_odds_never_approved():
    rows = card_rows(
        [
            {"match_id": "1", "selection": "Home", "final_recommendation": "Home", "odds": pd.NA, "confidence": "strong", "edge": 0.12},
        ]
    )
    audit = make_card(rows, warnings=[]).audit

    candidates = challenge_candidates(audit.rows, audit.warnings)

    assert candidates.empty


def test_challenge_staking_respects_bankroll_and_daily_risk():
    rows = card_rows(
        [
            {"match_id": "1", "selection": "Home", "final_recommendation": "Home", "odds": 1.8, "confidence": "strong", "edge": 0.12},
            {"match_id": "2", "selection": "Away", "final_recommendation": "Away", "odds": 2.1, "confidence": "strong", "edge": 0.11},
            {"match_id": "3", "selection": "Draw", "final_recommendation": "Draw", "odds": 3.4, "confidence": "strong", "edge": 0.10},
        ]
    )
    audit = make_card(rows, warnings=["No-bet rate is too low: 0.0%"]).audit
    candidates = challenge_candidates(audit.rows, audit.warnings)

    staked = apply_challenge_stakes(candidates, bankroll=300, warnings_present=True)

    assert staked["stake_dollars"].sum() <= 30
    assert staked["stake_dollars"].max() <= 45


def test_stake_sizing_works():
    strong = pd.Series({"confidence_score": 0.85, "edge": 0.09})
    half = pd.Series({"confidence_score": 0.70, "edge": 0.05})
    none = pd.Series({"confidence_score": 0.55, "edge": 0.20})

    assert suggested_stake_units(strong) == 1.0
    assert suggested_stake_units(half) == 0.5
    assert suggested_stake_units(none) == 0.0


def test_guardrails_and_pre_gate_display():
    rows = card_rows(
        [
            {
                "match_id": "1",
                "selection": "Under 2.5",
                "final_recommendation": "Under 2.5",
                "pre_gate_recommendation": "Under 2.5",
                "market": "total_2_5",
                "odds": 2.1,
                "confidence": "medium",
                "edge": 0.07,
                "guardrails_triggered": "missing_weather_total,early_group_market_anchor",
            },
        ]
    )

    report = format_bet_card(make_card(rows, warnings=[]))

    assert "missing_weather_total" in report
    assert "Pre-gate: Under 2.5" in report
    assert "Final: Under 2.5" in report


def test_lottery_draw_at_925_is_not_approved_by_default():
    rows = card_rows(
        [
            {
                "match_id": "lottery",
                "selection": "Draw",
                "final_recommendation": "Draw",
                "odds": 9.25,
                "confidence": "strong",
                "confidence_score": 0.74,
                "edge": 0.10,
            },
        ]
    )
    audit = make_card(rows, warnings=[]).audit

    candidates = challenge_candidates(audit.rows, audit.warnings)
    report = format_bet_card(make_card(rows, warnings=[]), mode="challenge", bankroll=300, target=1500)

    assert candidates.empty
    assert "## Approved Challenge Bets\n- None." in report


def test_lottery_draw_at_925_appears_under_lottery_leans():
    rows = card_rows(
        [
            {
                "match_id": "lottery",
                "selection": "Draw",
                "final_recommendation": "Draw",
                "odds": 9.25,
                "confidence": "strong",
                "confidence_score": 0.74,
                "edge": 0.10,
            },
        ]
    )

    report = format_bet_card(make_card(rows, warnings=[]), mode="challenge", bankroll=300, target=1500)

    assert "## Lottery Leans" in report
    assert "LOTTERY LEAN — not an official challenge bet" in report
    assert "Odds: 9.25" in report


def test_allow_lottery_can_approve_at_most_one_lottery_bet():
    rows = card_rows(
        [
            {
                "match_id": "lottery1",
                "selection": "Draw",
                "final_recommendation": "Draw",
                "odds": 9.25,
                "confidence": "strong",
                "confidence_score": 0.75,
                "edge": 0.11,
            },
            {
                "match_id": "lottery2",
                "selection": "Draw",
                "final_recommendation": "Draw",
                "odds": 7.5,
                "confidence": "strong",
                "confidence_score": 0.74,
                "edge": 0.10,
            },
        ]
    )
    audit = make_card(rows, warnings=[]).audit
    lottery = lottery_leans(audit.rows, audit.warnings)

    approved = approve_lottery_bets(lottery, bankroll=300)
    report = format_bet_card(make_card(rows, warnings=[]), mode="challenge", bankroll=300, target=1500, allow_lottery=True)

    assert len(approved) == 1
    assert report.count("APPROVED LOTTERY") == 1


def test_lottery_stake_is_capped_at_quarter_unit():
    rows = card_rows(
        [
            {
                "match_id": "lottery",
                "selection": "Draw",
                "final_recommendation": "Draw",
                "odds": 9.25,
                "confidence": "strong",
                "confidence_score": 0.75,
                "edge": 0.11,
            },
        ]
    )
    audit = make_card(rows, warnings=[]).audit
    approved = approve_lottery_bets(lottery_leans(audit.rows, audit.warnings), bankroll=300)

    assert approved.iloc[0]["stake_units"] == 0.25
    assert approved.iloc[0]["stake_dollars"] == 3.75


def test_draw_probability_warning_blocks_lottery_draws():
    rows = card_rows(
        [
            {
                "match_id": "lottery",
                "selection": "Draw",
                "final_recommendation": "Draw",
                "odds": 9.25,
                "confidence": "strong",
                "confidence_score": 0.75,
                "edge": 0.11,
            },
        ]
    )
    report = format_bet_card(
        make_card(rows, warnings=["Average draw probability is low: 20.0%"]),
        mode="challenge",
        bankroll=300,
        target=1500,
        allow_lottery=True,
    )

    assert "APPROVED LOTTERY" not in report
    assert "Blocked because draw probability audit warning is active." in report


def test_must_play_selects_one_pick_only_when_challenge_has_zero_official_bets():
    rows = card_rows(
        [
            {
                "match_id": "1",
                "selection": "no bet",
                "final_recommendation": "no bet",
                "pre_gate_recommendation": "Home",
                "candidate_selection": "Home",
                "candidate_market": "1x2",
                "candidate_odds": 1.85,
                "confidence": "medium",
                "confidence_score": 0.61,
                "edge": 0.04,
            },
            {
                "match_id": "2",
                "selection": "no bet",
                "final_recommendation": "no bet",
                "pre_gate_recommendation": "Under 2.5",
                "candidate_selection": "Under 2.5",
                "candidate_market": "total_2_5",
                "candidate_odds": 1.95,
                "confidence": "medium",
                "confidence_score": 0.60,
                "edge": 0.03,
            },
        ]
    )

    report = format_bet_card(make_card(rows, warnings=[]), mode="must-play", bankroll=300, target=1500)

    assert report.count("MUST-PLAY CHALLENGE PICK") == 1
    assert "This is action-mode, not strict model approval." in report


def test_must_play_does_not_add_extra_pick_when_official_exists():
    rows = card_rows(
        [
            {
                "match_id": "official",
                "selection": "Home",
                "final_recommendation": "Home",
                "odds": 1.8,
                "confidence": "strong",
                "confidence_score": 0.75,
                "edge": 0.09,
            },
            {
                "match_id": "forced",
                "selection": "no bet",
                "final_recommendation": "no bet",
                "candidate_selection": "Under 2.5",
                "candidate_market": "total_2_5",
                "candidate_odds": 1.9,
                "confidence": "medium",
                "confidence_score": 0.62,
                "edge": 0.04,
            },
        ]
    )

    report = format_bet_card(make_card(rows, warnings=[]), mode="must-play", bankroll=300, target=1500)

    assert "Normal challenge mode already found official approved bets" in report
    assert "MUST-PLAY CHALLENGE PICK" not in report


def test_must_play_hard_bans_missing_odds_lottery_draw_draw_warning_underdog_and_over_warning():
    rows = card_rows(
        [
            {"match_id": "missing", "candidate_selection": "Home", "candidate_market": "1x2", "candidate_odds": pd.NA, "confidence_score": 0.70, "edge": 0.10},
            {"match_id": "lottery", "candidate_selection": "Draw", "candidate_market": "1x2", "candidate_odds": 9.25, "confidence_score": 0.80, "edge": 0.20},
            {"match_id": "draw", "candidate_selection": "Draw", "candidate_market": "1x2", "candidate_odds": 4.2, "confidence_score": 0.70, "edge": 0.08},
            {"match_id": "dog", "candidate_selection": "Away", "candidate_market": "1x2", "candidate_odds": 3.0, "confidence_score": 0.70, "edge": 0.08},
            {"match_id": "over", "candidate_selection": "Over 2.5", "candidate_market": "total_2_5", "candidate_odds": 1.9, "confidence_score": 0.70, "edge": 0.08},
        ]
    )
    audit = make_card(
        rows,
        warnings=[
            "Average draw probability is low: 20.0%",
            "Underdog ML rate among recommended bets is high: 50.0%",
            "Over rate among total recommendations is high: 50.0%",
        ],
    ).audit

    candidates, rejected = must_play_candidates(audit.rows, audit.warnings)

    assert candidates.empty
    assert any("missing odds" in item for item in rejected)
    assert any("lottery draw banned" in item for item in rejected)
    assert any("draw probability audit warning active" in item for item in rejected)
    assert any("underdog ML audit warning active" in item for item in rejected)
    assert any("over-rate audit warning active" in item for item in rejected)


def test_must_play_confidence_below_058_is_banned():
    rows = card_rows(
        [
            {
                "match_id": "low",
                "candidate_selection": "Home",
                "candidate_market": "1x2",
                "candidate_odds": 1.8,
                "confidence": "thin",
                "confidence_score": 0.57,
                "edge": 0.10,
            }
        ]
    )
    audit = make_card(rows, warnings=[]).audit

    candidates, rejected = must_play_candidates(audit.rows, audit.warnings)

    assert candidates.empty
    assert any("confidence below 0.58" in item for item in rejected)


def test_must_play_stake_capped_around_10_for_300_bankroll_with_warnings():
    rows = card_rows(
        [
            {
                "match_id": "safe",
                "candidate_selection": "Home",
                "candidate_market": "1x2",
                "candidate_odds": 1.75,
                "confidence": "strong",
                "confidence_score": 0.80,
                "edge": 0.12,
            }
        ]
    )
    card = make_card(rows, warnings=["No-bet rate is too low: 0.0%"])

    selected, _ = select_must_play_pick(card, bankroll=300)

    assert len(selected) == 1
    assert selected.iloc[0]["stake_dollars"] <= 10


def test_must_play_prefers_derivative_market_over_lottery_draw():
    rows = card_rows(
        [
            {
                "match_id": "dnb",
                "selection": "no bet",
                "final_recommendation": "no bet",
                "candidate_selection": "Ecuador DNB",
                "candidate_market": "draw_no_bet",
                "candidate_odds": 1.75,
                "confidence": "medium",
                "confidence_score": 0.62,
                "edge": 0.04,
                "derivative_source_probabilities": '{"home_win": 0.50, "draw": 0.25, "away_win": 0.25}',
            },
            {
                "match_id": "lottery",
                "selection": "Draw",
                "final_recommendation": "Draw",
                "odds": 9.25,
                "confidence": "strong",
                "confidence_score": 0.80,
                "edge": 0.20,
            },
        ]
    )
    selected, rejected = select_must_play_pick(make_card(rows, warnings=[]), bankroll=300)

    assert selected.iloc[0]["final_recommendation"] == "Ecuador DNB"
    assert selected.iloc[0]["market"] == "draw_no_bet"


def test_missing_derivative_odds_are_skipped():
    rows = card_rows(
        [
            {
                "match_id": "dnb",
                "candidate_selection": "Ecuador DNB",
                "candidate_market": "draw_no_bet",
                "candidate_odds": pd.NA,
                "confidence_score": 0.70,
                "edge": 0.10,
                "derivative_source_probabilities": '{"home_win": 0.50, "draw": 0.25, "away_win": 0.25}',
            }
        ]
    )
    audit = make_card(rows, warnings=[]).audit

    candidates, rejected = must_play_candidates(audit.rows, audit.warnings)

    assert candidates.empty
    assert any("missing odds" in item for item in rejected)


def test_over_warning_blocks_btts_yes_and_aggressive_overs():
    rows = card_rows(
        [
            {
                "match_id": "btts",
                "candidate_selection": "BTTS Yes",
                "candidate_market": "btts",
                "candidate_odds": 1.9,
                "confidence_score": 0.70,
                "edge": 0.10,
            },
            {
                "match_id": "over",
                "candidate_selection": "Over 1.5",
                "candidate_market": "total_1_5",
                "candidate_odds": 1.5,
                "confidence_score": 0.70,
                "edge": 0.08,
            },
        ]
    )
    audit = make_card(rows, warnings=["Over rate among total recommendations is high: 50.0%"]).audit

    candidates, rejected = must_play_candidates(audit.rows, audit.warnings)

    assert candidates.empty
    assert any("BTTS Yes banned" in item for item in rejected)
    assert any("over-rate audit warning active" in item for item in rejected)
