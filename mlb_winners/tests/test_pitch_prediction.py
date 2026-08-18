from collections import Counter
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from mlb_winners.db import init_db, upsert_df
from mlb_winners.live import sample_pitch_type
from mlb_winners.pitch_prediction import PitchEvidence, evaluate_next_pitch_model, hierarchical_dirichlet_posterior
from mlb_winners.statcast import backfill_pitch_matchups, normalize_statcast_pitch_events
from mlb_winners.dashboard import _serialize


def evidence(**changes):
    values = dict(
        league_context={"FF": 60, "SL": 25, "CH": 15},
        pitcher_overall={"FF": 50, "SL": 30, "CH": 20},
        pitcher_count={}, exact_matchup={}, same_game={}, modifiers={}, modifier_reasons={},
    )
    values.update(changes)
    return PitchEvidence(**values)


def test_dirichlet_normalizes_and_count_evidence_changes_distribution():
    base, _ = hierarchical_dirichlet_posterior(evidence())
    ahead, diagnostic = hierarchical_dirichlet_posterior(evidence(pitcher_count={"SL": 35, "FF": 5}))
    assert sum(ahead.values()) == pytest.approx(1)
    assert ahead["SL"] > base["SL"]
    assert diagnostic["stage_l1_shifts"]["count"] > .1


def test_hierarchical_fallback_and_arsenal_filtering():
    probability, diagnostic = hierarchical_dirichlet_posterior(evidence(exact_matchup={"CH": 12}))
    assert probability["CH"] > diagnostic["stages"]["count"]["CH"]
    assert set(diagnostic["plausible_pitch_types"]) == {"FF", "SL", "CH"}
    assert sum(probability[p] for p in diagnostic["excluded_pitch_types"]) == pytest.approx(.005)
    unseen, unseen_diagnostic = hierarchical_dirichlet_posterior(evidence(pitcher_overall={}))
    assert sum(unseen.values()) == pytest.approx(1)
    assert len(unseen_diagnostic["plausible_pitch_types"]) >= 3


def test_live_modifiers_are_bounded():
    probability, _ = hierarchical_dirichlet_posterior(evidence(modifiers={"FF": 0, "SL": 50}))
    assert all(0 <= value <= 1 for value in probability.values())
    assert sum(probability.values()) == pytest.approx(1)


def test_seeded_sampler_uses_lower_ranked_pitches_at_expected_rates():
    pitches = [{"pitch_type": "FF", "probability": .46}, {"pitch_type": "CH", "probability": .30}, {"pitch_type": "SL", "probability": .24}]
    rng = np.random.default_rng(2026)
    observed = Counter(sample_pitch_type(rng, pitches) for _ in range(30_000))
    assert observed["FF"] / 30_000 == pytest.approx(.46, abs=.012)
    assert observed["CH"] / 30_000 == pytest.approx(.30, abs=.012)
    assert observed["SL"] / 30_000 == pytest.approx(.24, abs=.012)


def test_dashboard_serializes_top_three_pitch_distribution():
    payload = _serialize('[{"pitch_type":"FF","probability":0.46},{"pitch_type":"CH","probability":0.24},{"pitch_type":"SL","probability":0.17}]')
    assert [row["pitch_type"] for row in payload] == ["FF", "CH", "SL"]


def raw_pitch(day="2024-04-01", game=1, number=1, pitch="FF"):
    return {"game_pk": game, "game_date": day, "at_bat_number": 1, "pitch_number": number,
            "batter": 10, "pitcher": 20, "pitch_type": pitch, "stand": "R", "p_throws": "L",
            "balls": 0, "strikes": min(number - 1, 2), "description": "called_strike", "events": None}


def test_pitch_normalization_and_backfill_are_idempotent_and_resumable():
    con = duckdb.connect(":memory:")
    init_db(con)
    raw = pd.DataFrame([raw_pitch(number=1), raw_pitch(number=1), raw_pitch(number=2, pitch="SL")])
    assert len(normalize_statcast_pitch_events(raw)) == 2
    calls = []
    def fetcher(**kwargs):
        calls.append(kwargs)
        return raw
    first = backfill_pitch_matchups(con, date(2024, 4, 1), date(2024, 4, 2), fetcher=fetcher)
    second = backfill_pitch_matchups(con, date(2024, 4, 1), date(2024, 4, 2), fetcher=fetcher)
    assert first["completed_dates"] == 1 and first["no_data_dates"] == 1
    assert second["skipped_dates"] == 2
    assert len(calls) == 1
    assert con.execute("select count(*) from statcast_pitch_events").fetchone()[0] == 2


def test_chronological_evaluation_excludes_target_and_future_from_history():
    con = duckdb.connect(":memory:")
    init_db(con)
    rows = []
    for day, game, pitch in ((date(2024, 4, 1), 1, "FF"), (date(2025, 4, 1), 2, "SL"), (date(2026, 4, 1), 3, "CH")):
        rows.append({"game_pk": game, "game_date": day, "at_bat_number": 1, "pitch_number": 1,
                     "batter_id": 10, "pitcher_id": 20, "pitch_type": pitch, "batter_side": "R",
                     "pitcher_hand": "L", "balls": 0, "strikes": 0})
    upsert_df(con, "statcast_pitch_events", pd.DataFrame(rows))
    report = evaluate_next_pitch_model(con, date(2025, 4, 1), date(2025, 4, 1))
    assert report["pitches"] == 1
    assert "strictly earlier" in report["leakage_policy"]
