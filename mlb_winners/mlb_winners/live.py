from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import Settings
from .db import connect, upsert_df
from .mlb_api import MLB_API_BASE, fetch_schedule, get_json
from .odds import ODDS_API_BASE, SPORT_KEY, devig_two_way
from .pitch_prediction import hierarchical_dirichlet_posterior, load_live_pitch_evidence
from .team_map import normalize_team_name


MODEL_VERSION = "live-pitch-v3-dirichlet"
LIVE_STATUSES = ("in progress", "manager challenge", "review", "delayed", "suspended")
PITCH_NAMES = {
    "FF": "Four-seam", "SI": "Sinker", "SL": "Slider", "CH": "Changeup",
    "ST": "Sweeper", "FC": "Cutter", "CU": "Curveball", "KC": "Knuckle curve",
    "FS": "Splitter", "SV": "Slurve", "KN": "Knuckleball", "EP": "Eephus",
}
PITCH_OUTCOMES = (
    "ball", "called_strike", "swinging_strike", "foul", "in_play_out",
    "single", "double", "triple", "home_run", "hbp",
)
BASE_STATE_FALLBACK = {
    0: (0.51, 0.28, 0.11), 1: (0.63, 0.42, 0.21), 2: (0.65, 0.44, 0.23),
    3: (0.80, 0.65, 0.43), 4: (0.85, 0.67, 0.28), 5: (0.87, 0.72, 0.38),
    6: (0.88, 0.75, 0.42), 7: (0.91, 0.82, 0.63),
}


@dataclass(frozen=True)
class LiveState:
    game_pk: int
    captured_at: datetime
    game_date: date
    status: str
    inning: int
    half_inning: str
    batting_team_id: int | None
    home_team_id: int
    away_team_id: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    base_mask: int
    outs: int
    balls: int
    strikes: int
    batter_id: int | None
    batter_name: str | None
    pitcher_id: int | None
    pitcher_name: str | None
    at_bat_index: int
    event_index: int
    batter_side: str = "U"
    pitcher_hand: str = "U"
    pitcher_pitch_count: int = 0
    times_through_order: int = 1


def encode_base_state(first: bool, second: bool, third: bool) -> int:
    return int(first) | (int(second) << 1) | (int(third) << 2)


def decode_base_state(base_mask: int) -> dict[str, bool]:
    if base_mask not in range(8):
        raise ValueError("base_mask must be between 0 and 7")
    return {"first": bool(base_mask & 1), "second": bool(base_mask & 2), "third": bool(base_mask & 4)}


def state_number(base_mask: int, outs: int) -> int:
    if base_mask not in range(8) or outs not in range(3):
        raise ValueError("base/out state must use base_mask 0..7 and outs 0..2")
    return outs * 8 + base_mask + 1


def runners_from_matchup(matchup: dict[str, Any]) -> int:
    post = matchup.get("postOnFirst"), matchup.get("postOnSecond"), matchup.get("postOnThird")
    if any(v is not None for v in post):
        return encode_base_state(*(v is not None for v in post))
    return 0


def parse_live_feed(payload: dict[str, Any]) -> tuple[LiveState, list[dict[str, Any]]]:
    game_data = payload.get("gameData") or {}
    live_data = payload.get("liveData") or {}
    linescore = live_data.get("linescore") or {}
    plays = (live_data.get("plays") or {}).get("allPlays") or []
    current = (live_data.get("plays") or {}).get("currentPlay") or (plays[-1] if plays else {})
    about = current.get("about") or {}
    matchup = current.get("matchup") or {}
    count = current.get("count") or {}
    teams = game_data.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    offense = linescore.get("offense") or {}
    defense = linescore.get("defense") or {}
    base_mask = encode_base_state(
        bool(offense.get("first")), bool(offense.get("second")), bool(offense.get("third"))
    )
    events = current.get("playEvents") or []
    last_event = events[-1] if events else {}
    pitcher_id = _int_or_none((matchup.get("pitcher") or defense.get("pitcher") or {}).get("id"))
    pitcher_events = [
        e for play in plays
        if _int_or_none(((play.get("matchup") or {}).get("pitcher") or {}).get("id")) == pitcher_id
        for e in (play.get("playEvents") or []) if e.get("isPitch")
    ]
    batter_id = _int_or_none((matchup.get("batter") or offense.get("batter") or {}).get("id"))
    batter_pas = sum(
        1 for play in plays
        if _int_or_none(((play.get("matchup") or {}).get("batter") or {}).get("id")) == batter_id
        and int((play.get("about") or {}).get("atBatIndex") or -1) <= int(about.get("atBatIndex") or 0)
    )
    state = LiveState(
        game_pk=int(game_data.get("game", {}).get("pk") or payload.get("gamePk")),
        captured_at=datetime.now(timezone.utc).replace(microsecond=0),
        game_date=pd.to_datetime(game_data.get("datetime", {}).get("officialDate")).date(),
        status=str((game_data.get("status") or {}).get("detailedState") or "Unknown"),
        inning=int(linescore.get("currentInning") or about.get("inning") or 1),
        half_inning=str(linescore.get("inningHalf") or about.get("halfInning") or "Top"),
        batting_team_id=_int_or_none((offense.get("team") or {}).get("id")),
        home_team_id=int(home.get("id")),
        away_team_id=int(away.get("id")),
        home_team=normalize_team_name(home.get("name")),
        away_team=normalize_team_name(away.get("name")),
        home_score=int((linescore.get("teams", {}).get("home") or {}).get("runs") or 0),
        away_score=int((linescore.get("teams", {}).get("away") or {}).get("runs") or 0),
        base_mask=base_mask,
        outs=min(int(count.get("outs") or linescore.get("outs") or 0), 2),
        balls=min(int(count.get("balls") or 0), 3),
        strikes=min(int(count.get("strikes") or 0), 2),
        batter_id=batter_id,
        batter_name=(matchup.get("batter") or offense.get("batter") or {}).get("fullName"),
        pitcher_id=pitcher_id,
        pitcher_name=(matchup.get("pitcher") or defense.get("pitcher") or {}).get("fullName"),
        at_bat_index=int(about.get("atBatIndex") or current.get("atBatIndex") or 0),
        event_index=int(last_event.get("index") or 0),
        batter_side=str((matchup.get("batSide") or {}).get("code") or "U"),
        pitcher_hand=str((matchup.get("pitchHand") or {}).get("code") or "U"),
        pitcher_pitch_count=len(pitcher_events),
        times_through_order=max(1, min(4, batter_pas)),
    )
    return state, normalize_plays(
        state.game_pk, state.game_date, plays, state.captured_at,
        state.home_team_id, state.away_team_id,
    )


def normalize_plays(
    game_pk: int,
    game_date: date,
    plays: list[dict[str, Any]],
    fetched_at: datetime,
    home_team_id: int,
    away_team_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    half_groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for play in plays:
        about = play.get("about") or {}
        half_groups.setdefault((int(about.get("inning") or 0), str(about.get("halfInning") or "")), []).append(play)
    runs_after: dict[int, int] = {}
    for grouped in half_groups.values():
        half_runs = [sum(1 for r in (p.get("runners") or []) if ((r.get("movement") or {}).get("end") == "score")) for p in grouped]
        final_runs = sum(half_runs)
        scored_before = 0
        for play in grouped:
            runs_after[id(play)] = max(final_runs - scored_before, 0)
            scored_before += sum(1 for r in (play.get("runners") or []) if ((r.get("movement") or {}).get("end") == "score"))
    state_by_half: dict[tuple[int, str], tuple[int, int]] = {}
    for play in plays:
        about, matchup, result = play.get("about") or {}, play.get("matchup") or {}, play.get("result") or {}
        key = (int(about.get("inning") or 0), str(about.get("halfInning") or ""))
        starting_mask, starting_outs = state_by_half.get(key, (0, 0))
        ending_mask = starting_mask
        for runner in play.get("runners") or []:
            movement = runner.get("movement") or {}
            start, end = movement.get("start"), movement.get("end")
            bit = {"1B": 1, "2B": 2, "3B": 4}
            if start in bit:
                ending_mask &= ~bit[start]
            if end in bit:
                ending_mask |= bit[end]
        ending_outs = int((play.get("count") or {}).get("outs") or starting_outs)
        state_by_half[key] = (ending_mask, 0 if ending_outs >= 3 else ending_outs)
        at_bat = int(about.get("atBatIndex") or play.get("atBatIndex") or 0)
        events = play.get("playEvents") or [{}]
        for event in events:
            details, pitch_data, count = event.get("details") or {}, event.get("pitchData") or {}, event.get("count") or {}
            coords, breaks = pitch_data.get("coordinates") or {}, pitch_data.get("breaks") or {}
            rows.append({
                "game_pk": game_pk, "at_bat_index": at_bat, "event_index": int(event.get("index") or 0),
                "event_id": event.get("playId"), "game_date": game_date, "inning": int(about.get("inning") or 0),
                "half_inning": about.get("halfInning"),
                "batting_team_id": home_team_id if str(about.get("halfInning", "")).lower() == "bottom" else away_team_id,
                "fielding_team_id": away_team_id if str(about.get("halfInning", "")).lower() == "bottom" else home_team_id,
                "batter_id": _int_or_none((matchup.get("batter") or {}).get("id")),
                "batter_name": (matchup.get("batter") or {}).get("fullName"), "pitcher_id": _int_or_none((matchup.get("pitcher") or {}).get("id")),
                "pitcher_name": (matchup.get("pitcher") or {}).get("fullName"), "base_mask": starting_mask,
                "batter_side": str((matchup.get("batSide") or {}).get("code") or "U"),
                "pitcher_hand": str((matchup.get("pitchHand") or {}).get("code") or "U"),
                "outs": min(starting_outs, 2), "balls": min(int(count.get("balls") or 0), 3),
                "strikes": min(int(count.get("strikes") or 0), 2), "home_score": int(result.get("homeScore") or 0),
                "away_score": int(result.get("awayScore") or 0), "event_type": details.get("eventType") or result.get("eventType"),
                "event_description": details.get("description") or result.get("description"),
                "pitch_type": (details.get("type") or {}).get("code"), "start_speed": pitch_data.get("startSpeed"),
                "spin_rate": breaks.get("spinRate"), "plate_x": coords.get("pX"), "plate_z": coords.get("pZ"),
                "is_pitch": bool(event.get("isPitch")), "is_in_play": bool(details.get("isInPlay")),
                "runs_to_inning_end": runs_after.get(id(play)), "fetched_at": fetched_at,
                "raw_payload": json.dumps(event),
            })
    return rows


def team_state_rates(con, team_id: int, as_of: date, limit_games: int = 50) -> dict[str, Any]:
    game_rows = con.execute(
        """
        SELECT DISTINCT e.game_pk, e.game_date
        FROM live_play_events e JOIN games g USING (game_pk)
        WHERE e.batting_team_id = ? AND e.game_date < ? AND g.game_type = 'R'
        ORDER BY e.game_date DESC, e.game_pk DESC LIMIT ?
        """, [team_id, as_of, limit_games],
    ).fetchall()
    game_pks = [int(r[0]) for r in game_rows]
    rates: dict[tuple[int, int], tuple[float, list[float], int]] = {}
    if game_pks:
        marks = ",".join("?" for _ in game_pks)
        frame = con.execute(
            f"""
            SELECT base_mask, outs, runs_to_inning_end
            FROM live_play_events e
            WHERE batting_team_id = ? AND game_pk IN ({marks})
              AND event_index = (SELECT min(e2.event_index) FROM live_play_events e2
                                 WHERE e2.game_pk=e.game_pk AND e2.at_bat_index=e.at_bat_index)
              AND base_mask BETWEEN 0 AND 7 AND outs BETWEEN 0 AND 2
            """, [team_id, *game_pks],
        ).df()
        for (mask, outs), group in frame.groupby(["base_mask", "outs"]):
            values = group["runs_to_inning_end"].fillna(0).clip(0, 4).astype(int)
            n = len(values)
            prior_p = fallback_score_probability(int(mask), int(outs))
            score_p = (float((values > 0).sum()) + prior_p * 20) / (n + 20)
            dist = [float(((values == k).sum() + 2) / (n + 10)) for k in range(4)]
            dist.append(max(0.0, 1.0 - sum(dist)))
            rates[(int(mask), int(outs))] = (score_p, dist, n)
    output = []
    for outs in range(3):
        for mask in range(8):
            p, dist, n = rates.get((mask, outs), (fallback_score_probability(mask, outs), fallback_run_distribution(mask, outs), 0))
            output.append({"state": state_number(mask, outs), "base_mask": mask, "outs": outs, "score_probability": p, "run_distribution": dist, "sample_size": n})
    return {
        "team_id": team_id, "as_of": as_of.isoformat(), "games_used": len(game_pks),
        "window_start": game_rows[-1][1].isoformat() if game_rows else None,
        "window_end": game_rows[0][1].isoformat() if game_rows else None,
        "quality": "team" if len(game_pks) == limit_games else "shrunk_to_league",
        "states": output,
    }


def fallback_score_probability(base_mask: int, outs: int) -> float:
    return BASE_STATE_FALLBACK[base_mask][outs]


def fallback_run_distribution(base_mask: int, outs: int) -> list[float]:
    p = fallback_score_probability(base_mask, outs)
    multi = min(0.45, 0.10 + 0.06 * base_mask.bit_count() + 0.04 * (2 - outs))
    return [1 - p, p * (1 - multi), p * multi * 0.65, p * multi * 0.25, p * multi * 0.10]


def next_pitch_probabilities(con, state: LiveState) -> list[dict[str, Any]]:
    rows, _ = _next_pitch_estimate(con, state)
    return rows


def _next_pitch_estimate(con, state: LiveState) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence, supporting = load_live_pitch_evidence(con, state)
    posterior, diagnostics = hierarchical_dirichlet_posterior(evidence)
    promotion = con.execute(
        """SELECT report_json FROM pitch_prediction_evaluations
           WHERE start_date>=DATE '2026-01-01' ORDER BY end_date DESC LIMIT 1"""
    ).fetchone()
    promoted = bool(promotion and json.loads(promotion[0]).get("beats_overall_arsenal_log_loss"))
    if not promoted:
        posterior = diagnostics["stages"]["overall_arsenal"]
        posterior_total = sum(posterior.values()) or 1.0
        posterior = {key: value / posterior_total for key, value in posterior.items()}
    diagnostics["production_selection"] = {
        "candidate_promoted": promoted,
        "distribution": "hierarchical_posterior" if promoted else "pitcher_overall_arsenal",
        "reason": "2026 holdout log-loss gate" if promotion else "no completed 2026 holdout evaluation",
    }
    overall_total = sum(evidence.pitcher_overall.values()) or 1.0
    live_total = sum(evidence.same_game.values()) or 1.0
    expected, live = supporting["expected"], supporting["live"]
    rows = []
    plausible = set(diagnostics["plausible_pitch_types"])
    for pitch_type, probability in sorted(posterior.items(), key=lambda item: item[1], reverse=True):
        hist_velocity = _float_or_none((expected.get(pitch_type) or {}).get("velocity"))
        current_velocity = _float_or_none((live.get(pitch_type) or {}).get("velocity"))
        velocity_delta = (current_velocity - hist_velocity) if current_velocity is not None and hist_velocity is not None else None
        rows.append({
            "pitch_type": pitch_type,
            "pitch_name": PITCH_NAMES.get(pitch_type, pitch_type),
            "probability": probability,
            "is_plausible": pitch_type in plausible,
            "expected_usage": evidence.pitcher_overall.get(pitch_type, 0.0) / overall_total,
            "actual_usage": evidence.same_game.get(pitch_type, 0.0) / live_total if evidence.same_game else 0.0,
            "historical_pitches": int(evidence.pitcher_overall.get(pitch_type, 0.0)),
            "game_pitches": int(evidence.same_game.get(pitch_type, 0.0)),
            "game_whiff_rate": float((live.get(pitch_type) or {}).get("whiffs") or 0) / evidence.same_game[pitch_type] if evidence.same_game.get(pitch_type) else 0.0,
            "game_in_play_rate": float((live.get(pitch_type) or {}).get("in_play") or 0) / evidence.same_game[pitch_type] if evidence.same_game.get(pitch_type) else 0.0,
            "expected_velocity": hist_velocity,
            "current_velocity": current_velocity,
            "velocity_delta": velocity_delta,
            "expected_spin": _float_or_none((expected.get(pitch_type) or {}).get("spin")),
            "current_spin": _float_or_none((live.get(pitch_type) or {}).get("spin")),
            "posterior_evidence": {
                stage: values.get(pitch_type, 0.0) for stage, values in diagnostics["stages"].items()
            },
            "modifier_reasons": diagnostics["modifier_reasons"].get(pitch_type, []),
        })
    return rows, diagnostics


def pitch_outcome_probabilities(con, state: LiveState, pitch_type: str) -> dict[str, Any]:
    """Hierarchically shrink pitch results from league to pitcher, batter, then matchup."""
    probabilities = _default_pitch_outcomes(pitch_type, state.balls, state.strikes)
    sample_size = 0
    expected_woba = None
    run_value = None
    levels = [
        ("pitch_type=? AND pitcher_hand=? AND batter_side=?", [pitch_type, state.pitcher_hand, state.batter_side], 220.0),
        ("pitch_type=? AND pitcher_id=? AND batter_side=?", [pitch_type, state.pitcher_id, state.batter_side], 120.0),
        ("pitch_type=? AND batter_id=? AND pitcher_hand=?", [pitch_type, state.batter_id, state.pitcher_hand], 90.0),
        ("pitch_type=? AND batter_id=? AND pitcher_id=?", [pitch_type, state.batter_id, state.pitcher_id], 45.0),
    ]
    for where, params, prior_strength in levels:
        if any(value is None for value in params):
            continue
        row = _pitch_outcome_cell(con, state, where, params)
        if not row or row["pitch_count"] <= 0:
            continue
        n = row["pitch_count"]
        weight = n / (n + prior_strength)
        probabilities = {
            key: probabilities[key] * (1 - weight) + row[key] * weight for key in PITCH_OUTCOMES
        }
        sample_size = max(sample_size, int(n))
        if row["expected_woba"] is not None:
            expected_woba = row["expected_woba"] if expected_woba is None else expected_woba * (1 - weight) + row["expected_woba"] * weight
        if row["run_value"] is not None:
            run_value = row["run_value"] if run_value is None else run_value * (1 - weight) + row["run_value"] * weight
    probabilities = _normalize_probabilities(probabilities)
    return {
        "probabilities": probabilities,
        "sample_size": sample_size,
        "expected_woba": expected_woba,
        "run_value": run_value,
        "whiff_rate": probabilities["swinging_strike"],
    }


def build_pitch_matchup_analysis(con, state: LiveState) -> dict[str, Any]:
    pitches, diagnostics = _next_pitch_estimate(con, state)
    for pitch in pitches:
        outcome = pitch_outcome_probabilities(con, state, pitch["pitch_type"])
        pitch.update({
            "outcomes": outcome["probabilities"],
            "sample_size": outcome["sample_size"],
            "expected_woba": outcome["expected_woba"],
            "run_value": outcome["run_value"],
            "whiff_rate": outcome["whiff_rate"],
        })
    most_likely = pitches[0] if pitches else None
    return {
        "most_likely_pitch": most_likely["pitch_type"] if most_likely else None,
        "most_likely_pitch_name": most_likely["pitch_name"] if most_likely else None,
        "most_likely_pitch_prob": most_likely["probability"] if most_likely else None,
        "pitches": pitches,
        "top_three": [
            {"pitch_type": row["pitch_type"], "pitch_name": row["pitch_name"], "probability": row["probability"]}
            for row in pitches[:3]
        ],
        "posterior_diagnostics": diagnostics,
    }


def _pitch_outcome_cell(con, state: LiveState, where: str, params: list[Any]) -> dict[str, Any] | None:
    rate_columns = {
        "ball": "ball_rate", "called_strike": "called_strike_rate",
        "swinging_strike": "swinging_strike_rate", "foul": "foul_rate",
        "in_play_out": "in_play_out_rate", "single": "single_rate",
        "double": "double_rate", "triple": "triple_rate",
        "home_run": "home_run_rate", "hbp": "hbp_rate",
    }
    sums = ", ".join(f"sum({column}*pitch_count)/nullif(sum(pitch_count),0)" for column in rate_columns.values())
    row = con.execute(
        f"""SELECT sum(pitch_count), {sums},
                    sum(expected_woba*pitch_count)/nullif(sum(CASE WHEN expected_woba IS NOT NULL THEN pitch_count ELSE 0 END),0),
                    sum(run_value*pitch_count)/nullif(sum(CASE WHEN run_value IS NOT NULL THEN pitch_count ELSE 0 END),0)
             FROM pitch_type_matchup_daily
             WHERE game_date<? AND game_date>=? AND balls=? AND strikes=? AND {where}""",
        [state.game_date, state.game_date - timedelta(days=365), state.balls, state.strikes, *params],
    ).fetchone()
    if not row or row[0] is None:
        return None
    result = {"pitch_count": float(row[0])}
    for index, key in enumerate(rate_columns, start=1):
        result[key] = float(row[index] or 0.0)
    result["expected_woba"] = _float_or_none(row[-2])
    result["run_value"] = _float_or_none(row[-1])
    return result


def _default_pitch_outcomes(pitch_type: str, balls: int, strikes: int) -> dict[str, float]:
    outcomes = {
        "ball": .335, "called_strike": .155, "swinging_strike": .115, "foul": .170,
        "in_play_out": .145, "single": .047, "double": .014, "triple": .001,
        "home_run": .010, "hbp": .008,
    }
    if pitch_type in {"SL", "ST", "CU", "KC", "FS", "CH"}:
        outcomes["swinging_strike"] *= 1.18
        outcomes["called_strike"] *= .92
        outcomes["ball"] *= 1.05
    if pitch_type in {"FF", "SI", "FC"}:
        outcomes["called_strike"] *= 1.08
        outcomes["home_run"] *= 1.10
    if strikes == 2:
        outcomes["swinging_strike"] *= 1.22
        outcomes["ball"] *= 1.08
        outcomes["in_play_out"] *= .90
    if balls == 3:
        outcomes["called_strike"] *= 1.20
        outcomes["ball"] *= .78
    return _normalize_probabilities(outcomes)


def _normalize_probabilities(values: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(float(values.get(key, 0.0)), 0.0) for key in PITCH_OUTCOMES}
    total = sum(clipped.values()) or 1.0
    return {key: value / total for key, value in clipped.items()}


def plate_appearance_probabilities(con, state: LiveState) -> dict[str, float]:
    base = {"strikeout": .225, "walk_hbp": .095, "single": .145, "double": .045, "triple": .004, "home_run": .032}
    if state.batter_id:
        row = con.execute(
            """SELECT avg(k_rate), avg(bb_rate), avg(xba), avg(xslg), avg(barrel_rate)
               FROM player_statcast_daily WHERE player_id=? AND game_date<? AND game_date>=?""",
            [state.batter_id, state.game_date, state.game_date - timedelta(days=60)],
        ).fetchone()
        if row and row[0] is not None:
            base["strikeout"] = float(np.clip(row[0], .08, .45))
            base["walk_hbp"] = float(np.clip(row[1] + .01, .03, .22))
            base["single"] = float(np.clip((row[2] or .245) * .62, .07, .23))
            base["home_run"] = float(np.clip((row[4] or .07) * .43, .008, .10))
            base["double"] = float(np.clip(((row[3] or .400) - (row[2] or .245)) * .25, .02, .09))
    count_pressure = (state.strikes - state.balls) * .055
    base["strikeout"] = float(np.clip(base["strikeout"] + count_pressure, .06, .55))
    base["walk_hbp"] = float(np.clip(base["walk_hbp"] - count_pressure * .7, .02, .35))
    known = sum(base.values())
    base["other_out"] = max(.02, 1 - known)
    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


def predict_live(con, state: LiveState, sims: int = 2500) -> dict[str, Any]:
    rates = team_state_rates(con, state.batting_team_id or state.home_team_id, state.game_date)
    entry = next(s for s in rates["states"] if s["base_mask"] == state.base_mask and s["outs"] == state.outs)
    pa = plate_appearance_probabilities(con, state)
    pitch_analysis = build_pitch_matchup_analysis(con, state)
    contact_boost = pa["home_run"] * 1.5 + pa["single"] * .25 + pa["walk_hbp"] * .15 - pa["strikeout"] * .20
    score_p = float(np.clip(entry["score_probability"] + contact_boost - .035, .01, .99))
    pregame = _pregame_home_probability(con, state.game_pk)
    home_prob, samples = simulate_live_game(
        state, score_p, pregame, sims=sims, pitch_analysis=pitch_analysis
    )
    sensitivities = []
    for pitch in pitch_analysis["pitches"][:5]:
        forced_prob, _ = simulate_live_game(
            state, score_p, pregame, sims=400, pitch_analysis=pitch_analysis,
            forced_first_pitch=pitch["pitch_type"],
        )
        sensitivities.append({
            "pitch_type": pitch["pitch_type"], "pitch_name": pitch["pitch_name"],
            "home_win_prob": forced_prob, "delta": forced_prob - home_prob,
        })
    low, high = np.quantile(samples, [.05, .95])
    flags = []
    if rates["games_used"] < 50:
        flags.append(f"state rates use {rates['games_used']}/50 games")
    if state.batter_id is None or state.pitcher_id is None:
        flags.append("missing live matchup identity")
    return {
        "score_this_inning_prob": score_p, "home_win_prob": home_prob, "away_win_prob": 1 - home_prob,
        "uncertainty_low": float(low), "uncertainty_high": float(high), "state_sample_size": entry["sample_size"],
        "quality_flags": flags, "pa_outcomes": pa, "state_rates": rates,
        "pitch_analysis": pitch_analysis, "pitch_sensitivities": sensitivities,
    }


def simulate_live_game(
    state: LiveState,
    score_p: float,
    pregame_home: float,
    sims: int = 5000,
    pitch_analysis: dict[str, Any] | None = None,
    forced_first_pitch: str | None = None,
) -> tuple[float, np.ndarray]:
    seed_key = f"{state.game_pk}:{state.at_bat_index}:{state.event_index}:{state.balls}:{state.strikes}"
    rng = np.random.default_rng(int(hashlib.sha1(seed_key.encode()).hexdigest()[:8], 16))
    home = np.full(sims, state.home_score, dtype=int)
    away = np.full(sims, state.away_score, dtype=int)
    batting_home = state.half_inning.lower().startswith("bottom")
    if pitch_analysis and pitch_analysis.get("pitches"):
        current_runs = np.zeros(sims, dtype=int)
        for index in range(sims):
            mask, outs = state.base_mask, state.outs
            balls, strikes = state.balls, state.strikes
            first_pitch = forced_first_pitch
            plate_appearances = 0
            while outs < 3 and plate_appearances < 15:
                outcome = simulate_plate_appearance(
                    rng, pitch_analysis, balls, strikes, forced_first_pitch=first_pitch
                )
                first_pitch = None
                mask, outs, runs = apply_plate_appearance_outcome(mask, outs, outcome)
                current_runs[index] += runs
                plate_appearances += 1
                balls = strikes = 0
                if batting_home and state.inning >= 9 and home[index] + current_runs[index] > away[index]:
                    break
    else:
        current_runs = rng.binomial(1, score_p, sims) + rng.binomial(1, score_p * .22, sims)
    (home if batting_home else away)[:] += current_runs
    halves_left = max(0, (9 - state.inning) * 2 + (1 if state.half_inning.lower().startswith("top") else 0))
    home_rate = float(np.clip(.28 + (pregame_home - .5) * .30, .17, .45))
    away_rate = float(np.clip(.28 - (pregame_home - .5) * .30, .17, .45))
    for half in range(halves_left):
        if half % 2 == (0 if state.half_inning.lower().startswith("bottom") else 1):
            away += rng.binomial(1, away_rate, sims) + rng.binomial(1, away_rate * .18, sims)
        else:
            home += rng.binomial(1, home_rate, sims) + rng.binomial(1, home_rate * .18, sims)
    tied = home == away
    while tied.any():
        n = int(tied.sum())
        home[tied] += rng.binomial(1, .54, n)
        away[tied] += rng.binomial(1, .46, n)
        tied = home == away
    wins = (home > away).astype(float)
    # Blend a small pregame prior early; observed state dominates late.
    live_weight = min(.97, .45 + state.inning * .055)
    samples = live_weight * wins + (1 - live_weight) * pregame_home
    return float(samples.mean()), samples


def simulate_plate_appearance(
    rng: np.random.Generator,
    pitch_analysis: dict[str, Any],
    balls: int = 0,
    strikes: int = 0,
    forced_first_pitch: str | None = None,
) -> str:
    pitches = pitch_analysis.get("pitches") or []
    outcomes_by_pitch = {row["pitch_type"]: row["outcomes"] for row in pitches}
    for pitch_number in range(20):
        pitch_type = forced_first_pitch if pitch_number == 0 and forced_first_pitch in outcomes_by_pitch else sample_pitch_type(rng, pitches)
        outcomes = _count_adjusted_outcomes(outcomes_by_pitch[pitch_type], balls, strikes)
        names = list(PITCH_OUTCOMES)
        outcome = str(rng.choice(names, p=[outcomes[name] for name in names]))
        if outcome == "ball":
            balls += 1
            if balls >= 4:
                return "walk_hbp"
        elif outcome in {"called_strike", "swinging_strike"}:
            strikes += 1
            if strikes >= 3:
                return "strikeout"
        elif outcome == "foul":
            strikes = min(2, strikes + 1)
        elif outcome == "hbp":
            return "walk_hbp"
        else:
            return outcome
    return "other_out"


def sample_pitch_type(rng: np.random.Generator, pitches: list[dict[str, Any]]) -> str:
    """Sample the full distribution; this deliberately does not take argmax."""
    if not pitches:
        raise ValueError("Cannot sample an empty pitch distribution.")
    probability = np.asarray([max(0.0, float(row.get("probability", 0))) for row in pitches])
    total = float(probability.sum())
    probability = probability / total if total > 0 else np.full(len(pitches), 1 / len(pitches))
    return str(rng.choice([row["pitch_type"] for row in pitches], p=probability))


def apply_plate_appearance_outcome(base_mask: int, outs: int, outcome: str) -> tuple[int, int, int]:
    first, second, third = bool(base_mask & 1), bool(base_mask & 2), bool(base_mask & 4)
    if outcome in {"strikeout", "in_play_out", "other_out"}:
        return base_mask, min(3, outs + 1), 0
    if outcome == "walk_hbp":
        runs = int(first and second and third)
        new_third = third or (first and second)
        new_second = second or first
        return encode_base_state(True, new_second, new_third), outs, runs
    if outcome == "single":
        runs = int(second) + int(third)
        return encode_base_state(True, first, False), outs, runs
    if outcome == "double":
        runs = int(first) + int(second) + int(third)
        return encode_base_state(False, True, False), outs, runs
    if outcome == "triple":
        runs = int(first) + int(second) + int(third)
        return encode_base_state(False, False, True), outs, runs
    if outcome == "home_run":
        return 0, outs, 1 + int(first) + int(second) + int(third)
    return base_mask, min(3, outs + 1), 0


def _count_adjusted_outcomes(outcomes: dict[str, float], balls: int, strikes: int) -> dict[str, float]:
    adjusted = dict(outcomes)
    if strikes == 2:
        adjusted["swinging_strike"] *= 1.18
        adjusted["ball"] *= 1.08
        adjusted["in_play_out"] *= .92
    if balls == 3:
        adjusted["ball"] *= .80
        adjusted["called_strike"] *= 1.20
    return _normalize_probabilities(adjusted)


def snapshot_row(state: LiveState, prediction: dict[str, Any]) -> dict[str, Any]:
    raw = asdict(state)
    raw.update({k: prediction[k] for k in ("score_this_inning_prob", "home_win_prob", "away_win_prob", "uncertainty_low", "uncertainty_high", "state_sample_size")})
    raw["captured_at"] = pd.Timestamp(state.captured_at)
    raw["snapshot_id"] = hashlib.sha1(f"{state.game_pk}:{state.at_bat_index}:{state.event_index}:{state.balls}:{state.strikes}".encode()).hexdigest()
    raw["model_version"] = MODEL_VERSION
    raw["quality_flags"] = json.dumps(prediction["quality_flags"])
    raw["most_likely_pitch"] = prediction["pitch_analysis"].get("most_likely_pitch")
    raw["most_likely_pitch_prob"] = prediction["pitch_analysis"].get("most_likely_pitch_prob")
    raw["top_three_pitches_json"] = json.dumps(prediction["pitch_analysis"].get("top_three", []))
    raw["pitch_analysis_json"] = json.dumps(prediction["pitch_analysis"])
    raw["pitch_sensitivities_json"] = json.dumps(prediction["pitch_sensitivities"])
    raw["pa_outcomes_json"] = json.dumps(prediction["pa_outcomes"])
    raw.pop("at_bat_index")
    raw.pop("event_index")
    return raw


def fetch_live_game(con, game_pk: int, force: bool = True) -> tuple[LiveState, dict[str, Any]]:
    payload = get_json(con, "mlb_live_feed", f"live:{game_pk}", f"{MLB_API_BASE}.1/game/{game_pk}/feed/live", {}, force=force)
    state, events = parse_live_feed(payload)
    if events:
        upsert_df(con, "live_play_events", pd.DataFrame(events))
    prediction = predict_live(con, state)
    upsert_df(con, "live_game_snapshots", pd.DataFrame([snapshot_row(state, prediction)]))
    return state, prediction


def fetch_historical_play_by_play(con, game_pk: int, game_date: date, force: bool = False) -> int:
    payload = get_json(con, "mlb_live_feed", f"live:{game_pk}", f"{MLB_API_BASE}.1/game/{game_pk}/feed/live", {}, force=force)
    _, events = parse_live_feed(payload)
    return upsert_df(con, "live_play_events", pd.DataFrame(events)) if events else 0


def latest_market(con, game_pk: int) -> dict[str, Any] | None:
    row = con.execute(
        """SELECT fetched_at, bookmaker, home_moneyline, away_moneyline, home_no_vig, away_no_vig
           FROM live_odds_snapshots WHERE game_pk=? ORDER BY fetched_at DESC LIMIT 1""", [game_pk],
    ).fetchone()
    if not row:
        return None
    age = (datetime.now(timezone.utc).replace(tzinfo=None) - row[0]).total_seconds()
    return {"fetched_at": row[0].isoformat(), "bookmaker": row[1], "home_moneyline": row[2], "away_moneyline": row[3],
            "home_no_vig": row[4], "away_no_vig": row[5], "age_seconds": age, "stale": age > 90}


def refresh_live_odds(con, api_key: str | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    last = con.execute("SELECT requested_at FROM live_odds_refreshes ORDER BY requested_at DESC LIMIT 1").fetchone()
    if last and (now.replace(tzinfo=None) - last[0]).total_seconds() < 60:
        wait = math.ceil(60 - (now.replace(tzinfo=None) - last[0]).total_seconds())
        raise RuntimeError(f"Odds refresh cooldown active; try again in {wait} seconds.")
    refresh_id = hashlib.sha1(now.isoformat().encode()).hexdigest()
    con.execute("INSERT INTO live_odds_refreshes (refresh_id, requested_at, status, rows_written) VALUES (?, ?, 'running', 0)", [refresh_id, now])
    api_key = api_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        _finish_refresh(con, refresh_id, "error", 0, None, None, "ODDS_API_KEY is not set")
        raise RuntimeError("ODDS_API_KEY is not set")
    try:
        response = requests.get(f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds", params={
            "apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american", "dateFormat": "iso"
        }, timeout=30)
        response.raise_for_status()
        payload = response.json()
        remaining, used = response.headers.get("x-requests-remaining"), response.headers.get("x-requests-used")
        games = con.execute("SELECT game_pk, home_team, away_team, game_datetime FROM games WHERE game_date BETWEEN ? AND ?", [now.date() - timedelta(days=1), now.date() + timedelta(days=1)]).df()
        rows = []
        snapshot_id = refresh_id
        for event in payload:
            home, away = normalize_team_name(event.get("home_team")), normalize_team_name(event.get("away_team"))
            match = games[(games["home_team"] == home) & (games["away_team"] == away)]
            game_pk = int(match.iloc[0]["game_pk"]) if not match.empty else None
            for book in event.get("bookmakers") or []:
                market = next((m for m in book.get("markets") or [] if m.get("key") == "h2h"), None)
                prices = {normalize_team_name(o.get("name")): o.get("price") for o in (market or {}).get("outcomes") or []}
                if home not in prices or away not in prices:
                    continue
                hp, ap = devig_two_way(prices[home], prices[away])
                rows.append({"snapshot_id": snapshot_id, "event_id": event.get("id"), "game_pk": game_pk, "fetched_at": now,
                             "bookmaker": book.get("key") or book.get("title"), "home_team": home, "away_team": away,
                             "home_moneyline": int(prices[home]), "away_moneyline": int(prices[away]), "home_no_vig": hp,
                             "away_no_vig": ap, "api_remaining": remaining, "api_used": used, "raw_payload": json.dumps(event)})
        if rows:
            upsert_df(con, "live_odds_snapshots", pd.DataFrame(rows))
        _finish_refresh(con, refresh_id, "success", len(rows), remaining, used, None)
        return {"rows": len(rows), "fetched_at": now.isoformat(), "remaining": remaining, "used": used, "source": "api"}
    except Exception as exc:
        _finish_refresh(con, refresh_id, "error", 0, None, None, str(exc))
        raise


def _finish_refresh(con, refresh_id: str, status: str, rows: int, remaining: str | None, used: str | None, error: str | None) -> None:
    con.execute("""UPDATE live_odds_refreshes SET completed_at=now(), status=?, rows_written=?,
                   api_remaining=?, api_used=?, error=? WHERE refresh_id=?""", [status, rows, remaining, used, error, refresh_id])


def odds_status(con) -> dict[str, Any]:
    row = con.execute("""SELECT requested_at, completed_at, status, rows_written, api_remaining, api_used, error
                         FROM live_odds_refreshes ORDER BY requested_at DESC LIMIT 1""").fetchone()
    if not row:
        return {"status": "never", "cooldown_seconds": 0}
    age = max(0, (datetime.now(timezone.utc).replace(tzinfo=None) - row[0]).total_seconds())
    return {"requested_at": row[0].isoformat(), "completed_at": row[1].isoformat() if row[1] else None,
            "status": row[2], "rows": row[3], "remaining": row[4], "used": row[5], "error": row[6],
            "cooldown_seconds": max(0, math.ceil(60 - age))}


class LivePoller:
    def __init__(self, settings: Settings = Settings(), interval: int = 5):
        self.settings, self.interval = settings, interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_poll: datetime | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name="mlb-live-poller", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def status(self) -> dict[str, Any]:
        return {"running": bool(self.thread and self.thread.is_alive()), "last_poll": self.last_poll.isoformat() if self.last_poll else None, "last_error": self.last_error, "interval_seconds": self.interval}

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                with connect(self.settings) as con:
                    slate = fetch_schedule(con, date.today(), date.today(), date.today().year, force=True)
                    for row in slate.to_dict("records"):
                        if any(s in str(row.get("status", "")).lower() for s in LIVE_STATUSES):
                            try:
                                fetch_live_game(con, int(row["game_pk"]), force=True)
                            except Exception as exc:
                                self.last_error = f"{row['game_pk']}: {exc}"
                    self.last_poll = datetime.now(timezone.utc)
            except Exception as exc:
                self.last_error = str(exc)
            self.stop_event.wait(self.interval)


def _pregame_home_probability(con, game_pk: int) -> float:
    row = con.execute(
        """SELECT probability, selection FROM prediction_snapshots WHERE game_pk=? AND market='moneyline'
           ORDER BY created_at DESC LIMIT 1""", [game_pk],
    ).fetchone()
    if not row:
        return .54
    probability, selection = float(row[0]), str(row[1]).lower()
    game = con.execute("SELECT home_team FROM games WHERE game_pk=?", [game_pk]).fetchone()
    return probability if game and normalize_team_name(selection) == game[0] else 1 - probability


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
