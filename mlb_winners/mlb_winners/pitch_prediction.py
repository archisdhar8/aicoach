from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd


PITCH_NAMES = {
    "FF": "Four-seam", "SI": "Sinker", "SL": "Slider", "CH": "Changeup",
    "ST": "Sweeper", "FC": "Cutter", "CU": "Curveball", "KC": "Knuckle curve",
    "FS": "Splitter", "SV": "Slurve", "KN": "Knuckleball", "EP": "Eephus",
}
COMMON_PITCH_TYPES = tuple(PITCH_NAMES)
UNSEEN_MASS = 0.005


@dataclass(frozen=True)
class PitchEvidence:
    league_context: dict[str, float]
    pitcher_overall: dict[str, float]
    pitcher_count: dict[str, float]
    exact_matchup: dict[str, float]
    same_game: dict[str, float]
    modifiers: dict[str, float]
    modifier_reasons: dict[str, list[str]]


def hierarchical_dirichlet_posterior(
    evidence: PitchEvidence,
    *,
    league_strength: float = 20.0,
    count_prior_strength: float = 12.0,
    matchup_prior_strength: float = 40.0,
    same_game_prior_strength: float = 36.0,
    unseen_mass: float = UNSEEN_MASS,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Nested Dirichlet updates from broad context to exact live evidence.

    The pitcher's overall arsenal is the operational center: league context only
    regularizes the arsenal, then count, exact matchup and same-game evidence
    update that distribution in order.
    """
    all_types = sorted(set(COMMON_PITCH_TYPES) | _keys(evidence))
    plausible, excluded = _plausible_arsenal(all_types, evidence)
    base = _smoothed_distribution(evidence.league_context, plausible)
    overall = _dirichlet_update(base, evidence.pitcher_overall, league_strength, plausible)
    count = _dirichlet_update(overall, evidence.pitcher_count, count_prior_strength, plausible)
    matchup = _dirichlet_update(count, evidence.exact_matchup, matchup_prior_strength, plausible)
    live = _dirichlet_update(matchup, evidence.same_game, same_game_prior_strength, plausible, evidence_weight=1.25)
    adjusted = {
        pitch_type: live[pitch_type] * max(float(evidence.modifiers.get(pitch_type, 1.0)), .05)
        for pitch_type in plausible
    }
    adjusted = _normalize(adjusted)
    if excluded and unseen_mass > 0:
        adjusted = {key: value * (1.0 - unseen_mass) for key, value in adjusted.items()}
        league_excluded = _smoothed_distribution(evidence.league_context, excluded)
        adjusted.update({key: unseen_mass * value for key, value in league_excluded.items()})
    adjusted = _normalize(adjusted)
    stages = {"overall_arsenal": overall, "count": count, "matchup": matchup, "same_game": live, "final": adjusted}
    diagnostics = {
        "formula": "Dirichlet(league context → pitcher arsenal → count → matchup → same game) × live modifiers",
        "prior_strengths": {
            "league_to_arsenal": league_strength,
            "arsenal_to_count": count_prior_strength,
            "count_to_matchup": matchup_prior_strength,
            "matchup_to_same_game": same_game_prior_strength,
            "same_game_weight": 1.25,
            "unseen_mass": unseen_mass,
        },
        "sample_sizes": {
            "league_context": _total(evidence.league_context),
            "pitcher_overall": _total(evidence.pitcher_overall),
            "pitcher_count": _total(evidence.pitcher_count),
            "exact_matchup": _total(evidence.exact_matchup),
            "same_game": _total(evidence.same_game),
        },
        "plausible_pitch_types": plausible,
        "excluded_pitch_types": excluded,
        "stages": stages,
        "stage_l1_shifts": {
            "count": _l1(overall, count),
            "matchup": _l1(count, matchup),
            "same_game": _l1(matchup, live),
            "modifiers": _l1(live, adjusted),
        },
        "modifier_reasons": evidence.modifier_reasons,
        "most_specific_evidence": (
            "same_game" if _total(evidence.same_game) else
            "exact_matchup" if _total(evidence.exact_matchup) else
            "pitcher_count" if _total(evidence.pitcher_count) else
            "pitcher_overall" if _total(evidence.pitcher_overall) else
            "league_context"
        ),
    }
    return adjusted, diagnostics


def load_live_pitch_evidence(con, state: Any) -> tuple[PitchEvidence, dict[str, dict[str, float | None]]]:
    cutoff = state.game_date
    since = cutoff - timedelta(days=365)
    league = _query_counts(
        con,
        """SELECT pitch_type, sum(pitch_count) n FROM pitch_type_matchup_daily
           WHERE game_date<? AND game_date>=? AND batter_side=? AND pitcher_hand=?
             AND balls=? AND strikes=? GROUP BY pitch_type""",
        [cutoff, since, state.batter_side, state.pitcher_hand, state.balls, state.strikes],
    )
    overall = _query_counts(
        con,
        """SELECT pitch_type, sum(pitch_count) n FROM pitch_type_matchup_daily
           WHERE game_date<? AND game_date>=? AND pitcher_id=? GROUP BY pitch_type""",
        [cutoff, since, state.pitcher_id],
    ) if state.pitcher_id else {}
    if state.pitcher_id and not overall:
        overall = _query_counts(
            con,
            """SELECT pitch_type, sum(pitches) n FROM pitcher_pitch_mix
               WHERE game_date<? AND game_date>=? AND pitcher_id=? GROUP BY pitch_type""",
            [cutoff, since, state.pitcher_id],
        )
    count = _query_counts(
        con,
        """SELECT pitch_type, sum(pitch_count) n FROM pitch_type_matchup_daily
           WHERE game_date<? AND game_date>=? AND pitcher_id=? AND balls=? AND strikes=?
             AND batter_side=? GROUP BY pitch_type""",
        [cutoff, since, state.pitcher_id, state.balls, state.strikes, state.batter_side],
    ) if state.pitcher_id else {}
    matchup = _query_counts(
        con,
        """SELECT pitch_type, sum(pitch_count) n FROM pitch_type_matchup_daily
           WHERE game_date<? AND game_date>=? AND pitcher_id=? AND batter_id=?
             AND balls=? AND strikes=? GROUP BY pitch_type""",
        [cutoff, since, state.pitcher_id, state.batter_id, state.balls, state.strikes],
    ) if state.pitcher_id and state.batter_id else {}
    live_frame = con.execute(
        """SELECT pitch_type, count(*) n, avg(start_speed) velocity, avg(spin_rate) spin,
                  sum(CASE WHEN lower(event_description) LIKE '%swinging strike%' THEN 1 ELSE 0 END) whiffs,
                  sum(CASE WHEN is_in_play THEN 1 ELSE 0 END) in_play,
                  sum(CASE WHEN lower(event_description) LIKE '%run(s)%' THEN 1 ELSE 0 END) damage
           FROM live_play_events WHERE game_pk=? AND pitcher_id=? AND is_pitch
             AND pitch_type IS NOT NULL GROUP BY pitch_type""",
        [state.game_pk, state.pitcher_id],
    ).df() if state.pitcher_id else pd.DataFrame()
    same_game = {str(row.pitch_type): float(row.n) for row in live_frame.itertuples()}
    expected_frame = con.execute(
        """SELECT pitch_type, sum(pitches) pitches,
                  sum(avg_velocity*pitches)/nullif(sum(pitches),0) velocity,
                  sum(avg_spin_rate*pitches)/nullif(sum(pitches),0) spin
           FROM pitcher_pitch_mix WHERE pitcher_id=? AND game_date<? AND game_date>=?
           GROUP BY pitch_type""", [state.pitcher_id, cutoff, cutoff - timedelta(days=60)],
    ).df() if state.pitcher_id else pd.DataFrame()
    expected = {
        str(row.pitch_type): {"pitches": float(row.pitches), "velocity": _float(row.velocity), "spin": _float(row.spin)}
        for row in expected_frame.itertuples()
    }
    live_stats = {
        str(row.pitch_type): {
            "pitches": float(row.n), "velocity": _float(row.velocity), "spin": _float(row.spin),
            "whiffs": float(row.whiffs or 0), "in_play": float(row.in_play or 0), "damage": float(row.damage or 0),
        }
        for row in live_frame.itertuples()
    }
    modifiers, reasons = live_modifiers(
        set(COMMON_PITCH_TYPES) | set(overall) | set(same_game),
        pitcher_pitch_count=int(getattr(state, "pitcher_pitch_count", 0) or 0),
        times_through_order=int(getattr(state, "times_through_order", 1) or 1),
        expected=expected,
        live=live_stats,
    )
    return PitchEvidence(league, overall, count, matchup, same_game, modifiers, reasons), {"expected": expected, "live": live_stats}


def live_modifiers(
    pitch_types: Iterable[str], *, pitcher_pitch_count: int, times_through_order: int,
    expected: dict[str, dict[str, float | None]], live: dict[str, dict[str, float | None]],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    modifiers: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for pitch_type in pitch_types:
        factor, notes = 1.0, []
        if times_through_order >= 3 and pitch_type in {"SL", "CH", "ST", "CU", "FS"}:
            factor *= 1.04
            notes.append("third-time-through secondary-pitch adjustment +4%")
        if pitcher_pitch_count >= 80:
            change = .94 if pitch_type in {"FF", "SI"} else 1.04
            factor *= change
            notes.append(f"workload adjustment {change - 1:+.0%}")
        expected_v = _float((expected.get(pitch_type) or {}).get("velocity"))
        live_v = _float((live.get(pitch_type) or {}).get("velocity"))
        if expected_v is not None and live_v is not None and live_v - expected_v <= -1.5:
            change = .90 if pitch_type in {"FF", "SI", "FC"} else .97
            factor *= change
            notes.append(f"velocity loss {live_v - expected_v:+.1f} mph")
        game_n = float((live.get(pitch_type) or {}).get("pitches") or 0)
        if game_n >= 4:
            whiff_rate = float((live.get(pitch_type) or {}).get("whiffs") or 0) / game_n
            damage_rate = float((live.get(pitch_type) or {}).get("damage") or 0) / game_n
            whiff_factor = 1.0 + min(whiff_rate, .30) * .18
            damage_factor = 1.0 - min(damage_rate, .20) * .30
            factor *= whiff_factor * damage_factor
            if whiff_rate:
                notes.append(f"same-game whiffs {whiff_rate:.0%}")
            if damage_rate:
                notes.append(f"damaging contact {damage_rate:.0%}")
        modifiers[pitch_type] = min(1.25, max(.75, factor))
        reasons[pitch_type] = notes
    return modifiers, reasons


def evaluate_next_pitch_model(
    con, start_date: date, end_date: date, *, max_pitches: int | None = None,
    posterior_kwargs: dict[str, float] | None = None,
) -> dict[str, Any]:
    posterior_kwargs = posterior_kwargs or {}
    frame = con.execute(
        """SELECT game_pk, game_date, at_bat_number, pitch_number, batter_id, pitcher_id,
                  pitch_type, batter_side, pitcher_hand, balls, strikes
           FROM statcast_pitch_events WHERE game_date<=? AND pitch_type IS NOT NULL
           ORDER BY game_date, game_pk, at_bat_number, pitch_number""", [end_date],
    ).df()
    if frame.empty:
        raise RuntimeError("No pitch-level Statcast rows are available for evaluation.")
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    histories = _History()
    training = frame[frame["game_date"] < start_date]
    histories.add_rows(training.to_dict("records"))
    evaluation = frame[(frame["game_date"] >= start_date) & (frame["game_date"] <= end_date)]
    if max_pitches:
        evaluation = evaluation.head(max_pitches)
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for game_date, day in evaluation.groupby("game_date", sort=True):
        same_game: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for row in day.to_dict("records"):
            evidence = histories.evidence(row, dict(same_game[(int(row["game_pk"]), int(row["pitcher_id"]))]))
            model, diagnostic = hierarchical_dirichlet_posterior(evidence, **posterior_kwargs)
            classes = sorted(set(COMMON_PITCH_TYPES) | set(model) | set(evidence.league_context) | set(evidence.pitcher_overall))
            league = _smoothed_distribution(evidence.league_context, classes)
            overall = _smoothed_distribution(evidence.pitcher_overall, classes) if evidence.pitcher_overall else league
            count = _smoothed_distribution(evidence.pitcher_count, classes) if evidence.pitcher_count else overall
            most = _one_hot_baseline(evidence.pitcher_overall or evidence.league_context, set(model))
            actual = str(row["pitch_type"])
            sample_n = int(diagnostic["sample_sizes"]["pitcher_count"])
            record = {
                "game_date": game_date, "game_pk": int(row["game_pk"]), "actual": actual,
                "balls": int(row["balls"]), "strikes": int(row["strikes"]),
                "sample_size": sample_n, "sample_tier": _sample_tier(sample_n),
            }
            for name, probability in (("model", model), ("most_thrown", most), ("overall_arsenal", overall), ("count_specific", count)):
                predictions[name].append({"actual": actual, "probabilities": probability})
            records.append(record)
            same_game[(int(row["game_pk"]), int(row["pitcher_id"]))][actual] += 1
        # The whole target date remains excluded until all predictions for it are complete.
        histories.add_rows(day.to_dict("records"))
    metrics = {name: classification_metrics(items) for name, items in predictions.items()}
    report = {
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "pitches": len(records), "metrics": metrics,
        "posterior_parameters": diagnostic["prior_strengths"] if records else {},
        "beats_overall_arsenal_log_loss": metrics["model"]["log_loss"] < metrics["overall_arsenal"]["log_loss"],
        "calibration": calibration_table(predictions["model"]),
        "by_count": _segmented_metrics(records, predictions["model"], lambda r: f"{r['balls']}-{r['strikes']}"),
        "by_sample_tier": _segmented_metrics(records, predictions["model"], lambda r: r["sample_tier"]),
        "leakage_policy": "history game_date must be strictly earlier than target game_date; same-game pitches are added only after prediction",
    }
    evaluation_id = hashlib.sha1(
        f"{start_date}:{end_date}:{len(records)}:{json.dumps(posterior_kwargs, sort_keys=True)}".encode()
    ).hexdigest()
    con.execute(
        """INSERT OR REPLACE INTO pitch_prediction_evaluations
           (evaluation_id,start_date,end_date,pitches,report_json) VALUES (?,?,?,?,?)""",
        [evaluation_id, start_date, end_date, len(records), json.dumps(report)],
    )
    return report


def classification_metrics(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {"top1_accuracy": 0.0, "log_loss": 0.0, "brier_score": 0.0}
    top1 = log_loss = brier = 0.0
    classes = set(COMMON_PITCH_TYPES)
    for item in items:
        actual, probabilities = item["actual"], _normalize_with_floor(item["probabilities"], classes | {item["actual"]})
        top1 += max(probabilities, key=probabilities.get) == actual
        log_loss -= np.log(max(probabilities.get(actual, 1e-9), 1e-9))
        brier += sum((probabilities.get(key, 0.0) - float(key == actual)) ** 2 for key in probabilities)
    n = len(items)
    return {"top1_accuracy": float(top1 / n), "log_loss": float(log_loss / n), "brier_score": float(brier / n)}


def calibration_table(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for item in items:
        probabilities = _normalize_with_floor(item["probabilities"], set(COMMON_PITCH_TYPES) | {item["actual"]})
        for pitch_type, probability in probabilities.items():
            lower = min(int(probability * 10), 9) / 10
            buckets[f"{lower:.1f}-{lower + .1:.1f}"].append((probability, float(pitch_type == item["actual"])))
    return [
        {"bucket": key, "predicted": float(np.mean([x[0] for x in values])),
         "observed": float(np.mean([x[1] for x in values])), "n": len(values)}
        for key, values in sorted(buckets.items())
    ]


def diagnose_pitcher_counts(con, pitcher_id: int, as_of: date, batter_side: str = "R") -> dict[str, Any]:
    hand_row = con.execute(
        """SELECT pitcher_hand FROM pitch_type_matchup_daily
           WHERE pitcher_id=? AND game_date<? GROUP BY pitcher_hand ORDER BY sum(pitch_count) DESC LIMIT 1""",
        [pitcher_id, as_of],
    ).fetchone()
    pitcher_hand = str(hand_row[0]) if hand_row else "U"
    rows = []
    for balls, strikes in ((0, 0), (0, 2), (3, 0), (3, 2)):
        state = _DiagnosticState(pitcher_id, as_of, batter_side, pitcher_hand, balls, strikes)
        evidence, _ = load_live_pitch_evidence(con, state)
        probability, diagnostic = hierarchical_dirichlet_posterior(evidence)
        rows.append({
            "count": f"{balls}-{strikes}", "probabilities": probability,
            "top_three": sorted(probability.items(), key=lambda item: item[1], reverse=True)[:3],
            "evidence": diagnostic,
        })
    differences = [
        _l1(rows[i]["probabilities"], rows[j]["probabilities"])
        for i in range(len(rows)) for j in range(i + 1, len(rows))
    ]
    return {
        "pitcher_id": pitcher_id, "as_of": as_of.isoformat(), "batter_side": batter_side, "pitcher_hand": pitcher_hand,
        "counts": rows, "max_pairwise_l1": max(differences) if differences else 0.0,
        "materially_different": bool(differences and max(differences) >= .10),
    }


class _DiagnosticState:
    def __init__(self, pitcher_id: int, game_date: date, batter_side: str, pitcher_hand: str, balls: int, strikes: int):
        self.pitcher_id, self.game_date, self.batter_side = pitcher_id, game_date, batter_side
        self.pitcher_hand, self.balls, self.strikes = pitcher_hand, balls, strikes
        self.batter_id, self.game_pk = None, -1
        self.pitcher_pitch_count, self.times_through_order = 0, 1


class _History:
    def __init__(self):
        self.league = defaultdict(Counter)
        self.overall = defaultdict(Counter)
        self.count = defaultdict(Counter)
        self.matchup = defaultdict(Counter)

    def add_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            pitch = str(row["pitch_type"])
            side, hand = str(row.get("batter_side") or "U"), str(row.get("pitcher_hand") or "U")
            balls, strikes = int(row.get("balls") or 0), int(row.get("strikes") or 0)
            pitcher, batter = int(row["pitcher_id"]), int(row["batter_id"])
            self.league[(side, hand, balls, strikes)][pitch] += 1
            self.overall[pitcher][pitch] += 1
            self.count[(pitcher, side, balls, strikes)][pitch] += 1
            self.matchup[(batter, pitcher, balls, strikes)][pitch] += 1

    def evidence(self, row: dict[str, Any], same_game: dict[str, float]) -> PitchEvidence:
        side, hand = str(row.get("batter_side") or "U"), str(row.get("pitcher_hand") or "U")
        balls, strikes = int(row.get("balls") or 0), int(row.get("strikes") or 0)
        pitcher, batter = int(row["pitcher_id"]), int(row["batter_id"])
        return PitchEvidence(
            dict(self.league[(side, hand, balls, strikes)]), dict(self.overall[pitcher]),
            dict(self.count[(pitcher, side, balls, strikes)]), dict(self.matchup[(batter, pitcher, balls, strikes)]),
            dict(same_game), {}, {},
        )


def _query_counts(con, sql: str, params: list[Any]) -> dict[str, float]:
    return {str(row[0]): float(row[1]) for row in con.execute(sql, params).fetchall() if row[0] and row[1]}


def _plausible_arsenal(all_types: list[str], evidence: PitchEvidence) -> tuple[list[str], list[str]]:
    total = _total(evidence.pitcher_overall)
    plausible = {
        key for key, value in evidence.pitcher_overall.items()
        if value >= max(3.0, total * .005)
    }
    # A pitch observed in the current game is definitive arsenal evidence. Sparse
    # historical count/matchup cells alone do not resurrect retired or miscoded pitches.
    plausible |= {key for key, value in evidence.same_game.items() if value > 0}
    if not plausible:
        ranked = sorted(evidence.league_context.items(), key=lambda item: item[1], reverse=True)
        cumulative, league_total = 0.0, _total(evidence.league_context) or 1.0
        for key, value in ranked:
            plausible.add(key)
            cumulative += value / league_total
            if cumulative >= .95 and len(plausible) >= 3:
                break
    if not plausible:
        plausible = {"FF", "SI", "SL", "CH", "CU"}
    plausible_list = sorted(plausible)
    return plausible_list, sorted(set(all_types) - plausible)


def _smoothed_distribution(counts: dict[str, float], keys: list[str]) -> dict[str, float]:
    if not keys:
        return {}
    return _normalize({key: float(counts.get(key, 0.0)) + .5 for key in keys})


def _dirichlet_update(prior: dict[str, float], counts: dict[str, float], strength: float, keys: list[str], evidence_weight: float = 1.0) -> dict[str, float]:
    return _normalize({key: strength * prior.get(key, 0.0) + evidence_weight * float(counts.get(key, 0.0)) for key in keys})


def _one_hot_baseline(counts: dict[str, float], classes: set[str]) -> dict[str, float]:
    if not counts:
        return _smoothed_distribution({}, sorted(classes))
    top = max(counts, key=counts.get)
    epsilon = 1e-4
    values = {key: epsilon for key in classes}
    values[top] = 1.0
    return _normalize(values)


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(value), 0.0) for value in values.values()) or 1.0
    return {key: max(float(value), 0.0) / total for key, value in values.items()}


def _normalize_with_floor(values: dict[str, float], classes: set[str], floor: float = 1e-6) -> dict[str, float]:
    return _normalize({key: max(float(values.get(key, 0.0)), floor) for key in classes})


def _keys(evidence: PitchEvidence) -> set[str]:
    return set().union(evidence.league_context, evidence.pitcher_overall, evidence.pitcher_count, evidence.exact_matchup, evidence.same_game)


def _total(values: dict[str, float]) -> float:
    return float(sum(values.values()))


def _l1(left: dict[str, float], right: dict[str, float]) -> float:
    return float(sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in set(left) | set(right)))


def _float(value: Any) -> float | None:
    try:
        return None if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _sample_tier(n: int) -> str:
    if n < 10:
        return "0-9"
    if n < 30:
        return "10-29"
    if n < 75:
        return "30-74"
    return "75+"


def _segmented_metrics(records: list[dict[str, Any]], predictions: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record, prediction in zip(records, predictions, strict=True):
        groups[key_fn(record)].append(prediction)
    return {key: {"n": len(items), **classification_metrics(items)} for key, items in sorted(groups.items())}
