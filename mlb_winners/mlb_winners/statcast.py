from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .db import upsert_df
from .team_map import normalize_team_name


def fetch_statcast_range(con, start_date: date, end_date: date, chunk_days: int = 7) -> dict[str, int]:
    """Fetch Statcast pitch-level data and store daily aggregates.

    This uses pybaseball's Baseball Savant wrapper. We aggregate immediately so
    the local DuckDB stays compact enough for quick model iteration.
    """
    try:
        from pybaseball import cache, statcast
    except ImportError as exc:
        raise RuntimeError("Install pybaseball to fetch Statcast data: pip install pybaseball") from exc

    # pybaseball defaults to writing under the user's home directory, which may be
    # read-only or blocked in sandboxed environments. Keep the cache inside the
    # project so repeated runs stay fast and deterministic.
    cache_dir = Path.cwd() / "data" / "pybaseball_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.config.cache_directory = str(cache_dir)
    cache.enable()
    current = start_date
    team_rows = 0
    pitcher_rows = 0
    matchup_rows = pitch_rows = 0
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        raw = statcast(start_dt=current.isoformat(), end_dt=chunk_end.isoformat())
        if raw is not None and not raw.empty:
            team_daily, pitcher_daily = aggregate_statcast(raw)
            player_daily = aggregate_player_batting(raw)
            pitch_mix = aggregate_pitch_mix(raw)
            pitch_matchups = aggregate_pitch_type_matchups(raw)
            pitch_events = normalize_statcast_pitch_events(raw)
            team_rows += upsert_df(con, "statcast_team_daily", team_daily)
            pitcher_rows += upsert_df(con, "statcast_pitcher_daily", pitcher_daily)
            upsert_df(con, "player_statcast_daily", player_daily)
            upsert_df(con, "pitcher_pitch_mix", pitch_mix)
            matchup_rows += upsert_df(con, "pitch_type_matchup_daily", pitch_matchups)
            pitch_rows += upsert_df(con, "statcast_pitch_events", pitch_events)
        current = chunk_end + timedelta(days=1)
    return {
        "team_daily_rows": team_rows,
        "pitcher_daily_rows": pitcher_rows,
        "pitch_matchup_rows": matchup_rows,
        "pitch_event_rows": pitch_rows,
    }


def normalize_statcast_pitch_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one reproducible row per Statcast pitch."""
    required = {"game_pk", "game_date", "at_bat_number", "pitch_number", "batter", "pitcher", "pitch_type"}
    if raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame()
    frame = raw.copy()
    frame = frame.dropna(subset=list(required)).copy()
    if frame.empty:
        return pd.DataFrame()
    def column(name: str, default: Any = None) -> pd.Series:
        return frame[name] if name in frame else pd.Series(default, index=frame.index)
    result = pd.DataFrame({
        "game_pk": pd.to_numeric(column("game_pk"), errors="coerce"),
        "game_date": pd.to_datetime(column("game_date"), errors="coerce").dt.date,
        "at_bat_number": pd.to_numeric(column("at_bat_number"), errors="coerce"),
        "pitch_number": pd.to_numeric(column("pitch_number"), errors="coerce"),
        "batter_id": pd.to_numeric(column("batter"), errors="coerce"),
        "pitcher_id": pd.to_numeric(column("pitcher"), errors="coerce"),
        "pitch_type": column("pitch_type").astype(str),
        "batter_side": column("stand", "U").fillna("U").astype(str),
        "pitcher_hand": column("p_throws", "U").fillna("U").astype(str),
        "balls": pd.to_numeric(column("balls", 0), errors="coerce").fillna(0).clip(0, 3),
        "strikes": pd.to_numeric(column("strikes", 0), errors="coerce").fillna(0).clip(0, 2),
        "description": column("description").fillna("").astype(str),
        "events": column("events").fillna("").astype(str),
        "release_speed": pd.to_numeric(column("release_speed"), errors="coerce"),
        "release_spin_rate": pd.to_numeric(column("release_spin_rate"), errors="coerce"),
        "estimated_woba": pd.to_numeric(column("estimated_woba_using_speedangle"), errors="coerce"),
        "run_value": pd.to_numeric(column("delta_run_exp"), errors="coerce"),
    }).dropna(subset=["game_pk", "game_date", "at_bat_number", "pitch_number", "batter_id", "pitcher_id"])
    integer_columns = ["game_pk", "at_bat_number", "pitch_number", "batter_id", "pitcher_id", "balls", "strikes"]
    result[integer_columns] = result[integer_columns].astype(int)
    return result.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"], keep="last")


def backfill_pitch_matchups(
    con, start_date: date, end_date: date, *, chunk_days: int = 7, force: bool = False,
    fetcher: Callable[..., pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Resumably backfill pitch events and daily matchup cells."""
    if start_date > end_date or chunk_days < 1:
        raise ValueError("Backfill dates and chunk size are invalid.")
    if fetcher is None:
        try:
            from pybaseball import cache, statcast
        except ImportError as exc:
            raise RuntimeError("Install pybaseball to fetch Statcast data") from exc
        cache_dir = Path.cwd() / "data" / "pybaseball_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.config.cache_directory = str(cache_dir)
        cache.enable()
        fetcher = statcast
    covered = set()
    if not force:
        covered = {row[0] for row in con.execute(
            "SELECT game_date FROM statcast_backfill_dates WHERE game_date BETWEEN ? AND ? AND status IN ('complete','no_data')",
            [start_date, end_date],
        ).fetchall()}
    requested = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    missing = [day for day in requested if day not in covered]
    summary: dict[str, Any] = {"requested_dates": len(requested), "skipped_dates": len(covered),
                               "completed_dates": 0, "no_data_dates": 0, "failures": [],
                               "pitch_event_rows": 0, "pitch_matchup_rows": 0}
    index = 0
    while index < len(missing):
        chunk = [missing[index]]
        while index + len(chunk) < len(missing) and len(chunk) < chunk_days and missing[index + len(chunk)] == chunk[-1] + timedelta(days=1):
            chunk.append(missing[index + len(chunk)])
        index += len(chunk)
        try:
            raw = fetcher(start_dt=chunk[0].isoformat(), end_dt=chunk[-1].isoformat())
            raw = raw if raw is not None else pd.DataFrame()
            events = normalize_statcast_pitch_events(raw)
            matchups = aggregate_pitch_type_matchups(raw)
            summary["pitch_event_rows"] += upsert_df(con, "statcast_pitch_events", events)
            summary["pitch_matchup_rows"] += upsert_df(con, "pitch_type_matchup_daily", matchups)
            raw_dates = set(pd.to_datetime(raw["game_date"], errors="coerce").dt.date.dropna()) if "game_date" in raw else set()
            for day in chunk:
                status = "complete" if day in raw_dates else "no_data"
                pitch_n = int((events["game_date"] == day).sum()) if not events.empty else 0
                matchup_n = int((matchups["game_date"] == day).sum()) if not matchups.empty else 0
                con.execute("INSERT OR REPLACE INTO statcast_backfill_dates VALUES (?,?,?,?,?,?)",
                            [day, status, pitch_n, matchup_n, pd.Timestamp.now(tz="UTC"), None])
                summary["completed_dates" if status == "complete" else "no_data_dates"] += 1
            print(f"statcast {chunk[0]}..{chunk[-1]} pitches={len(events)} matchups={len(matchups)}", flush=True)
        except Exception as exc:
            for day in chunk:
                con.execute("INSERT OR REPLACE INTO statcast_backfill_dates VALUES (?,?,?,?,?,?)",
                            [day, "failed", 0, 0, pd.Timestamp.now(tz="UTC"), str(exc)])
            summary["failures"].append({"start": chunk[0].isoformat(), "end": chunk[-1].isoformat(), "error": str(exc)})
            print(f"statcast FAILED {chunk[0]}..{chunk[-1]}: {exc}", flush=True)
    coverage = con.execute(
        "SELECT min(game_date), max(game_date), count(*) FROM statcast_backfill_dates WHERE status='complete'"
    ).fetchone()
    summary.update({"coverage_start": coverage[0].isoformat() if coverage[0] else None,
                    "coverage_end": coverage[1].isoformat() if coverage[1] else None,
                    "covered_game_dates": int(coverage[2])})
    return summary


def aggregate_statcast(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["batting_team"] = np.where(df["inning_topbot"].astype(str).str.lower() == "top", df["away_team"], df["home_team"])
    df["pitching_team"] = np.where(df["inning_topbot"].astype(str).str.lower() == "top", df["home_team"], df["away_team"])
    df["batting_team"] = df["batting_team"].map(normalize_team_name)
    df["pitching_team"] = df["pitching_team"].map(normalize_team_name)
    df["is_bbe"] = df["launch_speed"].notna()
    df["is_hard_hit"] = df["launch_speed"].ge(95).fillna(False)
    # A compact barrel proxy suitable for daily aggregation. Savant's exact
    # barrel definition is more nuanced, but this catches the hard/elevated core.
    df["is_barrel_proxy"] = (df["launch_speed"].ge(98) & df["launch_angle"].between(24, 34)).fillna(False)
    df["is_pa_event"] = df["events"].notna()
    df["is_strikeout"] = df["events"].astype(str).str.contains("strikeout", na=False)
    df["is_walk"] = df["events"].astype(str).isin(["walk", "intent_walk"])

    batting = aggregate_team_batting(df)
    allowed = aggregate_team_allowed(df)
    team_daily = batting.merge(allowed, left_on=["game_date", "team_name"], right_on=["game_date", "team_name"], how="outer")
    team_daily["team_id"] = None
    ordered_cols = [
        "game_date",
        "team_name",
        "team_id",
        "batted_balls",
        "xwoba",
        "xba",
        "hard_hit_rate",
        "barrel_rate",
        "avg_exit_velocity",
        "avg_launch_angle",
        "k_rate",
        "bb_rate",
        "pitches_seen",
        "xwoba_allowed",
        "hard_hit_allowed",
        "barrel_allowed",
        "avg_pitch_velocity",
        "avg_spin_rate",
        "pitches_thrown",
    ]
    team_daily = team_daily.reindex(columns=ordered_cols)

    pitcher_daily = aggregate_pitchers(df)
    return team_daily, pitcher_daily


def aggregate_team_batting(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (game_date, team_name), group in df.groupby(["game_date", "batting_team"], dropna=True):
        bbe = group[group["is_bbe"]]
        pa = group[group["is_pa_event"]]
        rows.append(
            {
                "game_date": game_date,
                "team_name": team_name,
                "batted_balls": int(len(bbe)),
                "xwoba": safe_mean(bbe.get("estimated_woba_using_speedangle")),
                "xba": safe_mean(bbe.get("estimated_ba_using_speedangle")),
                "hard_hit_rate": safe_rate(bbe["is_hard_hit"].sum(), len(bbe)),
                "barrel_rate": safe_rate(bbe["is_barrel_proxy"].sum(), len(bbe)),
                "avg_exit_velocity": safe_mean(bbe.get("launch_speed")),
                "avg_launch_angle": safe_mean(bbe.get("launch_angle")),
                "k_rate": safe_rate(pa["is_strikeout"].sum(), len(pa)),
                "bb_rate": safe_rate(pa["is_walk"].sum(), len(pa)),
                "pitches_seen": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_team_allowed(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (game_date, team_name), group in df.groupby(["game_date", "pitching_team"], dropna=True):
        bbe = group[group["is_bbe"]]
        rows.append(
            {
                "game_date": game_date,
                "team_name": team_name,
                "xwoba_allowed": safe_mean(bbe.get("estimated_woba_using_speedangle")),
                "hard_hit_allowed": safe_rate(bbe["is_hard_hit"].sum(), len(bbe)),
                "barrel_allowed": safe_rate(bbe["is_barrel_proxy"].sum(), len(bbe)),
                "avg_pitch_velocity": safe_mean(group.get("release_speed")),
                "avg_spin_rate": safe_mean(group.get("release_spin_rate")),
                "pitches_thrown": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (game_date, pitcher_id), group in df.groupby(["game_date", "pitcher"], dropna=True):
        bbe = group[group["is_bbe"]]
        pa = group[group["is_pa_event"]]
        rows.append(
            {
                "game_date": game_date,
                "pitcher_id": int(pitcher_id),
                "pitcher_name": first_non_null(group.get("player_name")),
                "team_name": first_non_null(group.get("pitching_team")),
                "batters_faced": int(len(pa)),
                "batted_balls_allowed": int(len(bbe)),
                "xwoba_allowed": safe_mean(bbe.get("estimated_woba_using_speedangle")),
                "hard_hit_allowed": safe_rate(bbe["is_hard_hit"].sum(), len(bbe)),
                "barrel_allowed": safe_rate(bbe["is_barrel_proxy"].sum(), len(bbe)),
                "avg_exit_velocity_allowed": safe_mean(bbe.get("launch_speed")),
                "avg_pitch_velocity": safe_mean(group.get("release_speed")),
                "avg_spin_rate": safe_mean(group.get("release_spin_rate")),
                "k_rate": safe_rate(pa["is_strikeout"].sum(), len(pa)),
                "bb_rate": safe_rate(pa["is_walk"].sum(), len(pa)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_player_batting(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if df.empty or "batter" not in df.columns:
        return pd.DataFrame()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["batting_team"] = np.where(df["inning_topbot"].astype(str).str.lower() == "top", df["away_team"], df["home_team"])
    df["batting_team"] = df["batting_team"].map(normalize_team_name)
    df["is_bbe"] = df["launch_speed"].notna()
    df["is_hard_hit"] = df["launch_speed"].ge(95).fillna(False)
    df["is_barrel_proxy"] = (df["launch_speed"].ge(98) & df["launch_angle"].between(24, 34)).fillna(False)
    df["is_pa_event"] = df["events"].notna()
    df["is_strikeout"] = df["events"].astype(str).str.contains("strikeout", na=False)
    df["is_walk"] = df["events"].astype(str).isin(["walk", "intent_walk"])
    rows = []
    for (game_date, batter_id), group in df.groupby(["game_date", "batter"], dropna=True):
        bbe = group[group["is_bbe"]]
        pa = group[group["is_pa_event"]]
        rows.append(
            {
                "game_date": game_date,
                "player_id": int(batter_id),
                "player_name": first_non_null(group.get("player_name")),
                "team_name": first_non_null(group.get("batting_team")),
                "batted_balls": int(len(bbe)),
                "xwoba": safe_mean(bbe.get("estimated_woba_using_speedangle")),
                "xba": safe_mean(bbe.get("estimated_ba_using_speedangle")),
                "xslg": safe_mean(bbe.get("estimated_slg_using_speedangle")),
                "hard_hit_rate": safe_rate(bbe["is_hard_hit"].sum(), len(bbe)),
                "barrel_rate": safe_rate(bbe["is_barrel_proxy"].sum(), len(bbe)),
                "avg_exit_velocity": safe_mean(bbe.get("launch_speed")),
                "avg_launch_angle": safe_mean(bbe.get("launch_angle")),
                "k_rate": safe_rate(pa["is_strikeout"].sum(), len(pa)),
                "bb_rate": safe_rate(pa["is_walk"].sum(), len(pa)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_pitch_mix(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    total = df.groupby(["game_date", "pitcher"], dropna=True).size().rename("total_pitches")
    rows = []
    for (game_date, pitcher_id, pitch_type), group in df.groupby(["game_date", "pitcher", "pitch_type"], dropna=True):
        total_pitches = int(total.loc[(game_date, pitcher_id)])
        rows.append(
            {
                "game_date": game_date,
                "pitcher_id": int(pitcher_id),
                "pitcher_name": first_non_null(group.get("player_name")),
                "pitch_type": pitch_type,
                "pitches": int(len(group)),
                "avg_velocity": safe_mean(group.get("release_speed")),
                "avg_spin_rate": safe_mean(group.get("release_spin_rate")),
                "usage_rate": safe_rate(len(group), total_pitches),
            }
        )
    return pd.DataFrame(rows)


def aggregate_pitch_type_matchups(raw: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-auditable daily pitch outcome cells from Statcast pitches."""
    if raw.empty or not {"game_date", "batter", "pitcher", "pitch_type"}.issubset(raw.columns):
        return pd.DataFrame()
    df = raw.copy()
    df = df[df["pitch_type"].notna() & df["batter"].notna() & df["pitcher"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["batter_side"] = df.get("stand", pd.Series("U", index=df.index)).fillna("U").astype(str)
    df["pitcher_hand"] = df.get("p_throws", pd.Series("U", index=df.index)).fillna("U").astype(str)
    df["balls"] = pd.to_numeric(df.get("balls", 0), errors="coerce").fillna(0).clip(0, 3).astype(int)
    df["strikes"] = pd.to_numeric(df.get("strikes", 0), errors="coerce").fillna(0).clip(0, 2).astype(int)
    description = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    events = df.get("events", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    df["is_ball"] = description.isin(["ball", "blocked_ball", "pitchout", "intent_ball"])
    df["is_called_strike"] = description.eq("called_strike")
    df["is_swinging_strike"] = description.isin(
        ["swinging_strike", "swinging_strike_blocked", "missed_bunt"]
    )
    df["is_foul"] = description.str.contains("foul", na=False)
    df["is_single"] = events.eq("single")
    df["is_double"] = events.eq("double")
    df["is_triple"] = events.eq("triple")
    df["is_home_run"] = events.eq("home_run")
    df["is_hbp"] = events.eq("hit_by_pitch") | description.eq("hit_by_pitch")
    hit_events = {"single", "double", "triple", "home_run"}
    out_events = {
        "field_out", "force_out", "grounded_into_double_play", "double_play",
        "field_error", "fielders_choice", "fielders_choice_out", "sac_fly",
        "sac_bunt", "triple_play",
    }
    df["is_in_play_out"] = events.isin(out_events)
    df["is_in_play"] = events.isin(hit_events | out_events) | description.eq("hit_into_play")
    estimated = pd.to_numeric(
        df.get("estimated_woba_using_speedangle", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    observed = pd.to_numeric(df.get("woba_value", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["pitch_woba"] = estimated.fillna(observed)
    df["pitch_run_value"] = pd.to_numeric(
        df.get("delta_run_exp", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    group_cols = ["game_date", "batter", "pitcher", "pitch_type", "batter_side", "pitcher_hand", "balls", "strikes"]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        game_date, batter_id, pitcher_id, pitch_type, batter_side, pitcher_hand, balls, strikes = keys
        n = len(group)
        rows.append({
            "game_date": game_date,
            "batter_id": int(batter_id),
            "pitcher_id": int(pitcher_id),
            "pitch_type": str(pitch_type),
            "batter_side": str(batter_side),
            "pitcher_hand": str(pitcher_hand),
            "balls": int(balls),
            "strikes": int(strikes),
            "pitch_count": n,
            "ball_rate": float(group["is_ball"].mean()),
            "called_strike_rate": float(group["is_called_strike"].mean()),
            "swinging_strike_rate": float(group["is_swinging_strike"].mean()),
            "foul_rate": float(group["is_foul"].mean()),
            "in_play_rate": float(group["is_in_play"].mean()),
            "in_play_out_rate": float(group["is_in_play_out"].mean()),
            "single_rate": float(group["is_single"].mean()),
            "double_rate": float(group["is_double"].mean()),
            "triple_rate": float(group["is_triple"].mean()),
            "extra_base_hit_rate": float((group["is_double"] | group["is_triple"] | group["is_home_run"]).mean()),
            "home_run_rate": float(group["is_home_run"].mean()),
            "hbp_rate": float(group["is_hbp"].mean()),
            "expected_woba": safe_mean(group["pitch_woba"]),
            "run_value": safe_mean(group["pitch_run_value"]),
            "avg_velocity": safe_mean(group.get("release_speed")),
            "avg_spin_rate": safe_mean(group.get("release_spin_rate")),
        })
    return pd.DataFrame(rows)


def safe_mean(values: Any) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.mean()) if not series.empty else None


def safe_rate(numerator: Any, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def first_non_null(values: Any) -> Any:
    if values is None:
        return None
    series = pd.Series(values).dropna()
    return series.iloc[0] if not series.empty else None
