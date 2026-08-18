from __future__ import annotations

from datetime import date
import time
from typing import Any

import pandas as pd
import requests

from .db import cache_get, cache_put, upsert_df
from .net import ensure_host_resolves
from .team_map import normalize_team_name


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def get_json(con, source: str, cache_key: str, url: str, params: dict[str, Any], force: bool = False) -> Any:
    cached = cache_get(con, source, cache_key)
    if cached is not None and not force:
        return cached
    try:
        ensure_host_resolves(url)
    except RuntimeError as exc:
        if cached is not None:
            return cached
        raise RuntimeError(f"{exc} (request={cache_key})") from exc
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            cache_put(con, source, cache_key, payload)
            return payload
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    if cached is not None:
        return cached
    raise RuntimeError(f"MLB Stats API request failed after retries for {cache_key}: {last_error}") from last_error


def fetch_schedule(con, start_date: date, end_date: date, season: int, force: bool = False) -> pd.DataFrame:
    try:
        payload = get_json(
            con,
            "mlb_stats_api",
            f"schedule:{start_date}:{end_date}:{season}",
            f"{MLB_API_BASE}/schedule",
            {
                "sportId": 1,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "season": season,
                "gameTypes": "R",
                "hydrate": "probablePitcher,venue",
            },
            force=force,
        )
        df = normalize_schedule(payload, season)
        upsert_df(con, "games", df)
        return df
    except Exception:
        cached = con.execute(
            """
            SELECT *
            FROM games
            WHERE game_date BETWEEN ? AND ?
              AND season = ?
            ORDER BY game_datetime
            """,
            [start_date, end_date, season],
        ).df()
        if not cached.empty:
            return cached
        raise


def fetch_boxscore(con, game_pk: int, *, force: bool = False) -> pd.DataFrame:
    payload = get_json(
        con,
        "mlb_stats_api",
        f"boxscore:{game_pk}",
        f"{MLB_API_BASE}/game/{game_pk}/boxscore",
        {},
        force=force,
    )
    df = normalize_boxscore(game_pk, payload)
    upsert_df(con, "team_game_stats", df)
    players, lineups = normalize_player_boxscore(game_pk, payload)
    upsert_df(con, "player_game_stats", players)
    upsert_df(con, "lineups", lineups)
    return df


def fetch_history(con, start_year: int, end_year: int, include_boxscores: bool = True) -> dict[int, int]:
    counts: dict[int, int] = {}
    for season in range(start_year, end_year + 1):
        schedule = fetch_schedule(
            con,
            date(season, 3, 1),
            date(season, 11, 30),
            season,
        )
        final_games = schedule[schedule["status"].str.lower().str.contains("final", na=False)]
        counts[season] = len(final_games)
        if include_boxscores:
            for game_pk in final_games["game_pk"].dropna().astype(int).tolist():
                fetch_boxscore(con, game_pk)
    return counts


def normalize_schedule(payload: dict[str, Any], season: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_team = home.get("team", {})
            away_team = away.get("team", {})
            home_pp = home.get("probablePitcher") or {}
            away_pp = away.get("probablePitcher") or {}
            status = (game.get("status") or {}).get("detailedState")
            home_score = home.get("score")
            away_score = away.get("score")
            home_won = home.get("isWinner")
            if home_won is None and home_score is not None and away_score is not None:
                home_won = int(home_score) > int(away_score)
            rows.append(
                {
                    "game_pk": game.get("gamePk"),
                    "game_date": game.get("officialDate") or day.get("date"),
                    "game_datetime": game.get("gameDate"),
                    "season": season,
                    "game_type": game.get("gameType"),
                    "status": status,
                    "doubleheader": game.get("doubleHeader"),
                    "game_number": game.get("gameNumber"),
                    "venue_name": (game.get("venue") or {}).get("name"),
                    "day_night": game.get("dayNight"),
                    "home_team_id": home_team.get("id"),
                    "away_team_id": away_team.get("id"),
                    "home_team": normalize_team_name(home_team.get("name")),
                    "away_team": normalize_team_name(away_team.get("name")),
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_probable_pitcher_id": home_pp.get("id"),
                    "away_probable_pitcher_id": away_pp.get("id"),
                    "home_probable_pitcher": home_pp.get("fullName"),
                    "away_probable_pitcher": away_pp.get("fullName"),
                    "home_won": home_won,
                }
            )
    return pd.DataFrame(rows)


def normalize_boxscore(game_pk: int, payload: dict[str, Any]) -> pd.DataFrame:
    teams = payload.get("teams", {})
    rows = []
    for side in ["home", "away"]:
        team_block = teams.get(side, {})
        team = team_block.get("team", {})
        batting = ((team_block.get("teamStats") or {}).get("batting") or {})
        pitching = ((team_block.get("teamStats") or {}).get("pitching") or {})
        starter = find_starting_pitcher(team_block)
        starter_stats = starter.get("stats", {}).get("pitching", {}) if starter else {}
        starter_ip = innings_to_float(starter_stats.get("inningsPitched"))
        total_ip = innings_to_float(pitching.get("inningsPitched"))
        rows.append(
            {
                "game_pk": game_pk,
                "team_id": team.get("id"),
                "team_name": normalize_team_name(team.get("name")),
                "side": side,
                "starter_id": (starter.get("person") or {}).get("id") if starter else None,
                "starter_name": (starter.get("person") or {}).get("fullName") if starter else None,
                "runs": int_or_none(batting.get("runs")),
                "hits": int_or_none(batting.get("hits")),
                "errors": int_or_none(batting.get("errors")),
                "at_bats": int_or_none(batting.get("atBats")),
                "doubles": int_or_none(batting.get("doubles")),
                "triples": int_or_none(batting.get("triples")),
                "home_runs": int_or_none(batting.get("homeRuns")),
                "walks": int_or_none(batting.get("baseOnBalls")),
                "strikeouts": int_or_none(batting.get("strikeOuts")),
                "left_on_base": int_or_none(batting.get("leftOnBase")),
                "starter_ip": starter_ip,
                "starter_er": int_or_none(starter_stats.get("earnedRuns")),
                "starter_so": int_or_none(starter_stats.get("strikeOuts")),
                "starter_bb": int_or_none(starter_stats.get("baseOnBalls")),
                "starter_hits": int_or_none(starter_stats.get("hits")),
                "starter_home_runs": int_or_none(starter_stats.get("homeRuns")),
                "bullpen_ip": max(total_ip - starter_ip, 0.0) if total_ip is not None else None,
                "bullpen_er": max(
                    (int_or_none(pitching.get("earnedRuns")) or 0)
                    - (int_or_none(starter_stats.get("earnedRuns")) or 0),
                    0,
                ),
                "bullpen_so": max(
                    (int_or_none(pitching.get("strikeOuts")) or 0)
                    - (int_or_none(starter_stats.get("strikeOuts")) or 0),
                    0,
                ),
                "bullpen_bb": max(
                    (int_or_none(pitching.get("baseOnBalls")) or 0)
                    - (int_or_none(starter_stats.get("baseOnBalls")) or 0),
                    0,
                ),
            }
        )
    return pd.DataFrame(rows)


def normalize_player_boxscore(game_pk: int, payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = payload.get("teams", {})
    game_date = None
    player_rows: list[dict[str, Any]] = []
    lineup_rows: list[dict[str, Any]] = []
    for side in ["home", "away"]:
        team_block = teams.get(side, {})
        team = team_block.get("team", {})
        team_id = team.get("id")
        team_name = normalize_team_name(team.get("name"))
        for player in (team_block.get("players") or {}).values():
            person = player.get("person") or {}
            player_id = person.get("id")
            if not player_id:
                continue
            batting_order = parse_batting_order(player.get("battingOrder"))
            position = (player.get("position") or {}).get("abbreviation")
            batting = (player.get("stats") or {}).get("batting") or {}
            pitching = (player.get("stats") or {}).get("pitching") or {}
            if batting_order:
                lineup_rows.append(
                    {
                        "game_pk": game_pk,
                        "team_id": team_id,
                        "team_name": team_name,
                        "side": side,
                        "player_id": player_id,
                        "player_name": person.get("fullName"),
                        "batting_order": batting_order,
                        "position": position,
                        "confirmed": True,
                        "source": "mlb_boxscore",
                    }
                )
            if batting or pitching:
                singles = int_or_none(batting.get("hits")) or 0
                doubles = int_or_none(batting.get("doubles")) or 0
                triples = int_or_none(batting.get("triples")) or 0
                homers = int_or_none(batting.get("homeRuns")) or 0
                total_bases = singles + doubles + (2 * triples) + (3 * homers)
                player_rows.append(
                    {
                        "game_pk": game_pk,
                        "game_date": game_date,
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player_id,
                        "player_name": person.get("fullName"),
                        "side": side,
                        "batting_order": batting_order,
                        "position": position,
                        "at_bats": int_or_none(batting.get("atBats")),
                        "hits": int_or_none(batting.get("hits")),
                        "doubles": doubles,
                        "triples": triples,
                        "home_runs": homers,
                        "walks": int_or_none(batting.get("baseOnBalls")),
                        "strikeouts": int_or_none(batting.get("strikeOuts")),
                        "total_bases": total_bases,
                        "rbi": int_or_none(batting.get("rbi")),
                        "runs": int_or_none(batting.get("runs")),
                        "innings_pitched": innings_to_float(pitching.get("inningsPitched")) if pitching else None,
                        "earned_runs": int_or_none(pitching.get("earnedRuns")),
                        "batters_faced": int_or_none(pitching.get("battersFaced")),
                        "pitches_thrown": int_or_none(pitching.get("numberOfPitches")),
                    }
                )
    return pd.DataFrame(player_rows), pd.DataFrame(lineup_rows)


def find_starting_pitcher(team_block: dict[str, Any]) -> dict[str, Any] | None:
    players = team_block.get("players", {})
    for player in players.values():
        if player.get("position", {}).get("abbreviation") != "P":
            continue
        stats = player.get("stats", {}).get("pitching", {})
        if stats.get("gamesStarted") in [1, "1"]:
            return player
    pitchers = team_block.get("pitchers") or []
    if pitchers:
        return players.get(f"ID{pitchers[0]}")
    return None


def innings_to_float(value: Any) -> float:
    if value in [None, ""]:
        return 0.0
    text = str(value)
    if "." not in text:
        return float(text)
    whole, outs = text.split(".", 1)
    return float(whole) + float(outs[:1]) / 3.0


def int_or_none(value: Any) -> int | None:
    if value in [None, ""]:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_batting_order(value: Any) -> int | None:
    if value in [None, ""]:
        return None
    try:
        return int(value) // 100
    except (TypeError, ValueError):
        return None
