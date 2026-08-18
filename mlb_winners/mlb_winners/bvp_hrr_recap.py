from __future__ import annotations

import argparse
import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

from .notifications import send_telegram


BASE_URL = "https://statsapi.mlb.com/api/v1"
LIVE_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


@dataclass(frozen=True)
class Candidate:
    batter_id: int
    batter: str
    team: str
    opponent: str
    pitcher_id: int
    pitcher: str
    game_pk: int
    game: str
    lineup_source: str
    bvp_hits: int
    bvp_at_bats: int
    bvp_avg: str
    bvp_ops: str
    season_avg: str
    season_ops: str
    pitcher_era: str

    @property
    def season_250(self) -> bool:
        return _float_stat(self.season_avg) >= 0.250

    @property
    def pitcher_era_above_450(self) -> bool:
        return _float_stat(self.pitcher_era) > 4.50


def main() -> None:
    args = parse_args()
    load_dotenv()
    target_date = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
    candidates, missing = build_candidates(target_date)
    graded = grade_candidates(candidates)
    message = format_recap(target_date, graded, missing)
    print(message)
    if args.telegram:
        result = send_telegram(message)
        if result.get("queued"):
            print(f"telegram queued: {result.get('result', {}).get('outbox_path')} error={result.get('error')}")
        elif not result.get("ok", True):
            raise SystemExit(f"telegram send failed: {result}")
        else:
            print(f"telegram sent message_id={result.get('result', {}).get('message_id')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recap daily BvP HRR screen hit rate.")
    parser.add_argument("--date", help="Target game date to recap. Defaults to yesterday.")
    parser.add_argument("--telegram", action="store_true", help="Send the recap through Telegram.")
    return parser.parse_args()


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_candidates(target_date: date) -> tuple[list[Candidate], list[str]]:
    games = _schedule(target_date)
    fallback_cache: dict[int, tuple[list[int], dict, int | None, str | None]] = {}
    pitcher_cache: dict[int, tuple[str, str | None, str | None]] = {}
    hitter_cache: dict[int, tuple[str, int | None, int | None, str | None]] = {}
    candidates: list[Candidate] = []
    missing: list[str] = []

    for game in games:
        game_pk = int(game["gamePk"])
        feed = _json(LIVE_FEED_URL.format(game_pk=game_pk))
        away_team = feed["gameData"]["teams"]["away"]
        home_team = feed["gameData"]["teams"]["home"]
        away_name = away_team["name"]
        home_name = home_team["name"]
        box_teams = feed["liveData"]["boxscore"]["teams"]
        away_pitcher = game["teams"]["away"].get("probablePitcher") or {}
        home_pitcher = game["teams"]["home"].get("probablePitcher") or {}

        sides = [
            ("away", away_team, away_name, home_name, home_pitcher),
            ("home", home_team, home_name, away_name, away_pitcher),
        ]
        for side, team_obj, team_name, opponent_name, opposing_pitcher in sides:
            order, players = lineup_from_box(box_teams[side])
            source = "posted"
            if not order:
                team_id = int(team_obj["id"])
                if team_id not in fallback_cache:
                    fallback_cache[team_id] = fallback_lineup(team_id, target_date)
                order, players, _, fallback_used_date = fallback_cache[team_id]
                source = fallback_used_date or "fallback"
            if not order:
                missing.append(f"{team_name}: no lineup found for {target_date} or prior 2 days")
                continue
            if not opposing_pitcher:
                missing.append(f"{team_name}: no opposing probable pitcher listed")
                continue

            pitcher_id = int(opposing_pitcher["id"])
            pitcher_name = opposing_pitcher["fullName"]
            if pitcher_id not in pitcher_cache:
                pitcher_cache[pitcher_id] = pitcher_season(pitcher_id)
            pitcher_era, _, _ = pitcher_cache[pitcher_id]

            for batter_id in order:
                total = vs_player_total(int(batter_id), pitcher_id)
                if not total:
                    continue
                at_bats = int(total.get("atBats") or 0)
                bvp_avg = str(total.get("avg") or ".000")
                if at_bats < 5 or _float_stat(bvp_avg) <= 0.350:
                    continue
                if int(batter_id) not in hitter_cache:
                    hitter_cache[int(batter_id)] = hitter_season(int(batter_id))
                season_avg, _, _, season_ops = hitter_cache[int(batter_id)]
                player = players.get(f"ID{batter_id}", {})
                batter_name = player.get("person", {}).get("fullName", str(batter_id))
                candidates.append(
                    Candidate(
                        batter_id=int(batter_id),
                        batter=batter_name,
                        team=team_name,
                        opponent=opponent_name,
                        pitcher_id=pitcher_id,
                        pitcher=pitcher_name,
                        game_pk=game_pk,
                        game=f"{away_name} at {home_name}",
                        lineup_source=source,
                        bvp_hits=int(total.get("hits") or 0),
                        bvp_at_bats=at_bats,
                        bvp_avg=bvp_avg,
                        bvp_ops=str(total.get("ops") or ""),
                        season_avg=season_avg,
                        season_ops=str(season_ops or ""),
                        pitcher_era=pitcher_era,
                    )
                )

    candidates.sort(key=lambda row: (row.lineup_source != "posted", -_float_stat(row.bvp_avg), -row.bvp_at_bats, row.batter))
    return candidates, missing


def grade_candidates(candidates: list[Candidate]) -> list[tuple[Candidate, dict, bool]]:
    game_cache: dict[int, dict] = {}
    graded: list[tuple[Candidate, dict, bool]] = []
    for candidate in candidates:
        if candidate.game_pk not in game_cache:
            game_cache[candidate.game_pk] = _json(LIVE_FEED_URL.format(game_pk=candidate.game_pk))
        feed = game_cache[candidate.game_pk]
        stats = batter_game_stats(feed, candidate.batter_id)
        hrr = int(stats.get("hits") or 0) + int(stats.get("runs") or 0) + int(stats.get("rbi") or 0)
        graded.append((candidate, stats, hrr >= 2))
    return graded


def format_recap(target_date: date, graded: list[tuple[Candidate, dict, bool]], missing: list[str]) -> str:
    all_total = len(graded)
    all_hits = sum(1 for _, _, hit in graded if hit)
    season_rows = [row for row in graded if row[0].season_250]
    season_total = len(season_rows)
    season_hits = sum(1 for _, _, hit in season_rows if hit)
    high_era_rows = [row for row in graded if row[0].pitcher_era_above_450]
    high_era_total = len(high_era_rows)
    high_era_hits = sum(1 for _, _, hit in high_era_rows if hit)

    lines = [f"MLB BvP HRR recap for {target_date.isoformat()}"]
    lines.append(f"All qualifiers: {all_hits}/{all_total} hit 1.5 H+R+RBI ({_pct(all_hits, all_total)})")
    lines.append(f".250+ season BA subset: {season_hits}/{season_total} hit 1.5 H+R+RBI ({_pct(season_hits, season_total)})")
    lines.append(f"Opposing ERA > 4.50 subset: {high_era_hits}/{high_era_total} hit 1.5 H+R+RBI ({_pct(high_era_hits, high_era_total)})")

    if graded:
        lines.append("")
        lines.append("All qualifiers:")
        for candidate, stats, hit in graded:
            lines.append(_format_result_line(candidate, stats, hit))

    if season_rows:
        lines.append("")
        lines.append(".250+ subset:")
        for candidate, stats, hit in season_rows:
            lines.append(_format_result_line(candidate, stats, hit))

    if high_era_rows:
        lines.append("")
        lines.append("Opposing ERA > 4.50 subset:")
        for candidate, stats, hit in high_era_rows:
            lines.append(_format_result_line(candidate, stats, hit))

    if missing:
        lines.append("")
        lines.append("Notes:")
        for note in missing[:8]:
            lines.append(f"- {note}")
        if len(missing) > 8:
            lines.append(f"- {len(missing) - 8} more missing-data notes")
    return "\n".join(lines)


def _format_result_line(candidate: Candidate, stats: dict, hit: bool) -> str:
    hits = int(stats.get("hits") or 0)
    runs = int(stats.get("runs") or 0)
    rbi = int(stats.get("rbi") or 0)
    mark = "HIT" if hit else "MISS"
    return (
        f"- {mark}: {candidate.batter} ({candidate.lineup_source}) vs {candidate.pitcher} "
        f"ERA {candidate.pitcher_era}; BvP {candidate.bvp_hits}-for-{candidate.bvp_at_bats} "
        f"{candidate.bvp_avg}; 2026 BA {candidate.season_avg}; game H/R/RBI {hits}/{runs}/{rbi}"
    )


def batter_game_stats(feed: dict, batter_id: int) -> dict:
    for side in ("away", "home"):
        player = feed["liveData"]["boxscore"]["teams"][side].get("players", {}).get(f"ID{batter_id}")
        if player:
            return player.get("stats", {}).get("batting", {}) or {}
    return {}


def lineup_from_box(box: dict) -> tuple[list[int], dict]:
    players = box.get("players") or {}
    ids: list[int] = []
    for batter_id in box.get("battingOrder") or []:
        if int(batter_id) not in ids:
            ids.append(int(batter_id))

    extras: list[tuple[int, int]] = []
    for player in players.values():
        person = player.get("person") or {}
        player_id = person.get("id")
        batting_order = player.get("battingOrder")
        if not player_id or not batting_order:
            continue
        if player.get("gameStatus", {}).get("isSubstitute"):
            continue
        if int(player_id) in ids:
            continue
        try:
            order_num = int(batting_order)
        except ValueError:
            order_num = 9999
        extras.append((order_num, int(player_id)))

    for _, player_id in sorted(extras):
        if player_id not in ids:
            ids.append(player_id)
    return ids, players


def fallback_lineup(team_id: int, target_date: date, lookback_days: int = 2) -> tuple[list[int], dict, int | None, str | None]:
    for days_back in range(1, lookback_days + 1):
        fallback_date = target_date - timedelta(days=days_back)
        games = _schedule(fallback_date, team_id=team_id)
        games.sort(key=lambda game: game.get("gameDate", ""), reverse=True)
        for game in games:
            game_pk = int(game["gamePk"])
            feed = _json(LIVE_FEED_URL.format(game_pk=game_pk))
            side = "away" if int(feed["gameData"]["teams"]["away"]["id"]) == team_id else "home"
            order, players = lineup_from_box(feed["liveData"]["boxscore"]["teams"][side])
            if order:
                official_date = feed["gameData"]["datetime"].get("officialDate") or fallback_date.isoformat()
                return order, players, game_pk, official_date
    return [], {}, None, None


def vs_player_total(batter_id: int, pitcher_id: int) -> dict | None:
    payload = _json(
        f"{BASE_URL}/people/{batter_id}/stats?"
        + urlencode({"stats": "vsPlayer", "group": "hitting", "opposingPlayerId": pitcher_id})
    )
    for block in payload.get("stats", []):
        if block.get("type", {}).get("displayName") == "vsPlayerTotal" and block.get("splits"):
            return block["splits"][0]["stat"]
    return None


def hitter_season(batter_id: int) -> tuple[str, int | None, int | None, str | None]:
    payload = _json(
        f"{BASE_URL}/people/{batter_id}/stats?"
        + urlencode({"stats": "season", "group": "hitting", "season": "2026", "gameType": "R"})
    )
    for block in payload.get("stats", []):
        for split in block.get("splits", []):
            stat = split.get("stat", {})
            return str(stat.get("avg") or ".000"), stat.get("hits"), stat.get("atBats"), stat.get("ops")
    return ".000", None, None, None


def pitcher_season(pitcher_id: int) -> tuple[str, str | None, str | None]:
    payload = _json(
        f"{BASE_URL}/people/{pitcher_id}/stats?"
        + urlencode({"stats": "season", "group": "pitching", "season": "2026", "gameType": "R"})
    )
    for block in payload.get("stats", []):
        for split in block.get("splits", []):
            stat = split.get("stat", {})
            return str(stat.get("era") or ""), stat.get("inningsPitched"), stat.get("whip")
    return "", None, None


def _schedule(target_date: date, team_id: int | None = None) -> list[dict]:
    params: dict[str, str | int] = {"sportId": 1, "date": target_date.isoformat(), "hydrate": "probablePitcher,team"}
    if team_id is not None:
        params["teamId"] = team_id
    payload = _json(f"{BASE_URL}/schedule?" + urlencode(params))
    games: list[dict] = []
    for day in payload.get("dates", []):
        games.extend(day.get("games", []))
    return games


def _json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=25) as response:
        return json.load(response)


def _float_stat(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(count / total) * 100:.1f}%"


if __name__ == "__main__":
    main()
