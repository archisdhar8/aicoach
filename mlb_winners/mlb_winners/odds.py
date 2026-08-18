from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any

import pandas as pd
import requests

from .db import cache_get, cache_put, upsert_df
from .net import ensure_host_resolves
from .team_map import normalize_team_name


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"


@dataclass(frozen=True)
class OddsFetchResult:
    rows: int
    remaining: str | None
    used: str | None
    last: str | None
    from_cache: bool


def moneyline_to_implied_prob(moneyline: int | float) -> float:
    line = float(moneyline)
    if line < 0:
        return abs(line) / (abs(line) + 100.0)
    return 100.0 / (line + 100.0)


def implied_prob_to_moneyline(probability: float) -> int:
    if probability <= 0 or probability >= 1:
        raise ValueError("probability must be between 0 and 1")
    if probability >= 0.5:
        return round(-100 * probability / (1 - probability))
    return round(100 * (1 - probability) / probability)


def american_profit_per_dollar(moneyline: int | float) -> float:
    line = float(moneyline)
    return 100.0 / abs(line) if line < 0 else line / 100.0


def expected_value_per_dollar(win_probability: float, moneyline: int | float) -> float:
    profit = american_profit_per_dollar(moneyline)
    return win_probability * profit - (1.0 - win_probability)


def devig_two_way(home_ml: int | float, away_ml: int | float) -> tuple[float, float]:
    home = moneyline_to_implied_prob(home_ml)
    away = moneyline_to_implied_prob(away_ml)
    total = home + away
    if total <= 0:
        raise ValueError("invalid market probabilities")
    return home / total, away / total


def classify_edge(edge: float, ev: float) -> str:
    if ev <= 0 or edge <= 0:
        return "no bet"
    if edge >= 0.07 and ev >= 0.08:
        return "strong"
    if edge >= 0.05 and ev >= 0.04:
        return "medium"
    if edge >= 0.03 and ev > 0:
        return "thin"
    return "no bet"


def fetch_current_odds(
    con,
    snapshot_date: date,
    api_key: str | None = None,
    force: bool = False,
    bookmaker: str | None = None,
    markets: str = "h2h",
) -> OddsFetchResult:
    cache_key = f"{SPORT_KEY}:{snapshot_date.isoformat()}:{markets}:us"
    cached = cache_get(con, "the_odds_api", cache_key)
    if cached and not force:
        rows = normalize_odds_payload(
            cached["payload"],
            snapshot_date,
            bookmaker,
            fetched_at=cached.get("fetched_at") or f"{snapshot_date.isoformat()}T00:00:00Z",
        )
        upsert_df(con, "odds_snapshots", rows)
        headers = cached.get("headers", {})
        return OddsFetchResult(
            rows=len(rows),
            remaining=headers.get("x-requests-remaining"),
            used=headers.get("x-requests-used"),
            last=headers.get("x-requests-last"),
            from_cache=True,
        )

    api_key = api_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set ODDS_API_KEY or pass --api-key to fetch odds.")

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    try:
        ensure_host_resolves(ODDS_API_BASE)
        response = requests.get(f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        headers = {
            "x-requests-remaining": response.headers.get("x-requests-remaining"),
            "x-requests-used": response.headers.get("x-requests-used"),
            "x-requests-last": response.headers.get("x-requests-last"),
        }
        cache_put(con, "the_odds_api", cache_key, {"payload": payload, "headers": headers, "fetched_at": fetched_at})
        rows = normalize_odds_payload(payload, snapshot_date, bookmaker, fetched_at=fetched_at)
        upsert_df(con, "odds_snapshots", rows)
        return OddsFetchResult(
            rows=len(rows),
            remaining=headers["x-requests-remaining"],
            used=headers["x-requests-used"],
            last=headers["x-requests-last"],
            from_cache=False,
        )
    except (RuntimeError, requests.RequestException) as exc:
        # In restricted/offline environments, prefer returning cached odds if available
        # (even when force=True), rather than failing the entire workflow.
        if cached:
            rows = normalize_odds_payload(
                cached["payload"],
                snapshot_date,
                bookmaker,
                fetched_at=cached.get("fetched_at") or f"{snapshot_date.isoformat()}T00:00:00Z",
            )
            upsert_df(con, "odds_snapshots", rows)
            headers = cached.get("headers", {})
            return OddsFetchResult(
                rows=len(rows),
                remaining=headers.get("x-requests-remaining"),
                used=headers.get("x-requests-used"),
                last=headers.get("x-requests-last"),
                from_cache=True,
            )
        return OddsFetchResult(rows=0, remaining=None, used=None, last=str(exc), from_cache=True)


def fetch_player_prop_lines(
    con,
    snapshot_date: date,
    markets: str,
    api_key: str | None = None,
    force: bool = False,
    bookmaker: str | None = None,
    max_events: int | None = None,
) -> OddsFetchResult:
    api_key = api_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set ODDS_API_KEY or pass --api-key to fetch player props.")

    events_cache_key = f"{SPORT_KEY}:{snapshot_date.isoformat()}:events"
    cached_events = cache_get(con, "the_odds_api_events", events_cache_key)
    if cached_events and not force:
        events = cached_events["payload"]
    else:
        ensure_host_resolves(ODDS_API_BASE)
        response = requests.get(
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events",
            params={"apiKey": api_key, "dateFormat": "iso"},
            timeout=30,
        )
        response.raise_for_status()
        events = response.json()
        cache_put(con, "the_odds_api_events", events_cache_key, {"payload": events, "headers": {}, "fetched_at": datetime.now(timezone.utc).isoformat()})

    day_events = []
    for event in events:
        commence = pd.to_datetime(event.get("commence_time"), errors="coerce")
        if not pd.isna(commence) and commence.date() == snapshot_date:
            day_events.append(event)
    if max_events:
        day_events = day_events[:max_events]

    all_rows = []
    remaining = used = last = None
    fetched_any_from_api = False
    for event in day_events:
        event_id = event.get("id")
        if not event_id:
            continue
        cache_key = f"{SPORT_KEY}:{event_id}:{snapshot_date.isoformat()}:{markets}:us"
        cached = cache_get(con, "the_odds_api_player_props", cache_key)
        if cached and not force:
            payload = cached["payload"]
            fetched_at = cached.get("fetched_at") or f"{snapshot_date.isoformat()}T00:00:00Z"
            headers = cached.get("headers", {})
        else:
            ensure_host_resolves(ODDS_API_BASE)
            response = requests.get(
                f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds",
                params={
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": markets,
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                },
                timeout=30,
            )
            if response.status_code in {404, 422}:
                continue
            response.raise_for_status()
            payload = response.json()
            fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            headers = {
                "x-requests-remaining": response.headers.get("x-requests-remaining"),
                "x-requests-used": response.headers.get("x-requests-used"),
                "x-requests-last": response.headers.get("x-requests-last"),
            }
            cache_put(con, "the_odds_api_player_props", cache_key, {"payload": payload, "headers": headers, "fetched_at": fetched_at})
            fetched_any_from_api = True
        remaining = headers.get("x-requests-remaining", remaining) if headers else remaining
        used = headers.get("x-requests-used", used) if headers else used
        last = headers.get("x-requests-last", last) if headers else last
        all_rows.append(normalize_player_prop_payload(payload, snapshot_date, bookmaker, fetched_at=fetched_at))

    rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not rows.empty:
        upsert_df(con, "player_prop_lines", rows)
    return OddsFetchResult(rows=len(rows), remaining=remaining, used=used, last=last, from_cache=not fetched_any_from_api)


def normalize_player_prop_payload(
    payload: dict[str, Any],
    snapshot_date: date,
    bookmaker_filter: str | None = None,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    snapshot_token = fetched_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    event_id = payload.get("id")
    home_team = normalize_team_name(payload.get("home_team"))
    away_team = normalize_team_name(payload.get("away_team"))
    for book in payload.get("bookmakers", []):
        if bookmaker_filter and book.get("key") != bookmaker_filter:
            continue
        for market in book.get("markets", []):
            market_key = market.get("key")
            grouped: dict[tuple[str, float], dict[str, Any]] = {}
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description") or outcome.get("player") or outcome.get("name")
                side = str(outcome.get("name", "")).lower()
                point = outcome.get("point")
                if not player_name or point is None or side not in {"over", "under"}:
                    continue
                key = (str(player_name), float(point))
                grouped.setdefault(key, {"raw_outcomes": []})
                grouped[key]["raw_outcomes"].append(outcome)
                grouped[key][f"{side}_odds"] = outcome.get("price")
            for (player_name, line), values in grouped.items():
                if values.get("over_odds") is None and values.get("under_odds") is None:
                    continue
                row_key = f"{event_id}:{book.get('key') or book.get('title')}:{market_key}:{player_name}:{line}:{snapshot_token}"
                rows.append(
                    {
                        "prop_line_id": hashlib.sha1(row_key.encode()).hexdigest(),
                        "event_id": event_id,
                        "snapshot_date": snapshot_date,
                        "commence_time": payload.get("commence_time"),
                        "bookmaker": book.get("key") or book.get("title"),
                        "home_team": home_team,
                        "away_team": away_team,
                        "player_id": None,
                        "player_name": player_name,
                        "market": market_key,
                        "line": line,
                        "over_odds": int(values["over_odds"]) if values.get("over_odds") is not None else None,
                        "under_odds": int(values["under_odds"]) if values.get("under_odds") is not None else None,
                        "raw_payload": {"event": payload, "outcomes": values["raw_outcomes"]},
                        "fetched_at": pd.to_datetime(snapshot_token),
                    }
                )
    return pd.DataFrame(rows)


def fetch_historical_odds(
    con,
    snapshot_date: date,
    api_key: str | None = None,
    snapshot_time_utc: str = "16:00:00",
    force: bool = False,
    bookmaker: str | None = None,
) -> OddsFetchResult:
    api_key = api_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set ODDS_API_KEY or pass --api-key to fetch historical odds.")

    hour, minute, second = [int(part) for part in snapshot_time_utc.split(":")]
    snapshot_dt = datetime.combine(snapshot_date, time(hour, minute, second), tzinfo=timezone.utc)
    snapshot_iso = snapshot_dt.isoformat().replace("+00:00", "Z")
    cache_key = f"{SPORT_KEY}:{snapshot_iso}:h2h:us"
    cached = cache_get(con, "the_odds_api_historical", cache_key)
    if cached and not force:
        payload = cached["payload"]
        rows = normalize_odds_payload(
            historical_data(payload),
            snapshot_date,
            bookmaker,
            fetched_at=cached.get("fetched_at") or snapshot_iso,
        )
        upsert_df(con, "odds_snapshots", rows)
        headers = cached.get("headers", {})
        return OddsFetchResult(
            rows=len(rows),
            remaining=headers.get("x-requests-remaining"),
            used=headers.get("x-requests-used"),
            last=headers.get("x-requests-last"),
            from_cache=True,
        )

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": snapshot_iso,
    }
    ensure_host_resolves(ODDS_API_BASE)
    response = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{SPORT_KEY}/odds",
        params=params,
        timeout=45,
    )
    raise_for_status_safely(response)
    payload = response.json()
    fetched_at = snapshot_iso
    headers = {
        "x-requests-remaining": response.headers.get("x-requests-remaining"),
        "x-requests-used": response.headers.get("x-requests-used"),
        "x-requests-last": response.headers.get("x-requests-last"),
    }
    cache_put(con, "the_odds_api_historical", cache_key, {"payload": payload, "headers": headers, "fetched_at": fetched_at})
    rows = normalize_odds_payload(historical_data(payload), snapshot_date, bookmaker, fetched_at=fetched_at)
    upsert_df(con, "odds_snapshots", rows)
    return OddsFetchResult(
        rows=len(rows),
        remaining=headers["x-requests-remaining"],
        used=headers["x-requests-used"],
        last=headers["x-requests-last"],
        from_cache=False,
    )


def normalize_odds_payload(
    payload: list[dict[str, Any]],
    snapshot_date: date,
    bookmaker_filter: str | None = None,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    snapshot_token = fetched_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for event in payload:
        home_team = normalize_team_name(event.get("home_team"))
        away_team = normalize_team_name(event.get("away_team"))
        for book in event.get("bookmakers", []):
            if bookmaker_filter and book.get("key") != bookmaker_filter:
                continue
            h2h = next((m for m in book.get("markets", []) if m.get("key") == "h2h"), None)
            spreads = next((m for m in book.get("markets", []) if m.get("key") == "spreads"), None)
            totals = next((m for m in book.get("markets", []) if m.get("key") == "totals"), None)
            if not h2h and not spreads and not totals:
                continue
            prices = {normalize_team_name(o.get("name")): o.get("price") for o in (h2h or {}).get("outcomes", [])}
            spread_prices = {normalize_team_name(o.get("name")): o for o in (spreads or {}).get("outcomes", [])}
            total_prices = {str(o.get("name")).lower(): o for o in (totals or {}).get("outcomes", [])}
            if h2h and (home_team not in prices or away_team not in prices):
                continue
            rows.append(
                {
                    "event_id": f"{event.get('id')}:{snapshot_token}",
                    "snapshot_date": snapshot_date,
                    "commence_time": event.get("commence_time"),
                    "bookmaker": book.get("key") or book.get("title"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_moneyline": int(prices[home_team]) if home_team in prices and prices[home_team] is not None else None,
                    "away_moneyline": int(prices[away_team]) if away_team in prices and prices[away_team] is not None else None,
                    "home_spread": spread_prices.get(home_team, {}).get("point"),
                    "away_spread": spread_prices.get(away_team, {}).get("point"),
                    "home_spread_price": spread_prices.get(home_team, {}).get("price"),
                    "away_spread_price": spread_prices.get(away_team, {}).get("price"),
                    "total_points": total_prices.get("over", {}).get("point"),
                    "over_price": total_prices.get("over", {}).get("price"),
                    "under_price": total_prices.get("under", {}).get("price"),
                    "raw_payload": event,
                    "fetched_at": pd.to_datetime(snapshot_token),
                }
            )
    return pd.DataFrame(rows)


def import_historical_odds_csv(con, path: str, source: str = "csv_import") -> int:
    df = pd.read_csv(path)
    required = {"game_date", "home_team", "away_team", "home_moneyline", "away_moneyline"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required odds CSV columns: {sorted(missing)}")
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "event_id": row.get("event_id") or f"{source}:{row['game_date']}:{row['away_team']}@{row['home_team']}:{idx}",
                "snapshot_date": pd.to_datetime(row["game_date"]).date(),
                "commence_time": row.get("commence_time"),
                "bookmaker": row.get("bookmaker") or source,
                "home_team": normalize_team_name(row["home_team"]),
                "away_team": normalize_team_name(row["away_team"]),
                "home_moneyline": int(row["home_moneyline"]),
                "away_moneyline": int(row["away_moneyline"]),
                "raw_payload": row.to_dict(),
            }
        )
    return upsert_df(con, "odds_snapshots", pd.DataFrame(rows))


def historical_data(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        return data if isinstance(data, list) else []
    return payload if isinstance(payload, list) else []


def raise_for_status_safely(response: requests.Response) -> None:
    if response.ok:
        return
    text = response.text[:500]
    text = re.sub(r"apiKey=[^&\\s]+", "apiKey=<redacted>", text)
    raise RuntimeError(f"Odds API request failed: status={response.status_code} body={text}")
