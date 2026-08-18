from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from .db import upsert_df
from .net import ensure_host_resolves


VENUE_COORDS = {
    "Angel Stadium": (33.8003, -117.8827),
    "Busch Stadium": (38.6226, -90.1928),
    "Chase Field": (33.4455, -112.0667),
    "Citi Field": (40.7571, -73.8458),
    "Citizens Bank Park": (39.9061, -75.1665),
    "Comerica Park": (42.3390, -83.0485),
    "Coors Field": (39.7561, -104.9942),
    "Dodger Stadium": (34.0739, -118.2400),
    "Fenway Park": (42.3467, -71.0972),
    "Globe Life Field": (32.7473, -97.0842),
    "Great American Ball Park": (39.0979, -84.5081),
    "Guaranteed Rate Field": (41.8300, -87.6339),
    "Kauffman Stadium": (39.0517, -94.4803),
    "loanDepot park": (25.7781, -80.2197),
    "Minute Maid Park": (29.7573, -95.3555),
    "Nationals Park": (38.8730, -77.0074),
    "Oracle Park": (37.7786, -122.3893),
    "Oriole Park at Camden Yards": (39.2840, -76.6217),
    "Petco Park": (32.7073, -117.1566),
    "PNC Park": (40.4469, -80.0057),
    "Progressive Field": (41.4962, -81.6852),
    "Rate Field": (41.8300, -87.6339),
    "Rogers Centre": (43.6414, -79.3894),
    "T-Mobile Park": (47.5914, -122.3325),
    "Target Field": (44.9817, -93.2776),
    "Tropicana Field": (27.7682, -82.6534),
    "Truist Park": (33.8908, -84.4678),
    "Wrigley Field": (41.9484, -87.6553),
    "Yankee Stadium": (40.8296, -73.9262),
}


DOME_OR_RETRACTABLE = {
    "Chase Field",
    "Globe Life Field",
    "loanDepot park",
    "Minute Maid Park",
    "Rogers Centre",
    "T-Mobile Park",
    "Tropicana Field",
}


def fetch_weather_for_games(con, games: pd.DataFrame, force: bool = False) -> int:
    if games.empty:
        return 0
    existing = set()
    if not force:
        existing = set(con.execute("SELECT game_pk FROM game_weather").df()["game_pk"].tolist())
    rows = []
    for game in games.itertuples(index=False):
        if game.game_pk in existing:
            continue
        venue = getattr(game, "venue_name", None)
        coords = VENUE_COORDS.get(venue)
        if not coords:
            continue
        game_date = pd.to_datetime(game.game_date).date()
        if venue in DOME_OR_RETRACTABLE:
            rows.append(default_dome_weather(game.game_pk, game_date, venue))
            continue
        try:
            weather = fetch_open_meteo(coords[0], coords[1], game_date)
        except (requests.RequestException, RuntimeError):
            weather = {
                "temperature_2m": None,
                "relative_humidity_2m": None,
                "wind_speed_10m": None,
                "wind_direction_10m": None,
                "precipitation": None,
                "source": "open-meteo-error",
            }
        rows.append(
            {
                "game_pk": game.game_pk,
                "game_date": game_date,
                "venue_name": venue,
                "temperature_f": weather.get("temperature_2m"),
                "relative_humidity": weather.get("relative_humidity_2m"),
                "wind_speed_mph": weather.get("wind_speed_10m"),
                "wind_direction_degrees": weather.get("wind_direction_10m"),
                "precipitation_in": weather.get("precipitation"),
                "source": weather.get("source"),
            }
        )
    return upsert_df(con, "game_weather", pd.DataFrame(rows))


def fetch_open_meteo(latitude: float, longitude: float, game_date: date) -> dict:
    base = "https://archive-api.open-meteo.com/v1/archive"
    ensure_host_resolves(base)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": game_date.isoformat(),
        "end_date": game_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }
    response = requests.get(base, params=params, timeout=30)
    response.raise_for_status()
    data = response.json().get("hourly", {})
    idx = 19 if len(data.get("time", [])) > 19 else 12
    return {
        "temperature_2m": value_at(data.get("temperature_2m"), idx),
        "relative_humidity_2m": value_at(data.get("relative_humidity_2m"), idx),
        "wind_speed_10m": value_at(data.get("wind_speed_10m"), idx),
        "wind_direction_10m": value_at(data.get("wind_direction_10m"), idx),
        "precipitation": value_at(data.get("precipitation"), idx),
        "source": "open-meteo-archive",
    }


def default_dome_weather(game_pk: int, game_date: date, venue: str) -> dict:
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "venue_name": venue,
        "temperature_f": 72.0,
        "relative_humidity": 45.0,
        "wind_speed_mph": 0.0,
        "wind_direction_degrees": 0.0,
        "precipitation_in": 0.0,
        "source": "dome-default",
    }


def value_at(values, idx: int):
    if not values:
        return None
    return values[min(idx, len(values) - 1)]
