from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, load_env_file
from .db import connect
from .live import MODEL_VERSION, LivePoller, latest_market, odds_status, refresh_live_odds, team_state_rates
from .pitch_prediction import diagnose_pitcher_counts


STATIC_DIR = Path(__file__).with_name("dashboard_static")
settings = Settings()
poller = LivePoller(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_env_file()
    poller.start()
    yield
    poller.stop()


app = FastAPI(title="MLB Live Lab", version=MODEL_VERSION, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/game/{game_pk}")
def game_dashboard(game_pk: int) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/live/games")
def live_games() -> dict:
    with connect(settings) as con:
        rows = con.execute(
            """
            SELECT * EXCLUDE (rn, pitch_analysis_json, pitch_sensitivities_json, pa_outcomes_json) FROM (
              SELECT *, row_number() OVER (PARTITION BY game_pk ORDER BY captured_at DESC) rn
              FROM live_game_snapshots
              WHERE game_date BETWEEN current_date - INTERVAL 1 DAY AND current_date + INTERVAL 1 DAY
            ) WHERE rn=1 ORDER BY captured_at DESC
            """
        ).df().to_dict("records")
        games = [_serialize(row) for row in rows]
        for game in games:
            game["top_three_pitches"] = game.pop("top_three_pitches_json", []) or []
            market = latest_market(con, int(game["game_pk"]))
            game["market"] = market
            game["home_edge"] = None
            game["away_edge"] = None
            game["actionable_side"] = None
            if market:
                game["home_edge"] = float(game["home_win_prob"]) - market["home_no_vig"]
                game["away_edge"] = float(game["away_win_prob"]) - market["away_no_vig"]
                if not market["stale"]:
                    if game["home_edge"] >= .05:
                        game["actionable_side"] = "home"
                    elif game["away_edge"] >= .05:
                        game["actionable_side"] = "away"
        return {"generated_at": datetime.now(timezone.utc).isoformat(), "games": games}


@app.get("/api/live/games/{game_pk}")
def live_game(game_pk: int) -> dict:
    with connect(settings) as con:
        rows = con.execute(
            "SELECT * FROM live_game_snapshots WHERE game_pk=? ORDER BY captured_at DESC LIMIT 240", [game_pk]
        ).df()
        if rows.empty:
            raise HTTPException(404, "No live snapshots found for this game.")
        latest = _serialize(rows.iloc[0].to_dict())
        history = [_serialize(row) for row in rows.sort_values("captured_at").to_dict("records")]
        market_history = [
            _serialize(row) for row in con.execute(
                """SELECT fetched_at, bookmaker, home_moneyline, away_moneyline, home_no_vig, away_no_vig
                   FROM live_odds_snapshots WHERE game_pk=? ORDER BY fetched_at""", [game_pk]
            ).df().to_dict("records")
        ]
        events = [
            _serialize(row) for row in con.execute(
                """SELECT inning, half_inning, at_bat_index, event_index, balls, strikes, outs,
                          batter_name, pitcher_name, event_description, pitch_type, start_speed
                   FROM live_play_events WHERE game_pk=?
                   ORDER BY at_bat_index DESC, event_index DESC LIMIT 40""", [game_pk]
            ).df().to_dict("records")
        ]
        rates = team_state_rates(con, int(latest["batting_team_id"]), date.fromisoformat(str(latest["game_date"])[:10]))
        return {
            "game": latest,
            "history": history,
            "market_history": market_history,
            "events": events,
            "state_rates": rates,
            "pitch_analysis": latest.get("pitch_analysis_json") or {},
            "pitch_sensitivities": latest.get("pitch_sensitivities_json") or [],
            "pa_outcomes": latest.get("pa_outcomes_json") or {},
        }


@app.get("/api/state-rates/{team_id}")
def state_rates(team_id: int, as_of: date | None = None) -> dict:
    with connect(settings) as con:
        return team_state_rates(con, team_id, as_of or date.today())


@app.get("/api/pitch-diagnostics/{pitcher_id}")
def pitch_diagnostics(pitcher_id: int, as_of: date | None = None, batter_side: str = "R") -> dict:
    side = batter_side.upper()
    if side not in {"L", "R", "S"}:
        raise HTTPException(400, "batter_side must be L, R, or S")
    with connect(settings) as con:
        return diagnose_pitcher_counts(con, pitcher_id, as_of or date.today(), side)


@app.post("/api/live/odds/refresh")
def manual_odds_refresh(x_manual_odds_refresh: str | None = Header(default=None)) -> dict:
    if x_manual_odds_refresh != "confirmed":
        raise HTTPException(400, "Manual odds refresh confirmation header is required.")
    try:
        with connect(settings) as con:
            return refresh_live_odds(con)
    except RuntimeError as exc:
        raise HTTPException(429 if "cooldown" in str(exc).lower() else 400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Odds provider request failed: {exc}") from exc


@app.get("/api/live/odds/status")
def manual_odds_status() -> dict:
    with connect(settings) as con:
        return odds_status(con)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok" if poller.status()["running"] else "degraded", "model_version": MODEL_VERSION, "poller": poller.status()}


def _serialize(value):
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, str) and value[:1] in {"[", "{"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value
