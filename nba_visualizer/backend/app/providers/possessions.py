from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Protocol

from app.possessions.normalization import (
    normalize_nba_play_by_play,
    normalize_pbpstats_possessions,
)
from app.providers.base import NBADataProvider, ProviderError
from app.schemas.nba_data import NBAGame
from app.schemas.possessions import RealPossession


class PossessionDataProvider(Protocol):
    """Boundary for normalized historical possessions, independent of pbpstats."""

    def get_possessions(self, game: NBAGame) -> list[RealPossession]: ...


class PbpStatsPossessionProvider:
    def __init__(
        self,
        fetcher: Callable[[str], list[dict[str, Any]]] | None = None,
        timeout_seconds: float = 12,
    ) -> None:
        self._fetcher = fetcher or self._fetch_with_pbpstats
        self._timeout_seconds = timeout_seconds

    def get_possessions(self, game: NBAGame) -> list[RealPossession]:
        try:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pbpstats")
            future = executor.submit(self._fetcher, game.external_id)
            try:
                raw = future.result(timeout=self._timeout_seconds)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            if not raw:
                raise ValueError("pbpstats returned no possessions")
            return normalize_pbpstats_possessions(game.id, game.external_id, raw)
        except FutureTimeoutError as error:
            raise ProviderError(
                "pbpstats possession request timed out",
                code="pbpstats_timeout",
                retryable=True,
                attempts=1,
            ) from error
        except Exception as error:
            raise ProviderError(
                "pbpstats possession request failed",
                code="pbpstats_request_failed",
                retryable=True,
                attempts=1,
            ) from error

    @staticmethod
    def _fetch_with_pbpstats(game_external_id: str) -> list[dict[str, Any]]:
        from pbpstats.client import Client

        client = Client(
            {
                "Possessions": {
                    "source": "web",
                    "data_provider": "stats_nba",
                }
            }
        )
        game = client.Game(game_external_id)
        return [_possession_dict(item) for item in game.possessions.items]


class NBAApiPossessionProvider:
    def __init__(self, nba_provider: NBADataProvider) -> None:
        self._nba_provider = nba_provider

    def get_possessions(self, game: NBAGame) -> list[RealPossession]:
        events = self._nba_provider.get_play_by_play(game.external_id)
        return normalize_nba_play_by_play(game.id, game.external_id, events)


class FallbackPossessionProvider:
    def __init__(
        self,
        primary: PossessionDataProvider,
        fallback: PossessionDataProvider,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    def get_possessions(self, game: NBAGame) -> list[RealPossession]:
        try:
            return self._primary.get_possessions(game)
        except ProviderError:
            return self._fallback.get_possessions(game)


def _possession_dict(possession: Any) -> dict[str, Any]:
    events = list(getattr(possession, "events", []))
    current_players = getattr(events[0], "current_players", {}) if events else {}
    offense_team_id = str(getattr(possession, "offense_team_id", ""))
    defense_team_id = str(getattr(possession, "defense_team_id", ""))
    return {
        "id": getattr(possession, "number", None),
        "period": getattr(possession, "period", None),
        "start_clock": getattr(possession, "start_time", None),
        "end_clock": getattr(possession, "end_time", None),
        "offense_team_id": offense_team_id or None,
        "defense_team_id": defense_team_id or None,
        "offensive_lineup": [
            {"id": player_id, "team_id": offense_team_id}
            for player_id in _team_lineup(current_players, offense_team_id)
        ],
        "defensive_lineup": [
            {"id": player_id, "team_id": defense_team_id}
            for player_id in _team_lineup(current_players, defense_team_id)
        ],
        "lineups_available": bool(current_players),
        "events": [_event_dict(event, index) for index, event in enumerate(events, start=1)],
    }


def _event_dict(event: Any, index: int) -> dict[str, Any]:
    try:
        values = vars(event)
    except TypeError:
        values = {}
    raw = {
        key: value
        for key, value in values.items()
        if isinstance(value, str | int | float | bool | type(None))
    }
    return {
        **raw,
        "event_id": raw.get("event_num") or index,
        "period": raw.get("period"),
        "clock": raw.get("clock") or raw.get("time"),
        "description": raw.get("description"),
        "event_type": event.__class__.__name__,
        "player_id": raw.get("player_id"),
        "team_id": raw.get("team_id"),
        "shot_x": raw.get("loc_x"),
        "shot_y": raw.get("loc_y"),
    }


def _team_lineup(current_players: Any, team_id: str) -> list[Any]:
    if not isinstance(current_players, dict):
        return []
    return next(
        (list(players) for key, players in current_players.items() if str(key) == team_id),
        [],
    )
