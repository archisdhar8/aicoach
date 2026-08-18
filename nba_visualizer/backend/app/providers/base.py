from datetime import date
from typing import Protocol

from app.schemas.nba_data import (
    NBAGame,
    NBAPlayer,
    NBATeam,
    PlayByPlayEvent,
    PlayerGameStats,
    PreferredLineupCandidate,
    ShotAttempt,
    TeamStats,
)


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool,
        attempts: int,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.attempts = attempts


class NBADataProvider(Protocol):
    """Provider-neutral boundary for normalized professional basketball data."""

    def get_teams(self) -> list[NBATeam]: ...
    def get_player_directory(self, season: str) -> list[NBAPlayer]: ...
    def get_roster(self, team_external_id: str, season: str) -> list[NBAPlayer]: ...
    def get_preferred_lineup(
        self, team_external_id: str, season: str
    ) -> list[PreferredLineupCandidate]: ...
    def get_player(self, player_external_id: str) -> NBAPlayer: ...
    def get_games(
        self,
        *,
        game_date: date | None = None,
        season: str | None = None,
        team_external_id: str | None = None,
    ) -> list[NBAGame]: ...
    def get_box_score(self, game_external_id: str) -> list[PlayerGameStats]: ...
    def get_play_by_play(self, game_external_id: str) -> list[PlayByPlayEvent]: ...
    def get_shot_chart(self, *, player_external_id: str, season: str) -> list[ShotAttempt]: ...
    def get_player_stats(self, season: str) -> list[NBAPlayer]: ...
    def get_team_stats(self, season: str) -> list[TeamStats]: ...
