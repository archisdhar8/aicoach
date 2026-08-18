from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid5

from pydantic import Field

from app.schemas.domain import DomainModel

NBA_NAMESPACE = UUID("cde443b8-20ef-4d46-8b90-180baf0a9714")


def stable_nba_id(entity: str, external_id: str) -> UUID:
    return uuid5(NBA_NAMESPACE, f"nba.com:{entity}:{external_id}")


class DataSource(StrEnum):
    NBA_COM = "nba.com"


class CachedRecord(DomainModel):
    source: DataSource = DataSource.NBA_COM
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NBATeam(CachedRecord):
    id: UUID
    external_id: str
    name: str
    full_name: str
    abbreviation: str
    city: str | None = None
    state: str | None = None
    year_founded: int | None = None


class PlayerShootingProfile(DomainModel):
    season: str
    games_played: int = Field(default=0, ge=0)
    field_goal_attempts: int = Field(default=0, ge=0)
    three_point_attempts: int = Field(default=0, ge=0)
    two_point_percentage: float | None = Field(default=None, ge=0, le=1)
    three_point_percentage: float | None = Field(default=None, ge=0, le=1)
    free_throw_percentage: float | None = Field(default=None, ge=0, le=1)
    provenance: str = "nba.com_season_totals"


class NBAPlayer(CachedRecord):
    id: UUID
    external_id: str
    team_id: UUID | None = None
    first_name: str
    last_name: str
    display_name: str
    position: str | None = None
    height: str | None = None
    jersey_number: int | None = Field(default=None, ge=0, le=99)
    active: bool = True
    shooting_profile: PlayerShootingProfile | None = None


class RosterEntry(CachedRecord):
    team_id: UUID
    player_id: UUID
    season: str


class NBAGame(CachedRecord):
    id: UUID
    external_id: str
    season: str
    game_date: date
    home_team_id: UUID | None = None
    away_team_id: UUID | None = None
    status: str | None = None


class PlayerGameStats(CachedRecord):
    game_id: UUID
    player_id: UUID
    team_id: UUID | None = None
    minutes: str | None = None
    points: int | None = None
    rebounds: int | None = None
    assists: int | None = None
    field_goals_made: int | None = None
    field_goals_attempted: int | None = None


class PlayByPlayEvent(DomainModel):
    game_external_id: str
    action_number: int
    period: int | None = None
    clock: str | None = None
    description: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ShotAttempt(DomainModel):
    player_external_id: str
    game_external_id: str | None = None
    x: float | None = None
    y: float | None = None
    made: bool | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TeamStats(CachedRecord):
    team_id: UUID
    season: str
    values: dict[str, int | float | str | None] = Field(default_factory=dict)


class CachedResult(DomainModel):
    cache_status: str
    retrieved_at: datetime | None = None


class TeamListResponse(CachedResult):
    teams: list[NBATeam]


class RosterResponse(CachedResult):
    team: NBATeam
    season: str
    players: list[NBAPlayer]


class PreferredLineupCandidate(DomainModel):
    player: NBAPlayer
    games_started: int = Field(default=0, ge=0)
    total_minutes: float = Field(default=0, ge=0)


class PreferredLineupResponse(CachedResult):
    team: NBATeam
    season: str
    players: list[NBAPlayer]
    selection_method: str = "current_roster_latest_starters_with_recent_role_fallback"
    disclaimer: str = (
        "Current-roster players ranked by the latest recorded starters and recent "
        "NBA roles; not a confirmed next-game lineup."
    )


class GameListResponse(CachedResult):
    games: list[NBAGame]


class BoxScoreResponse(CachedResult):
    game: NBAGame
    player_stats: list[PlayerGameStats]
