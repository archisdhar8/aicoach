from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from app.schemas.domain import DomainModel, Play


class FieldOrigin(StrEnum):
    OBSERVED = "observed"
    DERIVED = "derived"
    MANUAL = "manual"
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"


class FieldProvenance(DomainModel):
    origin: FieldOrigin
    source: str
    source_field: str | None = None
    note: str | None = None


class PossessionPlayer(DomainModel):
    id: UUID
    external_id: str
    display_name: str | None = None
    team_id: UUID | None = None
    team_external_id: str | None = None


class RealPossessionEvent(DomainModel):
    source_event_id: str
    sequence: int
    period: int | None = None
    clock: str | None = None
    event_type: str
    description: str | None = None
    team_external_id: str | None = None
    player_external_id: str | None = None
    shooter_external_id: str | None = None
    passer_external_id: str | None = None
    assist_external_id: str | None = None
    is_turnover: bool = False
    is_foul: bool = False
    is_rebound: bool = False
    shot_x: float | None = None
    shot_y: float | None = None
    shot_type: str | None = None
    shot_result: str | None = None
    points: int | None = None


class PossessionResult(DomainModel):
    result_type: str
    points: int = 0
    made: bool | None = None
    turnover: bool = False


class PossessionProvenance(DomainModel):
    provider: str
    source_game_id: str
    source_possession_id: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    movement_available: bool = False
    field_origins: dict[str, FieldProvenance]
    raw_reference: dict[str, Any] = Field(default_factory=dict)


class RealPossession(DomainModel):
    id: UUID
    game_id: UUID
    game_external_id: str
    period: int
    start_clock: str | None = None
    end_clock: str | None = None
    offense_team_external_id: str | None = None
    defense_team_external_id: str | None = None
    offensive_lineup: list[PossessionPlayer] = Field(default_factory=list)
    defensive_lineup: list[PossessionPlayer] = Field(default_factory=list)
    events: list[RealPossessionEvent]
    result: PossessionResult
    provenance: PossessionProvenance

    @model_validator(mode="after")
    def ensure_event_order(self) -> "RealPossession":
        sequences = [event.sequence for event in self.events]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("possession events must have unique increasing sequence numbers")
        return self


class PossessionListResponse(DomainModel):
    game_id: UUID
    possessions: list[RealPossession]
    cache_status: str
    retrieved_at: datetime | None = None


class PossessionReconstruction(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    possession_id: UUID
    play: Play
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PossessionReconstructionRequest(DomainModel):
    play: Play
