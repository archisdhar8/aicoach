from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.basketball.rules import COURT_LENGTH_FEET, COURT_WIDTH_FEET, validate_lineup


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(word.capitalize() for word in rest)


class DomainModel(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)


class TeamSide(StrEnum):
    OFFENSE = "offense"
    DEFENSE = "defense"


class Coverage(StrEnum):
    DROP = "drop"
    SWITCH = "switch"
    HEDGE = "hedge"
    BLITZ = "blitz"
    ICE = "ice"


class ActionType(StrEnum):
    MOVE = "move"
    CUT = "cut"
    DRIBBLE = "dribble"
    SCREEN = "screen"
    PASS = "pass"
    SHOOT = "shoot"
    HOLD = "hold"


class Team(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=100)
    abbreviation: str = Field(min_length=2, max_length=4)


class Player(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    team_id: UUID
    name: str = Field(min_length=1, max_length=100)
    jersey_number: int | None = Field(default=None, ge=0, le=99)
    position: str | None = Field(default=None, max_length=10)
    height: str | None = Field(default=None, max_length=20)
    external_id: str | None = None
    source: str | None = None


class CourtPosition(DomainModel):
    """Position in feet: x baseline-to-baseline, y sideline-to-sideline."""

    x: float = Field(ge=0, le=COURT_LENGTH_FEET)
    y: float = Field(ge=0, le=COURT_WIDTH_FEET)


class PlayerState(DomainModel):
    player: Player
    team_side: TeamSide
    position: CourtPosition
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    facing_degrees: float = Field(default=0.0, ge=0, lt=360)


class BallStateBase(DomainModel):
    position: CourtPosition
    height_feet: float = Field(default=3.5, ge=0, le=50)


class PossessedBallState(BallStateBase):
    state: Literal["possessed"] = "possessed"
    player_id: UUID


class PassingBallState(BallStateBase):
    state: Literal["traveling_between_players"] = "traveling_between_players"
    from_player_id: UUID
    to_player_id: UUID
    progress: float = Field(ge=0, le=1)


class ShootingBallState(BallStateBase):
    state: Literal["traveling_to_basket"] = "traveling_to_basket"
    shooter_player_id: UUID
    progress: float = Field(ge=0, le=1)
    deterministic_result: Literal["made", "missed"] = "made"


class LooseBallState(BallStateBase):
    state: Literal["loose"] = "loose"


BallState = Annotated[
    PossessedBallState | PassingBallState | ShootingBallState | LooseBallState,
    Field(discriminator="state"),
]


class ActionBase(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    player_id: UUID
    start_time: float = Field(ge=0)
    duration: float = Field(gt=0)
    source: CourtPosition
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


class MovementActionBase(ActionBase):
    target: CourtPosition
    waypoints: list[CourtPosition] = Field(default_factory=list)


class MoveAction(MovementActionBase):
    action_type: Literal["move"] = "move"


class CutAction(MovementActionBase):
    action_type: Literal["cut"] = "cut"


class DribbleAction(MovementActionBase):
    action_type: Literal["dribble"] = "dribble"


class ScreenAction(ActionBase):
    action_type: Literal["screen"] = "screen"
    screen_location: CourtPosition
    orientation_degrees: float = Field(default=90, ge=0, lt=360)
    target_player_id: UUID | None = None


class PassAction(ActionBase):
    action_type: Literal["pass"] = "pass"
    target: CourtPosition
    target_player_id: UUID


class ShootAction(ActionBase):
    action_type: Literal["shoot"] = "shoot"
    target: CourtPosition
    deterministic_result: Literal["made", "missed"] = "made"


class HoldAction(ActionBase):
    action_type: Literal["hold"] = "hold"


PlayAction = Annotated[
    MoveAction | CutAction | DribbleAction | ScreenAction | PassAction | ShootAction | HoldAction,
    Field(discriminator="action_type"),
]


class RoutePoint(DomainModel):
    time_seconds: float = Field(ge=0)
    position: CourtPosition


class PlayerRoute(DomainModel):
    player_id: UUID
    points: list[RoutePoint] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_route(self) -> "PlayerRoute":
        if self.points[0].time_seconds != 0:
            raise ValueError("a route must start at time zero")
        for previous, current in zip(self.points, self.points[1:], strict=False):
            if current.time_seconds <= previous.time_seconds:
                raise ValueError("route times must be strictly increasing")
        return self


class PossessionState(DomainModel):
    offense_team_id: UUID
    defense_team_id: UUID
    game_clock_seconds: float = Field(ge=0, le=720)
    shot_clock_seconds: float = Field(ge=0, le=24)
    coverage: Coverage


class SimulationFrame(DomainModel):
    timestamp_seconds: float = Field(ge=0)
    players: list[PlayerState]
    ball: BallState
    current_actions: list[PlayAction] = Field(default_factory=list)
    possession: PossessionState
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_frame(self) -> "SimulationFrame":
        validate_lineup(self.players)
        player_ids = {state.player.id for state in self.players}
        if isinstance(self.ball, PossessedBallState) and self.ball.player_id not in player_ids:
            raise ValueError("ball possessor must be present in the frame")
        return self


class Play(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=120)
    initial_frame: SimulationFrame
    routes: list[PlayerRoute] = Field(default_factory=list)
    actions: list[PlayAction] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_actions(self) -> "Play":
        player_ids = {state.player.id for state in self.initial_frame.players}
        for action in self.actions:
            if action.player_id not in player_ids:
                raise ValueError(f"action player {action.player_id} is not in the play")
            if isinstance(action, (PassAction, ScreenAction)):
                target_id = action.target_player_id
                if target_id is not None and target_id not in player_ids:
                    raise ValueError(f"target player {target_id} is not in the play")
        for player_id in player_ids:
            actions = sorted(
                (action for action in self.actions if action.player_id == player_id),
                key=lambda action: action.start_time,
            )
            for previous, current in zip(actions, actions[1:], strict=False):
                if current.start_time < previous.end_time:
                    raise ValueError(f"player {player_id} has overlapping actions")
        self._validate_possession()
        return self

    def _validate_possession(self) -> None:
        possessor: UUID | None = (
            self.initial_frame.ball.player_id
            if isinstance(self.initial_frame.ball, PossessedBallState)
            else None
        )
        available_at = 0.0
        for action in sorted(self.actions, key=lambda item: (item.start_time, item.end_time)):
            if isinstance(action, (PassAction, ShootAction, DribbleAction)) and (
                action.start_time < available_at or action.player_id != possessor
            ):
                raise ValueError(f"cannot {action.action_type}: player does not possess the ball")
            if isinstance(action, PassAction):
                possessor = action.target_player_id
                available_at = action.end_time
            elif isinstance(action, ShootAction):
                possessor = None
                available_at = action.end_time
