from copy import deepcopy
from uuid import UUID

from app.schemas.domain import (
    CourtPosition,
    CutAction,
    DribbleAction,
    HoldAction,
    MoveAction,
    PassAction,
    Play,
    ScreenAction,
    ShootAction,
)
from app.simulation.example import build_example_frame

PG = UUID("10000000-0000-0000-0000-000000000001")
SG = UUID("10000000-0000-0000-0000-000000000002")
SF = UUID("10000000-0000-0000-0000-000000000003")
PF = UUID("10000000-0000-0000-0000-000000000004")
C = UUID("10000000-0000-0000-0000-000000000005")
ON_BALL_DEFENDER = UUID("20000000-0000-0000-0000-000000000001")
RIM = CourtPosition(x=88.75, y=25)


def _position(x: float, y: float) -> CourtPosition:
    return CourtPosition(x=x, y=y)


def build_seed_plays() -> list[Play]:
    """Deterministic, generic teaching plays; not NBA tracking data."""
    return [
        Play(
            id=UUID("30000000-0000-0000-0000-000000000001"),
            name="High pick-and-roll",
            initial_frame=deepcopy(build_example_frame()),
            actions=[
                DribbleAction(
                    player_id=PG,
                    start_time=0,
                    duration=2,
                    source=_position(68, 25),
                    target=_position(78, 22),
                ),
                ScreenAction(
                    player_id=PF,
                    start_time=0.5,
                    duration=1,
                    source=_position(73, 25),
                    screen_location=_position(74, 24),
                    orientation_degrees=90,
                    target_player_id=ON_BALL_DEFENDER,
                ),
                CutAction(
                    player_id=PF,
                    start_time=1.5,
                    duration=1,
                    source=_position(74, 24),
                    target=_position(87, 25),
                ),
                PassAction(
                    player_id=PG,
                    start_time=2,
                    duration=0.5,
                    source=_position(78, 22),
                    target=_position(87, 25),
                    target_player_id=PF,
                ),
                ShootAction(
                    player_id=PF,
                    start_time=2.5,
                    duration=0.8,
                    source=_position(87, 25),
                    target=RIM,
                    deterministic_result="made",
                    metadata={"demo_result": True},
                ),
            ],
        ),
        Play(
            id=UUID("30000000-0000-0000-0000-000000000002"),
            name="Pick-and-pop",
            initial_frame=deepcopy(build_example_frame()),
            actions=[
                DribbleAction(
                    player_id=PG,
                    start_time=0,
                    duration=2,
                    source=_position(68, 25),
                    target=_position(77, 24),
                ),
                ScreenAction(
                    player_id=PF,
                    start_time=0.5,
                    duration=1,
                    source=_position(73, 25),
                    screen_location=_position(74, 24),
                    orientation_degrees=90,
                    target_player_id=ON_BALL_DEFENDER,
                ),
                MoveAction(
                    player_id=PF,
                    start_time=1.5,
                    duration=1,
                    source=_position(74, 24),
                    target=_position(72, 39),
                ),
                PassAction(
                    player_id=PG,
                    start_time=2,
                    duration=0.5,
                    source=_position(77, 24),
                    target=_position(72, 39),
                    target_player_id=PF,
                ),
                ShootAction(
                    player_id=PF,
                    start_time=2.5,
                    duration=0.8,
                    source=_position(72, 39),
                    target=RIM,
                    deterministic_result="made",
                    metadata={"demo_result": True},
                ),
            ],
        ),
        Play(
            id=UUID("30000000-0000-0000-0000-000000000003"),
            name="Horns",
            initial_frame=deepcopy(build_example_frame()),
            actions=[
                DribbleAction(
                    player_id=PG,
                    start_time=0,
                    duration=1.5,
                    source=_position(68, 25),
                    target=_position(72, 25),
                ),
                ScreenAction(
                    player_id=PF,
                    start_time=0.5,
                    duration=1,
                    source=_position(73, 25),
                    screen_location=_position(75, 19),
                    orientation_degrees=75,
                ),
                ScreenAction(
                    player_id=C,
                    start_time=0.5,
                    duration=1,
                    source=_position(87, 34),
                    screen_location=_position(75, 31),
                    orientation_degrees=105,
                ),
                CutAction(
                    player_id=SG,
                    start_time=0,
                    duration=1.5,
                    source=_position(77, 8),
                    target=_position(84, 19),
                ),
                PassAction(
                    player_id=PG,
                    start_time=1.5,
                    duration=0.5,
                    source=_position(72, 25),
                    target=_position(84, 19),
                    target_player_id=SG,
                ),
                ShootAction(
                    player_id=SG,
                    start_time=2,
                    duration=0.8,
                    source=_position(84, 19),
                    target=RIM,
                    deterministic_result="made",
                    metadata={"demo_result": True},
                ),
            ],
        ),
        Play(
            id=UUID("30000000-0000-0000-0000-000000000004"),
            name="Drive-and-kick",
            initial_frame=deepcopy(build_example_frame()),
            actions=[
                DribbleAction(
                    player_id=PG,
                    start_time=0,
                    duration=2,
                    source=_position(68, 25),
                    target=_position(86, 18),
                    waypoints=[_position(78, 22)],
                ),
                HoldAction(player_id=SG, start_time=0, duration=2.6, source=_position(77, 8)),
                PassAction(
                    player_id=PG,
                    start_time=2,
                    duration=0.6,
                    source=_position(86, 18),
                    target=_position(77, 8),
                    target_player_id=SG,
                ),
                ShootAction(
                    player_id=SG,
                    start_time=2.6,
                    duration=0.8,
                    source=_position(77, 8),
                    target=RIM,
                    deterministic_result="made",
                    metadata={"demo_result": True},
                ),
            ],
        ),
    ]
