import pytest
from pydantic import ValidationError

from app.schemas.domain import (
    CourtPosition,
    CutAction,
    MoveAction,
    PassAction,
    Play,
    PlayerRoute,
    RoutePoint,
    ScreenAction,
    ShootAction,
    SimulationFrame,
)
from app.simulation.example import build_example_frame


def test_position_rejects_coordinates_outside_regulation_court() -> None:
    with pytest.raises(ValidationError):
        CourtPosition(x=95, y=25)


def test_frame_requires_exactly_five_players_per_side() -> None:
    frame = build_example_frame()
    payload = frame.model_dump()
    payload["players"] = payload["players"][:-1]

    with pytest.raises(ValidationError, match="exactly 10 players"):
        SimulationFrame.model_validate(payload)


def test_route_requires_strictly_increasing_time_from_zero() -> None:
    player_id = build_example_frame().players[0].player.id
    with pytest.raises(ValidationError, match="strictly increasing"):
        PlayerRoute(
            player_id=player_id,
            points=[
                RoutePoint(time_seconds=0, position=CourtPosition(x=60, y=20)),
                RoutePoint(time_seconds=0, position=CourtPosition(x=70, y=20)),
            ],
        )


def test_rejects_pass_and_shot_without_possession() -> None:
    frame = build_example_frame()
    non_handler = frame.players[1]
    with pytest.raises(ValidationError, match="does not possess"):
        Play(
            name="Invalid possession",
            initial_frame=frame,
            actions=[
                PassAction(
                    player_id=non_handler.player.id,
                    target_player_id=frame.players[2].player.id,
                    start_time=0,
                    duration=0.5,
                    source=non_handler.position,
                    target=frame.players[2].position,
                ),
                ShootAction(
                    player_id=non_handler.player.id,
                    start_time=1,
                    duration=0.8,
                    source=non_handler.position,
                    target=CourtPosition(x=88.75, y=25),
                ),
            ],
        )


def test_screen_has_a_typed_representation() -> None:
    frame = build_example_frame()
    action = ScreenAction(
        player_id=frame.players[3].player.id,
        start_time=0.5,
        duration=1,
        source=frame.players[3].position,
        screen_location=CourtPosition(x=75, y=24),
        orientation_degrees=105,
        target_player_id=frame.players[5].player.id,
    )

    payload = action.model_dump(by_alias=True)
    assert payload["actionType"] == "screen"
    assert payload["screenLocation"] == {"x": 75.0, "y": 24.0}
    assert payload["orientationDegrees"] == 105
    assert "target" not in payload


def test_rejects_overlapping_actions_for_one_player() -> None:
    frame = build_example_frame()
    player = frame.players[1]
    with pytest.raises(ValidationError, match="overlapping actions"):
        Play(
            name="Invalid overlap",
            initial_frame=frame,
            actions=[
                MoveAction(
                    player_id=player.player.id,
                    start_time=0,
                    duration=2,
                    source=player.position,
                    target=CourtPosition(x=80, y=12),
                ),
                CutAction(
                    player_id=player.player.id,
                    start_time=1,
                    duration=2,
                    source=CourtPosition(x=80, y=12),
                    target=CourtPosition(x=86, y=20),
                ),
            ],
        )
