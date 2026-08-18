from pathlib import Path

from app.persistence.sqlite import SQLitePlayRepository
from app.schemas.domain import CourtPosition, Play, ScreenAction
from app.simulation.example import build_example_frame


def test_sqlite_repository_round_trip(tmp_path: Path) -> None:
    repository = SQLitePlayRepository(tmp_path / "plays.sqlite3")
    repository.initialize()
    screener = build_example_frame().players[3]
    play = Play(
        name="Horns foundation",
        initial_frame=build_example_frame(),
        actions=[
            ScreenAction(
                player_id=screener.player.id,
                start_time=0,
                duration=1.5,
                source=screener.position,
                screen_location=CourtPosition(x=75, y=24),
                orientation_degrees=90,
            )
        ],
    )

    repository.save(play)

    assert repository.get(play.id) == play
    assert repository.list() == [play]
    loaded = repository.get(play.id)
    assert loaded is not None
    screen = loaded.actions[0]
    assert isinstance(screen, ScreenAction)
    assert screen.screen_location == CourtPosition(x=75, y=24)

    assert repository.delete(play.id)
    assert repository.get(play.id) is None
