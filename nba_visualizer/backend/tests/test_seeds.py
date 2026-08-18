from app.schemas.domain import ScreenAction, ShootAction
from app.simulation.seeds import build_seed_plays


def test_seed_library_contains_four_valid_structured_plays() -> None:
    plays = build_seed_plays()

    assert {play.name for play in plays} == {
        "High pick-and-roll",
        "Pick-and-pop",
        "Horns",
        "Drive-and-kick",
    }
    assert all(play.actions for play in plays)
    assert all(
        any(isinstance(action, ScreenAction) for action in play.actions) for play in plays[:3]
    )
    shots = [action for play in plays for action in play.actions if isinstance(action, ShootAction)]
    assert shots
    assert all(action.target.x == 88.75 and action.target.y == 25 for action in shots)
