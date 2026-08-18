from math import hypot

from app.schemas.domain import CourtPosition, PlayerState, TeamSide


def pairwise_offensive_distances(players: list[PlayerState]) -> list[float]:
    """Return geometric spacing inputs in feet; this is not an outcome model."""
    offense = [state.position for state in players if state.team_side is TeamSide.OFFENSE]
    return [
        _distance(first, second)
        for index, first in enumerate(offense)
        for second in offense[index + 1 :]
    ]


def _distance(first: CourtPosition, second: CourtPosition) -> float:
    return hypot(second.x - first.x, second.y - first.y)
