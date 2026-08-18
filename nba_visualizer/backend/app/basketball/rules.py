from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.schemas.domain import CourtPosition, PlayerState

COURT_LENGTH_FEET = 94.0
COURT_WIDTH_FEET = 50.0
PLAYERS_PER_TEAM = 5
PLAYERS_ON_COURT = PLAYERS_PER_TEAM * 2


def is_on_court(position: CourtPosition) -> bool:
    """Return whether a position lies within the regulation full-court bounds."""
    return 0 <= position.x <= COURT_LENGTH_FEET and 0 <= position.y <= COURT_WIDTH_FEET


def validate_lineup(players: list[PlayerState]) -> None:
    """Enforce a unique five-on-five lineup for a simulation frame."""
    if len(players) != PLAYERS_ON_COURT:
        raise ValueError(f"a simulation frame requires exactly {PLAYERS_ON_COURT} players")

    player_ids = [state.player.id for state in players]
    if len(set(player_ids)) != len(player_ids):
        raise ValueError("player IDs must be unique within a simulation frame")

    for side in ("offense", "defense"):
        count = sum(state.team_side.value == side for state in players)
        if count != PLAYERS_PER_TEAM:
            raise ValueError(f"a simulation frame requires {PLAYERS_PER_TEAM} {side} players")

    if any(not is_on_court(state.position) for state in players):
        raise ValueError("all players must be positioned on the court")
