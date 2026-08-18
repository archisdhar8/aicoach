from uuid import UUID

from app.schemas.domain import (
    CourtPosition,
    Coverage,
    Player,
    PlayerState,
    PossessedBallState,
    PossessionState,
    SimulationFrame,
    TeamSide,
)

OFFENSE_ID = UUID("10000000-0000-0000-0000-000000000001")
DEFENSE_ID = UUID("20000000-0000-0000-0000-000000000001")


def _player_state(
    *,
    player_number: int,
    name: str,
    jersey: int,
    team_id: UUID,
    team_side: TeamSide,
    x: float,
    y: float,
) -> PlayerState:
    namespace = "1" if team_side is TeamSide.OFFENSE else "2"
    return PlayerState(
        player=Player(
            id=UUID(f"{namespace}0000000-0000-0000-0000-{player_number:012d}"),
            team_id=team_id,
            name=name,
            jersey_number=jersey,
            position=("PG", "SG", "SF", "PF", "C")[player_number - 1],
        ),
        team_side=team_side,
        position=CourtPosition(x=x, y=y),
        facing_degrees=0 if team_side is TeamSide.OFFENSE else 180,
    )


def build_example_frame() -> SimulationFrame:
    """Return deterministic state for UI/integration development, not a prediction."""
    offense = [
        _player_state(
            player_number=1,
            name="Point Guard",
            jersey=1,
            team_id=OFFENSE_ID,
            team_side=TeamSide.OFFENSE,
            x=68,
            y=25,
        ),
        _player_state(
            player_number=2,
            name="Right Wing",
            jersey=2,
            team_id=OFFENSE_ID,
            team_side=TeamSide.OFFENSE,
            x=77,
            y=8,
        ),
        _player_state(
            player_number=3,
            name="Left Wing",
            jersey=3,
            team_id=OFFENSE_ID,
            team_side=TeamSide.OFFENSE,
            x=77,
            y=42,
        ),
        _player_state(
            player_number=4,
            name="Screener",
            jersey=4,
            team_id=OFFENSE_ID,
            team_side=TeamSide.OFFENSE,
            x=73,
            y=25,
        ),
        _player_state(
            player_number=5,
            name="Dunker",
            jersey=5,
            team_id=OFFENSE_ID,
            team_side=TeamSide.OFFENSE,
            x=87,
            y=34,
        ),
    ]
    defense = [
        _player_state(
            player_number=1,
            name="On-ball Defender",
            jersey=11,
            team_id=DEFENSE_ID,
            team_side=TeamSide.DEFENSE,
            x=70,
            y=25,
        ),
        _player_state(
            player_number=2,
            name="Nail Defender",
            jersey=12,
            team_id=DEFENSE_ID,
            team_side=TeamSide.DEFENSE,
            x=79,
            y=13,
        ),
        _player_state(
            player_number=3,
            name="Wing Defender",
            jersey=13,
            team_id=DEFENSE_ID,
            team_side=TeamSide.DEFENSE,
            x=79,
            y=38,
        ),
        _player_state(
            player_number=4,
            name="Screen Defender",
            jersey=14,
            team_id=DEFENSE_ID,
            team_side=TeamSide.DEFENSE,
            x=76,
            y=26,
        ),
        _player_state(
            player_number=5,
            name="Rim Defender",
            jersey=15,
            team_id=DEFENSE_ID,
            team_side=TeamSide.DEFENSE,
            x=85,
            y=28,
        ),
    ]
    ball_handler = offense[0]
    return SimulationFrame(
        timestamp_seconds=0,
        players=[*offense, *defense],
        ball=PossessedBallState(
            position=CourtPosition(
                x=ball_handler.position.x + 0.5, y=ball_handler.position.y + 2.5
            ),
            player_id=ball_handler.player.id,
        ),
        possession=PossessionState(
            offense_team_id=OFFENSE_ID,
            defense_team_id=DEFENSE_ID,
            game_clock_seconds=420,
            shot_clock_seconds=18,
            coverage=Coverage.DROP,
        ),
        metadata={"source": "deterministic_example", "predictive": False},
    )
