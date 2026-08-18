from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from app.persistence.nba_data import SQLiteNBADataRepository
from app.providers.base import ProviderError
from app.providers.nba_api import NBAApiProvider
from app.providers.service import NBADataService
from app.schemas.nba_data import (
    NBAGame,
    NBAPlayer,
    NBATeam,
    PlayerGameStats,
    PlayerShootingProfile,
    PreferredLineupCandidate,
    stable_nba_id,
)


def team() -> NBATeam:
    return NBATeam(
        id=stable_nba_id("team", "1610612744"),
        external_id="1610612744",
        name="Warriors",
        full_name="Golden State Warriors",
        abbreviation="GSW",
        city="Golden State",
        state="California",
        year_founded=1946,
        retrieved_at=datetime(2026, 8, 17, tzinfo=UTC),
    )


def player() -> NBAPlayer:
    return NBAPlayer(
        id=stable_nba_id("player", "201939"),
        external_id="201939",
        team_id=team().id,
        first_name="Stephen",
        last_name="Curry",
        display_name="Stephen Curry",
        position="G",
        height="6-2",
        jersey_number=30,
        retrieved_at=datetime(2026, 8, 17, tzinfo=UTC),
    )


def test_nba_api_roster_response_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = NBAApiProvider(timeout_seconds=1, max_attempts=1)
    raw = {
        "PLAYER_ID": 201939,
        "PLAYER": "Stephen Curry",
        "POSITION": "G",
        "HEIGHT": "6-2",
        "NUM": "30",
    }
    monkeypatch.setattr(provider, "_rows", lambda build, dataset: [raw])

    normalized = provider.get_roster("1610612744", "2025-26")

    assert len(normalized) == 1
    assert normalized[0].source == "nba.com"
    assert normalized[0].external_id == "201939"
    assert normalized[0].display_name == "Stephen Curry"
    assert normalized[0].team_id == team().id
    assert normalized[0].height == "6-2"
    assert normalized[0].jersey_number == 30


def test_provider_retries_are_bounded() -> None:
    provider = NBAApiProvider(timeout_seconds=1, max_attempts=2)
    attempts = 0

    def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise TimeoutError

    with pytest.raises(ProviderError) as error:
        provider._attempt(fail)

    assert attempts == 2
    assert error.value.attempts == 2
    assert error.value.retryable


def test_preferred_lineup_uses_most_recent_recorded_starters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = NBAApiProvider(timeout_seconds=1, max_attempts=1)
    game_rows = [
        {"GAME_ID": "0022500001", "GAME_DATE": "2026-04-10", "WL": "W"},
        {"GAME_ID": "0022500000", "GAME_DATE": "2026-04-09", "WL": "L"},
    ]
    box_rows = [
        {
            "PLAYER_ID": index,
            "PLAYER_NAME": f"Player {index}",
            "TEAM_ID": "1610612744",
            "START_POSITION": position,
            "MIN": "30:00",
        }
        for index, position in [
            (1, "G"),
            (2, "G"),
            (3, "F"),
            (4, "F"),
            (5, "C"),
            (6, ""),
        ]
    ]
    monkeypatch.setattr(
        provider,
        "_rows",
        lambda build, dataset: game_rows if dataset == "LeagueGameFinderResults" else box_rows,
    )

    lineup = provider.get_preferred_lineup("1610612744", "2025-26")

    assert [candidate.player.external_id for candidate in lineup] == ["1", "2", "3", "4", "5"]
    assert all(candidate.games_started == 1 for candidate in lineup)


def test_preferred_lineup_falls_back_to_season_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = NBAApiProvider(timeout_seconds=1, max_attempts=1)
    stats_rows = [
        {
            "PLAYER_ID": index,
            "PLAYER_NAME": f"Player {index}",
            "TEAM_ID": "1610612744",
            "MIN": minutes,
        }
        for index, minutes in [(1, 100), (2, 500), (3, 650), (4, 700), (5, 800), (6, 900)]
    ]
    monkeypatch.setattr(
        provider,
        "_rows",
        lambda build, dataset: [] if dataset == "LeagueGameFinderResults" else stats_rows,
    )

    lineup = provider.get_preferred_lineup("1610612744", "2025-26")

    assert [candidate.player.external_id for candidate in lineup] == ["6", "5", "4", "3", "2"]


def test_offseason_lineup_excludes_departed_players_and_uses_current_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = NBAApiProvider(timeout_seconds=1, max_attempts=1)
    game_rows = [{"GAME_ID": "last-game", "GAME_DATE": "2026-06-01", "WL": "W"}]
    box_rows = [
        {
            "PLAYER_ID": index,
            "PLAYER_NAME": f"Player {index}",
            "TEAM_ID": "1610612744",
            "START_POSITION": "G" if index < 3 else "F",
            "MIN": "30:00",
        }
        for index in range(1, 6)
    ]
    roster_rows = [
        {
            "PLAYER_ID": index,
            "PLAYER": f"Current {index}",
            "POSITION": "G" if index < 4 else "F",
            "NUM": str(index),
        }
        for index in range(2, 8)
    ]
    prior_stats = [
        {
            "PLAYER_ID": index,
            "PLAYER_NAME": f"Current {index}",
            "TEAM_ID": "prior-team",
            "GS": 20 - index,
            "MIN": 1000 - index,
            "GP": 70,
            "FGM": 400,
            "FGA": 800,
            "FG3M": 140,
            "FG3A": 400,
            "FG3_PCT": 0.35,
            "FT_PCT": 0.8,
        }
        for index in range(2, 8)
    ]
    stats_calls = 0
    offseason_zero_stats = [
        {
            "PLAYER_ID": index,
            "PLAYER_NAME": f"Current {index}",
            "TEAM_ID": "1610612744",
            "GS": 0,
            "MIN": 0,
        }
        for index in reversed(range(2, 8))
    ]

    def rows(build: Any, dataset: str) -> list[dict[str, Any]]:
        nonlocal stats_calls
        if dataset == "CommonTeamRoster":
            return roster_rows
        if dataset == "LeagueGameFinderResults":
            return game_rows
        if dataset == "PlayerStats":
            return box_rows
        stats_calls += 1
        return offseason_zero_stats if stats_calls == 1 else prior_stats

    monkeypatch.setattr(provider, "_rows", rows)

    lineup = provider.get_preferred_lineup("1610612744", "2026-27")

    selected_ids = [candidate.player.external_id for candidate in lineup]
    assert "1" not in selected_ids
    assert selected_ids == ["2", "3", "4", "5", "6"]
    assert [candidate.player.jersey_number for candidate in lineup] == [2, 3, 4, 5, 6]
    profile = lineup[0].player.shooting_profile
    assert profile is not None
    assert profile.season == "2025-26"
    assert profile.three_point_percentage == 0.35
    assert profile.two_point_percentage == 0.65


def test_preferred_lineup_cache_round_trip(tmp_path: Path) -> None:
    repository = SQLiteNBADataRepository(tmp_path / "lineup.sqlite3")
    repository.initialize()
    candidates = [
        PreferredLineupCandidate(
            player=player().model_copy(
                update={
                    "id": stable_nba_id("player", str(index)),
                    "external_id": str(index),
                    "shooting_profile": PlayerShootingProfile(
                        season="2025-26",
                        games_played=70,
                        field_goal_attempts=900,
                        three_point_attempts=400,
                        two_point_percentage=0.55,
                        three_point_percentage=0.38,
                        free_throw_percentage=0.82,
                    ),
                }
            ),
            games_started=30 - index,
            total_minutes=1000 - index,
        )
        for index in range(5)
    ]

    repository.replace_preferred_lineup(team().id, "2025-26", candidates)

    assert repository.list_preferred_lineup(team().id, "2025-26") == candidates
    assert repository.preferred_lineup_retrieved_at(team().id, "2025-26") is not None


def test_normalized_cache_round_trip_and_roster_membership(tmp_path: Path) -> None:
    repository = SQLiteNBADataRepository(tmp_path / "nba.sqlite3")
    repository.initialize()
    repository.save_teams([team()])
    repository.replace_roster(team().id, "2025-26", [player()])

    assert repository.list_teams() == [team()]
    assert repository.list_roster(team().id, "2025-26") == [player()]
    assert repository.roster_retrieved_at(team().id, "2025-26") == player().retrieved_at


def test_game_and_player_stats_cache_round_trip(tmp_path: Path) -> None:
    repository = SQLiteNBADataRepository(tmp_path / "stats.sqlite3")
    repository.initialize()
    game = NBAGame(
        id=stable_nba_id("game", "0022500001"),
        external_id="0022500001",
        season="2025-26",
        game_date=date(2025, 10, 21),
        home_team_id=team().id,
        status="final",
        retrieved_at=datetime(2026, 8, 17, tzinfo=UTC),
    )
    stats = PlayerGameStats(
        game_id=game.id,
        player_id=player().id,
        team_id=team().id,
        minutes="31:20",
        points=24,
        rebounds=5,
        assists=8,
        field_goals_made=8,
        field_goals_attempted=17,
        retrieved_at=datetime(2026, 8, 17, tzinfo=UTC),
    )
    repository.save_games([game])
    repository.save_player_game_stats([stats])

    assert repository.get_game(game.id) == game
    assert repository.list_games(season="2025-26") == [game]
    assert repository.list_player_game_stats(game.id) == [stats]


class FakeProvider:
    def __init__(self) -> None:
        self.fail = False

    def get_teams(self) -> list[NBATeam]:
        if self.fail:
            raise ProviderError("timeout", code="timeout", retryable=True, attempts=2)
        return [team()]

    def get_roster(self, team_external_id: str, season: str) -> list[NBAPlayer]:
        if self.fail:
            raise ProviderError("timeout", code="timeout", retryable=True, attempts=2)
        assert team_external_id == team().external_id
        assert season == "2025-26"
        return [player()]

    def get_player_directory(self, season: str) -> list[NBAPlayer]:
        return [player()]

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(name)


def test_service_falls_back_to_cache_when_provider_fails(tmp_path: Path) -> None:
    repository = SQLiteNBADataRepository(tmp_path / "fallback.sqlite3")
    repository.initialize()
    provider = FakeProvider()
    service = NBADataService(repository, provider)  # type: ignore[arg-type]
    service.get_teams(refresh=True)
    service.get_roster(team().id, "2025-26", refresh=True)
    provider.fail = True

    teams = service.get_teams(refresh=True)
    roster = service.get_roster(team().id, "2025-26", refresh=True)

    assert teams.cache_status == "cache_fallback"
    assert roster.cache_status == "cache_fallback"
    assert roster.players == [player()]
