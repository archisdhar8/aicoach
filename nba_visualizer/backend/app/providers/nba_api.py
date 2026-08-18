from collections.abc import Callable
from datetime import UTC, date, datetime
from time import sleep
from typing import Any, TypeVar, cast

from app.providers.base import ProviderError
from app.schemas.nba_data import (
    NBAGame,
    NBAPlayer,
    NBATeam,
    PlayByPlayEvent,
    PlayerGameStats,
    PlayerShootingProfile,
    PreferredLineupCandidate,
    ShotAttempt,
    TeamStats,
    stable_nba_id,
)

T = TypeVar("T")


class NBAApiProvider:
    """Normalize nba_api responses without leaking nba_api objects past this module."""

    def __init__(self, *, timeout_seconds: float = 10, max_attempts: int = 2) -> None:
        if timeout_seconds <= 0 or max_attempts < 1:
            raise ValueError("timeout and retry bounds must be positive")
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts

    def _attempt(self, operation: Callable[[], T]) -> T:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return operation()
            except Exception as error:  # nba_api exposes requests exceptions inconsistently.
                last_error = error
                if attempt < self._max_attempts:
                    sleep(0.25 * attempt)
        raise ProviderError(
            f"NBA.com data request failed after {self._max_attempts} attempts",
            code="nba_data_request_failed",
            retryable=True,
            attempts=self._max_attempts,
        ) from last_error

    def _rows(self, build: Callable[[], Any], dataset: str) -> list[dict[str, Any]]:
        def fetch() -> list[dict[str, Any]]:
            payload = cast(dict[str, list[dict[str, Any]]], build().get_normalized_dict())
            return payload.get(dataset, [])

        return self._attempt(fetch)

    def get_teams(self) -> list[NBATeam]:
        from nba_api.stats.static import teams

        retrieved_at = datetime.now(UTC)
        rows = self._attempt(teams.get_teams)
        return [
            NBATeam(
                id=stable_nba_id("team", str(row["id"])),
                external_id=str(row["id"]),
                name=str(row["nickname"]),
                full_name=str(row["full_name"]),
                abbreviation=str(row["abbreviation"]),
                city=str(row["city"]),
                state=str(row["state"]),
                year_founded=int(row["year_founded"]),
                retrieved_at=retrieved_at,
            )
            for row in rows
        ]

    def get_player_directory(self, season: str) -> list[NBAPlayer]:
        from nba_api.stats.endpoints import leaguedashplayerstats

        rows = self._rows(
            lambda: leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                timeout=self._timeout_seconds,
            ),
            "LeagueDashPlayerStats",
        )
        return [self._player_from_stats(row) for row in rows]

    def get_roster(self, team_external_id: str, season: str) -> list[NBAPlayer]:
        from nba_api.stats.endpoints import commonteamroster

        rows = self._rows(
            lambda: commonteamroster.CommonTeamRoster(
                team_id=int(team_external_id),
                season=season,
                timeout=self._timeout_seconds,
            ),
            "CommonTeamRoster",
        )
        team_id = stable_nba_id("team", team_external_id)
        return [self._player_from_roster(row, team_id) for row in rows]

    def get_player(self, player_external_id: str) -> NBAPlayer:
        from nba_api.stats.endpoints import commonplayerinfo

        rows = self._rows(
            lambda: commonplayerinfo.CommonPlayerInfo(
                player_id=int(player_external_id), timeout=self._timeout_seconds
            ),
            "CommonPlayerInfo",
        )
        if not rows:
            raise ProviderError(
                "NBA.com returned no player record",
                code="player_not_found",
                retryable=False,
                attempts=1,
            )
        row = rows[0]
        team_external_id = _optional_string(row.get("TEAM_ID"))
        return NBAPlayer(
            id=stable_nba_id("player", player_external_id),
            external_id=player_external_id,
            team_id=(stable_nba_id("team", team_external_id) if team_external_id else None),
            first_name=str(row.get("FIRST_NAME") or ""),
            last_name=str(row.get("LAST_NAME") or ""),
            display_name=str(row.get("DISPLAY_FIRST_LAST") or player_external_id),
            position=_optional_string(row.get("POSITION")),
            height=_optional_string(row.get("HEIGHT")),
            jersey_number=_jersey_number(row.get("JERSEY")),
            active=str(row.get("ROSTERSTATUS", "Active")).lower() == "active",
        )

    def get_games(
        self,
        *,
        game_date: date | None = None,
        season: str | None = None,
        team_external_id: str | None = None,
    ) -> list[NBAGame]:
        from nba_api.stats.endpoints import leaguegamefinder

        rows = self._rows(
            lambda: leaguegamefinder.LeagueGameFinder(
                date_from_nullable=game_date.strftime("%m/%d/%Y") if game_date else "",
                date_to_nullable=game_date.strftime("%m/%d/%Y") if game_date else "",
                season_nullable=season or "",
                team_id_nullable=team_external_id or "",
                timeout=self._timeout_seconds,
            ),
            "LeagueGameFinderResults",
        )
        games: dict[str, NBAGame] = {}
        for row in rows:
            external_id = str(row["GAME_ID"])
            matchup = str(row.get("MATCHUP") or "")
            row_team = stable_nba_id("team", str(row["TEAM_ID"]))
            existing = games.get(external_id)
            games[external_id] = NBAGame(
                id=stable_nba_id("game", external_id),
                external_id=external_id,
                season=season or str(row.get("SEASON_ID") or "unknown"),
                game_date=datetime.strptime(str(row["GAME_DATE"]), "%Y-%m-%d").date(),
                home_team_id=(
                    row_team if " vs. " in matchup else existing.home_team_id if existing else None
                ),
                away_team_id=(
                    row_team if " @ " in matchup else existing.away_team_id if existing else None
                ),
                status="final" if row.get("WL") else None,
            )
        return list(games.values())

    def get_box_score(self, game_external_id: str) -> list[PlayerGameStats]:
        from nba_api.stats.endpoints import boxscoretraditionalv2

        rows = self._rows(
            lambda: boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=game_external_id, timeout=self._timeout_seconds
            ),
            "PlayerStats",
        )
        return [
            PlayerGameStats(
                game_id=stable_nba_id("game", game_external_id),
                player_id=stable_nba_id("player", str(row["PLAYER_ID"])),
                team_id=stable_nba_id("team", str(row["TEAM_ID"])),
                minutes=_optional_string(row.get("MIN")),
                points=_optional_int(row.get("PTS")),
                rebounds=_optional_int(row.get("REB")),
                assists=_optional_int(row.get("AST")),
                field_goals_made=_optional_int(row.get("FGM")),
                field_goals_attempted=_optional_int(row.get("FGA")),
            )
            for row in rows
        ]

    def get_play_by_play(self, game_external_id: str) -> list[PlayByPlayEvent]:
        from nba_api.stats.endpoints import playbyplayv3

        rows = self._rows(
            lambda: playbyplayv3.PlayByPlayV3(
                game_id=game_external_id, timeout=self._timeout_seconds
            ),
            "PlayByPlay",
        )
        return [
            PlayByPlayEvent(
                game_external_id=game_external_id,
                action_number=int(row.get("actionNumber") or row.get("ACTIONNUMBER") or index),
                period=_optional_int(row.get("period") or row.get("PERIOD")),
                clock=_optional_string(row.get("clock") or row.get("PCTIMESTRING")),
                description=_optional_string(
                    row.get("description")
                    or row.get("HOMEDESCRIPTION")
                    or row.get("VISITORDESCRIPTION")
                ),
                raw=row,
            )
            for index, row in enumerate(rows)
        ]

    def get_shot_chart(self, *, player_external_id: str, season: str) -> list[ShotAttempt]:
        from nba_api.stats.endpoints import shotchartdetail

        rows = self._rows(
            lambda: shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=int(player_external_id),
                season_nullable=season,
                context_measure_simple="FGA",
                timeout=self._timeout_seconds,
            ),
            "Shot_Chart_Detail",
        )
        return [
            ShotAttempt(
                player_external_id=player_external_id,
                game_external_id=_optional_string(row.get("GAME_ID")),
                x=_optional_float(row.get("LOC_X")),
                y=_optional_float(row.get("LOC_Y")),
                made=(
                    bool(row["SHOT_MADE_FLAG"]) if row.get("SHOT_MADE_FLAG") is not None else None
                ),
                raw=row,
            )
            for row in rows
        ]

    def get_player_stats(self, season: str) -> list[NBAPlayer]:
        return self.get_player_directory(season)

    def get_preferred_lineup(
        self,
        team_external_id: str,
        season: str,
    ) -> list[PreferredLineupCandidate]:
        from nba_api.stats.endpoints import (
            boxscoretraditionalv2,
            leaguedashplayerstats,
            leaguegamefinder,
        )

        current_roster = self.get_roster(team_external_id, season)
        roster_by_external_id = {player.external_id: player for player in current_roster}
        starter_rows: list[dict[str, Any]] = []

        game_rows = self._rows(
            lambda: leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                team_id_nullable=team_external_id,
                timeout=self._timeout_seconds,
            ),
            "LeagueGameFinderResults",
        )
        completed_games = [row for row in game_rows if row.get("GAME_ID") and row.get("WL")]
        if completed_games:
            latest = max(completed_games, key=lambda row: str(row.get("GAME_DATE") or ""))
            game_external_id = str(latest["GAME_ID"])
            box_rows = self._rows(
                lambda: boxscoretraditionalv2.BoxScoreTraditionalV2(
                    game_id=game_external_id,
                    timeout=self._timeout_seconds,
                ),
                "PlayerStats",
            )
            starter_rows = [
                row
                for row in box_rows
                if str(row.get("TEAM_ID")) == team_external_id
                and str(row.get("PLAYER_ID")) in roster_by_external_id
                and _optional_string(row.get("START_POSITION")) is not None
            ]
        rows = self._rows(
            lambda: leaguedashplayerstats.LeagueDashPlayerStats(
                team_id_nullable=team_external_id,
                season=season,
                per_mode_detailed="Totals",
                timeout=self._timeout_seconds,
            ),
            "LeagueDashPlayerStats",
        )
        eligible_rows = [
            {**row, "_PROFILE_SEASON": season}
            for row in rows
            if str(row.get("PLAYER_ID")) in roster_by_external_id
            and (
                (_optional_int(row.get("GS")) or 0) > 0
                or (_optional_float(row.get("MIN")) or 0) > 0
            )
        ]
        if len(eligible_rows) < 5:
            previous_rows = self._rows(
                lambda: leaguedashplayerstats.LeagueDashPlayerStats(
                    season=_previous_season(season),
                    per_mode_detailed="Totals",
                    timeout=self._timeout_seconds,
                ),
                "LeagueDashPlayerStats",
            )
            seen = {str(row.get("PLAYER_ID")) for row in eligible_rows}
            eligible_rows.extend(
                {**row, "_PROFILE_SEASON": _previous_season(season)}
                for row in previous_rows
                if str(row.get("PLAYER_ID")) in roster_by_external_id
                and str(row.get("PLAYER_ID")) not in seen
            )
        profile_rows = {
            str(row.get("PLAYER_ID")): row for row in eligible_rows
        }
        if len(starter_rows) == 5:
            return [
                PreferredLineupCandidate(
                    player=_with_shooting_profile(
                        roster_by_external_id[str(row["PLAYER_ID"])],
                        profile_rows.get(str(row["PLAYER_ID"])),
                    ),
                    games_started=1,
                    total_minutes=_optional_float(row.get("MIN")) or 0,
                )
                for row in starter_rows
            ]

        ranked = sorted(
            eligible_rows,
            key=lambda row: (
                _optional_int(row.get("GS")) or 0,
                _optional_float(row.get("MIN")) or 0,
            ),
            reverse=True,
        )
        candidates = [
            PreferredLineupCandidate(
                player=_with_shooting_profile(
                    roster_by_external_id[str(row["PLAYER_ID"])], row
                ),
                games_started=_optional_int(row.get("GS")) or 0,
                total_minutes=_optional_float(row.get("MIN")) or 0,
            )
            for row in ranked[:5]
        ]
        selected_ids = {candidate.player.external_id for candidate in candidates}
        candidates.extend(
            PreferredLineupCandidate(player=player)
            for player in current_roster
            if player.external_id not in selected_ids
        )
        return candidates[:5]

    def get_team_stats(self, season: str) -> list[TeamStats]:
        from nba_api.stats.endpoints import leaguedashteamstats

        rows = self._rows(
            lambda: leaguedashteamstats.LeagueDashTeamStats(
                season=season, timeout=self._timeout_seconds
            ),
            "LeagueDashTeamStats",
        )
        return [
            TeamStats(
                team_id=stable_nba_id("team", str(row["TEAM_ID"])),
                season=season,
                values={str(key): value for key, value in row.items()},
            )
            for row in rows
        ]

    @staticmethod
    def _player_from_roster(row: dict[str, Any], team_id: Any) -> NBAPlayer:
        external_id = str(row["PLAYER_ID"])
        display_name = str(row.get("PLAYER") or external_id)
        first_name, _, last_name = display_name.partition(" ")
        return NBAPlayer(
            id=stable_nba_id("player", external_id),
            external_id=external_id,
            team_id=team_id,
            first_name=first_name,
            last_name=last_name,
            display_name=display_name,
            position=_optional_string(row.get("POSITION")),
            height=_optional_string(row.get("HEIGHT")),
            jersey_number=_jersey_number(row.get("NUM")),
        )

    @staticmethod
    def _player_from_stats(row: dict[str, Any]) -> NBAPlayer:
        external_id = str(row["PLAYER_ID"])
        display_name = str(row.get("PLAYER_NAME") or external_id)
        first_name, _, last_name = display_name.partition(" ")
        team_external_id = _optional_string(row.get("TEAM_ID"))
        return NBAPlayer(
            id=stable_nba_id("player", external_id),
            external_id=external_id,
            team_id=(stable_nba_id("team", team_external_id) if team_external_id else None),
            first_name=first_name,
            last_name=last_name,
            display_name=display_name,
        )

    @staticmethod
    def _player_from_box_score(row: dict[str, Any]) -> NBAPlayer:
        external_id = str(row["PLAYER_ID"])
        display_name = str(row.get("PLAYER_NAME") or external_id)
        first_name, _, last_name = display_name.partition(" ")
        team_external_id = _optional_string(row.get("TEAM_ID"))
        return NBAPlayer(
            id=stable_nba_id("player", external_id),
            external_id=external_id,
            team_id=(stable_nba_id("team", team_external_id) if team_external_id else None),
            first_name=first_name,
            last_name=last_name,
            display_name=display_name,
            position=_optional_string(row.get("START_POSITION")),
        )


def _optional_string(value: Any) -> str | None:
    if value is None or str(value).strip() in {"", "0", "None"}:
        return None
    return str(value).strip()


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _jersey_number(value: Any) -> int | None:
    number = _optional_int(value)
    return number if number is not None and 0 <= number <= 99 else None


def _with_shooting_profile(
    player: NBAPlayer,
    row: dict[str, Any] | None,
) -> NBAPlayer:
    if row is None:
        return player
    field_goal_attempts = _optional_int(row.get("FGA")) or 0
    three_point_attempts = _optional_int(row.get("FG3A")) or 0
    if field_goal_attempts <= 0:
        return player
    field_goals_made = _optional_int(row.get("FGM")) or 0
    three_point_made = _optional_int(row.get("FG3M")) or 0
    two_point_attempts = max(field_goal_attempts - three_point_attempts, 0)
    two_point_percentage = (
        max(field_goals_made - three_point_made, 0) / two_point_attempts
        if two_point_attempts > 0
        else None
    )
    profile = PlayerShootingProfile(
        season=str(row.get("_PROFILE_SEASON") or "unknown"),
        games_played=_optional_int(row.get("GP")) or 0,
        field_goal_attempts=field_goal_attempts,
        three_point_attempts=three_point_attempts,
        two_point_percentage=two_point_percentage,
        three_point_percentage=_optional_float(row.get("FG3_PCT")),
        free_throw_percentage=_optional_float(row.get("FT_PCT")),
    )
    return player.model_copy(update={"shooting_profile": profile})


def _previous_season(season: str) -> str:
    start_year = int(season[:4]) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"
