from datetime import UTC, date, datetime
from uuid import UUID

from app.persistence.base import NBADataRepository
from app.providers.base import NBADataProvider, ProviderError
from app.schemas.nba_data import (
    BoxScoreResponse,
    GameListResponse,
    PreferredLineupCandidate,
    PreferredLineupResponse,
    RosterResponse,
    TeamListResponse,
)


class DataUnavailableError(RuntimeError):
    def __init__(self, resource: str, provider_error: ProviderError | None = None) -> None:
        super().__init__(f"{resource} is unavailable and no cached data exists")
        self.resource = resource
        self.provider_error = provider_error


class NBADataService:
    """Cache-first application service. The UI never invokes a provider directly."""

    def __init__(self, repository: NBADataRepository, provider: NBADataProvider) -> None:
        self._repository = repository
        self._provider = provider

    def get_teams(self, *, refresh: bool = False) -> TeamListResponse:
        cached = self._repository.list_teams()
        if cached and not refresh:
            return TeamListResponse(
                teams=cached,
                cache_status="cache",
                retrieved_at=max(team.retrieved_at for team in cached),
            )
        try:
            teams = self._provider.get_teams()
            self._repository.save_teams(teams)
            return TeamListResponse(
                teams=teams,
                cache_status="refreshed",
                retrieved_at=max((team.retrieved_at for team in teams), default=None),
            )
        except ProviderError as error:
            if cached:
                return TeamListResponse(
                    teams=cached,
                    cache_status="cache_fallback",
                    retrieved_at=max(team.retrieved_at for team in cached),
                )
            raise DataUnavailableError("NBA teams", error) from error

    def get_roster(self, team_id: UUID, season: str, *, refresh: bool = False) -> RosterResponse:
        team = self._repository.get_team(team_id)
        if team is None:
            raise DataUnavailableError("NBA team")
        cached = self._repository.list_roster(team_id, season)
        retrieved_at = self._repository.roster_retrieved_at(team_id, season)
        if cached and not refresh:
            return RosterResponse(
                team=team,
                season=season,
                players=cached,
                cache_status="cache",
                retrieved_at=retrieved_at,
            )
        try:
            players = self._provider.get_roster(team.external_id, season)
            self._repository.replace_roster(team_id, season, players)
            return RosterResponse(
                team=team,
                season=season,
                players=players,
                cache_status="refreshed",
                retrieved_at=max(
                    (player.retrieved_at for player in players),
                    default=datetime.now(UTC),
                ),
            )
        except ProviderError as error:
            if cached:
                return RosterResponse(
                    team=team,
                    season=season,
                    players=cached,
                    cache_status="cache_fallback",
                    retrieved_at=retrieved_at,
                )
            raise DataUnavailableError(f"{team.full_name} {season} roster", error) from error

    def preload(self, season: str) -> tuple[int, int]:
        team_result = self.get_teams(refresh=True)
        directory = self._provider.get_player_directory(season)
        self._repository.save_players(directory)
        roster_players = 0
        for team in team_result.teams:
            roster_players += len(self.get_roster(team.id, season, refresh=True).players)
        return len(directory), roster_players

    def get_preferred_lineup(
        self,
        team_id: UUID,
        season: str,
        *,
        refresh: bool = False,
    ) -> PreferredLineupResponse:
        team = self._repository.get_team(team_id)
        if team is None:
            raise DataUnavailableError("NBA team")
        cached = self._repository.list_preferred_lineup(team_id, season)
        retrieved_at = self._repository.preferred_lineup_retrieved_at(team_id, season)
        if len(cached) == 5 and not refresh:
            return PreferredLineupResponse(
                team=team,
                season=season,
                players=[candidate.player for candidate in cached],
                cache_status="cache",
                retrieved_at=retrieved_at,
            )
        try:
            candidates = self._provider.get_preferred_lineup(team.external_id, season)
            if len(candidates) != 5:
                raise ProviderError(
                    "player stats did not provide five lineup candidates",
                    code="preferred_lineup_incomplete",
                    retryable=False,
                    attempts=1,
                )
            roster_by_id = {
                player.id: player for player in self._repository.list_roster(team_id, season)
            }
            enriched = [
                PreferredLineupCandidate(
                    player=(
                        roster_by_id[candidate.player.id].model_copy(
                            update={"shooting_profile": candidate.player.shooting_profile}
                        )
                        if candidate.player.id in roster_by_id
                        else candidate.player
                    ),
                    games_started=candidate.games_started,
                    total_minutes=candidate.total_minutes,
                )
                for candidate in candidates
            ]
            self._repository.replace_preferred_lineup(team_id, season, enriched)
            return PreferredLineupResponse(
                team=team,
                season=season,
                players=[candidate.player for candidate in enriched],
                cache_status="refreshed",
                retrieved_at=max(
                    (candidate.player.retrieved_at for candidate in enriched),
                    default=datetime.now(UTC),
                ),
            )
        except ProviderError as error:
            if len(cached) == 5:
                return PreferredLineupResponse(
                    team=team,
                    season=season,
                    players=[candidate.player for candidate in cached],
                    cache_status="cache_fallback",
                    retrieved_at=retrieved_at,
                )
            raise DataUnavailableError(
                f"typical {team.full_name} lineup for {season}", error
            ) from error

    def get_games(
        self,
        *,
        game_date: date | None = None,
        season: str | None = None,
        team_id: UUID | None = None,
        refresh: bool = False,
    ) -> GameListResponse:
        cached = self._repository.list_games(game_date=game_date, season=season, team_id=team_id)
        if cached and not refresh:
            return GameListResponse(
                games=cached,
                cache_status="cache",
                retrieved_at=max(game.retrieved_at for game in cached),
            )
        team_external_id: str | None = None
        if team_id is not None:
            team = self._repository.get_team(team_id)
            if team is None:
                raise DataUnavailableError("NBA team")
            team_external_id = team.external_id
        try:
            games = self._provider.get_games(
                game_date=game_date,
                season=season,
                team_external_id=team_external_id,
            )
            self._repository.save_games(games)
            return GameListResponse(
                games=games,
                cache_status="refreshed",
                retrieved_at=max((game.retrieved_at for game in games), default=None),
            )
        except ProviderError as error:
            if cached:
                return GameListResponse(
                    games=cached,
                    cache_status="cache_fallback",
                    retrieved_at=max(game.retrieved_at for game in cached),
                )
            raise DataUnavailableError("NBA games", error) from error

    def get_box_score(self, game_id: UUID, *, refresh: bool = False) -> BoxScoreResponse:
        game = self._repository.get_game(game_id)
        if game is None:
            raise DataUnavailableError("NBA game")
        cached = self._repository.list_player_game_stats(game_id)
        if cached and not refresh:
            return BoxScoreResponse(
                game=game,
                player_stats=cached,
                cache_status="cache",
                retrieved_at=max(item.retrieved_at for item in cached),
            )
        try:
            stats = self._provider.get_box_score(game.external_id)
            self._repository.save_player_game_stats(stats)
            return BoxScoreResponse(
                game=game,
                player_stats=stats,
                cache_status="refreshed",
                retrieved_at=max((item.retrieved_at for item in stats), default=None),
            )
        except ProviderError as error:
            if cached:
                return BoxScoreResponse(
                    game=game,
                    player_stats=cached,
                    cache_status="cache_fallback",
                    retrieved_at=max(item.retrieved_at for item in cached),
                )
            raise DataUnavailableError(f"box score for {game.external_id}", error) from error
