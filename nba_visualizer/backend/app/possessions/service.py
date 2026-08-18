from datetime import UTC, datetime
from uuid import UUID

from app.persistence.base import NBADataRepository, PossessionRepository
from app.providers.base import ProviderError
from app.providers.possessions import PossessionDataProvider
from app.providers.service import DataUnavailableError
from app.schemas.domain import Play
from app.schemas.possessions import (
    PossessionListResponse,
    PossessionReconstruction,
    RealPossession,
)


class PossessionService:
    def __init__(
        self,
        nba_repository: NBADataRepository,
        possession_repository: PossessionRepository,
        provider: PossessionDataProvider,
    ) -> None:
        self._nba_repository = nba_repository
        self._possession_repository = possession_repository
        self._provider = provider

    def list_for_game(self, game_id: UUID, *, refresh: bool = False) -> PossessionListResponse:
        game = self._nba_repository.get_game(game_id)
        if game is None:
            raise DataUnavailableError("NBA game")
        cached = self._possession_repository.list_possessions(game_id)
        if cached and not refresh:
            return self._response(game_id, cached, "cache")
        try:
            possessions = self._provider.get_possessions(game)
            self._possession_repository.save_possessions(possessions)
            return self._response(game_id, possessions, "refreshed")
        except ProviderError as error:
            if cached:
                return self._response(game_id, cached, "cache_fallback")
            raise DataUnavailableError(f"possessions for game {game.external_id}", error) from error

    def get(self, possession_id: UUID) -> RealPossession:
        possession = self._possession_repository.get_possession(possession_id)
        if possession is None:
            raise DataUnavailableError("real possession")
        return possession

    def save_reconstruction(
        self,
        possession_id: UUID,
        play: Play,
    ) -> PossessionReconstruction:
        self.get(possession_id)
        metadata = {
            **play.initial_frame.metadata,
            "realPossessionId": str(possession_id),
            "reconstructionOrigin": "manual_reconstruction",
            "historicalMovementAvailable": False,
        }
        initial_frame = play.initial_frame.model_copy(
            deep=True,
            update={"metadata": metadata},
        )
        play = play.model_copy(deep=True, update={"initial_frame": initial_frame})
        now = datetime.now(UTC)
        reconstruction = PossessionReconstruction(
            possession_id=possession_id,
            play=play,
            created_at=now,
            updated_at=now,
        )
        return self._possession_repository.save_reconstruction(reconstruction)

    def list_reconstructions(self, possession_id: UUID) -> list[PossessionReconstruction]:
        self.get(possession_id)
        return self._possession_repository.list_reconstructions(possession_id)

    @staticmethod
    def _response(
        game_id: UUID,
        possessions: list[RealPossession],
        cache_status: str,
    ) -> PossessionListResponse:
        return PossessionListResponse(
            game_id=game_id,
            possessions=possessions,
            cache_status=cache_status,
            retrieved_at=max(
                (item.provenance.retrieved_at for item in possessions),
                default=None,
            ),
        )
