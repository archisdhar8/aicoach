from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from app.api.data_routes import unavailable
from app.api.dependencies import get_possession_service
from app.possessions.service import PossessionService
from app.providers.service import DataUnavailableError
from app.schemas.possessions import (
    PossessionListResponse,
    PossessionReconstruction,
    PossessionReconstructionRequest,
    RealPossession,
)

router = APIRouter(prefix="/api/v1", tags=["Real possessions"])
ServiceDependency = Annotated[PossessionService, Depends(get_possession_service)]


@router.get(
    "/nba/games/{game_id}/possessions",
    response_model=PossessionListResponse,
    response_model_by_alias=True,
)
def list_game_possessions(
    game_id: UUID,
    service: ServiceDependency,
    refresh: bool = Query(default=False),
) -> PossessionListResponse:
    try:
        return service.list_for_game(game_id, refresh=refresh)
    except DataUnavailableError as error:
        raise unavailable(error) from error


@router.get(
    "/real-possessions/{possession_id}",
    response_model=RealPossession,
    response_model_by_alias=True,
)
def get_real_possession(
    possession_id: UUID,
    service: ServiceDependency,
) -> RealPossession:
    try:
        return service.get(possession_id)
    except DataUnavailableError as error:
        raise unavailable(error) from error


@router.get(
    "/real-possessions/{possession_id}/reconstructions",
    response_model=list[PossessionReconstruction],
    response_model_by_alias=True,
)
def list_reconstructions(
    possession_id: UUID,
    service: ServiceDependency,
) -> list[PossessionReconstruction]:
    try:
        return service.list_reconstructions(possession_id)
    except DataUnavailableError as error:
        raise unavailable(error) from error


@router.post(
    "/real-possessions/{possession_id}/reconstructions",
    response_model=PossessionReconstruction,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
def save_reconstruction(
    possession_id: UUID,
    request: PossessionReconstructionRequest,
    service: ServiceDependency,
) -> PossessionReconstruction:
    try:
        return service.save_reconstruction(possession_id, request.play)
    except DataUnavailableError as error:
        raise unavailable(error) from error
