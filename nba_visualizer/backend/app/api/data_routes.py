from datetime import date
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_nba_data_service
from app.providers.service import DataUnavailableError, NBADataService
from app.schemas.nba_data import (
    BoxScoreResponse,
    GameListResponse,
    PreferredLineupResponse,
    RosterResponse,
    TeamListResponse,
)

router = APIRouter(prefix="/api/v1/nba", tags=["NBA data"])
ServiceDependency = Annotated[NBADataService, Depends(get_nba_data_service)]


def unavailable(error: DataUnavailableError) -> HTTPException:
    provider = error.provider_error
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": provider.code if provider else "nba_data_unavailable",
            "message": str(error),
            "retryable": provider.retryable if provider else False,
            "attempts": provider.attempts if provider else 0,
        },
    )


@router.get("/teams", response_model=TeamListResponse, response_model_by_alias=True)
def list_nba_teams(
    service: ServiceDependency,
    refresh: bool = Query(default=False),
) -> TeamListResponse:
    try:
        return service.get_teams(refresh=refresh)
    except DataUnavailableError as error:
        raise unavailable(error) from error


@router.get(
    "/teams/{team_id}/roster",
    response_model=RosterResponse,
    response_model_by_alias=True,
)
def get_nba_roster(
    team_id: UUID,
    service: ServiceDependency,
    season: str = Query(default="2025-26", pattern=r"^\d{4}-\d{2}$"),
    refresh: bool = Query(default=False),
) -> RosterResponse:
    try:
        return service.get_roster(team_id, season, refresh=refresh)
    except DataUnavailableError as error:
        raise unavailable(error) from error


@router.get(
    "/teams/{team_id}/preferred-lineup",
    response_model=PreferredLineupResponse,
    response_model_by_alias=True,
)
def get_preferred_lineup(
    team_id: UUID,
    service: ServiceDependency,
    season: str = Query(default="2025-26", pattern=r"^\d{4}-\d{2}$"),
    refresh: bool = Query(default=False),
) -> PreferredLineupResponse:
    try:
        return service.get_preferred_lineup(team_id, season, refresh=refresh)
    except DataUnavailableError as error:
        raise unavailable(error) from error


@router.get("/games", response_model=GameListResponse, response_model_by_alias=True)
def get_nba_games(
    service: ServiceDependency,
    game_date: Annotated[date | None, Query()] = None,
    season: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}$")] = None,
    team_id: Annotated[UUID | None, Query()] = None,
    refresh: bool = Query(default=False),
) -> GameListResponse:
    try:
        return service.get_games(
            game_date=game_date,
            season=season,
            team_id=team_id,
            refresh=refresh,
        )
    except DataUnavailableError as error:
        raise unavailable(error) from error


@router.get(
    "/games/{game_id}/box-score",
    response_model=BoxScoreResponse,
    response_model_by_alias=True,
)
def get_nba_box_score(
    game_id: UUID,
    service: ServiceDependency,
    refresh: bool = Query(default=False),
) -> BoxScoreResponse:
    try:
        return service.get_box_score(game_id, refresh=refresh)
    except DataUnavailableError as error:
        raise unavailable(error) from error
