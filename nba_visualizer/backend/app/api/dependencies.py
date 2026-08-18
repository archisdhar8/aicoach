from functools import lru_cache

from app.config import settings
from app.persistence.base import PlayRepository
from app.persistence.nba_data import SQLiteNBADataRepository
from app.persistence.sqlite import SQLitePlayRepository
from app.possessions.service import PossessionService
from app.providers.nba_api import NBAApiProvider
from app.providers.possessions import (
    FallbackPossessionProvider,
    NBAApiPossessionProvider,
    PbpStatsPossessionProvider,
)
from app.providers.service import NBADataService


@lru_cache
def get_play_repository() -> PlayRepository:
    return SQLitePlayRepository(settings.database_path)


@lru_cache
def get_nba_data_repository() -> SQLiteNBADataRepository:
    return SQLiteNBADataRepository(settings.database_path)


@lru_cache
def get_nba_provider() -> NBAApiProvider:
    return NBAApiProvider()


@lru_cache
def get_nba_data_service() -> NBADataService:
    return NBADataService(get_nba_data_repository(), get_nba_provider())


@lru_cache
def get_possession_service() -> PossessionService:
    repository = get_nba_data_repository()
    provider = FallbackPossessionProvider(
        PbpStatsPossessionProvider(),
        NBAApiPossessionProvider(get_nba_provider()),
    )
    return PossessionService(repository, repository, provider)
