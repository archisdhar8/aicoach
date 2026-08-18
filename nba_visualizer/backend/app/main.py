from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.data_routes import router as data_router
from app.api.dependencies import get_nba_data_repository, get_play_repository
from app.api.possession_routes import router as possession_router
from app.api.routes import router
from app.config import settings
from app.persistence.sqlite import SQLitePlayRepository
from app.simulation.seeds import build_seed_plays


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    repository = get_play_repository()
    if isinstance(repository, SQLitePlayRepository):
        repository.initialize()
        for play in build_seed_plays():
            repository.save(play)
    get_nba_data_repository().initialize()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(data_router)
app.include_router(possession_router)
