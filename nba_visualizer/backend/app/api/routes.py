from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.dependencies import get_play_repository
from app.persistence.base import PlayRepository
from app.schemas.domain import Play, SimulationFrame
from app.simulation.example import build_example_frame

router = APIRouter(prefix="/api/v1")
RepositoryDependency = Annotated[PlayRepository, Depends(get_play_repository)]


class HealthResponse(BaseModel):
    status: str
    service: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="nba-play-lab-api")


@router.get("/simulation/example", response_model=SimulationFrame, response_model_by_alias=True)
def example_simulation_frame() -> SimulationFrame:
    return build_example_frame()


@router.get("/plays", response_model=list[Play], response_model_by_alias=True)
def list_plays(repository: RepositoryDependency) -> list[Play]:
    return repository.list()


@router.post(
    "/plays",
    response_model=Play,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
def save_play(play: Play, repository: RepositoryDependency) -> Play:
    return repository.save(play)


@router.post(
    "/plays/{play_id}/duplicate",
    response_model=Play,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
def duplicate_play(play_id: UUID, repository: RepositoryDependency) -> Play:
    play = repository.get(play_id)
    if play is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Play not found")
    now = datetime.now(UTC)
    duplicate = play.model_copy(
        deep=True,
        update={
            "id": uuid4(),
            "name": f"{play.name} copy",
            "created_at": now,
            "updated_at": now,
        },
    )
    return repository.save(duplicate)


@router.delete("/plays/{play_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_play(play_id: UUID, repository: RepositoryDependency) -> None:
    if not repository.delete(play_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Play not found")


@router.get("/plays/{play_id}", response_model=Play, response_model_by_alias=True)
def get_play(play_id: UUID, repository: RepositoryDependency) -> Play:
    play = repository.get(play_id)
    if play is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Play not found")
    return play
