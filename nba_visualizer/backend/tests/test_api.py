from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_nba_data_service, get_play_repository
from app.main import app
from app.persistence.sqlite import SQLitePlayRepository
from app.schemas.nba_data import TeamListResponse
from app.simulation.example import build_example_frame


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "nba-play-lab-api"}


def test_example_frame_contains_five_on_five_state() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/simulation/example")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["players"]) == 10
    assert {player["teamSide"] for player in payload["players"]} == {"offense", "defense"}
    assert payload["metadata"] == {"source": "deterministic_example", "predictive": False}


def test_play_create_load_duplicate_and_delete(tmp_path: Path) -> None:
    repository = SQLitePlayRepository(tmp_path / "api-plays.sqlite3")
    repository.initialize()
    app.dependency_overrides[get_play_repository] = lambda: repository
    payload = {
        "name": "API horns",
        "initialFrame": build_example_frame().model_dump(mode="json", by_alias=True),
        "routes": [],
        "actions": [],
    }
    try:
        with TestClient(app) as client:
            created = client.post("/api/v1/plays", json=payload)
            assert created.status_code == 201
            play_id = created.json()["id"]

            loaded = client.get(f"/api/v1/plays/{play_id}")
            assert loaded.status_code == 200
            assert loaded.json()["name"] == "API horns"

            duplicated = client.post(f"/api/v1/plays/{play_id}/duplicate")
            assert duplicated.status_code == 201
            assert duplicated.json()["id"] != play_id
            assert duplicated.json()["name"] == "API horns copy"

            deleted = client.delete(f"/api/v1/plays/{play_id}")
            assert deleted.status_code == 204
            assert client.get(f"/api/v1/plays/{play_id}").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_nba_team_endpoint_uses_application_service() -> None:
    class FakeDataService:
        def get_teams(self, *, refresh: bool = False) -> TeamListResponse:
            assert not refresh
            return TeamListResponse(teams=[], cache_status="cache", retrieved_at=None)

    app.dependency_overrides[get_nba_data_service] = FakeDataService
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/nba/teams")
        assert response.status_code == 200
        assert response.json() == {
            "teams": [],
            "cacheStatus": "cache",
            "retrievedAt": None,
        }
    finally:
        app.dependency_overrides.clear()
