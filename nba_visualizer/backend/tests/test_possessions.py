import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from app.persistence.nba_data import SQLiteNBADataRepository
from app.possessions.normalization import normalize_pbpstats_possessions
from app.possessions.service import PossessionService
from app.providers.base import ProviderError
from app.schemas.nba_data import NBAGame, stable_nba_id
from app.schemas.possessions import FieldOrigin
from app.simulation.seeds import build_seed_plays

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pbpstats_possessions.json"
GAME_ID = stable_nba_id("game", "0022500001")
RETRIEVED_AT = datetime(2026, 8, 17, tzinfo=UTC)


def raw_possessions() -> list[dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text())


def normalized():
    return normalize_pbpstats_possessions(
        GAME_ID,
        "0022500001",
        raw_possessions(),
        retrieved_at=RETRIEVED_AT,
    )


def game() -> NBAGame:
    return NBAGame(
        id=GAME_ID,
        external_id="0022500001",
        season="2025-26",
        game_date=date(2025, 10, 21),
        retrieved_at=RETRIEVED_AT,
    )


def test_possession_normalization_orders_events_and_parses_lineups() -> None:
    possession = normalized()[0]

    assert [event.source_event_id for event in possession.events] == ["101", "102"]
    assert len(possession.offensive_lineup) == 5
    assert possession.offensive_lineup[0].display_name == "Stephen Curry"
    assert possession.offensive_lineup[0].team_id == stable_nba_id("team", "1610612744")
    assert possession.result.result_type == "made_shot"
    assert possession.result.points == 3
    assert possession.events[-1].shot_x == -8.0


def test_provenance_distinguishes_observed_derived_and_unavailable() -> None:
    first, missing = normalized()

    assert first.provenance.movement_available is False
    assert first.provenance.field_origins["events.2.shot_x"].origin == FieldOrigin.OBSERVED
    assert (
        first.provenance.field_origins["events.2.passer_external_id"].origin == FieldOrigin.DERIVED
    )
    assert first.provenance.field_origins["result"].origin == FieldOrigin.DERIVED
    assert missing.provenance.field_origins["offensive_lineup"].origin == FieldOrigin.UNAVAILABLE
    assert missing.offensive_lineup == []
    assert missing.result.turnover


def test_source_snapshots_and_user_reconstruction_storage_are_separate(
    tmp_path: Path,
) -> None:
    repository = SQLiteNBADataRepository(tmp_path / "possessions.sqlite3")
    repository.initialize()
    repository.save_games([game()])
    source = normalized()
    repository.save_possessions(source)

    before = repository.get_possession(source[0].id)
    assert before == source[0]
    service = PossessionService(repository, repository, FailingProvider())  # type: ignore[arg-type]
    saved = service.save_reconstruction(source[0].id, build_seed_plays()[0])

    reconstructions = repository.list_reconstructions(source[0].id)
    assert reconstructions == [saved]
    assert saved.play.initial_frame.metadata["reconstructionOrigin"] == ("manual_reconstruction")
    assert saved.play.initial_frame.metadata["historicalMovementAvailable"] is False
    assert repository.get_possession(source[0].id) == before


class FailingProvider:
    def get_possessions(self, _: NBAGame):
        raise ProviderError("timeout", code="timeout", retryable=True, attempts=2)


def test_provider_failure_uses_cached_fixture_possessions(tmp_path: Path) -> None:
    repository = SQLiteNBADataRepository(tmp_path / "fallback.sqlite3")
    repository.initialize()
    repository.save_games([game()])
    repository.save_possessions(normalized())
    service = PossessionService(repository, repository, FailingProvider())  # type: ignore[arg-type]

    response = service.list_for_game(GAME_ID, refresh=True)

    assert response.cache_status == "cache_fallback"
    assert len(response.possessions) == 2
