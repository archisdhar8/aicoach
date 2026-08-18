from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.schemas.nba_data import PlayByPlayEvent, stable_nba_id
from app.schemas.possessions import (
    FieldOrigin,
    FieldProvenance,
    PossessionPlayer,
    PossessionProvenance,
    PossessionResult,
    RealPossession,
    RealPossessionEvent,
)

EVENT_FIELDS = tuple(RealPossessionEvent.model_fields)
TOP_LEVEL_FIELDS = (
    "game_id",
    "game_external_id",
    "period",
    "start_clock",
    "end_clock",
    "offense_team_external_id",
    "defense_team_external_id",
    "offensive_lineup",
    "defensive_lineup",
    "events",
    "result",
)


def normalize_pbpstats_possessions(
    game_id: UUID,
    game_external_id: str,
    raw_possessions: Iterable[dict[str, Any]],
    *,
    retrieved_at: datetime | None = None,
) -> list[RealPossession]:
    timestamp = retrieved_at or datetime.now(UTC)
    return [
        _normalize_source_possession(
            game_id,
            game_external_id,
            raw,
            provider="pbpstats",
            retrieved_at=timestamp,
            index=index,
        )
        for index, raw in enumerate(raw_possessions, start=1)
    ]


def normalize_nba_play_by_play(
    game_id: UUID,
    game_external_id: str,
    events: list[PlayByPlayEvent],
    *,
    retrieved_at: datetime | None = None,
) -> list[RealPossession]:
    timestamp = retrieved_at or datetime.now(UTC)
    groups: list[list[PlayByPlayEvent]] = []
    current: list[PlayByPlayEvent] = []
    current_key: str | None = None
    for event in sorted(events, key=lambda item: item.action_number):
        key_value = event.raw.get("possession") or event.raw.get("possessionId")
        key = str(key_value) if key_value is not None else None
        if current and key is not None and current_key is not None and key != current_key:
            groups.append(current)
            current = []
        current.append(event)
        current_key = key or current_key
        if key is None and _is_terminal(event.raw):
            groups.append(current)
            current = []
            current_key = None
    if current:
        groups.append(current)

    raw_possessions = []
    for index, group in enumerate(groups, start=1):
        first = group[0]
        team_id = next(
            (
                _text(item.raw.get("teamId") or item.raw.get("TEAM_ID"))
                for item in group
                if item.raw.get("teamId") or item.raw.get("TEAM_ID")
            ),
            None,
        )
        raw_possessions.append(
            {
                "id": str(first.raw.get("possession") or first.raw.get("possessionId") or index),
                "period": first.period or 1,
                "start_clock": first.clock,
                "end_clock": group[-1].clock,
                "offense_team_id": team_id,
                "events": [
                    {
                        **item.raw,
                        "actionNumber": item.action_number,
                        "period": item.period,
                        "clock": item.clock,
                        "description": item.description,
                    }
                    for item in group
                ],
                "offensive_lineup": [],
                "defensive_lineup": [],
                "lineups_available": False,
            }
        )
    return [
        _normalize_source_possession(
            game_id,
            game_external_id,
            raw,
            provider="nba_api",
            retrieved_at=timestamp,
            index=index,
        )
        for index, raw in enumerate(raw_possessions, start=1)
    ]


def _normalize_source_possession(
    game_id: UUID,
    game_external_id: str,
    raw: dict[str, Any],
    *,
    provider: str,
    retrieved_at: datetime,
    index: int,
) -> RealPossession:
    source_id = str(raw.get("id") or raw.get("possession_id") or index)
    raw_events = list(raw.get("events") or [])
    events = [
        _normalize_event(event, sequence, provider)
        for sequence, event in enumerate(sorted(raw_events, key=_event_sequence), start=1)
    ]
    lineups_available = bool(
        raw.get(
            "lineups_available", bool(raw.get("offensive_lineup") or raw.get("defensive_lineup"))
        )
    )
    offensive_lineup = [_player(value) for value in raw.get("offensive_lineup") or []]
    defensive_lineup = [_player(value) for value in raw.get("defensive_lineup") or []]
    origins = _origins(provider, events, lineups_available)
    return RealPossession(
        id=stable_nba_id("possession", f"{game_external_id}:{source_id}"),
        game_id=game_id,
        game_external_id=game_external_id,
        period=int(raw.get("period") or (events[0].period if events else None) or 1),
        start_clock=_optional_text(raw.get("start_clock") or (events[0].clock if events else None)),
        end_clock=_optional_text(raw.get("end_clock") or (events[-1].clock if events else None)),
        offense_team_external_id=_optional_text(raw.get("offense_team_id")),
        defense_team_external_id=_optional_text(raw.get("defense_team_id")),
        offensive_lineup=offensive_lineup,
        defensive_lineup=defensive_lineup,
        events=events,
        result=_result(events),
        provenance=PossessionProvenance(
            provider=provider,
            source_game_id=game_external_id,
            source_possession_id=source_id,
            retrieved_at=retrieved_at,
            movement_available=False,
            field_origins=origins,
            raw_reference={"event_count": len(raw_events)},
        ),
    )


def _normalize_event(raw: dict[str, Any], sequence: int, provider: str) -> RealPossessionEvent:
    action_type = _text(raw.get("actionType") or raw.get("event_type") or raw.get("EVENTMSGTYPE"))
    subtype = _optional_text(raw.get("subType") or raw.get("shot_type"))
    description = _optional_text(raw.get("description") or raw.get("DESCRIPTION"))
    shot_result = _optional_text(raw.get("shotResult") or raw.get("shot_result"))
    event_type = _event_type(action_type, description)
    player_id = _optional_text(raw.get("personId") or raw.get("player_id") or raw.get("PLAYER1_ID"))
    assist_id = _optional_text(raw.get("assistPersonId") or raw.get("assist_player_id"))
    shooter_id = player_id if event_type == "shot" else None
    return RealPossessionEvent(
        source_event_id=str(raw.get("actionNumber") or raw.get("event_id") or sequence),
        sequence=sequence,
        period=_optional_int(raw.get("period") or raw.get("PERIOD")),
        clock=_optional_text(raw.get("clock") or raw.get("PCTIMESTRING")),
        event_type=event_type,
        description=description,
        team_external_id=_optional_text(raw.get("teamId") or raw.get("team_id")),
        player_external_id=player_id,
        shooter_external_id=shooter_id,
        passer_external_id=assist_id,
        assist_external_id=assist_id,
        is_turnover=event_type == "turnover",
        is_foul=event_type == "foul",
        is_rebound=event_type == "rebound",
        shot_x=_optional_float(raw.get("xLegacy") or raw.get("shot_x") or raw.get("LOC_X")),
        shot_y=_optional_float(raw.get("yLegacy") or raw.get("shot_y") or raw.get("LOC_Y")),
        shot_type=subtype,
        shot_result=shot_result,
        points=_optional_int(raw.get("points") or raw.get("shot_value")),
    )


def _origins(
    provider: str,
    events: list[RealPossessionEvent],
    lineups_available: bool,
) -> dict[str, FieldProvenance]:
    observed = FieldProvenance(origin=FieldOrigin.OBSERVED, source=provider)
    derived = FieldProvenance(origin=FieldOrigin.DERIVED, source=provider)
    unavailable = FieldProvenance(
        origin=FieldOrigin.UNAVAILABLE,
        source=provider,
        note="Field was not supplied by this public source.",
    )
    origins = {field: observed for field in TOP_LEVEL_FIELDS}
    origins["offense_team_external_id"] = derived
    origins["defense_team_external_id"] = derived
    origins["result"] = derived
    origins["offensive_lineup"] = derived if lineups_available else unavailable
    origins["defensive_lineup"] = derived if lineups_available else unavailable
    for event in events:
        for field in EVENT_FIELDS:
            value = getattr(event, field)
            origin = observed if value is not None else unavailable
            if field == "passer_external_id" and value is not None:
                origin = derived
            origins[f"events.{event.sequence}.{field}"] = origin
    return origins


def _result(events: list[RealPossessionEvent]) -> PossessionResult:
    turnover = next((event for event in reversed(events) if event.is_turnover), None)
    if turnover is not None:
        return PossessionResult(result_type="turnover", turnover=True)
    shot = next((event for event in reversed(events) if event.event_type == "shot"), None)
    if shot is not None:
        made = (shot.shot_result or "").lower() == "made"
        return PossessionResult(
            result_type="made_shot" if made else "missed_shot",
            points=shot.points or (2 if made else 0),
            made=made,
        )
    foul = next((event for event in reversed(events) if event.is_foul), None)
    return PossessionResult(result_type="foul" if foul else "other")


def _player(value: Any) -> PossessionPlayer:
    if isinstance(value, dict):
        external_id = str(value.get("id") or value.get("player_id"))
        team_external_id = _optional_text(value.get("team_id"))
        return PossessionPlayer(
            id=stable_nba_id("player", external_id),
            external_id=external_id,
            display_name=_optional_text(value.get("name") or value.get("display_name")),
            team_id=(stable_nba_id("team", team_external_id) if team_external_id else None),
            team_external_id=team_external_id,
        )
    external_id = str(value)
    return PossessionPlayer(
        id=stable_nba_id("player", external_id),
        external_id=external_id,
    )


def _event_sequence(raw: dict[str, Any]) -> int:
    return _optional_int(raw.get("actionNumber") or raw.get("event_id")) or 0


def _event_type(action_type: str, description: str | None) -> str:
    text = f"{action_type} {description or ''}".lower()
    if "turnover" in text:
        return "turnover"
    if "rebound" in text:
        return "rebound"
    if "foul" in text:
        return "foul"
    if any(value in text for value in ("2pt", "3pt", "shot", "jump", "layup", "dunk")):
        return "shot"
    if "pass" in text:
        return "pass"
    if "substitution" in text:
        return "substitution"
    return "other"


def _is_terminal(raw: dict[str, Any]) -> bool:
    text = str(raw).lower()
    return any(value in text for value in ("turnover", "made", "defensive rebound", "period end"))


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_text(value: Any) -> str | None:
    text = _text(value).strip()
    return text if text and text not in {"0", "None"} else None


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
