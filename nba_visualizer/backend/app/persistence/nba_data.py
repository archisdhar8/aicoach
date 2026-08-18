import sqlite3
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from uuid import UUID

from app.schemas.domain import Play
from app.schemas.nba_data import (
    DataSource,
    NBAGame,
    NBAPlayer,
    NBATeam,
    PlayerGameStats,
    PlayerShootingProfile,
    PreferredLineupCandidate,
    stable_nba_id,
)
from app.schemas.possessions import PossessionReconstruction, RealPossession


class SQLiteNBADataRepository:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS nba_teams (
                    id TEXT PRIMARY KEY, external_id TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
                    full_name TEXT NOT NULL, abbreviation TEXT NOT NULL, city TEXT, state TEXT,
                    year_founded INTEGER, source TEXT NOT NULL, retrieved_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nba_players (
                    id TEXT PRIMARY KEY, external_id TEXT UNIQUE NOT NULL, team_id TEXT,
                    first_name TEXT NOT NULL, last_name TEXT NOT NULL, display_name TEXT NOT NULL,
                    position TEXT, height TEXT, jersey_number INTEGER, active INTEGER NOT NULL,
                    source TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                    shooting_profile_json TEXT
                );
                CREATE TABLE IF NOT EXISTS nba_rosters (
                    team_id TEXT NOT NULL, player_id TEXT NOT NULL, season TEXT NOT NULL,
                    source TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                    PRIMARY KEY (team_id, player_id, season)
                );
                CREATE TABLE IF NOT EXISTS nba_preferred_lineups (
                    team_id TEXT NOT NULL, season TEXT NOT NULL, slot INTEGER NOT NULL,
                    player_id TEXT NOT NULL, games_started INTEGER NOT NULL,
                    total_minutes REAL NOT NULL, source TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    PRIMARY KEY (team_id, season, slot)
                );
                CREATE TABLE IF NOT EXISTS nba_games (
                    id TEXT PRIMARY KEY, external_id TEXT UNIQUE NOT NULL, season TEXT NOT NULL,
                    game_date TEXT NOT NULL, home_team_id TEXT, away_team_id TEXT, status TEXT,
                    source TEXT NOT NULL, retrieved_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nba_player_game_stats (
                    game_id TEXT NOT NULL, player_id TEXT NOT NULL, team_id TEXT, minutes TEXT,
                    points INTEGER, rebounds INTEGER, assists INTEGER, field_goals_made INTEGER,
                    field_goals_attempted INTEGER, source TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                    PRIMARY KEY (game_id, player_id)
                );
                CREATE TABLE IF NOT EXISTS real_possession_snapshots (
                    snapshot_id TEXT PRIMARY KEY, possession_id TEXT NOT NULL,
                    game_id TEXT NOT NULL, source_possession_id TEXT NOT NULL,
                    provider TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE (possession_id, retrieved_at)
                );
                CREATE INDEX IF NOT EXISTS idx_real_possessions_game
                    ON real_possession_snapshots (game_id, possession_id, retrieved_at);
                CREATE TABLE IF NOT EXISTS possession_reconstructions (
                    id TEXT PRIMARY KEY, possession_id TEXT NOT NULL,
                    play_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(nba_players)").fetchall()
            }
            if "shooting_profile_json" not in columns:
                connection.execute(
                    "ALTER TABLE nba_players ADD COLUMN shooting_profile_json TEXT"
                )

    def save_teams(self, teams: Iterable[NBATeam]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO nba_teams VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET external_id=excluded.external_id,
                    name=excluded.name, full_name=excluded.full_name,
                    abbreviation=excluded.abbreviation, city=excluded.city, state=excluded.state,
                    year_founded=excluded.year_founded, source=excluded.source,
                    retrieved_at=excluded.retrieved_at
                """,
                [
                    (
                        str(team.id),
                        team.external_id,
                        team.name,
                        team.full_name,
                        team.abbreviation,
                        team.city,
                        team.state,
                        team.year_founded,
                        team.source.value,
                        team.retrieved_at.isoformat(),
                    )
                    for team in teams
                ],
            )

    def list_teams(self) -> list[NBATeam]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM nba_teams ORDER BY full_name").fetchall()
        return [self._team(row) for row in rows]

    def get_team(self, team_id: UUID) -> NBATeam | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM nba_teams WHERE id = ?", (str(team_id),)
            ).fetchone()
        return None if row is None else self._team(row)

    def save_players(self, players: Iterable[NBAPlayer]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO nba_players (
                    id, external_id, team_id, first_name, last_name, display_name,
                    position, height, jersey_number, active, source, retrieved_at,
                    shooting_profile_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET external_id=excluded.external_id,
                    team_id=excluded.team_id, first_name=excluded.first_name,
                    last_name=excluded.last_name, display_name=excluded.display_name,
                    position=COALESCE(excluded.position, nba_players.position),
                    height=COALESCE(excluded.height, nba_players.height),
                    jersey_number=COALESCE(excluded.jersey_number, nba_players.jersey_number),
                    active=excluded.active, source=excluded.source,
                    retrieved_at=excluded.retrieved_at,
                    shooting_profile_json=COALESCE(
                        excluded.shooting_profile_json,
                        nba_players.shooting_profile_json
                    )
                """,
                [
                    (
                        str(player.id),
                        player.external_id,
                        str(player.team_id) if player.team_id else None,
                        player.first_name,
                        player.last_name,
                        player.display_name,
                        player.position,
                        player.height,
                        player.jersey_number,
                        int(player.active),
                        player.source.value,
                        player.retrieved_at.isoformat(),
                        (
                            player.shooting_profile.model_dump_json(by_alias=True)
                            if player.shooting_profile is not None
                            else None
                        ),
                    )
                    for player in players
                ],
            )

    def replace_roster(self, team_id: UUID, season: str, players: list[NBAPlayer]) -> None:
        self.save_players(players)
        retrieved_at = max((player.retrieved_at for player in players), default=datetime.now())
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM nba_rosters WHERE team_id = ? AND season = ?",
                (str(team_id), season),
            )
            connection.executemany(
                "INSERT INTO nba_rosters VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        str(team_id),
                        str(player.id),
                        season,
                        player.source.value,
                        retrieved_at.isoformat(),
                    )
                    for player in players
                ],
            )

    def list_roster(self, team_id: UUID, season: str) -> list[NBAPlayer]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.* FROM nba_players p JOIN nba_rosters r ON r.player_id = p.id
                WHERE r.team_id = ? AND r.season = ? ORDER BY p.display_name
                """,
                (str(team_id), season),
            ).fetchall()
        return [self._player(row) for row in rows]

    def replace_preferred_lineup(
        self,
        team_id: UUID,
        season: str,
        candidates: list[PreferredLineupCandidate],
    ) -> None:
        self.save_players(candidate.player for candidate in candidates)
        retrieved_at = max(
            (candidate.player.retrieved_at for candidate in candidates),
            default=datetime.now(),
        )
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM nba_preferred_lineups WHERE team_id = ? AND season = ?",
                (str(team_id), season),
            )
            connection.executemany(
                "INSERT INTO nba_preferred_lineups VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        str(team_id),
                        season,
                        slot,
                        str(candidate.player.id),
                        candidate.games_started,
                        candidate.total_minutes,
                        candidate.player.source.value,
                        retrieved_at.isoformat(),
                    )
                    for slot, candidate in enumerate(candidates)
                ],
            )

    def list_preferred_lineup(self, team_id: UUID, season: str) -> list[PreferredLineupCandidate]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.*, lineup.games_started, lineup.total_minutes
                FROM nba_preferred_lineups lineup
                JOIN nba_players p ON p.id = lineup.player_id
                WHERE lineup.team_id = ? AND lineup.season = ? ORDER BY lineup.slot
                """,
                (str(team_id), season),
            ).fetchall()
        return [
            PreferredLineupCandidate(
                player=self._player(row),
                games_started=row["games_started"],
                total_minutes=row["total_minutes"],
            )
            for row in rows
        ]

    def preferred_lineup_retrieved_at(self, team_id: UUID, season: str) -> datetime | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT MAX(retrieved_at) FROM nba_preferred_lineups
                WHERE team_id = ? AND season = ?
                """,
                (str(team_id), season),
            ).fetchone()
        return None if row is None or row[0] is None else datetime.fromisoformat(row[0])

    def save_games(self, games: Iterable[NBAGame]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO nba_games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status,
                    source=excluded.source, retrieved_at=excluded.retrieved_at
                """,
                [
                    (
                        str(game.id),
                        game.external_id,
                        game.season,
                        game.game_date.isoformat(),
                        str(game.home_team_id) if game.home_team_id else None,
                        str(game.away_team_id) if game.away_team_id else None,
                        game.status,
                        game.source.value,
                        game.retrieved_at.isoformat(),
                    )
                    for game in games
                ],
            )

    def list_games(
        self,
        *,
        game_date: date | None = None,
        season: str | None = None,
        team_id: UUID | None = None,
    ) -> list[NBAGame]:
        clauses: list[str] = []
        values: list[str] = []
        if game_date is not None:
            clauses.append("game_date = ?")
            values.append(game_date.isoformat())
        if season is not None:
            clauses.append("season = ?")
            values.append(season)
        if team_id is not None:
            clauses.append("(home_team_id = ? OR away_team_id = ?)")
            values.extend([str(team_id), str(team_id)])
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM nba_games{where} ORDER BY game_date DESC", values
            ).fetchall()
        return [self._game(row) for row in rows]

    def get_game(self, game_id: UUID) -> NBAGame | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM nba_games WHERE id = ?", (str(game_id),)
            ).fetchone()
        return None if row is None else self._game(row)

    def save_player_game_stats(self, stats: Iterable[PlayerGameStats]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO nba_player_game_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id, player_id) DO UPDATE SET minutes=excluded.minutes,
                    points=excluded.points, rebounds=excluded.rebounds, assists=excluded.assists,
                    field_goals_made=excluded.field_goals_made,
                    field_goals_attempted=excluded.field_goals_attempted,
                    retrieved_at=excluded.retrieved_at
                """,
                [
                    (
                        str(item.game_id),
                        str(item.player_id),
                        str(item.team_id) if item.team_id else None,
                        item.minutes,
                        item.points,
                        item.rebounds,
                        item.assists,
                        item.field_goals_made,
                        item.field_goals_attempted,
                        item.source.value,
                        item.retrieved_at.isoformat(),
                    )
                    for item in stats
                ],
            )

    def list_player_game_stats(self, game_id: UUID) -> list[PlayerGameStats]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM nba_player_game_stats WHERE game_id = ? ORDER BY player_id",
                (str(game_id),),
            ).fetchall()
        return [self._player_game_stats(row) for row in rows]

    def roster_retrieved_at(self, team_id: UUID, season: str) -> datetime | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MAX(retrieved_at) FROM nba_rosters WHERE team_id = ? AND season = ?",
                (str(team_id), season),
            ).fetchone()
        return None if row is None or row[0] is None else datetime.fromisoformat(row[0])

    def save_possessions(self, possessions: Iterable[RealPossession]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO real_possession_snapshots
                    (snapshot_id, possession_id, game_id, source_possession_id,
                     provider, retrieved_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(
                            stable_nba_id(
                                "possession_snapshot",
                                f"{item.id}:{item.provenance.retrieved_at.isoformat()}",
                            )
                        ),
                        str(item.id),
                        str(item.game_id),
                        item.provenance.source_possession_id,
                        item.provenance.provider,
                        item.provenance.retrieved_at.isoformat(),
                        item.model_dump_json(by_alias=True),
                    )
                    for item in possessions
                ],
            )

    def list_possessions(self, game_id: UUID) -> list[RealPossession]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT snapshots.payload_json
                FROM real_possession_snapshots snapshots
                JOIN (
                    SELECT possession_id, MAX(retrieved_at) AS latest
                    FROM real_possession_snapshots WHERE game_id = ? GROUP BY possession_id
                ) selected ON selected.possession_id = snapshots.possession_id
                    AND selected.latest = snapshots.retrieved_at
                WHERE snapshots.game_id = ? ORDER BY snapshots.source_possession_id
                """,
                (str(game_id), str(game_id)),
            ).fetchall()
        return [RealPossession.model_validate_json(row[0]) for row in rows]

    def get_possession(self, possession_id: UUID) -> RealPossession | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM real_possession_snapshots
                WHERE possession_id = ? ORDER BY retrieved_at DESC LIMIT 1
                """,
                (str(possession_id),),
            ).fetchone()
        return None if row is None else RealPossession.model_validate_json(row[0])

    def save_reconstruction(
        self, reconstruction: PossessionReconstruction
    ) -> PossessionReconstruction:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO possession_reconstructions
                    (id, possession_id, play_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET play_json=excluded.play_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(reconstruction.id),
                    str(reconstruction.possession_id),
                    reconstruction.play.model_dump_json(by_alias=True),
                    reconstruction.created_at.isoformat(),
                    reconstruction.updated_at.isoformat(),
                ),
            )
        return reconstruction

    def list_reconstructions(self, possession_id: UUID) -> list[PossessionReconstruction]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, play_json, created_at, updated_at
                FROM possession_reconstructions WHERE possession_id = ?
                ORDER BY updated_at DESC
                """,
                (str(possession_id),),
            ).fetchall()
        return [
            PossessionReconstruction(
                id=row["id"],
                possession_id=possession_id,
                play=Play.model_validate_json(row["play_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def _team(row: sqlite3.Row) -> NBATeam:
        return NBATeam(
            id=row["id"],
            external_id=row["external_id"],
            name=row["name"],
            full_name=row["full_name"],
            abbreviation=row["abbreviation"],
            city=row["city"],
            state=row["state"],
            year_founded=row["year_founded"],
            source=DataSource(row["source"]),
            retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
        )

    @staticmethod
    def _player(row: sqlite3.Row) -> NBAPlayer:
        return NBAPlayer(
            id=row["id"],
            external_id=row["external_id"],
            team_id=row["team_id"],
            first_name=row["first_name"],
            last_name=row["last_name"],
            display_name=row["display_name"],
            position=row["position"],
            height=row["height"],
            jersey_number=row["jersey_number"],
            active=bool(row["active"]),
            shooting_profile=(
                PlayerShootingProfile.model_validate_json(row["shooting_profile_json"])
                if row["shooting_profile_json"] is not None
                else None
            ),
            source=DataSource(row["source"]),
            retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
        )

    @staticmethod
    def _game(row: sqlite3.Row) -> NBAGame:
        return NBAGame(
            id=row["id"],
            external_id=row["external_id"],
            season=row["season"],
            game_date=date.fromisoformat(row["game_date"]),
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            status=row["status"],
            source=DataSource(row["source"]),
            retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
        )

    @staticmethod
    def _player_game_stats(row: sqlite3.Row) -> PlayerGameStats:
        return PlayerGameStats(
            game_id=row["game_id"],
            player_id=row["player_id"],
            team_id=row["team_id"],
            minutes=row["minutes"],
            points=row["points"],
            rebounds=row["rebounds"],
            assists=row["assists"],
            field_goals_made=row["field_goals_made"],
            field_goals_attempted=row["field_goals_attempted"],
            source=DataSource(row["source"]),
            retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection
