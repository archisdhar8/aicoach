from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .team_map import normalize_team_name


FEATURE_COLUMNS = [
    "win_pct_diff",
    "run_diff_per_game_diff",
    "last7_run_diff_diff",
    "last14_run_diff_diff",
    "last30_run_diff_diff",
    "last7_runs_for_diff",
    "last14_runs_for_diff",
    "last30_runs_for_diff",
    "ewma_runs_for_diff",
    "ewma_run_prevention_diff",
    "venue_split_win_pct_diff",
    "rest_days_diff",
    "bullpen_ip_last3_diff",
    "bullpen_ip_last7_diff",
    "bullpen_er_last3_diff",
    "bullpen_er_last7_diff",
    "bullpen_fatigue_advantage",
    "bullpen_quality_advantage",
    "bullpen_kbb_advantage",
    "starter_era_advantage",
    "starter_whip_advantage",
    "starter_kbb_advantage",
    "starter_fip_advantage",
    "starter_last5_fip_advantage",
    "starter_history_diff",
    "starter_rest_advantage",
    "starter_workload_advantage",
    "both_lineups_confirmed",
    "lineup_offense_advantage",
    "park_run_factor",
    "is_day_game",
    "doubleheader_game",
]


PARK_RUN_FACTORS = {
    "Coors Field": 1.18,
    "Great American Ball Park": 1.08,
    "Fenway Park": 1.06,
    "Kauffman Stadium": 1.04,
    "Wrigley Field": 1.04,
    "Citizens Bank Park": 1.03,
    "Yankee Stadium": 1.03,
    "Globe Life Field": 1.02,
    "Chase Field": 1.01,
    "Dodger Stadium": 1.00,
    "Busch Stadium": 0.99,
    "Minute Maid Park": 0.99,
    "Citi Field": 0.98,
    "T-Mobile Park": 0.97,
    "Petco Park": 0.96,
    "Oracle Park": 0.94,
    "Tropicana Field": 0.94,
}


@dataclass
class TeamState:
    wins: int = 0
    losses: int = 0
    runs_for: int = 0
    runs_against: int = 0
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    last_date: pd.Timestamp | None = None
    recent: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=30))
    statcast_recent: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=30))

    def win_pct(self) -> float:
        games = self.wins + self.losses
        return self.wins / games if games else 0.5

    def side_win_pct(self, side: str) -> float:
        if side == "home":
            games = self.home_wins + self.home_losses
            return self.home_wins / games if games else 0.5
        games = self.away_wins + self.away_losses
        return self.away_wins / games if games else 0.5

    def run_diff_per_game(self) -> float:
        games = self.wins + self.losses
        return (self.runs_for - self.runs_against) / games if games else 0.0

    def rolling_run_diff(self, n: int) -> float:
        games = list(self.recent)[-n:]
        if not games:
            return 0.0
        return float(np.mean([g["runs_for"] - g["runs_against"] for g in games]))

    def ewma(self, key: str, n: int = 14, alpha: float = 0.35, default: float = 0.0) -> float:
        values = [float(g.get(key, 0.0) or 0.0) for g in list(self.recent)[-n:]]
        if not values:
            return default
        current = values[0]
        for value in values[1:]:
            current = alpha * value + (1.0 - alpha) * current
        return float(current)

    def rolling_sum(self, key: str, n: int) -> float:
        games = list(self.recent)[-n:]
        return float(sum(g.get(key, 0.0) or 0.0 for g in games))

    def rolling_mean(self, key: str, n: int, default: float = 0.0, statcast: bool = False) -> float:
        source = self.statcast_recent if statcast else self.recent
        values = [g.get(key) for g in list(source)[-n:]]
        values = [float(v) for v in values if v is not None and not pd.isna(v)]
        return float(np.mean(values)) if values else default

    def rest_days(self, game_date: pd.Timestamp) -> int:
        if self.last_date is None:
            return 3
        return int((game_date - self.last_date).days)

    def update(
        self,
        game_date: pd.Timestamp,
        side: str,
        runs_for: int,
        runs_against: int,
        bullpen_ip: float,
        bullpen_er: int,
        bullpen_so: int = 0,
        bullpen_bb: int = 0,
    ) -> None:
        won = runs_for > runs_against
        self.wins += int(won)
        self.losses += int(not won)
        self.runs_for += runs_for
        self.runs_against += runs_against
        if side == "home":
            self.home_wins += int(won)
            self.home_losses += int(not won)
        else:
            self.away_wins += int(won)
            self.away_losses += int(not won)
        self.recent.append(
            {
                "runs_for": runs_for,
                "runs_against": runs_against,
                "bullpen_ip": bullpen_ip,
                "bullpen_er": bullpen_er,
                "bullpen_so": bullpen_so,
                "bullpen_bb": bullpen_bb,
            }
        )
        self.last_date = game_date

    def update_statcast(self, values: dict[str, Any]) -> None:
        if values:
            self.statcast_recent.append(values)


@dataclass
class PitcherState:
    ip: float = 0.0
    er: int = 0
    hits: int = 0
    walks: int = 0
    strikeouts: int = 0
    home_runs: int = 0
    starts: int = 0
    season: int | None = None
    season_ip: float = 0.0
    season_er: int = 0
    season_hits: int = 0
    season_walks: int = 0
    season_strikeouts: int = 0
    season_home_runs: int = 0
    season_starts: int = 0
    last_start_date: pd.Timestamp | None = None
    recent_starts: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5))
    statcast_recent: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=10))

    def era(self) -> float:
        return 4.50 if self.ip <= 0 else 9.0 * self.er / self.ip

    def whip(self) -> float:
        return 1.30 if self.ip <= 0 else (self.walks + self.hits) / self.ip

    def kbb(self) -> float:
        return self.strikeouts / max(self.walks, 1)

    def fip_proxy(self) -> float:
        if self.ip <= 0:
            return 4.50
        return ((13 * self.home_runs) + (3 * self.walks) - (2 * self.strikeouts)) / self.ip + 3.20

    def season_era(self, season: int | None) -> float:
        return 4.50 if self.season != season or self.season_ip <= 0 else 9.0 * self.season_er / self.season_ip

    def season_whip(self, season: int | None) -> float:
        return 1.30 if self.season != season or self.season_ip <= 0 else (self.season_walks + self.season_hits) / self.season_ip

    def season_kbb(self, season: int | None) -> float:
        return 2.20 if self.season != season else self.season_strikeouts / max(self.season_walks, 1)

    def season_fip_proxy(self, season: int | None) -> float:
        if self.season != season or self.season_ip <= 0:
            return 4.50
        return ((13 * self.season_home_runs) + (3 * self.season_walks) - (2 * self.season_strikeouts)) / self.season_ip + 3.20

    def season_starts_for(self, season: int | None) -> int:
        return self.season_starts if self.season == season else 0

    def last5_fip(self) -> float:
        values = [g.get("fip_proxy") for g in self.recent_starts]
        values = [float(v) for v in values if v is not None and not pd.isna(v)]
        return float(np.mean(values)) if values else self.fip_proxy()

    def rest_days(self, game_date: pd.Timestamp) -> int:
        if self.last_start_date is None:
            return 5
        return int(max(0, min((game_date - self.last_start_date).days, 14)))

    def workload_trend(self) -> float:
        values = [float(g.get("pitches_proxy", 0.0) or 0.0) for g in self.recent_starts]
        if len(values) < 2:
            return 0.0
        return float(values[-1] - np.mean(values[:-1]))

    def rolling_mean(self, key: str, n: int, default: float) -> float:
        values = [g.get(key) for g in list(self.statcast_recent)[-n:]]
        values = [float(v) for v in values if v is not None and not pd.isna(v)]
        return float(np.mean(values)) if values else default

    def update(self, game_date: pd.Timestamp, season: int | None, ip: float, er: int, hits: int, walks: int, strikeouts: int, home_runs: int = 0) -> None:
        if self.season != season:
            self.season = season
            self.season_ip = 0.0
            self.season_er = 0
            self.season_hits = 0
            self.season_walks = 0
            self.season_strikeouts = 0
            self.season_home_runs = 0
            self.season_starts = 0
        self.ip += ip or 0.0
        self.er += er or 0
        self.hits += hits or 0
        self.walks += walks or 0
        self.strikeouts += strikeouts or 0
        self.home_runs += home_runs or 0
        self.season_ip += ip or 0.0
        self.season_er += er or 0
        self.season_hits += hits or 0
        self.season_walks += walks or 0
        self.season_strikeouts += strikeouts or 0
        self.season_home_runs += home_runs or 0
        start_ip = float(ip or 0.0)
        if start_ip > 0:
            self.starts += 1
            self.season_starts += 1
            start_fip = ((13 * (home_runs or 0)) + (3 * (walks or 0)) - (2 * (strikeouts or 0))) / start_ip + 3.20
            self.recent_starts.append(
                {
                    "ip": start_ip,
                    "er": er or 0,
                    "fip_proxy": start_fip,
                    "pitches_proxy": max(start_ip * 15.5, (hits or 0) + (walks or 0) + (strikeouts or 0)),
                }
            )
            self.last_start_date = game_date

    def update_statcast(self, values: dict[str, Any]) -> None:
        if values:
            self.statcast_recent.append(values)


def build_training_frame(
    games: pd.DataFrame,
    team_stats: pd.DataFrame | None = None,
    statcast_team: pd.DataFrame | None = None,
    statcast_pitchers: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    player_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()
    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    stats_lookup = build_stats_lookup(team_stats)
    statcast_team_lookup = build_statcast_team_lookup(statcast_team)
    statcast_pitcher_lookup = build_statcast_pitcher_lookup(statcast_pitchers)
    weather_lookup = build_weather_lookup(weather)
    lineup_lookup = build_lineup_lookup(lineups)
    player_states: dict[int, PlayerState] = defaultdict(PlayerState)
    team_states: dict[int, TeamState] = defaultdict(TeamState)
    pitcher_states: dict[int, PitcherState] = defaultdict(PitcherState)
    rows: list[dict[str, Any]] = []

    for game in games.itertuples(index=False):
        if pd.isna(game.home_team_id) or pd.isna(game.away_team_id):
            continue
        home_state = team_states[int(game.home_team_id)]
        away_state = team_states[int(game.away_team_id)]
        home_pitcher_id = first_present(
            getattr(game, "home_probable_pitcher_id", None),
            stats_lookup.get((game.game_pk, int(game.home_team_id)), {}).get("starter_id"),
        )
        away_pitcher_id = first_present(
            getattr(game, "away_probable_pitcher_id", None),
            stats_lookup.get((game.game_pk, int(game.away_team_id)), {}).get("starter_id"),
        )
        home_pitcher = pitcher_states[int(home_pitcher_id)] if home_pitcher_id else PitcherState()
        away_pitcher = pitcher_states[int(away_pitcher_id)] if away_pitcher_id else PitcherState()
        game_weather = weather_lookup.get(game.game_pk, {})
        home_lineup = lineup_features(lineup_lookup.get((game.game_pk, int(game.home_team_id)), []), player_states)
        away_lineup = lineup_features(lineup_lookup.get((game.game_pk, int(game.away_team_id)), []), player_states)
        row = {
            "game_pk": game.game_pk,
            "game_date": game.game_date,
            "season": game.season,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "home_team_id": int(game.home_team_id),
            "away_team_id": int(game.away_team_id),
            "home_probable_pitcher_id": home_pitcher_id,
            "away_probable_pitcher_id": away_pitcher_id,
            "home_probable_pitcher": getattr(game, "home_probable_pitcher", None),
            "away_probable_pitcher": getattr(game, "away_probable_pitcher", None),
            "target_home_win": bool(game.home_won) if not pd.isna(game.home_won) else None,
            "target_total_runs": total_runs_target(game),
            "home_win_pct": home_state.win_pct(),
            "away_win_pct": away_state.win_pct(),
            "home_team_games_prior": home_state.wins + home_state.losses,
            "away_team_games_prior": away_state.wins + away_state.losses,
            "home_run_diff_per_game": home_state.run_diff_per_game(),
            "away_run_diff_per_game": away_state.run_diff_per_game(),
            "home_last7_run_diff": home_state.rolling_run_diff(7),
            "away_last7_run_diff": away_state.rolling_run_diff(7),
            "home_last14_run_diff": home_state.rolling_run_diff(14),
            "away_last14_run_diff": away_state.rolling_run_diff(14),
            "home_last30_run_diff": home_state.rolling_run_diff(30),
            "away_last30_run_diff": away_state.rolling_run_diff(30),
            "home_last7_runs_for": home_state.rolling_mean("runs_for", 7, 4.4),
            "away_last7_runs_for": away_state.rolling_mean("runs_for", 7, 4.4),
            "home_last14_runs_for": home_state.rolling_mean("runs_for", 14, 4.4),
            "away_last14_runs_for": away_state.rolling_mean("runs_for", 14, 4.4),
            "home_last30_runs_for": home_state.rolling_mean("runs_for", 30, 4.4),
            "away_last30_runs_for": away_state.rolling_mean("runs_for", 30, 4.4),
            "home_ewma_runs_for": home_state.ewma("runs_for", default=4.4),
            "away_ewma_runs_for": away_state.ewma("runs_for", default=4.4),
            "home_ewma_runs_against": home_state.ewma("runs_against", default=4.4),
            "away_ewma_runs_against": away_state.ewma("runs_against", default=4.4),
            "home_home_win_pct": home_state.side_win_pct("home"),
            "away_away_win_pct": away_state.side_win_pct("away"),
            "home_rest_days": clip_rest(home_state.rest_days(game.game_date)),
            "away_rest_days": clip_rest(away_state.rest_days(game.game_date)),
            "home_bullpen_ip_last3": home_state.rolling_sum("bullpen_ip", 3),
            "away_bullpen_ip_last3": away_state.rolling_sum("bullpen_ip", 3),
            "home_bullpen_ip_last7": home_state.rolling_sum("bullpen_ip", 7),
            "away_bullpen_ip_last7": away_state.rolling_sum("bullpen_ip", 7),
            "home_bullpen_er_last3": home_state.rolling_sum("bullpen_er", 3),
            "away_bullpen_er_last3": away_state.rolling_sum("bullpen_er", 3),
            "home_bullpen_er_last7": home_state.rolling_sum("bullpen_er", 7),
            "away_bullpen_er_last7": away_state.rolling_sum("bullpen_er", 7),
            "home_bullpen_fatigue_score": bullpen_fatigue(home_state),
            "away_bullpen_fatigue_score": bullpen_fatigue(away_state),
            "home_bullpen_quality_score": bullpen_quality(home_state),
            "away_bullpen_quality_score": bullpen_quality(away_state),
            "home_starter_era_prior": home_pitcher.era(),
            "away_starter_era_prior": away_pitcher.era(),
            "home_starter_season_era": home_pitcher.season_era(game.season),
            "away_starter_season_era": away_pitcher.season_era(game.season),
            "home_starter_season_whip": home_pitcher.season_whip(game.season),
            "away_starter_season_whip": away_pitcher.season_whip(game.season),
            "home_starter_season_kbb": home_pitcher.season_kbb(game.season),
            "away_starter_season_kbb": away_pitcher.season_kbb(game.season),
            "home_starter_season_fip_proxy": home_pitcher.season_fip_proxy(game.season),
            "away_starter_season_fip_proxy": away_pitcher.season_fip_proxy(game.season),
            "home_starter_season_starts": home_pitcher.season_starts_for(game.season),
            "away_starter_season_starts": away_pitcher.season_starts_for(game.season),
            "home_starter_whip_prior": home_pitcher.whip(),
            "away_starter_whip_prior": away_pitcher.whip(),
            "home_starter_kbb_prior": home_pitcher.kbb(),
            "away_starter_kbb_prior": away_pitcher.kbb(),
            "home_starter_fip_proxy": home_pitcher.fip_proxy(),
            "away_starter_fip_proxy": away_pitcher.fip_proxy(),
            "home_starter_last5_fip": home_pitcher.last5_fip(),
            "away_starter_last5_fip": away_pitcher.last5_fip(),
            "home_starter_games_prior": home_pitcher.starts,
            "away_starter_games_prior": away_pitcher.starts,
            "home_starter_rest_days": home_pitcher.rest_days(game.game_date),
            "away_starter_rest_days": away_pitcher.rest_days(game.game_date),
            "home_starter_workload_trend": home_pitcher.workload_trend(),
            "away_starter_workload_trend": away_pitcher.workload_trend(),
            "home_starter_xwoba_allowed": home_pitcher.rolling_mean("xwoba_allowed", 5, 0.320),
            "away_starter_xwoba_allowed": away_pitcher.rolling_mean("xwoba_allowed", 5, 0.320),
            "home_starter_hard_hit_allowed": home_pitcher.rolling_mean("hard_hit_allowed", 5, 0.39),
            "away_starter_hard_hit_allowed": away_pitcher.rolling_mean("hard_hit_allowed", 5, 0.39),
            "home_starter_barrel_allowed": home_pitcher.rolling_mean("barrel_allowed", 5, 0.075),
            "away_starter_barrel_allowed": away_pitcher.rolling_mean("barrel_allowed", 5, 0.075),
            "home_starter_velocity": home_pitcher.rolling_mean("avg_pitch_velocity", 5, 92.5),
            "away_starter_velocity": away_pitcher.rolling_mean("avg_pitch_velocity", 5, 92.5),
            "home_starter_spin_rate": home_pitcher.rolling_mean("avg_spin_rate", 5, 2250.0),
            "away_starter_spin_rate": away_pitcher.rolling_mean("avg_spin_rate", 5, 2250.0),
            "home_team_xwoba_last14": home_state.rolling_mean("xwoba", 14, 0.320, statcast=True),
            "away_team_xwoba_last14": away_state.rolling_mean("xwoba", 14, 0.320, statcast=True),
            "home_team_xba_last14": home_state.rolling_mean("xba", 14, 0.245, statcast=True),
            "away_team_xba_last14": away_state.rolling_mean("xba", 14, 0.245, statcast=True),
            "home_team_hard_hit_last14": home_state.rolling_mean("hard_hit_rate", 14, 0.39, statcast=True),
            "away_team_hard_hit_last14": away_state.rolling_mean("hard_hit_rate", 14, 0.39, statcast=True),
            "home_team_barrel_last14": home_state.rolling_mean("barrel_rate", 14, 0.075, statcast=True),
            "away_team_barrel_last14": away_state.rolling_mean("barrel_rate", 14, 0.075, statcast=True),
            "home_team_k_rate_last14": home_state.rolling_mean("k_rate", 14, 0.22, statcast=True),
            "away_team_k_rate_last14": away_state.rolling_mean("k_rate", 14, 0.22, statcast=True),
            "home_team_bb_rate_last14": home_state.rolling_mean("bb_rate", 14, 0.085, statcast=True),
            "away_team_bb_rate_last14": away_state.rolling_mean("bb_rate", 14, 0.085, statcast=True),
            "home_pitching_xwoba_allowed_last14": home_state.rolling_mean("xwoba_allowed", 14, 0.320, statcast=True),
            "away_pitching_xwoba_allowed_last14": away_state.rolling_mean("xwoba_allowed", 14, 0.320, statcast=True),
            "home_pitching_hard_hit_allowed_last14": home_state.rolling_mean("hard_hit_allowed", 14, 0.39, statcast=True),
            "away_pitching_hard_hit_allowed_last14": away_state.rolling_mean("hard_hit_allowed", 14, 0.39, statcast=True),
            "home_pitching_barrel_allowed_last14": home_state.rolling_mean("barrel_allowed", 14, 0.075, statcast=True),
            "away_pitching_barrel_allowed_last14": away_state.rolling_mean("barrel_allowed", 14, 0.075, statcast=True),
            "home_bullpen_kbb_last7": safe_ratio(home_state.rolling_sum("bullpen_so", 7), home_state.rolling_sum("bullpen_bb", 7), default=2.2),
            "away_bullpen_kbb_last7": safe_ratio(away_state.rolling_sum("bullpen_so", 7), away_state.rolling_sum("bullpen_bb", 7), default=2.2),
            "home_lineup_confirmed": home_lineup["confirmed"],
            "away_lineup_confirmed": away_lineup["confirmed"],
            "home_lineup_offense_rating": home_lineup["offense_rating"],
            "away_lineup_offense_rating": away_lineup["offense_rating"],
            "home_lineup_platoon_score": home_lineup["platoon_score"],
            "away_lineup_platoon_score": away_lineup["platoon_score"],
            "home_missing_lineup_penalty": 1.0 - home_lineup["confirmed"],
            "away_missing_lineup_penalty": 1.0 - away_lineup["confirmed"],
            "temperature_f": game_weather.get("temperature_f", 72.0),
            "wind_speed_mph": game_weather.get("wind_speed_mph", 0.0),
            "wind_out_proxy": wind_out_proxy(game_weather.get("wind_speed_mph", 0.0), game_weather.get("wind_direction_degrees")),
            "precipitation_in": game_weather.get("precipitation_in", 0.0),
            "park_run_factor": PARK_RUN_FACTORS.get(getattr(game, "venue_name", None), 1.0),
            "same_division": 0,
            "is_day_game": int(str(getattr(game, "day_night", "")).lower() == "day"),
            "doubleheader_game": int((getattr(game, "doubleheader", "N") or "N") != "N"),
        }
        row.update(matchup_difference_features(row))
        rows.append(row)

        if has_final_score(game):
            update_states_for_game(game, stats_lookup, statcast_team_lookup, statcast_pitcher_lookup, team_states, pitcher_states)
            update_player_states_for_game(game.game_pk, player_stats, player_states)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].fillna(0.0)
    return frame


def build_prediction_frame(
    games: pd.DataFrame,
    history_games: pd.DataFrame,
    team_stats: pd.DataFrame | None = None,
    statcast_team: pd.DataFrame | None = None,
    statcast_pitchers: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    player_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    history = history_games[history_games["home_won"].notna()].copy()
    combined = pd.concat([history, games], ignore_index=True, sort=False)
    frame = build_training_frame(combined, team_stats, statcast_team, statcast_pitchers, weather, lineups, player_stats)
    wanted = set(games["game_pk"].tolist())
    return frame[frame["game_pk"].isin(wanted)].reset_index(drop=True)


def build_stats_lookup(team_stats: pd.DataFrame | None) -> dict[tuple[int, int], dict[str, Any]]:
    if team_stats is None or team_stats.empty:
        return {}
    lookup = {}
    for row in team_stats.to_dict("records"):
        lookup[(row["game_pk"], row["team_id"])] = row
    return lookup


def build_statcast_team_lookup(statcast_team: pd.DataFrame | None) -> dict[tuple[pd.Timestamp, str], dict[str, Any]]:
    if statcast_team is None or statcast_team.empty:
        return {}
    df = statcast_team.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["team_name"] = df["team_name"].map(normalize_team_name)
    return {(row["game_date"], row["team_name"]): row for row in df.to_dict("records")}


def build_statcast_pitcher_lookup(statcast_pitchers: pd.DataFrame | None) -> dict[tuple[pd.Timestamp, int], dict[str, Any]]:
    if statcast_pitchers is None or statcast_pitchers.empty:
        return {}
    df = statcast_pitchers.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    return {
        (row["game_date"], int(row["pitcher_id"])): row
        for row in df.to_dict("records")
        if row.get("pitcher_id") is not None and not pd.isna(row.get("pitcher_id"))
    }


def build_weather_lookup(weather: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
    if weather is None or weather.empty:
        return {}
    return {row["game_pk"]: row for row in weather.to_dict("records")}


@dataclass
class PlayerState:
    at_bats: int = 0
    walks: int = 0
    total_bases: int = 0
    strikeouts: int = 0

    def offense_rating(self) -> float:
        pa = self.at_bats + self.walks
        if pa <= 0:
            return 0.700
        return float((self.total_bases + 0.7 * self.walks) / pa)

    def update(self, at_bats: int, walks: int, total_bases: int, strikeouts: int) -> None:
        self.at_bats += at_bats or 0
        self.walks += walks or 0
        self.total_bases += total_bases or 0
        self.strikeouts += strikeouts or 0


def build_lineup_lookup(lineups: pd.DataFrame | None) -> dict[tuple[int, int], list[dict[str, Any]]]:
    if lineups is None or lineups.empty:
        return {}
    df = lineups.copy()
    return {
        key: group.sort_values("batting_order").to_dict("records")
        for key, group in df.groupby(["game_pk", "team_id"], dropna=False)
        if key[0] is not None and key[1] is not None
    }


def lineup_features(players: list[dict[str, Any]], player_states: dict[int, PlayerState]) -> dict[str, float]:
    if not players:
        return {"confirmed": 0.0, "offense_rating": 0.700, "platoon_score": 0.0}
    ratings = []
    confirmed_count = 0
    for player in players[:9]:
        player_id = player.get("player_id")
        if player.get("confirmed"):
            confirmed_count += 1
        if player_id is not None and not pd.isna(player_id):
            ratings.append(player_states[int(player_id)].offense_rating())
    return {
        "confirmed": float(confirmed_count >= 8),
        "offense_rating": float(np.mean(ratings)) if ratings else 0.700,
        "platoon_score": 0.0,
    }


def update_player_states_for_game(game_pk: int, player_stats: pd.DataFrame | None, player_states: dict[int, PlayerState]) -> None:
    if player_stats is None or player_stats.empty:
        return
    rows = player_stats[player_stats["game_pk"] == game_pk]
    for row in rows.to_dict("records"):
        player_id = row.get("player_id")
        if player_id is None or pd.isna(player_id):
            continue
        player_states[int(player_id)].update(
            row.get("at_bats", 0) or 0,
            row.get("walks", 0) or 0,
            row.get("total_bases", 0) or 0,
            row.get("strikeouts", 0) or 0,
        )


def matchup_difference_features(row: dict[str, Any]) -> dict[str, float]:
    def value(name: str, default: float = 0.0) -> float:
        raw = row.get(name, default)
        if raw is None or pd.isna(raw):
            return default
        return float(raw)

    return {
        "win_pct_diff": value("home_win_pct", 0.5) - value("away_win_pct", 0.5),
        "run_diff_per_game_diff": value("home_run_diff_per_game") - value("away_run_diff_per_game"),
        "last7_run_diff_diff": value("home_last7_run_diff") - value("away_last7_run_diff"),
        "last14_run_diff_diff": value("home_last14_run_diff") - value("away_last14_run_diff"),
        "last30_run_diff_diff": value("home_last30_run_diff") - value("away_last30_run_diff"),
        "last7_runs_for_diff": value("home_last7_runs_for", 4.4) - value("away_last7_runs_for", 4.4),
        "last14_runs_for_diff": value("home_last14_runs_for", 4.4) - value("away_last14_runs_for", 4.4),
        "last30_runs_for_diff": value("home_last30_runs_for", 4.4) - value("away_last30_runs_for", 4.4),
        "ewma_runs_for_diff": value("home_ewma_runs_for", 4.4) - value("away_ewma_runs_for", 4.4),
        # Lower runs allowed is better, so away minus home favors the home team.
        "ewma_run_prevention_diff": value("away_ewma_runs_against", 4.4) - value("home_ewma_runs_against", 4.4),
        "venue_split_win_pct_diff": value("home_home_win_pct", 0.5) - value("away_away_win_pct", 0.5),
        "rest_days_diff": value("home_rest_days", 3.0) - value("away_rest_days", 3.0),
        # Positive workload/fatigue differences favor the team with the fresher bullpen.
        "bullpen_ip_last3_diff": value("away_bullpen_ip_last3") - value("home_bullpen_ip_last3"),
        "bullpen_ip_last7_diff": value("away_bullpen_ip_last7") - value("home_bullpen_ip_last7"),
        "bullpen_er_last3_diff": value("away_bullpen_er_last3") - value("home_bullpen_er_last3"),
        "bullpen_er_last7_diff": value("away_bullpen_er_last7") - value("home_bullpen_er_last7"),
        "bullpen_fatigue_advantage": value("away_bullpen_fatigue_score") - value("home_bullpen_fatigue_score"),
        "bullpen_quality_advantage": value("home_bullpen_quality_score") - value("away_bullpen_quality_score"),
        # Lower starter run-prevention stats are better for the home side.
        "starter_era_advantage": value("away_starter_era_prior", 4.5) - value("home_starter_era_prior", 4.5),
        "starter_whip_advantage": value("away_starter_whip_prior", 1.3) - value("home_starter_whip_prior", 1.3),
        "starter_kbb_advantage": value("home_starter_kbb_prior", 2.2) - value("away_starter_kbb_prior", 2.2),
        "starter_fip_advantage": value("away_starter_fip_proxy", 4.5) - value("home_starter_fip_proxy", 4.5),
        "starter_last5_fip_advantage": value("away_starter_last5_fip", 4.5) - value("home_starter_last5_fip", 4.5),
        "starter_history_diff": value("home_starter_games_prior") - value("away_starter_games_prior"),
        "starter_rest_advantage": value("home_starter_rest_days", 5.0) - value("away_starter_rest_days", 5.0),
        "starter_workload_advantage": value("home_starter_workload_trend") - value("away_starter_workload_trend"),
        # Keep Statcast in derived form only; raw default-heavy columns are excluded from FEATURE_COLUMNS.
        "team_xwoba_advantage": value("home_team_xwoba_last14", 0.320) - value("away_team_xwoba_last14", 0.320),
        "team_k_rate_advantage": value("away_team_k_rate_last14", 0.22) - value("home_team_k_rate_last14", 0.22),
        "team_bb_rate_advantage": value("home_team_bb_rate_last14", 0.085) - value("away_team_bb_rate_last14", 0.085),
        "pitching_xwoba_prevention_advantage": value("away_pitching_xwoba_allowed_last14", 0.320) - value("home_pitching_xwoba_allowed_last14", 0.320),
        "bullpen_kbb_advantage": value("home_bullpen_kbb_last7", 2.2) - value("away_bullpen_kbb_last7", 2.2),
        "both_lineups_confirmed": min(value("home_lineup_confirmed"), value("away_lineup_confirmed")),
        "lineup_offense_advantage": value("home_lineup_offense_rating", 0.700) - value("away_lineup_offense_rating", 0.700),
        "lineup_platoon_advantage": value("home_lineup_platoon_score") - value("away_lineup_platoon_score"),
    }


def bullpen_fatigue(state: TeamState) -> float:
    ip3 = state.rolling_sum("bullpen_ip", 3)
    ip7 = state.rolling_sum("bullpen_ip", 7)
    short_term = min(ip3 / 13.5, 1.0)
    week_term = min(ip7 / 34.0, 1.0)
    return float(np.clip((0.65 * short_term) + (0.35 * week_term), 0.0, 1.0))


def bullpen_quality(state: TeamState) -> float:
    er7 = state.rolling_sum("bullpen_er", 7)
    ip7 = state.rolling_sum("bullpen_ip", 7)
    kbb = safe_ratio(state.rolling_sum("bullpen_so", 7), state.rolling_sum("bullpen_bb", 7), default=2.2)
    era = 9.0 * er7 / ip7 if ip7 else 4.20
    return float(np.clip((4.20 - era) / 4.0 + (kbb - 2.2) / 5.0, -1.0, 1.0))


def update_states_for_game(game, stats_lookup, statcast_team_lookup, statcast_pitcher_lookup, team_states, pitcher_states) -> None:
    game_date = game.game_date
    home_id = int(game.home_team_id)
    away_id = int(game.away_team_id)
    home_runs = int(game.home_score)
    away_runs = int(game.away_score)
    home_stats = stats_lookup.get((game.game_pk, home_id), {})
    away_stats = stats_lookup.get((game.game_pk, away_id), {})
    team_states[home_id].update(
        game_date,
        "home",
        home_runs,
        away_runs,
        home_stats.get("bullpen_ip", 0.0),
        home_stats.get("bullpen_er", 0) or 0,
        home_stats.get("bullpen_so", 0) or 0,
        home_stats.get("bullpen_bb", 0) or 0,
    )
    team_states[away_id].update(
        game_date,
        "away",
        away_runs,
        home_runs,
        away_stats.get("bullpen_ip", 0.0),
        away_stats.get("bullpen_er", 0) or 0,
        away_stats.get("bullpen_so", 0) or 0,
        away_stats.get("bullpen_bb", 0) or 0,
    )
    team_states[home_id].update_statcast(statcast_team_lookup.get((game_date, game.home_team), {}))
    team_states[away_id].update_statcast(statcast_team_lookup.get((game_date, game.away_team), {}))
    update_pitcher_state(pitcher_states, home_stats, statcast_pitcher_lookup, game_date)
    update_pitcher_state(pitcher_states, away_stats, statcast_pitcher_lookup, game_date)


def update_pitcher_state(pitcher_states, stats: dict[str, Any], statcast_pitcher_lookup, game_date: pd.Timestamp) -> None:
    starter_id = stats.get("starter_id")
    if not starter_id:
        return
    pitcher_states[int(starter_id)].update(
        game_date,
        int(pd.Timestamp(game_date).year),
        stats.get("starter_ip", 0.0),
        stats.get("starter_er", 0) or 0,
        stats.get("starter_hits", 0) or 0,
        stats.get("starter_bb", 0) or 0,
        stats.get("starter_so", 0) or 0,
        stats.get("starter_home_runs", 0) or 0,
    )
    pitcher_states[int(starter_id)].update_statcast(statcast_pitcher_lookup.get((game_date, int(starter_id)), {}))


def has_final_score(game) -> bool:
    return not pd.isna(game.home_score) and not pd.isna(game.away_score)


def total_runs_target(game) -> int | None:
    if not has_final_score(game):
        return None
    return int(game.home_score) + int(game.away_score)


def clip_rest(days: int) -> int:
    return int(max(0, min(days, 7)))


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator) / denominator if denominator else default


def wind_out_proxy(wind_speed: float | None, wind_direction: float | None) -> float:
    if wind_speed is None or wind_direction is None or pd.isna(wind_speed) or pd.isna(wind_direction):
        return 0.0
    # Without exact park orientation, encode wind strength as a smooth signal and
    # let the model learn whether it helps scoring at each park.
    return float(wind_speed)


def first_present(*values):
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return None
