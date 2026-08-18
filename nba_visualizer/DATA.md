# NBA data integration

The application uses a provider-neutral, cache-first path:

`nba_api / NBA.com → NBAApiProvider normalization → SQLite cache → FastAPI → frontend`

React components never import `nba_api` and never call NBA.com. The provider returns internal Pydantic models, and the SQLite repository stores normalized teams, players, roster memberships, games, and player game statistics with source IDs, source name, season where applicable, and retrieval timestamps.

## Package calls

| Operation | `nba_api` call / result set |
| --- | --- |
| Teams | `nba_api.stats.static.teams.get_teams()` (packaged directory; no NBA.com request) |
| Player directory / player stats | `LeagueDashPlayerStats` / `LeagueDashPlayerStats` |
| Team roster | `CommonTeamRoster` / `CommonTeamRoster` |
| Preferred starting five | Current `CommonTeamRoster`, filtered latest `LeagueGameFinder` + `BoxScoreTraditionalV2` starters; recent `LeagueDashPlayerStats` role fallback |
| Player details | `CommonPlayerInfo` / `CommonPlayerInfo` |
| Games | `LeagueGameFinder` / `LeagueGameFinderResults` |
| Box score | `BoxScoreTraditionalV2` / `PlayerStats` |
| Play-by-play | `PlayByPlayV3` / `PlayByPlay` |
| Shot chart | `ShotChartDetail` / `Shot_Chart_Detail` |
| Team stats | `LeagueDashTeamStats` / `LeagueDashTeamStats` |

These are unofficially exposed NBA.com statistics endpoints wrapped by the community `nba_api` package. NBA.com can throttle them, time out, change response columns, or return partial data without notice. They should not be treated as a guaranteed production SLA.

## Cache and failure behavior

- Normal reads use SQLite first and make no external request when cached data exists.
- Explicit refresh and preload calls use a 10-second request timeout and at most two attempts.
- A failed refresh returns existing cached data with `cacheStatus: "cache_fallback"`.
- If neither provider nor cache can supply a resource, FastAPI returns HTTP 503 with a structured code, message, retryability flag, and attempt count.
- Missing fields remain `null`; the application does not invent heights, positions, numbers, games, or statistics.
- Cached data has no artificial expiry. Refresh is explicit so normal visualizer usage remains deterministic.

Preload teams, the current player directory, and all team rosters from `backend/`:

```bash
.venv/bin/python -m app.cli preload-nba --season 2026-27
```

The command can take time and may be partially limited by NBA.com. Data successfully written before a later failure remains in SQLite.

Normalized cache APIs currently exposed to applications are:

- `GET /api/v1/nba/teams`
- `GET /api/v1/nba/teams/{team_id}/roster?season=2026-27`
- `GET /api/v1/nba/teams/{team_id}/preferred-lineup?season=2026-27`
- `GET /api/v1/nba/games` with optional date, season, and team filters
- `GET /api/v1/nba/games/{game_id}/box-score`

The roster editor uses only these FastAPI endpoints. It never calls NBA.com from the browser.

Preferred-lineup responses also carry a normalized shooting profile when season
totals are available: games played, field-goal attempts, three-point attempts,
derived two-point percentage, three-point percentage, free-throw percentage,
season, and provenance. Profiles are cached with the normalized player record;
missing shooting data remains missing and causes an explicit league fallback in
the shot heuristic.

When a team is selected, the editor first loads that season's current roster.
It uses the most recently recorded starters only when those players still appear
on that roster. If the selected season has not begun, it ranks current-roster
players using their most recent prior-season roles, then fills any remaining
slots from the current roster. Jersey numbers always come from the current
roster response. This is a setup convenience, not a confirmed lineup for a
future game; injuries, new transactions, rest, and matchup decisions can still
change the next starting five. Historical possession imports use the recorded
on-court lineup when that source data is available.

## Availability and tracking limitation

Roster responses commonly provide name, position, listed height, and jersey number, but fields can be missing or stale. Public play-by-play describes events, and public shot charts provide shot locations; neither is equivalent to optical tracking.

**Standard NBA public statistics APIs do not provide full frame-by-frame x/y coordinates for all ten players and the ball. They cannot reconstruct complete possessions as tracking data.** Accurate possession reconstruction requires a licensed tracking source or an appropriately licensed local dataset behind the same provider interface.
