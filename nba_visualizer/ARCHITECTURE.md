# NBA Play Lab architecture

## Request flow

1. Next.js loads named `PlayDefinition` records through the typed API client.
2. FastAPI validates discriminated action and ball-state schemas, then stores
   the structured JSON through `PlayRepository`.
3. The editor builds an immutable definition with typed basketball actions;
   Phase 1 `PlayerRoute` waypoints remain supported for compatibility.
4. The pure frontend engine snapshots the play and samples `SimulationFrame`
   output from deterministic simulation time.
5. React passes each output frame into court/player rendering modules.
6. The coordinate adapter alone maps court feet to SVG units and pointer input
   back to court coordinates.

## Frontend modules

- `app`: Next.js routing, page composition, and global styles.
- `frontend/court`: SVG court geometry and coordinate transforms.
- `frontend/players`: player marker rendering.
- `frontend/editor`: play library, action composer, action timeline, and editor
  orchestration.
- `frontend/animation`: action/route interpolation, ball-state transitions,
  simulation sampling, and the frame-rate-independent clock.
- `frontend/defense`: screen-event detection, explicit matchup timelines,
  coverage strategy implementations, and defensive waypoint generation.
- `frontend/analytics`: typed frame analytics, pure court geometry, heuristic
  scoring, and cadence-based snapshot memoization.
- `frontend/controls`: coverage and later play controls.
- `frontend/data`: HTTP client boundary.
- `frontend/domain`: TypeScript versions of API/domain contracts.

## Backend modules

- `app/api`: HTTP-only concerns and dependency wiring.
- `app/schemas`: canonical domain and API objects.
- `app/simulation`: deterministic examples and four generic seed plays.
- `app/basketball`: court constants and basketball invariants.
- `app/providers`: external NBA data provider interfaces.
- `app/persistence`: repository interfaces and SQLite implementation.
- `app/analytics`: pure analysis functions over structured state.

Dependencies point inward: transport and storage depend on domain contracts;
domain rules never depend on FastAPI, SQLite, pandas, React, or SVG.

## NBA data flow

`NBAApiProvider` is the only module aware of `nba_api`. It converts NBA.com
result sets into internal team, player, roster, game, and player-game-stat
schemas. `NBADataService` resolves normal reads from SQLite, performs explicit
bounded refreshes, and falls back to cached records when the provider fails.
FastAPI exposes only normalized cache results to the frontend.

The provider protocol also defines play-by-play, shot-chart, player-stat, and
team-stat operations so another licensed or local provider can replace
NBA.com without changing editor components. Public NBA data is not treated as
player-tracking data.

## Intentionally deferred

- Undo/redo and multi-select editing
- Editing route waypoint timing and route speed per player
- Help rotations beyond the two defenders directly involved in a screen
- Passing-lane, driving-lane, openness, and spacing visual layers
- Real-possession ingestion and licensed tracking-data adapters
- Expected-points models and model evaluation
- Authentication, multi-user storage, and Postgres implementation
- Generated TypeScript contracts
- Any LLM explanation layer
