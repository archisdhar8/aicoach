# NBA Play Lab

An isolated foundation for an interactive NBA play visualizer. The frontend is
Next.js/TypeScript with SVG rendering; the backend is FastAPI/Pydantic with a
SQLite repository behind a storage interface.

The editor opens as a blank basketball whiteboard, with four deterministic demo
plays available in Advanced. There is no action-type toolbar: a short drag
repositions a player, a longer ball-handler path becomes DRIBBLE, and a longer
off-ball path becomes CUT. Clicking an offensive teammate creates a PASS and
selects the receiver; clicking the rim with the ball handler creates SHOOT.
“Add screen” opens a simple off-ball player picker; choosing the screener creates
a visible SCREEN at that player's current location and returns control to the
ball handler. Timing is derived from route distance and ball speed, while the
detailed action form and timeline stay available in Advanced.

Team selection is kept at the top of the workspace. Choosing a team immediately
loads current-roster players with their current jersey numbers. Recorded
starters are used only if they still belong to that roster; during the offseason,
recent playing roles provide the fallback ordering.
Formation mode then allows unrestricted player placement before “Done
positioning” switches the court into action-drawing mode.

Passes transfer possession only when the ball arrives, dribbles keep the ball
with the handler, and demo shots travel to the rim using an explicit made or
missed result. No trained make/miss model, tracking-data claim, expected-points
model, or LLM is present.

Whiteboard-created shots use a deterministic player-aware teaching heuristic.
When cached NBA season totals are available, the engine uses the shooter's 2P%
or 3P% and attempt volume, shrinks small samples toward a league reference, then
adjusts for shot distance and nearest-defender proximity. The result is stable
for the saved action and shown as `+2`, `+3`, or `MISS`. A visible audit card
identifies the season, sample size, distance, defender distance, and fallback.
This is not a tracking-derived or empirically calibrated expected field-goal model.

Cached NBA.com roster data can be applied to either five-player lineup without
changing the simulation coordinate system. See [DATA.md](./DATA.md) for the
exact `nba_api` calls, cache behavior, failure semantics, preload command, and
tracking-data limitations.

The visualizer also generates deterministic pick-and-roll defense for Drop,
Switch, Hedge, Blitz, and side-only ICE. Offense-only playback, configurable
teaching parameters, matchup transitions, and debug overlays are available.
See [DEFENSE.md](./DEFENSE.md) for the rule and assignment model.

Each generated frame also carries deterministic nearest-defender, passing-lane,
spacing, driving-lane, and heuristic shot-openness analytics. Selective SVG
overlays and the Drop-versus-Blitz comparison consume typed results from the
domain engine. See [ANALYTICS.md](./ANALYTICS.md) for definitions and cadence.

Public NBA play-by-play can now be searched by game and normalized into
provenance-aware possessions. Known lineups and events can seed a separate
manual reconstruction, but the application never invents historical player
movement. See [POSSESSIONS.md](./POSSESSIONS.md) for providers, cache behavior,
provenance, and public-data limitations.

## Local development

Terminal 1:

```bash
cd nba_visualizer/backend
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Terminal 2:

```bash
cd nba_visualizer
npm install
npm run dev
```

Open <http://localhost:3001>. API documentation is at
<http://localhost:8000/docs>.

## Checks

```bash
npm run lint
npm run typecheck
npm test
npm run build
npm run test:e2e

cd backend
.venv/bin/ruff check app tests
.venv/bin/mypy app
.venv/bin/pytest
```

## Boundaries

- Domain coordinates are feet on a regulation 94-by-50-foot court. The origin
  is the left baseline/sideline corner; `x` runs baseline to baseline and `y`
  runs sideline to sideline.
- Only `frontend/court/rendering/coordinates.ts` converts domain coordinates to
  SVG pixels.
- React components render and edit structured state. Pure deterministic frame
  sampling lives in `frontend/animation`; Pydantic independently validates the
  same persisted contracts at the API boundary.
- `PlayRepository` isolates persistence so SQLite can later be replaced by
  Postgres without changing API or simulation code.
- Pydantic and TypeScript contracts are duplicated for now. OpenAPI-based type
  generation is intentionally deferred until the API stabilizes.
- `PlayDefinition` is immutable simulation input. `SimulationResult` snapshots
  it, and `SimulationFrame` is sampled output; playback never edits the source
  play.
- Actions and ball states are discriminated unions rather than optional-field
  dictionaries. SVG markup is never persisted.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the module map and phase boundaries.
