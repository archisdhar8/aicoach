# AI Coach

Sports analytics, decision-support, and simulation projects for MLB and NBA
workflows. Each application is self-contained, uses deterministic numerical
logic for its core outputs, and keeps external data access behind explicit
provider and caching boundaries.

## Projects

| Project | Purpose | Stack |
| --- | --- | --- |
| [MLB Winners](./mlb_winners/) | Pregame MLB probability modeling, market comparison, simulations, props, historical evaluation, and a live dashboard | Python, DuckDB, scikit-learn, XGBoost, FastAPI |
| [NBA Play Lab](./nba_visualizer/) | Interactive half-court play design, structured basketball actions, deterministic defensive coverages, spatial analytics, and player-aware shot simulation | Next.js, React, TypeScript, SVG, FastAPI, Pydantic, SQLite |

## MLB Winners

The MLB project builds leakage-safe pregame features from historical games,
team form, probable starters, bullpen usage, Statcast, weather, and park
context. It can generate win probabilities, compare them with available market
prices, run deterministic simulations, evaluate several player-prop markets,
and serve a local live dashboard.

Core principles:

- Chronological features use only information available before first pitch.
- MLB, Statcast, weather, and odds requests are cached locally.
- Current odds are market-comparison inputs, not automatic model labels.
- Betting ROI is not reported without valid historical or forward-collected
  pregame odds.
- Missing starters, lineups, weather, or prices remain visible in outputs.

Quick start:

```bash
cd mlb_winners
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python -m mlb_winners.cli --help
```

See the [MLB documentation](./mlb_winners/README.md) for data commands,
training, backtesting, predictions, simulations, props, and dashboard usage.

## NBA Play Lab

The NBA application is a basketball whiteboard and deterministic play
simulator. Users can import current NBA rosters, position ten players, draw
cuts and dribbles, add passes, screens, and shots, choose defensive coverage,
and animate the possession on an SVG half-court.

Current capabilities include:

- Structured MOVE, CUT, DRIBBLE, SCREEN, PASS, SHOOT, and HOLD actions.
- Drop, Switch, Hedge, Blitz, ICE, and offense-only defensive modes.
- Explicit ball ownership and timed pass/shot flight.
- Player-aware shooting estimates using cached NBA season totals, shot
  distance, sample size, and defender proximity.
- Passing-lane, spacing, driving-lane, shot-openness, and matchup overlays.
- Saved plays and provenance-aware manual reconstruction of public
  play-by-play possessions.

Quick start:

```bash
cd nba_visualizer/backend
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```bash
cd nba_visualizer
npm install
npm run dev
```

Open `http://localhost:3001`. See the
[NBA documentation](./nba_visualizer/README.md) for architecture, data-source
limitations, testing, defensive rules, and analytics definitions.

## Repository Layout

```text
.
├── mlb_winners/      # MLB models, data workflows, CLI, dashboard, tests
└── nba_visualizer/   # NBA web editor, simulation engine, API, tests
```

Local databases, downloaded datasets, virtual environments, dependency
folders, credentials, test artifacts, and build output are intentionally not
versioned.

## Important Boundaries

- Probabilities and simulations are decision-support outputs, not guarantees.
- The NBA public statistics APIs do not provide complete frame-by-frame player
  and ball tracking; the visualizer never presents reconstructed routes as
  observed historical movement.
- The player-aware NBA shot estimate is a transparent teaching heuristic, not
  a calibrated expected field-goal model.
- LLMs are not used to invent numeric sports outcomes.
