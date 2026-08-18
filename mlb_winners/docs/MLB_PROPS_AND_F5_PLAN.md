# MLB Props, NRFI, And F5 Expansion

This document tracks the move beyond full-game moneyline into markets that are more directly modelable from pitcher, lineup, park, and weather inputs. This expansion reuses shared MLB data utilities, DuckDB caching, odds math, and feature generation. It does not replace or modify the production moneyline model.

## Phase 1: Pitcher Strikeout Props

Implemented first because pitcher strikeouts have a clean player-level target and a direct projection path.

### Target

- `starter_strikeouts`
- `projected_strikeouts`
- `over_result = starter_strikeouts > prop_line`
- Push when `starter_strikeouts == prop_line`

### Projection

The baseline formula is:

```text
expected_Ks = expected_batters_faced * projected_K_rate
```

`expected_batters_faced` uses:

- projected starter innings
- pitch/workload trend
- starter role and season starts
- starter run prevention quality
- opponent offensive strength

`projected_K_rate` uses:

- pitcher K/BB profile
- recent pitcher K rate from cached player logs
- opponent team K rate
- projected lineup K rate when available
- velocity and spin when available

The daily model now blends the formula baseline with a regression model:

- Primary: XGBoost regressor when available
- Fallback: scikit-learn histogram gradient regressor
- Blend: learned projection plus baseline projection

The projection is converted into over/under probabilities with a Poisson approximation, then compared against current prop odds using no-vig implied probability, edge, EV per dollar, and risk filters.

### Risk Filters

No pitcher K bet when:

- probable starter is missing or uncertain
- prop odds are missing
- prop odds are stale for today's slate
- edge is below the dynamic threshold
- model uncertainty is high
- starter history is too thin
- line movement is sharply against the selected side

### CLI

- `fetch-player-props --date YYYY-MM-DD --market strikeouts`
- `predict-pitcher-k-today --date YYYY-MM-DD --fetch-odds`
- Existing compatible path: `predict-props-today --date YYYY-MM-DD --market strikeouts --fetch-odds`
- `grade-props --date YYYY-MM-DD`

### Evaluation

Pitcher K evaluation should report:

- MAE
- RMSE
- over/under accuracy by line bucket
- calibration by projected probability bucket
- ROI only when historical prop odds or forward-collected pregame prop odds exist

## Phase 2: NRFI Scaffold

Implemented as scaffold only. Do not treat as bettable until trained and validated.

### Target

- No run scored in first inning
- Model components:
  - `P(away scores top 1st)`
  - `P(home scores bottom 1st)`
  - `NRFI = (1 - away_score_prob) * (1 - home_score_prob)`

### Features

- top-lineup strength
- starter first-inning proxy from K/BB/FIP/barrel profile
- park run factor
- weather
- recent offense quality

### CLI

- `predict-nrfi-today --date YYYY-MM-DD`

## Phase 3: First Five Scaffold

Implemented as scaffold only. It is starter-heavy by design and avoids bullpen-heavy features.

### Targets

- first five inning run differential
- first five inning total runs

### Features

- starter FIP/xwOBA/contact quality
- lineup strength
- recent team offensive quality
- park and weather

### CLI

- `predict-f5-today --date YYYY-MM-DD`

## Phase 4: Hitter Total Bases Scaffold

Hitter total bases uses the existing player prop path and requires confirmed lineups.

### Target

- hitter total bases
- over/under versus current prop line

### Features

- confirmed lineup flag
- batting order
- hitter total bases per PA from cached logs
- park run factor
- opposing starter context
- future additions: hitter xSLG, xwOBA, barrel rate, hard-hit rate, pitcher handedness, pitcher xwOBA allowed

### CLI

- `fetch-player-props --date YYYY-MM-DD --market total_bases`
- `predict-hitter-tb-today --date YYYY-MM-DD --fetch-odds`

## Phase 5: Qualified Player Prop Value Layer

Implemented as a conservative daily qualification layer for individually priced player props. It does not force a slate size and may return no plays.

### Markets

- Batter Hits + Runs + RBIs over 1.5 (`batter_hits_runs_rbis`)
- Starting pitcher strikeouts over sportsbook main and alternate lines (`pitcher_strikeouts`)

### Batter HRR Projection

The HRR model removes batter-vs-pitcher batting average as a primary rule. Current inputs are limited to cached data:

- confirmed lineup status
- batting order and projected plate appearances
- player historical hits+runs+RBI per PA with shrinkage
- cached player Statcast quality when available: xBA, xwOBA, xSLG, hard-hit rate, barrel rate
- recent team run context and opposing starter run-prevention context
- park/weather flags when available

When player Statcast is sparse, the command labels the row as a fallback calculation and lowers data quality. Raw batting average is not used as a dominant qualification rule.

### Pitcher K Projection

Pitcher strikeouts reuse the existing starter strikeout projection path and regression blend. The value layer then recomputes the over probability for each available sportsbook line rather than applying one default probability to every line. Rows are downgraded or rejected for missing starters, thin history, short workload/opening roles, stale odds, low data quality, or insufficient price edge.

### Odds And Qualification

For every candidate with odds, the command stores:

- sportsbook, market, line, odds and odds timestamp
- no-vig market probability from over/under prices
- model probability and fair odds
- probability edge and EV per dollar
- data-quality score and rejection reason

Default qualification thresholds:

- minimum probability edge: 4 percentage points
- minimum EV: 5%
- minimum data quality: 0.75
- maximum odds age: 8 hours
- confirmed lineup required for HRR props

Thresholds are CLI-configurable and should not be optimized on the same games used for evaluation.

### Parlays

Parlays are secondary. The command builds 2-, 3-, and 4-leg combinations only from already qualified positive-EV singles. Same-game combinations receive explicit correlation haircuts, including a negative adjustment for a pitcher strikeout over paired with an opposing batter HRR over. Same-team HRR legs are also adjusted because they share run environment and scoring opportunities. If the correlation-adjusted joint EV is not positive, no parlay is returned.

### CLI

```bash
python -m mlb_winners.cli predict-qualified-player-props --date YYYY-MM-DD
python -m mlb_winners.cli predict-qualified-player-props --date YYYY-MM-DD --fetch-odds
python -m mlb_winners.cli predict-qualified-player-props --date YYYY-MM-DD --telegram
```

Useful threshold overrides:

```bash
python -m mlb_winners.cli predict-qualified-player-props --date YYYY-MM-DD \
  --min-edge 0.04 \
  --min-ev 0.05 \
  --min-data-quality 0.75 \
  --max-odds-age-hours 8
```

Outputs:

- `qualified_player_props_YYYY-MM-DD.csv`
- `rejected_player_props_YYYY-MM-DD.csv`
- `qualified_player_prop_parlays_2leg_YYYY-MM-DD.csv`
- `qualified_player_prop_parlays_3leg_YYYY-MM-DD.csv`
- `qualified_player_prop_parlays_4leg_YYYY-MM-DD.csv`

The same run writes `qualified_player_prop_snapshots` for auditability and stores qualified singles in the existing prop recommendation tables so `grade-props` can settle them when boxscores are available.

## Database Tables

Added or supported:

- `player_prop_lines`
- `player_prop_predictions`
- `player_prop_recommendations`
- `qualified_player_prop_snapshots`
- `nrfi_predictions`
- `f5_predictions`

Existing compatible tables remain:

- `props_predictions`
- `prop_recommendations`

## Betting Discipline

These markets should be handled more conservatively than moneyline until enough forward-collected odds exist. Do not claim ROI without:

- imported historical prop odds
- authorized historical prop odds access
- forward-collected prop snapshots saved before first pitch

The first production candidate is pitcher strikeouts. NRFI, F5, and hitter total bases are scaffolded for workflow and data collection, not final betting.
