# MLB Winners

Daily MLB moneyline winner and value-bet model.

This project uses free MLB Stats API data for schedules, results, boxscores, and
game history. Current odds are optional and fetched from The Odds API with a
cache-first workflow so a limited monthly quota is not wasted.

## Setup

```bash
cd mlb_winners
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional odds key:

```bash
export ODDS_API_KEY="your-key"
```

Optional SMS alerts use Twilio:

```bash
export TWILIO_ACCOUNT_SID="AC..."
export TWILIO_AUTH_TOKEN="..."
export TWILIO_FROM_NUMBER="+15551234567"
export ALERT_TO_NUMBER="+15557654321"
```

Optional Telegram alerts:

```bash
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"
```

You can also put those same `KEY=value` lines in a local `.env` file at the
project root. The CLI loads `.env` automatically.

## Commands

```bash
python -m mlb_winners.cli fetch-history --start-year 2021 --end-year 2025
python -m mlb_winners.cli fetch-statcast --start-date 2021-03-01 --end-date 2025-11-30
python -m mlb_winners.cli fetch-weather --start-date 2021-03-01 --end-date 2025-11-30
python -m mlb_winners.cli train --train-through 2025 --test-year 2026
python -m mlb_winners.cli analyze-features --train-through 2025 --test-year 2026
python -m mlb_winners.cli backtest --years 2021,2022,2023,2024,2025
python -m mlb_winners.cli backtest-totals --years 2021,2022,2023,2024,2025
python -m mlb_winners.cli fetch-totals-odds --date 2026-05-18
python -m mlb_winners.cli predict-totals-today --date 2026-05-18 --fetch-odds
python -m mlb_winners.cli fetch-odds --date 2026-05-18
python -m mlb_winners.cli market-report --date 2026-05-18
python -m mlb_winners.cli predict-today --date 2026-05-18
python -m mlb_winners.cli send-alerts --date 2026-05-18 --telegram --window-minutes 60
python -m mlb_winners.cli send-lotto-parlay --date 2026-05-18 --telegram --stake-units 0.1
python -m mlb_winners.cli data-integrity --date 2026-05-18
python -m mlb_winners.cli simulate-slate --date 2026-05-18 --sims 20000
python -m mlb_winners.cli simulate-game --date 2026-05-18 --game-pk 823465 --sims 20000
python -m mlb_winners.cli predict-props --date 2026-05-18 --market strikeouts
python -m mlb_winners.cli fetch-player-logs --start-year 2021 --end-year 2025
python -m mlb_winners.cli fetch-player-props --date 2026-05-18 --market strikeouts --max-events 3
python -m mlb_winners.cli predict-props-today --date 2026-05-18 --market strikeouts
python -m mlb_winners.cli import-odds-csv --path data/odds/historical/odds.csv
python -m mlb_winners.cli backtest-portfolio --year 2026 --staking flat
python -m mlb_winners.cli record-results --year 2026
python -m mlb_winners.cli fetch-play-by-play --start-date 2024-03-20 --end-date 2026-07-24
python -m mlb_winners.cli serve-live-dashboard
```

## Live Dashboard

`serve-live-dashboard` starts the local dashboard at
`http://127.0.0.1:8765`. The free MLB live feed is refreshed every five
seconds. The Odds API is never called by a page load, timer, or background
worker; use the **Refresh live odds** button when you intentionally want one
quota-consuming slate request.

Run `fetch-play-by-play` before relying on team-specific state rates. The
command is incremental and idempotent, and the model uses each team's latest
50 completed regular-season games across season boundaries. Until the history
is populated, the dashboard visibly marks the rate as league-shrunk.

`fetch-statcast` also builds count-specific batter/pitcher/pitch-type outcome
cells. The live detail view uses those cells for next-pitch probabilities,
pitch-by-pitch plate-appearance simulation, expected-versus-actual arsenal
usage, velocity changes, whiff/xwOBA results, and win-probability sensitivity.
All rolling pitch features are restricted to dates before the live game.

For the resumable pitch-level history and chronological diagnostics:

```bash
python -m mlb_winners.cli backfill-pitch-matchups --start-date 2024-03-20 --end-date 2026-08-16
python -m mlb_winners.cli evaluate-next-pitch --start-date 2026-03-01 --end-date 2026-08-16
python -m mlb_winners.cli pitch-diagnostic --pitcher-id 123456 --as-of-date 2026-08-17
```

The dashboard labels the leader as the highest-probability pitch and displays
the top three probabilities; simulation samples the entire distribution.

## Data And Quota Policy

- MLB data is cached as raw JSON and normalized into DuckDB tables.
- The Odds API is only used for current/future `baseball_mlb` moneyline odds.
- `fetch-odds` refuses to re-fetch a cached date unless `--force` is passed.
- Historical odds are not fetched automatically because The Odds API historical
  endpoint is quota-expensive and may require a paid tier.

## Stronger Feature Sources

- `fetch-history` stores game results, starter boxscore lines, bullpen workload,
  and team run context.
- `fetch-statcast` stores rolling team and pitcher contact-quality metrics:
  xwOBA, xBA, hard-hit rate, barrel proxy, exit velocity, pitch velocity, spin,
  strikeout rate, and walk rate.
- `fetch-weather` stores park weather context from Open-Meteo plus dome defaults:
  temperature, wind speed/direction, precipitation, and park run factor inputs.
- Training and prediction automatically use these enriched tables when present.
- `analyze-features` writes permutation importance, optional SHAP importance,
  leakage-risk flags, noise candidates, and over-dominant feature warnings.
- Engineered feature snapshots are saved to DuckDB so predictions can be
  reproduced and audited later.

## Simulation, Props, And Portfolio

- `simulate-slate` and `simulate-game` run deterministic Monte Carlo simulations
  from the current feature frame. Outputs include moneyline probabilities,
  total/team-total means, first-five scoring, and total-run percentiles.
- `predict-props` and `predict-props-today` support starter strikeouts, hits
  allowed, earned runs, outs recorded, batter total bases, hits, HR, RBI, and
  runs. Batter markets require lineup/player game stat data from full boxscore
  fetches.
- `fetch-player-props` uses The Odds API event-odds endpoint for one prop market
  at a time. This can cost one request per event, so use `--max-events` while
  testing.
- `data-integrity` flags missing starters, odds, weather, confirmed lineups,
  suspicious lines, and other slate readiness issues.
- `import-odds-csv` provides the historical odds path needed for true ROI
  backtests when the API key does not include paid historical odds access.
- Repeated `fetch-odds --force` calls preserve intraday snapshots; use
  `market-report` to inspect opening/latest lines, movement, velocity, and book
  disagreement. By default odds fetches only `h2h`; pass
  `--markets h2h,spreads,totals` when you intentionally want spreads/totals too.
- `backtest-portfolio` applies exposure limits and flat/Kelly staking to value
  candidates. `record-results` settles stored recommendations from final scores.

## Backtesting

Backtests are model-only unless historical moneyline CSV files are supplied.
Totals backtests are also model-only unless historical totals lines or
forward-collected pregame totals snapshots are supplied.
Place CSV files in `data/odds/historical/` with columns:

```text
game_date,home_team,away_team,home_moneyline,away_moneyline,bookmaker
```

Without historical odds, the report includes accuracy, log loss, Brier score,
and calibration, but skips ROI/profit claims.
