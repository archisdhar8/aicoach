# Soccer World Cup Agent

This module lives in `/Users/archisdhar/aicoach/soccerworldcup`, next to the MLB project. It is a decision-support agent for FIFA World Cup 2026 matches. It predicts match outcome, over/under 2.5 goals, and both-teams-to-score probabilities, then compares those probabilities to market odds when odds are available.

It is not an auto-betting bot. It does not place bets, size wagers automatically, or claim profitability without valid historical odds.

## Setup

Run commands from `/Users/archisdhar/aicoach/mlb_winners`.

Required for API fetches:

```bash
export API_FOOTBALL_KEY="your-api-football-key"
export ODDS_API_KEY="your-the-odds-api-key"
```

Optional FIFA ranking source:

```bash
export FOOTBALLDATA_IO_API_KEY="your-footballdata-io-key"
```

The CLI loads the project `.env` file automatically, so those values can also be stored there.

## Commands

Fetch schedule, results, teams, and venues:

```bash
python -m mlb_winners.cli fetch-soccer-schedule --start-date 2026-06-11 --end-date 2026-07-19
```

Fetch optional FIFA rankings and build local ratings from cached matches:

```bash
python -m mlb_winners.cli fetch-soccer-ratings --as-of-date 2026-06-11
```

Fetch current World Cup odds:

```bash
python -m mlb_winners.cli fetch-soccer-odds --date 2026-06-14 --markets h2h,totals
```

Predict a slate and write CSV plus Markdown reports:

```bash
python -m mlb_winners.cli predict-soccer-today --date 2026-06-14 --fetch-odds
python -m mlb_winners.cli predict-world-cup --date 2026-06-14 --fetch-odds
```

Run a model-only backtest on completed cached matches:

```bash
python -m mlb_winners.cli backtest-soccer --start-date 2026-06-11 --end-date 2026-07-19
```

Reports are written to `data/reports/`.

## Data Sources And Cache Behavior

- Schedule/results/venues use API-FOOTBALL World Cup 2026 fixtures with `league=1` and `season=2026`.
- Odds use The Odds API sport key `soccer_fifa_world_cup`; V1 supports `h2h` and `totals` from the odds endpoint and will use BTTS prices if a supported feed supplies them.
- Weather uses Open-Meteo when cached venue latitude and longitude are available.
- The root-level `/Users/archisdhar/aicoach/soccerworldcup/worldcup2026.db` schedule is used as a local fallback/source when API-Football does not expose 2026 fixtures for the current plan.
- Raw API payloads are cached in DuckDB `raw_api_cache`.
- Normalized rows are stored in `soccer_matches`, `soccer_team_ratings`, `soccer_odds_snapshots`, `soccer_predictions`, and `soccer_recommendations`.

If API keys or network access are unavailable, fetch commands use cached payloads when possible and otherwise return zero rows with a clear message.

## Model

V1 is intentionally simple and explainable:

- Build team ratings from matches completed before the prediction kickoff.
- Estimate team attack, defense, recent form, goals for/against, rest days, Elo, and optional FIFA ranking.
- Convert those inputs into expected goals for each team.
- Enumerate a Poisson scoreline grid, normalize the grid, and calculate:
  - home/draw/away probability
  - over and under 2.5 probability
  - both-teams-to-score probability
  - top correct scores

Current odds are only used after the model prediction, for no-vig market comparison, edge, EV, confidence, and no-bet filtering.

## Risk Filters

The Risk Agent returns no-bet when:

- supported odds are missing
- odds are stale
- team ratings are missing or very thin
- squad uncertainty is high
- weather uncertainty is high and edge is not large
- model uncertainty is high
- edge or EV is too small

Confidence tiers are `strong`, `medium`, `thin`, `no bet`, and `no odds`.

## Backtesting Limits

`backtest-soccer` is model-only unless valid historical odds or forward-collected pregame odds snapshots exist. It may report prediction metrics such as accuracy and Brier score. It must not report betting ROI, CLV, or profit claims without historical odds coverage saved before kickoff.
