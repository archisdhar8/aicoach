# MLB Moneyline Reset Analysis - 2026-07-08

This audit reviewed saved `moneyline_candidate_snapshots` from 2026-06-08 through 2026-07-08 and compared official strong/medium plays against available final scores in DuckDB.

## What Went Wrong

- Official strong/medium plays were too frequent: 97 candidates, 87 final-graded after requiring `status = Final`.
- Final-status-corrected official record: 28-59, -23.35 units.
- Strong plays were worse than medium plays:
  - Strong: 12-34, -17.28 units.
  - Medium: 16-25, -6.07 units.
- The worst segment was high projected EV / huge edge, not low edge:
  - Edge 10%+: 10-33, -18.36 units.
  - EV 20%+: 6-27, -15.75 units.
- Underdogs drove the largest losses:
  - Underdogs: 18-44, -16.37 units.
  - Strong underdogs included many unrealistic prices such as +800, +1300, +2500, +3300.
- The old live probability blend was too model-heavy:
  - `0.9 * model + 0.1 * market`
  - This allowed the model to fight the sportsbook too aggressively and produce fake value on extreme prices.
- Weekly stats previously graded rows with scores even when game status was not final. That could count scheduled or incomplete 0-0 rows incorrectly.

## Implemented Changes

- Live moneyline probability now uses the market as the anchor:
  - `market_adjusted = market_prob + 0.30 * (model_prob - market_prob)`
  - This keeps the model as an adjustment layer instead of the whole opinion.
- Tightened official moneyline thresholds:
  - Base edge threshold: 4%.
  - Favorite threshold: 6%.
  - Underdog threshold: 5.5%.
  - Strong requires at least 10% edge and 10% EV.
  - Medium requires at least 7% edge and 5% EV.
- Added hard no-bet filters:
  - `longshot_market_outlier` for official underdog prices above +250.
  - `heavy_favorite_price` for official favorite prices below -300.
  - `market_model_dislocation` for edge >= 18% or EV >= 35%.
- Applied the same market-prior probability to V2 predictions.
- Fixed weekly grading so only games with final status count as graded.
- Kept prop betting behavior separate with a prop-specific filter config, so pitcher strikeout props are not accidentally over-tightened by moneyline rules.

## Backward Simulation

Applying the new rules to the same saved final-graded candidates would have reduced official plays from 87 to 4. That is intentionally conservative. The old system's problem was not missing enough plays; it was letting too many fragile disagreements become official bets.

## Next Improvements

- Rebuild candidate tracking for two weeks using the new market-prior rules before loosening thresholds.
- Track skipped plays that would have been old strong/medium and compare their record separately.
- Add a bookmaker sanity check using multiple books when available, rather than trusting one extreme latest price.
- Consider making official alerts moneyline-only no-play by default unless the edge survives:
  - market-prior adjustment,
  - no longshot outlier,
  - confirmed starters/lineups,
  - no adverse line movement,
  - no bullpen/weather risk.
- Keep exploring pitcher strikeout props, because they are more directly modelable than full-game moneylines.
