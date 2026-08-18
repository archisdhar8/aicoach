# Next-pitch evaluation — 2026-08-17

## Data coverage

- Pitch events: 2,073,842 rows, 2024-03-20 through 2026-08-16.
- Matchup aggregates: 1,876,637 rows over the same range.
- Backfill ledger: 585 completed game dates, 295 no-data dates, zero remaining failures.
- Backfill writes during the final resumable run: 1,934,088 pitch rows and 1,749,618 matchup rows; earlier cached/idempotent rows account for the database totals above.

## Selected posterior

Three parameter sets were compared on the first 50,000 chronologically evaluated
2025 pitches. The selected strengths were league 20, count 12, matchup 40, and
same-game 36; its log loss was 1.3696 versus 1.3708 and 1.3725 for the alternatives.
The complete 2025 validation contained 752,687 pitches.

| 2025 model | Top-1 | Log loss | Brier |
|---|---:|---:|---:|
| Hierarchical posterior | 44.54% | 1.2787 | 0.6643 |
| Most-thrown pitch | 40.68% | 5.4633 | 1.1847 |
| Overall arsenal | 40.70% | 1.4177 | 0.7073 |
| Count-specific history | 44.01% | 1.3936 | 0.6796 |

## Untouched 2026 forward holdout

The holdout covered 586,380 pitches from 2026-03-04 through 2026-08-16.

| 2026 model | Top-1 | Log loss | Brier |
|---|---:|---:|---:|
| Hierarchical posterior | 42.43% | 1.3395 | 0.6834 |
| Most-thrown pitch | 38.34% | 5.6797 | 1.2318 |
| Overall arsenal | 38.33% | 1.4894 | 0.7279 |
| Count-specific history | 41.87% | 1.4421 | 0.6978 |

The posterior beat every requested baseline and passed the production promotion
gate. Calibration was close through the 0.0–0.9 buckets. The 0.9–1.0 bucket was
overconfident (97.96% predicted versus 77.97% observed over 1,875 pitch-type
probabilities), so extreme probabilities remain a known limitation.

## Count diagnostic

Pitcher 656302 had 3,183 pitches in the rolling evidence window. Against a
right-handed batter, the leading distributions were:

- 0–0: SL 30.0%, FF 29.7%, SI 15.5%.
- 0–2: SL 52.5%, FF 34.0%, ST 6.7%.
- 3–0: FF 64.6%, SL 19.1%, KC 4.8%.
- 3–2: FF 53.0%, SL 37.4%, ST 3.3%.

Maximum pairwise L1 distance was 0.777, demonstrating material count sensitivity
without imposing artificial variation.
