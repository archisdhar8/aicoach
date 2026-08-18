# Next-pitch probability model

## Previous calculation

The first live implementation began with a league pitch mix and applied additive
weights for pitcher usage, count, handedness, same-game usage, workload, and
velocity. Because those weights were accumulated on a common score scale, the
large pitcher and league fastball samples dominated sparse count cells. The UI
then displayed only `argmax(pitch probability)`, making a distribution such as
FF 46%, CH 24%, SL 17%, CU 13% look like a deterministic four-seam call.

## Hierarchical posterior

The replacement is a sequence of conjugate categorical/Dirichlet updates. For
pitch types `k`, each level uses the normalized preceding posterior as its prior:

`p_k = (alpha * prior_k + w * count_k) / (alpha + w * sum(count))`

The levels are league handedness/count context, pitcher overall arsenal, pitcher
count and batter-side mix, exact batter/pitcher/count history, and pitches already
observed from that pitcher in the current game. Bounded multiplicative likelihood
ratios then represent workload, times through the order, velocity loss, recent
whiffs, and damaging contact before a final normalization.

The individual pitcher's arsenal defines plausible support. Pitch types outside
that support share 0.5% probability so data errors or genuinely new offerings do
not receive impossible zero probability. With no pitcher history, the model falls
back to the handedness/count league distribution.

All historical database queries require `game_date < target_game_date`. During
evaluation, a target-game pitch enters same-game evidence only after its own
prediction, and other games on the same date remain excluded until the date is
complete.

## Promotion rule

Prior strengths are selected using 2025 validation data. The untouched 2026
forward period determines promotion: the posterior must beat the pitcher-overall
arsenal baseline on multiclass log loss. If it does not, the failure is reported
and the better validated distribution remains the production choice.
