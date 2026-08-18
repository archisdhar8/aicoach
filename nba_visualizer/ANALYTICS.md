# Deterministic spatial analytics

Every sampled simulation frame includes a typed `FrameAnalytics` snapshot. Calculations use canonical court feet and generated player velocities; SVG components only render the resulting objects.

The engine currently returns:

- `NearestDefenderEvaluation` for every offensive player, including distance, angle relative to the basket line, and signed defender closing speed.
- `PassingLaneEvaluation` for each ball-handler-to-teammate lane, including segment clearance, corridor intersections, pass distance, and a 0–100 interception-risk heuristic.
- `SpacingEvaluation` with nearest teammates, all offensive pair distances, occupied regions, paint/corner occupancy, and strong-/weak-side indicators.
- `DriveLaneEvaluation` with geometric blockers, nearby help, minimum lateral clearance, and a 0–100 lane-openness heuristic.
- `ShotOpennessEvaluation` for each offensive location, including shot distance, nearest defender, defender closing speed and approach angle, nearby-defender count, and a 0–100 openness heuristic.

These scores are deterministic teaching heuristics. Shot openness is **not** expected field-goal percentage, and no score is trained or calibrated from outcomes.

## Performance

Visual positions continue sampling at browser animation cadence. Analytics uses a fixed 0.1-second cadence (10 Hz); each quantized snapshot is memoized on the immutable simulation result and reused for intervening visual frames and repeated scrubs. Historical snapshots are never recomputed.

The local profile for a 240-frame, 60 Hz-style pick-and-roll sampling loop generated 21 unique analytics snapshots in 2.338 ms total on the development machine. The test budget is intentionally looser than that observed value to avoid treating one machine as a benchmark guarantee.

## Visualization

Passing lanes, nearest-defender distance, driving lane, spacing, shot openness, and matchup overlays can be toggled independently. Only nearest-defender distance is enabled by default to preserve visual hierarchy.

The seeded High pick-and-roll includes a Drop-versus-Blitz comparison at 1.9 seconds. Both rows are computed from the same offensive definition, with only the deterministic coverage response changed.
