# Deterministic pick-and-roll defense

The defensive engine is rule-based and contains no LLM decision-making. It detects explicit screen interactions from overlapping `DRIBBLE` and `SCREEN` actions, combines them with time-indexed matchup assignments, and dispatches the resulting event to one isolated coverage strategy.

## Strategies

- **Drop:** the point-of-attack defender navigates and recovers while the screener defender targets a configurable point between the handler and rim.
- **Switch:** after reaction and navigation delay, the two involved defenders exchange matchup assignments.
- **Hedge:** the screener defender briefly contains the handler and then recovers to the screener.
- **Blitz:** both involved defenders target the handler and the screener is marked temporarily exposed.
- **ICE:** supported only when a detected screen is in side pick-and-roll geometry. Middle screens return an explicit unsupported response.

Each strategy returns defensive instructions with absolute simulation times, court-coordinate waypoints, coverage states, assignment history, detected events, and exposed-player metadata. The animation layer samples these instructions independently of browser frame rate and never mutates the offensive play definition.

Defenders without a coverage-specific instruction continuously track their
explicit matchup with a reaction delay, speed limit, acceleration limit,
configurable cushion, and a stance biased toward the rim. This keeps all five defenders moving on cuts, drives, and passes even
when no screen is detected. A valid screen replaces the involved defenders'
generic matchup tracking with the selected coverage behavior. Offense-only mode
still leaves every defender stationary. Editing offense or coverage builds a
new defensive response immediately; faint route previews show that response on
the stopped whiteboard.

## Parameters

The Advanced drawer exposes defender speed, maximum acceleration, defensive
cushion, reaction delay, screen-navigation delay, recovery speed, drop depth,
and help distance. Defaults are plausible teaching values only. They have not
been fitted to tracking data and are not claimed to be empirically calibrated.

Initial assignments pair the five defenders and offensive players in lineup order. Assignment output remains explicit as `defenderId → offensivePlayerId` intervals, and debug overlays show the active assignment, screen event, defensive target, and coverage state.

The seeded **High pick-and-roll** is the coverage lab scenario. The same offensive definition can be animated against Drop, Switch, Hedge, Blitz, or ICE; because its screen is in the middle, ICE deliberately reports unsupported.
