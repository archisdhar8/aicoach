import type { CourtPosition, Coverage, PlayDefinition, PlayerState } from '@/frontend/domain/models'
import { actionEndTime } from '@/frontend/domain/models'
import {
  assignmentAt,
  detectScreenInteractions,
  initialAssignments,
  offensivePositionAt,
} from './events'
import { distance, interpolate, pointToward, RIM_POSITION } from './geometry'
import { BlitzCoverageStrategy } from './strategies/blitz'
import { DropCoverageStrategy } from './strategies/drop'
import { HedgeCoverageStrategy } from './strategies/hedge'
import { IceCoverageStrategy } from './strategies/ice'
import { SwitchCoverageStrategy } from './strategies/switch'
import {
  DEFAULT_DEFENSIVE_PARAMETERS,
  type CoverageStrategy,
  type DefenseMode,
  type DefensiveDebugState,
  type DefensiveParameters,
  type DefensiveResponse,
} from './types'

const STRATEGIES: Readonly<Record<Coverage, CoverageStrategy>> = {
  drop: new DropCoverageStrategy(),
  switch: new SwitchCoverageStrategy(),
  hedge: new HedgeCoverageStrategy(),
  blitz: new BlitzCoverageStrategy(),
  ice: new IceCoverageStrategy(),
}

export function generateDefensiveResponse(
  play: PlayDefinition,
  mode: DefenseMode = 'coverage',
  parameters: DefensiveParameters = DEFAULT_DEFENSIVE_PARAMETERS,
): DefensiveResponse | null {
  if (mode === 'offense_only') return null
  const assignments = initialAssignments(play)
  const events = detectScreenInteractions(play, assignments)
  const coverage = play.initialFrame.possession.coverage
  const response = STRATEGIES[coverage].generateResponse({
    play,
    assignments,
    events,
    parameters,
  })
  return response.supported
    ? addMatchupTracking(play, response, parameters)
    : response
}

function addMatchupTracking(
  play: PlayDefinition,
  response: DefensiveResponse,
  parameters: DefensiveParameters,
): DefensiveResponse {
  const instructed = new Set(response.instructions.map((item) => item.defenderId))
  const duration = play.actions.reduce(
    (latest, action) => Math.max(latest, actionEndTime(action)),
    play.routes.reduce(
      (latest, route) => Math.max(latest, route.points.at(-1)?.timeSeconds ?? 0),
      0,
    ),
  )
  if (duration <= 0) return response
  const cadence = 0.2
  const tracking = play.initialFrame.players
    .filter((state) => state.teamSide === 'defense' && !instructed.has(state.player.id))
    .flatMap((defender) => {
      const waypoints = [{
        timeSeconds: 0,
        position: { ...defender.position },
        state: 'tracking_assignment',
      }]
      let current = { ...defender.position }
      let velocity = { x: 0, y: 0 }
      for (let time = cadence; time <= duration + 1e-9; time += cadence) {
        const reactedTime = Math.max(0, time - parameters.reactionDelaySeconds)
        const assignment = assignmentAt(response.assignments, defender.player.id, reactedTime)
        if (assignment === undefined) continue
        const offense = offensivePositionAt(play, assignment.offensivePlayerId, reactedTime)
        const desired = pointToward(
          offense,
          RIM_POSITION,
          parameters.defensiveCushionFeet,
        )
        const remainingDistance = distance(current, desired)
        const desiredSpeed = Math.min(
          parameters.defenderSpeedFeetPerSecond,
          remainingDistance / cadence,
        )
        const direction = remainingDistance <= 0.001
          ? { x: 0, y: 0 }
          : {
              x: (desired.x - current.x) / remainingDistance,
              y: (desired.y - current.y) / remainingDistance,
            }
        const desiredVelocity = {
          x: direction.x * desiredSpeed,
          y: direction.y * desiredSpeed,
        }
        const velocityDelta = {
          x: desiredVelocity.x - velocity.x,
          y: desiredVelocity.y - velocity.y,
        }
        const deltaMagnitude = Math.hypot(velocityDelta.x, velocityDelta.y)
        const maximumVelocityChange = parameters.maximumAccelerationFeetPerSecondSquared * cadence
        const accelerationScale = deltaMagnitude <= maximumVelocityChange || deltaMagnitude === 0
          ? 1
          : maximumVelocityChange / deltaMagnitude
        velocity = {
          x: velocity.x + velocityDelta.x * accelerationScale,
          y: velocity.y + velocityDelta.y * accelerationScale,
        }
        const proposed = {
          x: current.x + velocity.x * cadence,
          y: current.y + velocity.y * cadence,
        }
        current = distance(current, proposed) >= remainingDistance ? desired : proposed
        waypoints.push({
          timeSeconds: Math.min(time, duration),
          position: { ...current },
          state: 'tracking_assignment',
        })
      }
      return waypoints.length < 2 ? [] : [{
        defenderId: defender.player.id,
        targetOffensivePlayerId: assignmentAt(
          response.assignments,
          defender.player.id,
          0,
        )?.offensivePlayerId,
        waypoints,
      }]
    })
  return { ...response, instructions: [...response.instructions, ...tracking] }
}

export function defensivePositionAt(
  response: DefensiveResponse | null,
  initialState: PlayerState,
  timeSeconds: number,
): CourtPosition {
  const instruction = response?.instructions.find(
    (candidate) => candidate.defenderId === initialState.player.id,
  )
  if (instruction === undefined || instruction.waypoints.length === 0) {
    return { ...initialState.position }
  }
  const points = instruction.waypoints
  if (timeSeconds <= points[0].timeSeconds) return { ...initialState.position }
  for (let index = 1; index < points.length; index += 1) {
    const previous = points[index - 1]
    const current = points[index]
    if (timeSeconds <= current.timeSeconds) {
      const progress = (timeSeconds - previous.timeSeconds)
        / Math.max(current.timeSeconds - previous.timeSeconds, 0.001)
      return interpolate(previous.position, current.position, progress)
    }
  }
  return { ...points.at(-1)?.position ?? initialState.position }
}

export function defensiveDuration(response: DefensiveResponse | null): number {
  return response?.instructions.reduce(
    (duration, instruction) => Math.max(
      duration,
      instruction.waypoints.at(-1)?.timeSeconds ?? 0,
    ),
    0,
  ) ?? 0
}

export function defensiveDebugAt(
  response: DefensiveResponse | null,
  timeSeconds: number,
): DefensiveDebugState | null {
  if (response === null) return null
  const targets = response.instructions.flatMap((instruction) => {
    const next = instruction.waypoints.find((waypoint) => waypoint.timeSeconds >= timeSeconds)
      ?? instruction.waypoints.at(-1)
    return next === undefined ? [] : [{
      defenderId: instruction.defenderId,
      position: { ...next.position },
      state: next.state,
    }]
  })
  const assignments = [...new Set(response.assignments.map((item) => item.defenderId))]
    .flatMap((defenderId) => {
      const assignment = assignmentAt(response.assignments, defenderId, timeSeconds)
      return assignment === undefined ? [] : [{
        defenderId,
        offensivePlayerId: assignment.offensivePlayerId,
      }]
    })
  const activeState = targets.find((target) => target.state !== 'reacting')?.state
  return {
    coverage: response.coverage,
    supported: response.supported,
    unsupportedReason: response.unsupportedReason,
    coverageState: response.supported ? activeState ?? 'waiting_for_screen' : 'unsupported',
    assignments,
    targets,
    screenEvents: response.events,
  }
}

export { DEFAULT_DEFENSIVE_PARAMETERS }
export type { DefenseMode, DefensiveDebugState, DefensiveParameters, DefensiveResponse }
