import type { CourtPosition, PlayerState, SimulationFrame } from '@/frontend/domain/models'
import { RIM_POSITION } from '@/frontend/defense/geometry'
import {
  DEFAULT_ANALYTICS_CONFIG,
  type AnalyticsConfig,
  type CourtRegion,
  type DriveLaneEvaluation,
  type FrameAnalytics,
  type NearestDefenderEvaluation,
  type PassingLaneEvaluation,
  type RiskBand,
  type ShotOpennessEvaluation,
  type SpacingEvaluation,
} from './types'
import {
  clampScore,
  approachAngleDegrees,
  closingSpeed,
  distance,
  pointToSegmentDistance,
  relativeAngleToBasket,
} from './geometry'

export function calculateFrameAnalytics(
  frame: SimulationFrame,
  config: AnalyticsConfig = DEFAULT_ANALYTICS_CONFIG,
  cadenceSeconds = 0.1,
): FrameAnalytics {
  const started = performance.now()
  const offense = frame.players.filter((state) => state.teamSide === 'offense')
  const defense = frame.players.filter((state) => state.teamSide === 'defense')
  const handlerId = frame.ball.state === 'possessed' ? frame.ball.playerId : null
  const nearestDefenders = offense.map((player) => nearestDefender(player, defense))
  return {
    sampledAtSeconds: frame.timestampSeconds,
    cadenceSeconds,
    nearestDefenders,
    passingLanes: handlerId === null
      ? []
      : passingLanes(handlerId, offense, defense, config.passCorridorWidthFeet),
    spacing: spacing(offense, handlerId),
    driveLane: handlerId === null
      ? null
      : driveLane(handlerId, offense, defense, config),
    shotOpenness: offense.map((player) => evaluateShotOpenness(
      player,
      defense,
      config.nearbyDefenderRadiusFeet,
    )),
    computationDurationMs: performance.now() - started,
    heuristicOnly: true,
  }
}

function nearestDefender(
  offensivePlayer: PlayerState,
  defenders: PlayerState[],
): NearestDefenderEvaluation {
  const nearest = [...defenders].sort(
    (left, right) => distance(left.position, offensivePlayer.position)
      - distance(right.position, offensivePlayer.position),
  )[0]
  if (nearest === undefined) {
    return {
      offensivePlayerId: offensivePlayer.player.id,
      defenderId: null,
      distanceFeet: null,
      relativeAngleDegrees: null,
      defenderClosingSpeedFeetPerSecond: null,
    }
  }
  return {
    offensivePlayerId: offensivePlayer.player.id,
    defenderId: nearest.player.id,
    distanceFeet: distance(offensivePlayer.position, nearest.position),
    relativeAngleDegrees: relativeAngleToBasket(
      offensivePlayer.position,
      nearest.position,
      RIM_POSITION,
    ),
    defenderClosingSpeedFeetPerSecond: closingSpeed(
      nearest.position,
      { x: nearest.velocityX, y: nearest.velocityY },
      offensivePlayer.position,
    ),
  }
}

function passingLanes(
  handlerId: string,
  offense: PlayerState[],
  defense: PlayerState[],
  corridorWidthFeet: number,
): PassingLaneEvaluation[] {
  const handler = offense.find((state) => state.player.id === handlerId)
  if (handler === undefined) return []
  return offense.filter((target) => target.player.id !== handlerId).map((target) => {
    const distances = defense.map((defender) => ({
      defenderId: defender.player.id,
      ...pointToSegmentDistance(defender.position, handler.position, target.position),
    }))
    const minimum = distances.reduce(
      (value, item) => Math.min(value, item.distanceFeet),
      Number.POSITIVE_INFINITY,
    )
    const intersecting = distances.filter(
      (item) => item.projection > 0.05
        && item.projection < 0.95
        && item.distanceFeet <= corridorWidthFeet,
    )
    const passDistanceFeet = distance(handler.position, target.position)
    const minimumDistance = Number.isFinite(minimum) ? minimum : null
    const risk = clampScore(
      (minimumDistance === null ? 0 : Math.max(0, 1 - minimumDistance / corridorWidthFeet) * 55)
      + intersecting.length * 18
      + Math.min(passDistanceFeet / 40, 1) * 15,
    )
    return {
      handlerId,
      targetPlayerId: target.player.id,
      source: { ...handler.position },
      target: { ...target.position },
      passDistanceFeet,
      minimumDefenderDistanceFeet: minimumDistance,
      intersectingDefenderIds: intersecting.map((item) => item.defenderId),
      corridorWidthFeet,
      interceptionRiskScore: risk,
      riskBand: riskBand(risk),
    }
  })
}

function spacing(offense: PlayerState[], handlerId: string | null): SpacingEvaluation {
  const pairDistances = offense.flatMap((first, index) => offense.slice(index + 1).map((second) => ({
    firstPlayerId: first.player.id,
    secondPlayerId: second.player.id,
    distanceFeet: distance(first.position, second.position),
  })))
  const nearestTeammates = offense.map((player) => {
    const options = offense
      .filter((teammate) => teammate.player.id !== player.player.id)
      .map((teammate) => ({
        teammateId: teammate.player.id,
        distanceFeet: distance(player.position, teammate.position),
      }))
      .sort((left, right) => left.distanceFeet - right.distanceFeet)
    return {
      playerId: player.player.id,
      teammateId: options[0]?.teammateId ?? null,
      distanceFeet: options[0]?.distanceFeet ?? null,
    }
  })
  const occupiedRegions = emptyRegions()
  offense.forEach((player) => occupiedRegions[regionFor(player.position)].push(player.player.id))
  const handler = offense.find((player) => player.player.id === handlerId)
  const strongSide = handler === undefined || Math.abs(handler.position.y - 25) < 3
    ? 'middle'
    : handler.position.y < 25 ? 'right' : 'left'
  const strong = strongSide === 'middle'
    ? offense
    : offense.filter((player) => strongSide === 'right'
      ? player.position.y < 25
      : player.position.y > 25)
  const weak = strongSide === 'middle'
    ? []
    : offense.filter((player) => !strong.includes(player))
  const weakPairs = weak.flatMap((first, index) => weak.slice(index + 1).map(
    (second) => distance(first.position, second.position),
  ))
  return {
    nearestTeammates,
    pairDistances,
    occupiedRegions,
    paintOccupancy: occupiedRegions.paint.length,
    cornerOccupancy: {
      right: occupiedRegions.right_corner.length,
      left: occupiedRegions.left_corner.length,
    },
    strongSide,
    strongSidePlayerCount: strong.length,
    weakSidePlayerCount: weak.length,
    weakSideAverageSeparationFeet: weakPairs.length === 0
      ? null
      : weakPairs.reduce((sum, value) => sum + value, 0) / weakPairs.length,
  }
}

function driveLane(
  handlerId: string,
  offense: PlayerState[],
  defense: PlayerState[],
  config: AnalyticsConfig,
): DriveLaneEvaluation | null {
  const handler = offense.find((state) => state.player.id === handlerId)
  if (handler === undefined) return null
  const distances = defense.map((defender) => ({
    defenderId: defender.player.id,
    ...pointToSegmentDistance(defender.position, handler.position, RIM_POSITION),
  }))
  const blockers = distances.filter((item) => item.projection > 0.05
    && item.projection < 1
    && item.distanceFeet <= config.driveCorridorWidthFeet)
  const help = distances.filter((item) => item.projection > 0.05
    && item.projection < 1
    && item.distanceFeet > config.driveCorridorWidthFeet
    && item.distanceFeet <= config.helpDistanceFeet)
  const minimum = distances.reduce(
    (value, item) => Math.min(value, item.distanceFeet),
    Number.POSITIVE_INFINITY,
  )
  return {
    handlerId,
    source: { ...handler.position },
    target: { ...RIM_POSITION },
    corridorWidthFeet: config.driveCorridorWidthFeet,
    blockingDefenderIds: blockers.map((item) => item.defenderId),
    helpDefenderIds: help.map((item) => item.defenderId),
    minimumLateralDefenderDistanceFeet: Number.isFinite(minimum) ? minimum : null,
    opennessScore: clampScore(100 - blockers.length * 36 - help.length * 11
      - Math.max(0, config.driveCorridorWidthFeet - minimum) * 5),
  }
}

export function evaluateShotOpenness(
  shooter: PlayerState,
  defense: PlayerState[],
  nearbyRadiusFeet: number,
): ShotOpennessEvaluation {
  const nearest = nearestDefender(shooter, defense)
  const nearestState = defense.find((defender) => defender.player.id === nearest.defenderId)
  const nearby = defense.filter(
    (defender) => distance(defender.position, shooter.position) <= nearbyRadiusFeet,
  )
  const nearestDistance = nearest.distanceFeet
  return {
    offensivePlayerId: shooter.player.id,
    location: { ...shooter.position },
    nearestDefenderId: nearest.defenderId,
    nearestDefenderDistanceFeet: nearestDistance,
    defenderClosingSpeedFeetPerSecond: nearest.defenderClosingSpeedFeetPerSecond,
    defenderApproachAngleDegrees: nearestState === undefined
      ? null
      : approachAngleDegrees(
          nearestState.position,
          { x: nearestState.velocityX, y: nearestState.velocityY },
          shooter.position,
        ),
    shotDistanceFeet: distance(shooter.position, RIM_POSITION),
    nearbyDefenderCount: nearby.length,
    opennessScore: clampScore(
      (nearestDistance === null ? 100 : Math.min(nearestDistance / 8, 1) * 82)
      + (nearest.defenderClosingSpeedFeetPerSecond ?? 0) * -1.5
      - Math.max(nearby.length - 1, 0) * 12,
    ),
  }
}

function emptyRegions(): Record<CourtRegion, string[]> {
  return {
    paint: [], right_corner: [], left_corner: [], right_wing: [], left_wing: [],
    top: [], other: [],
  }
}

function regionFor(position: CourtPosition): CourtRegion {
  if (position.x >= 75 && position.y >= 17 && position.y <= 33) return 'paint'
  if (position.x >= 84 && position.y <= 7) return 'right_corner'
  if (position.x >= 84 && position.y >= 43) return 'left_corner'
  if (position.y < 18) return 'right_wing'
  if (position.y > 32) return 'left_wing'
  if (position.x < 75) return 'top'
  return 'other'
}

function riskBand(score: number): RiskBand {
  return score >= 66 ? 'high' : score >= 33 ? 'moderate' : 'low'
}
