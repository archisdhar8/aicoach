import type { CourtPosition, Player, PlayerState } from '@/frontend/domain/models'

const RIM: CourtPosition = { x: 88.75, y: 25 }

export interface HeuristicShotOutcome {
  result: 'made' | 'missed'
  pointValue: 2 | 3
  shotDistanceFeet: number
  nearestDefenderDistanceFeet: number | null
  makeProbabilityHeuristic: number
  deterministicRoll: number
  profileSource: 'nba_season_totals' | 'league_fallback'
  profileSeason: string | null
  profileAttempts: number
  playerBaselineProbability: number
  distanceAdjustment: number
  pressureAdjustment: number
}

export function evaluatePlayerAwareShot(
  actionId: string,
  player: Player,
  source: CourtPosition,
  defenders: PlayerState[],
): HeuristicShotOutcome {
  const shotDistanceFeet = distance(source, RIM)
  const pointValue = shotPointValue(source)
  const nearestDefenderDistanceFeet = defenders.length === 0
    ? null
    : Math.min(...defenders.map((defender) => distance(source, defender.position)))
  const locationProbability = shotDistanceFeet <= 4
    ? 0.7
    : shotDistanceFeet <= 10
      ? 0.53
      : shotDistanceFeet <= 16
        ? 0.45
        : shotDistanceFeet < 23.75
          ? 0.4
          : 0.35
  const leagueReference = pointValue === 3 ? 0.36 : 0.55
  const profileRate = pointValue === 3
    ? player.shootingProfile?.threePointPercentage
    : player.shootingProfile?.twoPointPercentage
  const profileAttempts = pointValue === 3
    ? player.shootingProfile?.threePointAttempts ?? 0
    : Math.max(
        (player.shootingProfile?.fieldGoalAttempts ?? 0)
          - (player.shootingProfile?.threePointAttempts ?? 0),
        0,
      )
  const hasPlayerProfile = profileRate !== null
    && profileRate !== undefined
    && profileAttempts > 0
  const reliability = hasPlayerProfile
    ? profileAttempts / (profileAttempts + (pointValue === 3 ? 180 : 240))
    : 0
  const playerBaselineProbability = hasPlayerProfile
    ? leagueReference + (profileRate - leagueReference) * reliability
    : leagueReference
  const distanceAdjustment = locationProbability - leagueReference
  const pressureAdjustment = nearestDefenderDistanceFeet === null
    ? 0.03
    : nearestDefenderDistanceFeet < 2
      ? -0.24
      : nearestDefenderDistanceFeet < 4
        ? -0.14
        : nearestDefenderDistanceFeet < 6
          ? -0.06
          : nearestDefenderDistanceFeet >= 8
            ? 0.04
            : 0
  const makeProbabilityHeuristic = clamp(
    playerBaselineProbability + distanceAdjustment + pressureAdjustment,
    0.08,
    0.82,
  )
  const deterministicRoll = stableUnitInterval(
    `${actionId}:${player.id}:${source.x.toFixed(2)}:${source.y.toFixed(2)}`,
  )
  return {
    result: deterministicRoll < makeProbabilityHeuristic ? 'made' : 'missed',
    pointValue,
    shotDistanceFeet,
    nearestDefenderDistanceFeet,
    makeProbabilityHeuristic,
    deterministicRoll,
    profileSource: hasPlayerProfile ? 'nba_season_totals' : 'league_fallback',
    profileSeason: hasPlayerProfile ? player.shootingProfile?.season ?? null : null,
    profileAttempts,
    playerBaselineProbability,
    distanceAdjustment,
    pressureAdjustment,
  }
}

export function shotPointValue(source: CourtPosition): 2 | 3 {
  const beyondArc = distance(source, RIM) >= 23.75
  const beyondCornerLine = (source.y <= 3 || source.y >= 47) && source.x <= RIM.x
  return beyondArc || beyondCornerLine ? 3 : 2
}

function stableUnitInterval(value: string): number {
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0) / 4294967296
}

function distance(first: CourtPosition, second: CourtPosition): number {
  return Math.hypot(second.x - first.x, second.y - first.y)
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum)
}
