import type { CourtPosition } from '@/frontend/domain/models'

export type RiskBand = 'low' | 'moderate' | 'high'
export type CourtRegion =
  | 'paint'
  | 'right_corner'
  | 'left_corner'
  | 'right_wing'
  | 'left_wing'
  | 'top'
  | 'other'

export interface NearestDefenderEvaluation {
  offensivePlayerId: string
  defenderId: string | null
  distanceFeet: number | null
  relativeAngleDegrees: number | null
  defenderClosingSpeedFeetPerSecond: number | null
}

export interface PassingLaneEvaluation {
  handlerId: string
  targetPlayerId: string
  source: CourtPosition
  target: CourtPosition
  passDistanceFeet: number
  minimumDefenderDistanceFeet: number | null
  intersectingDefenderIds: string[]
  corridorWidthFeet: number
  interceptionRiskScore: number
  riskBand: RiskBand
}

export interface OffensivePairDistance {
  firstPlayerId: string
  secondPlayerId: string
  distanceFeet: number
}

export interface SpacingEvaluation {
  nearestTeammates: Array<{
    playerId: string
    teammateId: string | null
    distanceFeet: number | null
  }>
  pairDistances: OffensivePairDistance[]
  occupiedRegions: Record<CourtRegion, string[]>
  paintOccupancy: number
  cornerOccupancy: { right: number; left: number }
  strongSide: 'right' | 'left' | 'middle'
  strongSidePlayerCount: number
  weakSidePlayerCount: number
  weakSideAverageSeparationFeet: number | null
}

export interface DriveLaneEvaluation {
  handlerId: string
  source: CourtPosition
  target: CourtPosition
  corridorWidthFeet: number
  blockingDefenderIds: string[]
  helpDefenderIds: string[]
  minimumLateralDefenderDistanceFeet: number | null
  opennessScore: number
}

export interface ShotOpennessEvaluation {
  offensivePlayerId: string
  location: CourtPosition
  nearestDefenderId: string | null
  nearestDefenderDistanceFeet: number | null
  defenderClosingSpeedFeetPerSecond: number | null
  defenderApproachAngleDegrees: number | null
  shotDistanceFeet: number
  nearbyDefenderCount: number
  opennessScore: number
}

export interface FrameAnalytics {
  sampledAtSeconds: number
  cadenceSeconds: number
  nearestDefenders: NearestDefenderEvaluation[]
  passingLanes: PassingLaneEvaluation[]
  spacing: SpacingEvaluation
  driveLane: DriveLaneEvaluation | null
  shotOpenness: ShotOpennessEvaluation[]
  computationDurationMs: number
  heuristicOnly: true
}

export interface AnalyticsConfig {
  passCorridorWidthFeet: number
  driveCorridorWidthFeet: number
  nearbyDefenderRadiusFeet: number
  helpDistanceFeet: number
}

export const DEFAULT_ANALYTICS_CONFIG: Readonly<AnalyticsConfig> = {
  passCorridorWidthFeet: 3,
  driveCorridorWidthFeet: 5,
  nearbyDefenderRadiusFeet: 6,
  helpDistanceFeet: 9,
}

export type AnalyticsOverlay =
  | 'passing_lanes'
  | 'defender_distance'
  | 'driving_lanes'
  | 'spacing'
  | 'shot_openness'
  | 'matchups'
