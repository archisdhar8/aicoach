import type { CourtPosition, Coverage, PlayDefinition } from '@/frontend/domain/models'

export type DefenseMode = 'coverage' | 'offense_only'

export interface DefensiveParameters {
  defenderSpeedFeetPerSecond: number
  maximumAccelerationFeetPerSecondSquared: number
  defensiveCushionFeet: number
  reactionDelaySeconds: number
  screenNavigationDelaySeconds: number
  recoverySpeedFeetPerSecond: number
  dropDepthFeet: number
  helpDistanceFeet: number
}

export const DEFAULT_DEFENSIVE_PARAMETERS: Readonly<DefensiveParameters> = {
  defenderSpeedFeetPerSecond: 12,
  maximumAccelerationFeetPerSecondSquared: 28,
  defensiveCushionFeet: 2.5,
  reactionDelaySeconds: 0.18,
  screenNavigationDelaySeconds: 0.35,
  recoverySpeedFeetPerSecond: 13,
  dropDepthFeet: 6,
  helpDistanceFeet: 4,
}

export interface DefensiveAssignment {
  defenderId: string
  offensivePlayerId: string
  startTime: number
  endTime?: number
  reason: 'initial_matchup' | 'switch'
}

export interface ScreenInteractionEvent {
  type: 'screen_interaction'
  screenActionId: string
  timeSeconds: number
  handlerId: string
  screenerId: string
  pointOfAttackDefenderId: string
  screenerDefenderId: string
  location: CourtPosition
  side: 'left' | 'right' | 'middle'
}

export interface DefensiveWaypoint {
  timeSeconds: number
  position: CourtPosition
  state: string
}

export interface DefensiveInstruction {
  defenderId: string
  targetOffensivePlayerId?: string
  waypoints: DefensiveWaypoint[]
}

export interface DefensiveResponse {
  coverage: Coverage
  supported: boolean
  unsupportedReason?: string
  events: ScreenInteractionEvent[]
  assignments: DefensiveAssignment[]
  instructions: DefensiveInstruction[]
  exposedOffensivePlayerIds: string[]
}

export interface CoverageContext {
  play: PlayDefinition
  assignments: DefensiveAssignment[]
  events: ScreenInteractionEvent[]
  parameters: DefensiveParameters
}

export interface CoverageStrategy {
  readonly coverage: Coverage
  generateResponse(context: CoverageContext): DefensiveResponse
}

export interface DefensiveDebugState {
  coverage: Coverage
  supported: boolean
  unsupportedReason?: string
  coverageState: string
  assignments: Array<{ defenderId: string; offensivePlayerId: string }>
  targets: Array<{ defenderId: string; position: CourtPosition; state: string }>
  screenEvents: ScreenInteractionEvent[]
}
