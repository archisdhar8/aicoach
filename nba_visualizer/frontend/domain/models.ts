export type Identifier = string

export type TeamSide = 'offense' | 'defense'
export type Coverage = 'drop' | 'switch' | 'hedge' | 'blitz' | 'ice'
export type ActionType = 'move' | 'cut' | 'dribble' | 'screen' | 'pass' | 'shoot' | 'hold'

export interface PlayerShootingProfile {
  season: string
  gamesPlayed: number
  fieldGoalAttempts: number
  threePointAttempts: number
  twoPointPercentage?: number | null
  threePointPercentage?: number | null
  freeThrowPercentage?: number | null
  provenance: 'nba.com_season_totals'
}

export interface Player {
  id: Identifier
  teamId: Identifier
  name: string
  jerseyNumber?: number | null
  position?: string | null
  height?: string | null
  externalId?: string | null
  source?: string | null
  shootingProfile?: PlayerShootingProfile | null
}

export interface Team {
  id: Identifier
  name: string
  abbreviation: string
}

export interface NBATeamData {
  id: Identifier
  externalId: string
  name: string
  fullName: string
  abbreviation: string
  city?: string | null
  state?: string | null
  yearFounded?: number | null
  source: 'nba.com'
  retrievedAt: string
}

export interface NBAPlayerData {
  id: Identifier
  externalId: string
  teamId?: Identifier | null
  firstName: string
  lastName: string
  displayName: string
  position?: string | null
  height?: string | null
  jerseyNumber?: number | null
  active: boolean
  source: 'nba.com'
  retrievedAt: string
  shootingProfile?: PlayerShootingProfile | null
}

export interface TeamListResponse {
  teams: NBATeamData[]
  cacheStatus: 'cache' | 'refreshed' | 'cache_fallback'
  retrievedAt?: string | null
}

export interface RosterResponse {
  team: NBATeamData
  season: string
  players: NBAPlayerData[]
  cacheStatus: 'cache' | 'refreshed' | 'cache_fallback'
  retrievedAt?: string | null
}

export interface PreferredLineupResponse {
  team: NBATeamData
  season: string
  players: NBAPlayerData[]
  selectionMethod: string
  disclaimer: string
  cacheStatus: 'cache' | 'refreshed' | 'cache_fallback'
  retrievedAt?: string | null
}

export interface NBAGameData {
  id: Identifier
  externalId: string
  season: string
  gameDate: string
  homeTeamId?: Identifier | null
  awayTeamId?: Identifier | null
  status?: string | null
  source: 'nba.com'
  retrievedAt: string
}

export interface GameListResponse {
  games: NBAGameData[]
  cacheStatus: 'cache' | 'refreshed' | 'cache_fallback'
  retrievedAt?: string | null
}

export type FieldOrigin = 'observed' | 'derived' | 'manual' | 'simulated' | 'unavailable'

export interface FieldProvenance {
  origin: FieldOrigin
  source: string
  sourceField?: string | null
  note?: string | null
}

export interface PossessionPlayer {
  id: Identifier
  externalId: string
  displayName?: string | null
  teamId?: Identifier | null
  teamExternalId?: string | null
}

export interface RealPossessionEvent {
  sourceEventId: string
  sequence: number
  period?: number | null
  clock?: string | null
  eventType: string
  description?: string | null
  teamExternalId?: string | null
  playerExternalId?: string | null
  shooterExternalId?: string | null
  passerExternalId?: string | null
  assistExternalId?: string | null
  isTurnover: boolean
  isFoul: boolean
  isRebound: boolean
  shotX?: number | null
  shotY?: number | null
  shotType?: string | null
  shotResult?: string | null
  points?: number | null
}

export interface RealPossession {
  id: Identifier
  gameId: Identifier
  gameExternalId: string
  period: number
  startClock?: string | null
  endClock?: string | null
  offenseTeamExternalId?: string | null
  defenseTeamExternalId?: string | null
  offensiveLineup: PossessionPlayer[]
  defensiveLineup: PossessionPlayer[]
  events: RealPossessionEvent[]
  result: { resultType: string; points: number; made?: boolean | null; turnover: boolean }
  provenance: {
    provider: string
    sourceGameId: string
    sourcePossessionId: string
    retrievedAt: string
    movementAvailable: false
    fieldOrigins: Record<string, FieldProvenance>
    rawReference: Record<string, unknown>
  }
}

export interface PossessionListResponse {
  gameId: Identifier
  possessions: RealPossession[]
  cacheStatus: 'cache' | 'refreshed' | 'cache_fallback'
  retrievedAt?: string | null
}

export interface PossessionReconstruction {
  id: Identifier
  possessionId: Identifier
  play: PlayDefinition
  createdAt: string
  updatedAt: string
}

export interface CourtPosition {
  x: number
  y: number
}

export interface PlayerState {
  player: Player
  teamSide: TeamSide
  position: CourtPosition
  velocityX: number
  velocityY: number
  facingDegrees: number
}

interface BallStateBase {
  position: CourtPosition
  heightFeet: number
}

export interface PossessedBallState extends BallStateBase {
  state: 'possessed'
  playerId: Identifier
}

export interface PassingBallState extends BallStateBase {
  state: 'traveling_between_players'
  fromPlayerId: Identifier
  toPlayerId: Identifier
  progress: number
}

export interface ShootingBallState extends BallStateBase {
  state: 'traveling_to_basket'
  shooterPlayerId: Identifier
  progress: number
  deterministicResult: 'made' | 'missed'
}

export interface LooseBallState extends BallStateBase {
  state: 'loose'
}

export type BallState =
  | PossessedBallState
  | PassingBallState
  | ShootingBallState
  | LooseBallState

interface ActionBase {
  id: Identifier
  playerId: Identifier
  startTime: number
  duration: number
  source: CourtPosition
  metadata: Record<string, string | number | boolean | null>
}

interface MovementActionBase extends ActionBase {
  target: CourtPosition
  waypoints: CourtPosition[]
}

export interface MoveAction extends MovementActionBase {
  actionType: 'move'
}

export interface CutAction extends MovementActionBase {
  actionType: 'cut'
}

export interface DribbleAction extends MovementActionBase {
  actionType: 'dribble'
}

export interface ScreenAction extends ActionBase {
  actionType: 'screen'
  screenLocation: CourtPosition
  orientationDegrees: number
  targetPlayerId?: Identifier | null
}

export interface PassAction extends ActionBase {
  actionType: 'pass'
  target: CourtPosition
  targetPlayerId: Identifier
}

export interface ShootAction extends ActionBase {
  actionType: 'shoot'
  target: CourtPosition
  deterministicResult: 'made' | 'missed'
}

export interface HoldAction extends ActionBase {
  actionType: 'hold'
}

export type PlayAction =
  | MoveAction
  | CutAction
  | DribbleAction
  | ScreenAction
  | PassAction
  | ShootAction
  | HoldAction

export interface RoutePoint {
  timeSeconds: number
  position: CourtPosition
}

export interface PlayerRoute {
  playerId: Identifier
  points: RoutePoint[]
}

export interface PossessionState {
  offenseTeamId: Identifier
  defenseTeamId: Identifier
  gameClockSeconds: number
  shotClockSeconds: number
  coverage: Coverage
}

export interface SimulationFrame {
  timestampSeconds: number
  players: PlayerState[]
  ball: BallState
  currentActions: PlayAction[]
  possession: PossessionState
  metadata: Record<string, unknown>
  analytics?: import('@/frontend/analytics/types').FrameAnalytics
}

export interface PlayDefinition {
  id: Identifier
  name: string
  initialFrame: SimulationFrame
  routes: PlayerRoute[]
  actions: PlayAction[]
  createdAt: string
  updatedAt: string
}

export type Play = PlayDefinition

export function actionEndTime(action: PlayAction): number {
  return action.startTime + action.duration
}
