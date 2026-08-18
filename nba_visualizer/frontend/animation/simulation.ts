import type {
  BallState,
  CourtPosition,
  PassAction,
  PlayAction,
  PlayDefinition,
  PlayerRoute,
  PlayerState,
  ShootAction,
  SimulationFrame,
} from '@/frontend/domain/models'
import { actionEndTime } from '@/frontend/domain/models'
import { assertValidPlay } from '@/frontend/domain/actions'
import { interpolateRoute, validateRoute } from './routes'
import {
  defensiveDuration,
  defensivePositionAt,
  generateDefensiveResponse,
  type DefenseMode,
  type DefensiveParameters,
  type DefensiveResponse,
} from '@/frontend/defense/engine'
import { calculateFrameAnalytics } from '@/frontend/analytics/engine'
import type { FrameAnalytics } from '@/frontend/analytics/types'

const RIM_POSITION: CourtPosition = { x: 88.75, y: 25 }

export interface SimulationResult {
  readonly playDefinition: PlayDefinition
  readonly durationSeconds: number
  readonly defensiveResponse: DefensiveResponse | null
  readonly analyticsCadenceSeconds: number
  readonly analyticsCache: Map<number, FrameAnalytics>
}

export interface SimulationOptions {
  defenseMode?: DefenseMode
  defensiveParameters?: DefensiveParameters
}

export function simulatePlay(play: PlayDefinition, options: SimulationOptions = {}): SimulationResult {
  const snapshot = clonePlay(play)
  assertValidPlay(snapshot)
  const playerIds = new Set(snapshot.initialFrame.players.map((state) => state.player.id))
  const routedPlayerIds = new Set<string>()

  snapshot.routes.forEach((route) => {
    validateRoute(route)
    if (!playerIds.has(route.playerId)) throw new Error(`route player ${route.playerId} is not in the play`)
    if (routedPlayerIds.has(route.playerId)) throw new Error(`player ${route.playerId} has multiple routes`)
    routedPlayerIds.add(route.playerId)
  })

  const routeDuration = snapshot.routes.reduce(
    (duration, route) => Math.max(duration, route.points.at(-1)?.timeSeconds ?? 0),
    0,
  )
  const actionDuration = snapshot.actions.reduce(
    (duration, action) => Math.max(duration, actionEndTime(action)),
    0,
  )
  const defensiveResponse = generateDefensiveResponse(
    snapshot,
    options.defenseMode,
    options.defensiveParameters,
  )
  return {
    playDefinition: snapshot,
    defensiveResponse,
    analyticsCadenceSeconds: 0.1,
    analyticsCache: new Map<number, FrameAnalytics>(),
    durationSeconds: Math.max(
      routeDuration,
      actionDuration,
      defensiveDuration(defensiveResponse),
    ),
  }
}

export function sampleSimulation(result: SimulationResult, timeSeconds: number): SimulationFrame {
  const timestampSeconds = Math.min(Math.max(timeSeconds, 0), result.durationSeconds)
  const frame = sampleKinematicFrame(result, timestampSeconds)
  const analyticsTime = Math.min(
    Math.floor((timestampSeconds + 1e-9) / result.analyticsCadenceSeconds)
      * result.analyticsCadenceSeconds,
    result.durationSeconds,
  )
  let analytics = result.analyticsCache.get(analyticsTime)
  if (analytics === undefined) {
    analytics = calculateFrameAnalytics(
      sampleKinematicFrame(result, analyticsTime),
      undefined,
      result.analyticsCadenceSeconds,
    )
    result.analyticsCache.set(analyticsTime, analytics)
  }
  return { ...frame, analytics }
}

function sampleKinematicFrame(
  result: SimulationResult,
  timestampSeconds: number,
): SimulationFrame {
  const play = result.playDefinition
  const players = play.initialFrame.players.map((state) => ({
    ...clonePlayerState(state),
    ...kinematicsAt(result, state, timestampSeconds),
  }))
  const currentActions = play.actions.filter(
    (action) => timestampSeconds >= action.startTime && timestampSeconds < actionEndTime(action),
  )
  return {
    ...play.initialFrame,
    timestampSeconds,
    players,
    ball: ballAt(play, players, timestampSeconds),
    currentActions: currentActions.map(cloneAction),
    metadata: {
      ...play.initialFrame.metadata,
      playDefinitionId: play.id,
      simulationOutput: true,
      defensiveCoverage: result.defensiveResponse?.coverage ?? 'offense_only',
      defensiveSupported: result.defensiveResponse?.supported ?? true,
    },
  }
}

function kinematicsAt(
  result: SimulationResult,
  state: PlayerState,
  timeSeconds: number,
): Pick<PlayerState, 'position' | 'velocityX' | 'velocityY'> {
  const position = state.teamSide === 'defense'
    ? defensivePositionAt(result.defensiveResponse, state, timeSeconds)
    : playerPositionAt(result.playDefinition, state.player.id, timeSeconds)
  const previousTime = Math.max(0, timeSeconds - 0.05)
  if (previousTime === timeSeconds) return { position, velocityX: 0, velocityY: 0 }
  const previous = state.teamSide === 'defense'
    ? defensivePositionAt(result.defensiveResponse, state, previousTime)
    : playerPositionAt(result.playDefinition, state.player.id, previousTime)
  const elapsed = timeSeconds - previousTime
  return {
    position,
    velocityX: (position.x - previous.x) / elapsed,
    velocityY: (position.y - previous.y) / elapsed,
  }
}

function playerPositionAt(
  play: PlayDefinition,
  playerId: string,
  timeSeconds: number,
): CourtPosition {
  const initial = play.initialFrame.players.find((state) => state.player.id === playerId)?.position
  if (initial === undefined) return { x: 47, y: 25 }

  const actions = play.actions
    .filter((action) => action.playerId === playerId)
    .sort((left, right) => left.startTime - right.startTime)
  if (actions.length === 0) {
    const route = play.routes.find((candidate) => candidate.playerId === playerId)
    return route === undefined ? { ...initial } : interpolateRoute(route, timeSeconds)
  }

  let position = { ...initial }
  for (const action of actions) {
    if (timeSeconds < action.startTime) return position
    if (timeSeconds < actionEndTime(action)) {
      return positionDuringAction(action, timeSeconds)
    }
    position = positionAfterAction(action, position)
  }
  return position
}

function positionDuringAction(action: PlayAction, timeSeconds: number): CourtPosition {
  const progress = Math.min(Math.max((timeSeconds - action.startTime) / action.duration, 0), 1)
  if (action.actionType === 'move' || action.actionType === 'cut' || action.actionType === 'dribble') {
    return interpolatePositions([action.source, ...action.waypoints, action.target], progress)
  }
  if (action.actionType === 'screen') {
    return interpolate(action.source, action.screenLocation, Math.min(progress * 3, 1))
  }
  return { ...action.source }
}

function positionAfterAction(action: PlayAction, fallback: CourtPosition): CourtPosition {
  if (action.actionType === 'move' || action.actionType === 'cut' || action.actionType === 'dribble') {
    return { ...action.target }
  }
  if (action.actionType === 'screen') return { ...action.screenLocation }
  return { ...fallback }
}

function ballAt(
  play: PlayDefinition,
  players: PlayerState[],
  timeSeconds: number,
): BallState {
  const activePass = play.actions.find(
    (action): action is PassAction => action.actionType === 'pass'
      && timeSeconds >= action.startTime
      && timeSeconds < actionEndTime(action),
  )
  if (activePass !== undefined) {
    const progress = (timeSeconds - activePass.startTime) / activePass.duration
    const from = playerPosition(players, activePass.playerId, activePass.source)
    const to = playerPosition(players, activePass.targetPlayerId, activePass.target)
    return {
      state: 'traveling_between_players',
      fromPlayerId: activePass.playerId,
      toPlayerId: activePass.targetPlayerId,
      progress,
      position: interpolate(from, to, progress),
      heightFeet: 4 + Math.sin(Math.PI * progress) * 4,
    }
  }

  const activeShot = play.actions.find(
    (action): action is ShootAction => action.actionType === 'shoot'
      && timeSeconds >= action.startTime
      && timeSeconds < actionEndTime(action),
  )
  if (activeShot !== undefined) {
    const progress = (timeSeconds - activeShot.startTime) / activeShot.duration
    const from = playerPosition(players, activeShot.playerId, activeShot.source)
    return {
      state: 'traveling_to_basket',
      shooterPlayerId: activeShot.playerId,
      progress,
      deterministicResult: activeShot.deterministicResult,
      position: interpolate(from, shotDestination(activeShot), progress),
      heightFeet: 7 + Math.sin(Math.PI * progress) * 8,
    }
  }

  let possessor = play.initialFrame.ball.state === 'possessed'
    ? play.initialFrame.ball.playerId
    : null
  let shot: ShootAction | undefined
  for (const action of [...play.actions].sort((left, right) => actionEndTime(left) - actionEndTime(right))) {
    if (timeSeconds < actionEndTime(action)) continue
    if (action.actionType === 'pass') possessor = action.targetPlayerId
    if (action.actionType === 'shoot') {
      possessor = null
      shot = action
    }
  }

  if (possessor !== null) {
    const position = playerPosition(players, possessor, play.initialFrame.ball.position)
    return {
      state: 'possessed',
      playerId: possessor,
      position: { x: position.x + 0.55, y: position.y + 1.45 },
      heightFeet: 3.5,
    }
  }

  if (shot !== undefined) {
    return {
      state: 'loose',
      position: shotDestination(shot),
      heightFeet: shot.deterministicResult === 'made' ? 8 : 10,
    }
  }
  return {
    state: 'loose',
    position: { ...play.initialFrame.ball.position },
    heightFeet: play.initialFrame.ball.heightFeet,
  }
}

function shotDestination(shot: ShootAction): CourtPosition {
  if (shot.deterministicResult === 'made') return { ...shot.target }
  const direction = shot.source.y <= shot.target.y ? 1 : -1
  return {
    x: shot.target.x - 2.5,
    y: Math.min(Math.max(shot.target.y + direction * 3.25, 0), 50),
  }
}

function playerPosition(
  players: PlayerState[],
  playerId: string,
  fallback: CourtPosition,
): CourtPosition {
  return players.find((state) => state.player.id === playerId)?.position ?? fallback
}

function interpolatePositions(points: CourtPosition[], progress: number): CourtPosition {
  if (points.length < 2) return { ...(points[0] ?? RIM_POSITION) }
  const scaled = progress * (points.length - 1)
  const index = Math.min(Math.floor(scaled), points.length - 2)
  return interpolate(points[index], points[index + 1], scaled - index)
}

function interpolate(source: CourtPosition, target: CourtPosition, progress: number): CourtPosition {
  return {
    x: source.x + (target.x - source.x) * progress,
    y: source.y + (target.y - source.y) * progress,
  }
}

function clonePlayerState(state: PlayerState): PlayerState {
  return { ...state, player: { ...state.player }, position: { ...state.position } }
}

function cloneAction(action: PlayAction): PlayAction {
  if (action.actionType === 'move' || action.actionType === 'cut' || action.actionType === 'dribble') {
    return {
      ...action,
      source: { ...action.source },
      metadata: { ...action.metadata },
      target: { ...action.target },
      waypoints: action.waypoints.map((position) => ({ ...position })),
    }
  }
  if (action.actionType === 'screen') {
    return {
      ...action,
      source: { ...action.source },
      metadata: { ...action.metadata },
      screenLocation: { ...action.screenLocation },
    }
  }
  if (action.actionType === 'pass' || action.actionType === 'shoot') {
    return {
      ...action,
      source: { ...action.source },
      metadata: { ...action.metadata },
      target: { ...action.target },
    }
  }
  return {
    ...action,
    source: { ...action.source },
    metadata: { ...action.metadata },
  }
}

function cloneRoute(route: PlayerRoute): PlayerRoute {
  return {
    playerId: route.playerId,
    points: route.points.map((point) => ({
      timeSeconds: point.timeSeconds,
      position: { ...point.position },
    })),
  }
}

function clonePlay(play: PlayDefinition): PlayDefinition {
  return {
    ...play,
    initialFrame: {
      ...play.initialFrame,
      players: play.initialFrame.players.map(clonePlayerState),
      ball: { ...play.initialFrame.ball, position: { ...play.initialFrame.ball.position } },
      currentActions: play.initialFrame.currentActions.map(cloneAction),
      possession: { ...play.initialFrame.possession },
      metadata: { ...play.initialFrame.metadata },
    },
    routes: play.routes.map(cloneRoute),
    actions: play.actions.map(cloneAction),
  }
}
