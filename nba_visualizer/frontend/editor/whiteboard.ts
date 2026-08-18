import type {
  CourtPosition,
  PlayAction,
  PlayDefinition,
  PlayerState,
} from '@/frontend/domain/models'
import { actionEndTime } from '@/frontend/domain/models'

export const WHITEBOARD_RIM: CourtPosition = { x: 88.75, y: 25 }
export const REPOSITION_MAX_DISPLACEMENT_FEET = 4.5
export const PASS_TARGET_RADIUS_FEET = 4.5

export interface CompletedDrawGesture {
  playerId: string
  positions: CourtPosition[]
}

export function futureBallOwner(play: PlayDefinition): string | null {
  let owner = play.initialFrame.ball.state === 'possessed'
    ? play.initialFrame.ball.playerId
    : null
  const actions = [...play.actions].sort((left, right) => (
    actionEndTime(left) - actionEndTime(right)
  ))
  actions.forEach((action) => {
    if (action.actionType === 'pass') owner = action.targetPlayerId
    if (action.actionType === 'shoot') owner = null
  })
  return owner
}

export function inferredMovementType(
  play: PlayDefinition,
  playerId: string,
): 'cut' | 'dribble' | null {
  const player = play.initialFrame.players.find((state) => state.player.id === playerId)
  if (player?.teamSide !== 'offense') return null
  return futureBallOwner(play) === playerId ? 'dribble' : 'cut'
}

export function shouldReposition(positions: CourtPosition[]): boolean {
  if (positions.length < 2) return false
  return distance(positions[0], positions.at(-1) ?? positions[0])
    <= REPOSITION_MAX_DISPLACEMENT_FEET
}

export function simplifyGesture(
  positions: CourtPosition[],
  toleranceFeet = 0.35,
): CourtPosition[] {
  const deduplicated = positions.filter((point, index) => (
    index === 0 || distance(point, positions[index - 1]) >= 0.12
  ))
  if (deduplicated.length <= 2) return deduplicated.map(clonePosition)
  return ramerDouglasPeucker(deduplicated, toleranceFeet).map(clonePosition)
}

export function nearestTarget(
  players: PlayerState[],
  position: CourtPosition,
  excludedPlayerId: string,
  side: 'offense' | 'defense',
  radiusFeet = PASS_TARGET_RADIUS_FEET,
): PlayerState | null {
  const candidates = players.filter((state) => (
    state.player.id !== excludedPlayerId && state.teamSide === side
  ))
  const nearest = candidates.reduce<PlayerState | null>((current, candidate) => (
    current === null || distance(candidate.position, position) < distance(current.position, position)
      ? candidate
      : current
  ), null)
  return nearest !== null && distance(nearest.position, position) <= radiusFeet ? nearest : null
}

export function playerAvailableAt(actions: PlayAction[], playerId: string): number {
  return actions.reduce((latest, action) => (
    action.playerId === playerId ? Math.max(latest, actionEndTime(action)) : latest
  ), 0)
}

export function ballAvailableAt(actions: PlayAction[]): number {
  return actions.reduce((latest, action) => (
    action.actionType === 'pass'
      || action.actionType === 'shoot'
      || action.actionType === 'dribble'
      ? Math.max(latest, actionEndTime(action))
      : latest
  ), 0)
}

export function playerPositionAfterActions(
  actions: PlayAction[],
  playerId: string,
  initial: CourtPosition,
): CourtPosition {
  let position = clonePosition(initial)
  actions
    .filter((action) => action.playerId === playerId)
    .sort((left, right) => actionEndTime(left) - actionEndTime(right))
    .forEach((action) => {
      if (action.actionType === 'move' || action.actionType === 'cut' || action.actionType === 'dribble') {
        position = clonePosition(action.target)
      } else if (action.actionType === 'screen') {
        position = clonePosition(action.screenLocation)
      }
    })
  return position
}

export function routeLength(positions: CourtPosition[]): number {
  return positions.slice(1).reduce(
    (total, point, index) => total + distance(positions[index], point),
    0,
  )
}

export function distance(first: CourtPosition, second: CourtPosition): number {
  return Math.hypot(second.x - first.x, second.y - first.y)
}

function ramerDouglasPeucker(
  points: CourtPosition[],
  tolerance: number,
): CourtPosition[] {
  const first = points[0]
  const last = points.at(-1) ?? first
  let greatestDistance = 0
  let splitIndex = 0
  for (let index = 1; index < points.length - 1; index += 1) {
    const candidateDistance = distanceToSegment(points[index], first, last)
    if (candidateDistance > greatestDistance) {
      greatestDistance = candidateDistance
      splitIndex = index
    }
  }
  if (greatestDistance <= tolerance) return [first, last]
  const left = ramerDouglasPeucker(points.slice(0, splitIndex + 1), tolerance)
  const right = ramerDouglasPeucker(points.slice(splitIndex), tolerance)
  return [...left.slice(0, -1), ...right]
}

function distanceToSegment(
  point: CourtPosition,
  start: CourtPosition,
  end: CourtPosition,
): number {
  const dx = end.x - start.x
  const dy = end.y - start.y
  const squaredLength = dx * dx + dy * dy
  if (squaredLength === 0) return distance(point, start)
  const progress = Math.min(1, Math.max(0, (
    (point.x - start.x) * dx + (point.y - start.y) * dy
  ) / squaredLength))
  return distance(point, {
    x: start.x + progress * dx,
    y: start.y + progress * dy,
  })
}

function clonePosition(position: CourtPosition): CourtPosition {
  return { x: position.x, y: position.y }
}
