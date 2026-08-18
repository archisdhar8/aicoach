import type { CourtPosition, PlayerRoute } from '@/frontend/domain/models'

const MIN_POINT_DISTANCE_FEET = 0.2
export const DEFAULT_ROUTE_SPEED_FEET_PER_SECOND = 8

export class InvalidRouteError extends Error {}

export function buildTimedRoute(
  playerId: string,
  positions: CourtPosition[],
  speedFeetPerSecond = DEFAULT_ROUTE_SPEED_FEET_PER_SECOND,
): PlayerRoute {
  if (!playerId) throw new InvalidRouteError('playerId is required')
  if (speedFeetPerSecond <= 0) throw new InvalidRouteError('route speed must be positive')

  const distinctPositions = positions.filter((position, index) => (
    index === 0 || distance(position, positions[index - 1]) >= MIN_POINT_DISTANCE_FEET
  ))
  if (distinctPositions.length < 2) {
    throw new InvalidRouteError('a route requires at least two distinct points')
  }

  let timeSeconds = 0
  const points = distinctPositions.map((position, index) => {
    if (index > 0) {
      timeSeconds += distance(distinctPositions[index - 1], position) / speedFeetPerSecond
    }
    return { timeSeconds, position: { ...position } }
  })
  return { playerId, points }
}

export function validateRoute(route: PlayerRoute): void {
  if (!route.playerId) throw new InvalidRouteError('playerId is required')
  if (route.points.length < 2) throw new InvalidRouteError('a route requires at least two points')
  if (route.points[0].timeSeconds !== 0) throw new InvalidRouteError('a route must start at time zero')

  route.points.forEach((point, index) => {
    if (!Number.isFinite(point.timeSeconds) || point.timeSeconds < 0) {
      throw new InvalidRouteError('route times must be finite and non-negative')
    }
    if (point.position.x < 47 || point.position.x > 94 || point.position.y < 0 || point.position.y > 50) {
      throw new InvalidRouteError('route points must stay on the visible half-court')
    }
    if (index > 0 && point.timeSeconds <= route.points[index - 1].timeSeconds) {
      throw new InvalidRouteError('route times must be strictly increasing')
    }
  })
}

export function interpolateRoute(route: PlayerRoute, timeSeconds: number): CourtPosition {
  validateRoute(route)
  const first = route.points[0]
  const last = route.points.at(-1) as PlayerRoute['points'][number]
  if (timeSeconds <= first.timeSeconds) return { ...first.position }
  if (timeSeconds >= last.timeSeconds) return { ...last.position }

  const endIndex = route.points.findIndex((point) => point.timeSeconds >= timeSeconds)
  const start = route.points[endIndex - 1]
  const end = route.points[endIndex]
  const progress = (timeSeconds - start.timeSeconds) / (end.timeSeconds - start.timeSeconds)
  return {
    x: start.position.x + (end.position.x - start.position.x) * progress,
    y: start.position.y + (end.position.y - start.position.y) * progress,
  }
}

function distance(first: CourtPosition, second: CourtPosition): number {
  return Math.hypot(second.x - first.x, second.y - first.y)
}
