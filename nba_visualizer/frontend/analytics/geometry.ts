import type { CourtPosition } from '@/frontend/domain/models'

export interface SegmentDistance {
  distanceFeet: number
  projection: number
  closestPoint: CourtPosition
}

export function distance(left: CourtPosition, right: CourtPosition): number {
  return Math.hypot(right.x - left.x, right.y - left.y)
}

export function pointToSegmentDistance(
  point: CourtPosition,
  source: CourtPosition,
  target: CourtPosition,
): SegmentDistance {
  const dx = target.x - source.x
  const dy = target.y - source.y
  const lengthSquared = dx * dx + dy * dy
  const rawProjection = lengthSquared === 0
    ? 0
    : ((point.x - source.x) * dx + (point.y - source.y) * dy) / lengthSquared
  const projection = Math.min(Math.max(rawProjection, 0), 1)
  const closestPoint = {
    x: source.x + dx * projection,
    y: source.y + dy * projection,
  }
  return { distanceFeet: distance(point, closestPoint), projection, closestPoint }
}

export function closingSpeed(
  defenderPosition: CourtPosition,
  defenderVelocity: CourtPosition,
  target: CourtPosition,
): number {
  const dx = target.x - defenderPosition.x
  const dy = target.y - defenderPosition.y
  const length = Math.hypot(dx, dy)
  if (length === 0) return 0
  return defenderVelocity.x * dx / length + defenderVelocity.y * dy / length
}

export function approachAngleDegrees(
  defenderPosition: CourtPosition,
  defenderVelocity: CourtPosition,
  target: CourtPosition,
): number | null {
  const targetVector = {
    x: target.x - defenderPosition.x,
    y: target.y - defenderPosition.y,
  }
  const velocityLength = Math.hypot(defenderVelocity.x, defenderVelocity.y)
  const targetLength = Math.hypot(targetVector.x, targetVector.y)
  if (velocityLength === 0 || targetLength === 0) return null
  const cosine = (
    defenderVelocity.x * targetVector.x + defenderVelocity.y * targetVector.y
  ) / (velocityLength * targetLength)
  return Math.acos(Math.min(Math.max(cosine, -1), 1)) * 180 / Math.PI
}

export function relativeAngleToBasket(
  player: CourtPosition,
  defender: CourtPosition,
  basket: CourtPosition,
): number {
  const defenderAngle = Math.atan2(defender.y - player.y, defender.x - player.x)
  const basketAngle = Math.atan2(basket.y - player.y, basket.x - player.x)
  let difference = (defenderAngle - basketAngle) * 180 / Math.PI
  while (difference > 180) difference -= 360
  while (difference < -180) difference += 360
  return difference
}

export function clampScore(score: number): number {
  return Math.round(Math.min(Math.max(score, 0), 100) * 10) / 10
}
