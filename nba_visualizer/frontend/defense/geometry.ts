import type { CourtPosition } from '@/frontend/domain/models'

export const RIM_POSITION: Readonly<CourtPosition> = { x: 88.75, y: 25 }

export function distance(left: CourtPosition, right: CourtPosition): number {
  return Math.hypot(right.x - left.x, right.y - left.y)
}

export function pointToward(
  source: CourtPosition,
  target: CourtPosition,
  distanceFeet: number,
): CourtPosition {
  const total = distance(source, target)
  if (total === 0) return { ...source }
  const ratio = Math.min(Math.max(distanceFeet / total, 0), 1)
  return {
    x: source.x + (target.x - source.x) * ratio,
    y: source.y + (target.y - source.y) * ratio,
  }
}

export function travelTime(
  source: CourtPosition,
  target: CourtPosition,
  speedFeetPerSecond: number,
): number {
  return distance(source, target) / Math.max(speedFeetPerSecond, 0.1)
}

export function interpolate(
  source: CourtPosition,
  target: CourtPosition,
  progress: number,
): CourtPosition {
  const clamped = Math.min(Math.max(progress, 0), 1)
  return {
    x: source.x + (target.x - source.x) * clamped,
    y: source.y + (target.y - source.y) * clamped,
  }
}
