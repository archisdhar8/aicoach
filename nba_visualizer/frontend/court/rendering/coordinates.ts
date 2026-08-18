import type { CourtPosition } from '@/frontend/domain/models'

export const COURT_LENGTH_FEET = 94
export const COURT_WIDTH_FEET = 50
export const HALF_COURT_START_X_FEET = 47
export const HALF_COURT_LENGTH_FEET = 47

export interface SvgPoint {
  x: number
  y: number
}

export interface SvgCourtDimensions {
  width: number
  height: number
}

export function courtToSvg(
  position: CourtPosition,
  dimensions: SvgCourtDimensions = { width: COURT_LENGTH_FEET, height: COURT_WIDTH_FEET },
): SvgPoint {
  return {
    x: (position.x / COURT_LENGTH_FEET) * dimensions.width,
    y: (position.y / COURT_WIDTH_FEET) * dimensions.height,
  }
}

export function halfCourtToSvg(position: CourtPosition): SvgPoint {
  return {
    x: position.x - HALF_COURT_START_X_FEET,
    y: position.y,
  }
}

export function svgToHalfCourt(
  point: SvgPoint,
  dimensions: SvgCourtDimensions,
): CourtPosition {
  return {
    x: HALF_COURT_START_X_FEET + (point.x / dimensions.width) * HALF_COURT_LENGTH_FEET,
    y: (point.y / dimensions.height) * COURT_WIDTH_FEET,
  }
}

export function clientPointToHalfCourt(
  clientPoint: SvgPoint,
  bounds: Pick<DOMRect, 'left' | 'top' | 'width' | 'height'>,
): CourtPosition {
  const localX = Math.min(Math.max(clientPoint.x - bounds.left, 0), bounds.width)
  const localY = Math.min(Math.max(clientPoint.y - bounds.top, 0), bounds.height)
  return svgToHalfCourt(
    { x: localX, y: localY },
    { width: bounds.width, height: bounds.height },
  )
}
