import { describe, expect, it } from 'vitest'
import {
  clientPointToHalfCourt,
  courtToSvg,
  halfCourtToSvg,
  svgToHalfCourt,
} from './coordinates'

describe('courtToSvg', () => {
  it('maps regulation-court corners into the SVG viewport', () => {
    expect(courtToSvg({ x: 0, y: 0 }, { width: 940, height: 500 })).toEqual({ x: 0, y: 0 })
    expect(courtToSvg({ x: 94, y: 50 }, { width: 940, height: 500 })).toEqual({ x: 940, y: 500 })
  })

  it('maps center court without changing domain coordinates', () => {
    const position = { x: 47, y: 25 }
    expect(courtToSvg(position, { width: 188, height: 100 })).toEqual({ x: 94, y: 50 })
    expect(position).toEqual({ x: 47, y: 25 })
  })

  it('maps the offensive half-court without changing canonical feet', () => {
    expect(halfCourtToSvg({ x: 47, y: 0 })).toEqual({ x: 0, y: 0 })
    expect(halfCourtToSvg({ x: 94, y: 50 })).toEqual({ x: 47, y: 50 })
    expect(svgToHalfCourt({ x: 23.5, y: 25 }, { width: 47, height: 50 })).toEqual({ x: 70.5, y: 25 })
  })

  it('converts pointer pixels to court feet only at the rendering boundary', () => {
    const bounds = { left: 100, top: 50, width: 470, height: 500 }
    expect(clientPointToHalfCourt({ x: 335, y: 300 }, bounds)).toEqual({ x: 70.5, y: 25 })
    expect(clientPointToHalfCourt({ x: 700, y: 0 }, bounds)).toEqual({ x: 94, y: 0 })
  })
})
