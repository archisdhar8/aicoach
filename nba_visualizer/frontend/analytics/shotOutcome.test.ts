import { describe, expect, it } from 'vitest'
import type { Player, PlayerState } from '@/frontend/domain/models'
import { evaluatePlayerAwareShot, shotPointValue } from './shotOutcome'

const shooter = (threePointPercentage?: number): Player => ({
  id: `p-${threePointPercentage ?? 'fallback'}`,
  teamId: 'offense',
  name: 'Shooter',
  shootingProfile: threePointPercentage === undefined ? null : {
    season: '2025-26',
    gamesPlayed: 75,
    fieldGoalAttempts: 1_200,
    threePointAttempts: 500,
    twoPointPercentage: 0.56,
    threePointPercentage,
    freeThrowPercentage: 0.8,
    provenance: 'nba.com_season_totals',
  },
})

const defender = (x: number, y: number): PlayerState => ({
  player: { id: `d-${x}-${y}`, teamId: 'defense', name: 'Defender' },
  teamSide: 'defense',
  position: { x, y },
  velocityX: 0,
  velocityY: 0,
  facingDegrees: 0,
})

describe('heuristic shot outcomes', () => {
  it('reduces the make heuristic for close pressure', () => {
    const source = { x: 75, y: 25 }
    const open = evaluatePlayerAwareShot('same-shot', shooter(), source, [defender(60, 5)])
    const contested = evaluatePlayerAwareShot('same-shot', shooter(), source, [defender(75.5, 25)])
    expect(open.makeProbabilityHeuristic).toBeGreaterThan(contested.makeProbabilityHeuristic)
    expect(contested.nearestDefenderDistanceFeet).toBeCloseTo(0.5)
  })

  it('gives stronger three-point shooters a higher probability at the same location', () => {
    const source = { x: 63, y: 25 }
    const strong = evaluatePlayerAwareShot('same', shooter(0.43), source, [])
    const weak = evaluatePlayerAwareShot('same', shooter(0.29), source, [])
    expect(strong.makeProbabilityHeuristic).toBeGreaterThan(weak.makeProbabilityHeuristic)
    expect(strong.profileSource).toBe('nba_season_totals')
    expect(strong.profileAttempts).toBe(500)
  })

  it('distinguishes two-point and three-point locations', () => {
    expect(shotPointValue({ x: 82, y: 25 })).toBe(2)
    expect(shotPointValue({ x: 63, y: 25 })).toBe(3)
    expect(shotPointValue({ x: 86, y: 2 })).toBe(3)
  })

  it('is deterministic while supporting both makes and misses', () => {
    const outcomes = Array.from({ length: 40 }, (_, index) => (
      evaluatePlayerAwareShot(`shot-${index}`, shooter(), { x: 66, y: 25 }, []).result
    ))
    expect(outcomes).toContain('made')
    expect(outcomes).toContain('missed')
    expect(evaluatePlayerAwareShot('repeat', shooter(), { x: 66, y: 25 }, []))
      .toEqual(evaluatePlayerAwareShot('repeat', shooter(), { x: 66, y: 25 }, []))
  })
})
