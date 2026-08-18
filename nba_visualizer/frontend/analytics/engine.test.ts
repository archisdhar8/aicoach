import { describe, expect, it } from 'vitest'
import type { PlayerState, SimulationFrame } from '@/frontend/domain/models'
import { calculateFrameAnalytics } from './engine'

describe('spatial basketball analytics', () => {
  it('detects a defender exactly on a pass segment', () => {
    const frame = makeFrame([
      offense('handler', 60, 25),
      offense('target', 80, 25),
      defense('defender', 70, 25),
    ])

    const lane = calculateFrameAnalytics(frame).passingLanes[0]

    expect(lane.minimumDefenderDistanceFeet).toBe(0)
    expect(lane.intersectingDefenderIds).toEqual(['defender'])
    expect(lane.interceptionRiskScore).toBeGreaterThan(70)
  })

  it('returns a low-risk lane when the defender is far away', () => {
    const frame = makeFrame([
      offense('handler', 60, 25),
      offense('target', 80, 25),
      defense('defender', 70, 45),
    ])

    const lane = calculateFrameAnalytics(frame).passingLanes[0]

    expect(lane.minimumDefenderDistanceFeet).toBe(20)
    expect(lane.intersectingDefenderIds).toEqual([])
    expect(lane.riskBand).toBe('low')
  })

  it('scores a completely open drive higher than a blocked drive', () => {
    const open = calculateFrameAnalytics(makeFrame([offense('handler', 60, 25)]))
    const blocked = calculateFrameAnalytics(makeFrame([
      offense('handler', 60, 25),
      defense('rim-help', 75, 25),
    ]))

    expect(open.driveLane?.blockingDefenderIds).toEqual([])
    expect(open.driveLane?.opennessScore).toBe(100)
    expect(blocked.driveLane?.blockingDefenderIds).toEqual(['rim-help'])
    expect(blocked.driveLane!.opennessScore).toBeLessThan(open.driveLane!.opennessScore)
  })

  it('distinguishes an unguarded shooter from a tightly guarded shooter', () => {
    const unguarded = calculateFrameAnalytics(makeFrame([offense('shooter', 75, 8)]))
    const guarded = calculateFrameAnalytics(makeFrame([
      offense('shooter', 75, 8),
      defense('closeout', 76, 8, -2, 0),
    ]))

    expect(unguarded.shotOpenness[0].opennessScore).toBe(100)
    expect(guarded.shotOpenness[0]).toMatchObject({
      nearestDefenderId: 'closeout',
      nearestDefenderDistanceFeet: 1,
      nearbyDefenderCount: 1,
    })
    expect(guarded.shotOpenness[0].opennessScore).toBeLessThan(20)
  })

  it('reports defender closing velocity and pairwise spacing', () => {
    const analytics = calculateFrameAnalytics(makeFrame([
      offense('handler', 60, 25),
      offense('wing', 60, 40),
      defense('defender', 65, 25, -2, 0),
    ]))

    expect(analytics.nearestDefenders[0]).toMatchObject({
      defenderId: 'defender',
      distanceFeet: 5,
      relativeAngleDegrees: 0,
      defenderClosingSpeedFeetPerSecond: 2,
    })
    expect(analytics.spacing.pairDistances[0].distanceFeet).toBe(15)
  })
})

function makeFrame(players: PlayerState[]): SimulationFrame {
  const handler = players.find((state) => state.teamSide === 'offense')
  return {
    timestampSeconds: 1,
    players,
    ball: handler === undefined
      ? { state: 'loose', position: { x: 60, y: 25 }, heightFeet: 0 }
      : {
          state: 'possessed',
          playerId: handler.player.id,
          position: { ...handler.position },
          heightFeet: 3.5,
        },
    currentActions: [],
    possession: {
      offenseTeamId: 'offense',
      defenseTeamId: 'defense',
      gameClockSeconds: 400,
      shotClockSeconds: 18,
      coverage: 'drop',
    },
    metadata: {},
  }
}

function offense(id: string, x: number, y: number): PlayerState {
  return player(id, 'offense', x, y, 0, 0)
}

function defense(
  id: string,
  x: number,
  y: number,
  velocityX = 0,
  velocityY = 0,
): PlayerState {
  return player(id, 'defense', x, y, velocityX, velocityY)
}

function player(
  id: string,
  teamSide: 'offense' | 'defense',
  x: number,
  y: number,
  velocityX: number,
  velocityY: number,
): PlayerState {
  return {
    player: { id, teamId: teamSide, name: id, jerseyNumber: 1 },
    teamSide,
    position: { x, y },
    velocityX,
    velocityY,
    facingDegrees: 0,
  }
}
