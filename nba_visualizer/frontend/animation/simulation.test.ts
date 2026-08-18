import { describe, expect, it } from 'vitest'
import type { PlayDefinition, PlayerRoute } from '@/frontend/domain/models'
import { buildTimedRoute, interpolateRoute, InvalidRouteError, validateRoute } from './routes'
import { sampleSimulation, simulatePlay } from './simulation'
import { makeStructuredPlay } from '@/frontend/test/fixtures'

const route: PlayerRoute = {
  playerId: 'player-1',
  points: [
    { timeSeconds: 0, position: { x: 60, y: 10 } },
    { timeSeconds: 2, position: { x: 70, y: 20 } },
    { timeSeconds: 4, position: { x: 80, y: 20 } },
  ],
}

describe('route interpolation', () => {
  it('returns exact starting and ending positions', () => {
    expect(interpolateRoute(route, 0)).toEqual({ x: 60, y: 10 })
    expect(interpolateRoute(route, 10)).toEqual({ x: 80, y: 20 })
  })

  it('interpolates between timed waypoints', () => {
    expect(interpolateRoute(route, 1)).toEqual({ x: 65, y: 15 })
    expect(interpolateRoute(route, 3)).toEqual({ x: 75, y: 20 })
  })

  it('rejects invalid routes', () => {
    expect(() => validateRoute({ playerId: 'p', points: [] })).toThrow(InvalidRouteError)
    expect(() => validateRoute({
      playerId: 'p',
      points: [
        { timeSeconds: 0, position: { x: 60, y: 10 } },
        { timeSeconds: 0, position: { x: 70, y: 10 } },
      ],
    })).toThrow('strictly increasing')
    expect(() => buildTimedRoute('p', [{ x: 60, y: 10 }])).toThrow('at least two')
  })
})

describe('play simulation', () => {
  it('uses the longest route as duration and does not mutate the play', () => {
    const play = makePlay(route)
    const original = structuredClone(play)
    const result = simulatePlay(play)

    expect(result.durationSeconds).toBe(4)
    expect(sampleSimulation(result, 4).players[0].position).toEqual({ x: 80, y: 20 })
    expect(play).toEqual(original)
  })

  it('keeps a pass airborne until arrival, then transfers possession', () => {
    const play = makeStructuredPlay()
    play.actions = [{
      id: 'pass-1',
      actionType: 'pass',
      playerId: 'player-1',
      targetPlayerId: 'player-2',
      startTime: 1,
      duration: 1,
      source: { x: 60, y: 20 },
      target: { x: 70, y: 20 },
      metadata: {},
    }]
    const result = simulatePlay(play)

    expect(sampleSimulation(result, 0.9).ball).toMatchObject({ state: 'possessed', playerId: 'player-1' })
    expect(sampleSimulation(result, 1.5).ball).toMatchObject({
      state: 'traveling_between_players',
      fromPlayerId: 'player-1',
      toPlayerId: 'player-2',
      progress: 0.5,
    })
    expect(sampleSimulation(result, 2).ball).toMatchObject({ state: 'possessed', playerId: 'player-2' })
  })

  it('animates a deterministic missed shot past the rim to a loose rebound', () => {
    const play = makeStructuredPlay()
    play.actions = [{
      id: 'shot-1',
      actionType: 'shoot',
      playerId: 'player-1',
      startTime: 0,
      duration: 1,
      source: { x: 60, y: 20 },
      target: { x: 88.75, y: 25 },
      deterministicResult: 'missed',
      metadata: { demoResult: true },
    }]
    const result = simulatePlay(play)

    expect(sampleSimulation(result, 0.5).ball).toMatchObject({
      state: 'traveling_to_basket',
      progress: 0.5,
      deterministicResult: 'missed',
    })
    expect(sampleSimulation(result, 1).ball).toMatchObject({
      state: 'loose',
      position: { x: 86.25, y: 28.25 },
    })
  })

  it('memoizes analytics snapshots at a lower cadence than visual sampling', () => {
    const play = makeStructuredPlay()
    play.actions = [{
      id: 'hold-1',
      actionType: 'hold',
      playerId: 'player-2',
      startTime: 0,
      duration: 1,
      source: { x: 70, y: 20 },
      metadata: {},
    }]
    const result = simulatePlay(play)
    const first = sampleSimulation(result, 0.121)
    const second = sampleSimulation(result, 0.189)

    expect(first.analytics?.sampledAtSeconds).toBeCloseTo(0.1)
    expect(second.analytics).toBe(first.analytics)
    expect(result.analyticsCache.size).toBe(1)
  })
})

function makePlay(playerRoute: PlayerRoute): PlayDefinition {
  return {
    id: 'play-1',
    name: 'Test play',
    routes: [playerRoute],
    actions: [],
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    initialFrame: {
      timestampSeconds: 0,
      players: [{
        player: { id: 'player-1', teamId: 'team-1', name: 'Guard', jerseyNumber: 1 },
        teamSide: 'offense',
        position: { x: 60, y: 10 },
        velocityX: 0,
        velocityY: 0,
        facingDegrees: 0,
      }],
      ball: { state: 'possessed', playerId: 'player-1', position: { x: 60, y: 11 }, heightFeet: 3.5 },
      currentActions: [],
      possession: {
        offenseTeamId: 'team-1',
        defenseTeamId: 'team-2',
        gameClockSeconds: 400,
        shotClockSeconds: 20,
        coverage: 'drop',
      },
      metadata: {},
    },
  }
}
