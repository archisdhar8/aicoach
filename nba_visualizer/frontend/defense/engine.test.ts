import { describe, expect, it } from 'vitest'
import type { Coverage, PlayDefinition, PlayerState } from '@/frontend/domain/models'
import { generateDefensiveResponse } from './engine'
import { distance, RIM_POSITION } from './geometry'
import { sampleSimulation, simulatePlay } from '@/frontend/animation/simulation'

describe('deterministic pick-and-roll defense', () => {
  it('switches the two screen defenders only after a valid screen interaction', () => {
    const response = responseFor('switch')

    expect(response?.events).toHaveLength(1)
    const changes = response?.assignments.filter((assignment) => assignment.reason === 'switch') ?? []
    expect(changes).toEqual([
      expect.objectContaining({ defenderId: 'd1', offensivePlayerId: 'o4' }),
      expect.objectContaining({ defenderId: 'd4', offensivePlayerId: 'o1' }),
    ])
    expect(changes.every((assignment) => assignment.startTime > 0.5)).toBe(true)
  })

  it('does not switch without a detected screen', () => {
    const play = makePickAndRoll('switch')
    play.actions = play.actions.filter((action) => action.actionType !== 'screen')
    const response = generateDefensiveResponse(play)

    expect(response?.events).toEqual([])
    expect(response?.assignments.some((assignment) => assignment.reason === 'switch')).toBe(false)
    expect(response?.instructions).toHaveLength(5)
    expect(response?.instructions.every((instruction) => (
      instruction.waypoints.some((waypoint) => waypoint.state === 'tracking_assignment')
    ))).toBe(true)
    const frame = sampleSimulation(simulatePlay(play), 1)
    const defender = frame.players.find((state) => state.player.id === 'd1')
    expect(defender?.position).not.toEqual(play.initialFrame.players.find(
      (state) => state.player.id === 'd1',
    )?.position)
  })

  it('keeps the drop big at the configured depth between the handler and rim', () => {
    const response = responseFor('drop')
    const big = response?.instructions.find((instruction) => instruction.defenderId === 'd4')
    const target = big?.waypoints.at(-1)?.position

    expect(target).toBeDefined()
    expect(distance(RIM_POSITION, target!)).toBeCloseTo(6, 5)
    expect(target!.x).toBeLessThan(RIM_POSITION.x)
  })

  it('sends two defenders toward the handler in a blitz', () => {
    const response = responseFor('blitz')

    expect(response?.instructions).toHaveLength(5)
    expect(response?.instructions.slice(0, 2).map((instruction) => instruction.targetOffensivePlayerId)).toEqual([
      'o1',
      'o1',
    ])
    expect(response?.exposedOffensivePlayerIds).toEqual(['o4'])
  })

  it('produces deterministic assignment changes', () => {
    const first = responseFor('switch')
    const second = responseFor('switch')

    expect(second).toEqual(first)
  })

  it('returns explicit unsupported output for middle ICE', () => {
    const response = responseFor('ice')

    expect(response).toMatchObject({
      supported: false,
      unsupportedReason: 'ICE currently supports side pick-and-roll geometry only.',
      instructions: [],
    })
  })

  it('animates generated defense while offense-only mode leaves defenders in place', () => {
    const play = makePickAndRoll('blitz')
    const withDefense = sampleSimulation(simulatePlay(play), 1.5)
    const offenseOnly = sampleSimulation(
      simulatePlay(play, { defenseMode: 'offense_only' }),
      1.5,
    )
    const initialBig = play.initialFrame.players.find((state) => state.player.id === 'd4')!.position
    const movingBig = withDefense.players.find((state) => state.player.id === 'd4')!.position
    const staticBig = offenseOnly.players.find((state) => state.player.id === 'd4')!.position

    expect(movingBig).not.toEqual(initialBig)
    expect(staticBig).toEqual(initialBig)
  })

  it('profiles four seconds of 60 Hz rendering with cadence-cached analytics', () => {
    const result = simulatePlay(makePickAndRoll('drop'))
    const started = performance.now()
    for (let frame = 0; frame < 240; frame += 1) {
      sampleSimulation(result, Math.min(frame / 60, result.durationSeconds))
    }
    const durationMs = performance.now() - started

    console.info(JSON.stringify({
      profile: '240 visual frames',
      durationMs: Number(durationMs.toFixed(3)),
      analyticsSnapshots: result.analyticsCache.size,
    }))
    expect(result.analyticsCache.size).toBeLessThanOrEqual(
      Math.ceil(result.durationSeconds / result.analyticsCadenceSeconds) + 1,
    )
    expect(durationMs).toBeLessThan(100)
  })
})

function responseFor(coverage: Coverage) {
  return generateDefensiveResponse(makePickAndRoll(coverage))
}

function makePickAndRoll(coverage: Coverage): PlayDefinition {
  const offense = Array.from({ length: 5 }, (_, index) => playerState(
    `o${index + 1}`,
    'offense',
    68 + index * 3,
    8 + index * 8,
  ))
  offense[0].position = { x: 68, y: 25 }
  offense[3].position = { x: 73, y: 25 }
  const defense = Array.from({ length: 5 }, (_, index) => playerState(
    `d${index + 1}`,
    'defense',
    70 + index * 3,
    10 + index * 7,
  ))
  defense[0].position = { x: 70, y: 25 }
  defense[3].position = { x: 76, y: 26 }
  return {
    id: 'pnr',
    name: 'High pick-and-roll',
    routes: [],
    actions: [
      {
        id: 'dribble',
        actionType: 'dribble',
        playerId: 'o1',
        startTime: 0,
        duration: 2,
        source: { x: 68, y: 25 },
        target: { x: 78, y: 22 },
        waypoints: [],
        metadata: {},
      },
      {
        id: 'screen',
        actionType: 'screen',
        playerId: 'o4',
        targetPlayerId: 'd1',
        startTime: 0.5,
        duration: 1,
        source: { x: 73, y: 25 },
        screenLocation: { x: 74, y: 24 },
        orientationDegrees: 90,
        metadata: {},
      },
    ],
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    initialFrame: {
      timestampSeconds: 0,
      players: [...offense, ...defense],
      ball: {
        state: 'possessed',
        playerId: 'o1',
        position: { x: 68.5, y: 26.5 },
        heightFeet: 3.5,
      },
      currentActions: [],
      possession: {
        offenseTeamId: 'offense-team',
        defenseTeamId: 'defense-team',
        gameClockSeconds: 420,
        shotClockSeconds: 18,
        coverage,
      },
      metadata: {},
    },
  }
}

function playerState(
  id: string,
  teamSide: 'offense' | 'defense',
  x: number,
  y: number,
): PlayerState {
  return {
    player: { id, teamId: `${teamSide}-team`, name: id, jerseyNumber: 1 },
    teamSide,
    position: { x, y },
    velocityX: 0,
    velocityY: 0,
    facingDegrees: 0,
  }
}
