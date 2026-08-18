import { describe, expect, it } from 'vitest'
import type { CutAction, MoveAction, PassAction, ShootAction } from './models'
import { validatePlayDefinition } from './actions'
import { makeStructuredPlay } from '@/frontend/test/fixtures'

describe('structured play validation', () => {
  it('rejects overlapping actions for the same player', () => {
    const play = makeStructuredPlay()
    play.actions = [
      movement('move-1', 0, 2),
      movement('cut-1', 1, 2, 'cut'),
    ]

    expect(validatePlayDefinition(play)).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'overlap', actionId: 'cut-1' }),
    ]))
  })

  it('rejects passes and shots by players without possession', () => {
    const play = makeStructuredPlay()
    const pass: PassAction = {
      id: 'bad-pass',
      actionType: 'pass',
      playerId: 'player-2',
      targetPlayerId: 'player-1',
      startTime: 0,
      duration: 0.5,
      source: { x: 70, y: 20 },
      target: { x: 60, y: 20 },
      metadata: {},
    }
    const shot: ShootAction = {
      id: 'bad-shot',
      actionType: 'shoot',
      playerId: 'player-2',
      startTime: 1,
      duration: 0.8,
      source: { x: 70, y: 20 },
      target: { x: 88.75, y: 25 },
      deterministicResult: 'made',
      metadata: {},
    }
    play.actions = [pass, shot]

    const possessionErrors = validatePlayDefinition(play).filter((error) => error.code === 'possession')
    expect(possessionErrors.map((error) => error.actionId)).toEqual(['bad-pass', 'bad-shot'])
  })
})

function movement(
  id: string,
  startTime: number,
  duration: number,
  actionType: 'move' | 'cut' = 'move',
): MoveAction | CutAction {
  const base = {
    id,
    playerId: 'player-1',
    startTime,
    duration,
    source: { x: 60, y: 20 },
    target: { x: 70, y: 20 },
    waypoints: [],
    metadata: {},
  }
  return actionType === 'move'
    ? { ...base, actionType: 'move' }
    : { ...base, actionType: 'cut' }
}
