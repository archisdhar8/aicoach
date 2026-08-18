import { describe, expect, it } from 'vitest'
import type { PassAction, PlayDefinition } from '@/frontend/domain/models'
import { makeStructuredPlay } from '@/frontend/test/fixtures'
import {
  futureBallOwner,
  inferredMovementType,
  shouldReposition,
  simplifyGesture,
} from './whiteboard'

describe('whiteboard gesture inference', () => {
  it('infers dribble for the future ball handler and cut for an off-ball player', () => {
    const play = makeStructuredPlay()
    const owner = play.initialFrame.ball.state === 'possessed'
      ? play.initialFrame.ball.playerId
      : ''
    const teammate = play.initialFrame.players.find(
      (state) => state.teamSide === 'offense' && state.player.id !== owner,
    )?.player.id ?? ''
    expect(inferredMovementType(play, owner)).toBe('dribble')
    expect(inferredMovementType(play, teammate)).toBe('cut')
  })

  it('transfers conceptual ownership after a pass and undo restores it', () => {
    const play = makeStructuredPlay()
    const passer = futureBallOwner(play) ?? ''
    const receiver = play.initialFrame.players.find(
      (state) => state.teamSide === 'offense' && state.player.id !== passer,
    )?.player.id ?? ''
    const pass: PassAction = {
      id: 'whiteboard-pass',
      actionType: 'pass',
      playerId: passer,
      targetPlayerId: receiver,
      startTime: 0,
      duration: 0.5,
      source: { x: 60, y: 25 },
      target: { x: 70, y: 20 },
      metadata: {},
    }
    const withPass: PlayDefinition = { ...play, actions: [pass] }
    expect(futureBallOwner(withPass)).toBe(receiver)
    expect(futureBallOwner({ ...withPass, actions: [] })).toBe(passer)
  })

  it('uses a short drag for repositioning and simplifies noisy paths', () => {
    expect(shouldReposition([{ x: 60, y: 20 }, { x: 63, y: 22 }])).toBe(true)
    expect(shouldReposition([{ x: 60, y: 20 }, { x: 70, y: 22 }])).toBe(false)
    expect(simplifyGesture([
      { x: 60, y: 20 },
      { x: 62, y: 20.05 },
      { x: 64, y: 20 },
    ])).toEqual([{ x: 60, y: 20 }, { x: 64, y: 20 }])
  })
})
