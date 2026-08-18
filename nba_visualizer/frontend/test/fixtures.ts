import type { PlayDefinition } from '@/frontend/domain/models'

export function makeStructuredPlay(): PlayDefinition {
  return {
    id: 'play-1',
    name: 'Validation play',
    routes: [],
    actions: [],
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    initialFrame: {
      timestampSeconds: 0,
      players: [
        {
          player: { id: 'player-1', teamId: 'team-1', name: 'Handler', jerseyNumber: 1 },
          teamSide: 'offense',
          position: { x: 60, y: 20 },
          velocityX: 0,
          velocityY: 0,
          facingDegrees: 0,
        },
        {
          player: { id: 'player-2', teamId: 'team-1', name: 'Wing', jerseyNumber: 2 },
          teamSide: 'offense',
          position: { x: 70, y: 20 },
          velocityX: 0,
          velocityY: 0,
          facingDegrees: 0,
        },
      ],
      ball: {
        state: 'possessed',
        playerId: 'player-1',
        position: { x: 60.5, y: 21.5 },
        heightFeet: 3.5,
      },
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
