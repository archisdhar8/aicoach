import { describe, expect, it, vi } from 'vitest'
import { makeStructuredPlay } from '@/frontend/test/fixtures'
import type { PlayerState, PossessionPlayer, RealPossession } from '@/frontend/domain/models'
import {
  createManualReconstruction,
  IncompleteHistoricalLineupError,
} from './reconstruction'

vi.stubGlobal('crypto', { randomUUID: () => '00000000-0000-4000-8000-000000000099' })

function lineup(prefix: string, teamId: string): PossessionPlayer[] {
  return Array.from({ length: 5 }, (_, index) => ({
    id: `00000000-0000-4000-8000-${prefix}${index}`,
    externalId: `${prefix}${index}`,
    displayName: `${prefix} Player ${index + 1}`,
    teamId,
  }))
}

function possession(): RealPossession {
  return {
    id: '00000000-0000-4000-8000-000000000090',
    gameId: '00000000-0000-4000-8000-000000000091',
    gameExternalId: '0022500001',
    period: 2,
    startClock: 'PT05M12.00S',
    offenseTeamExternalId: '1',
    defenseTeamExternalId: '2',
    offensiveLineup: lineup('10', '00000000-0000-4000-8000-000000000001'),
    defensiveLineup: lineup('20', '00000000-0000-4000-8000-000000000002'),
    events: [],
    result: { resultType: 'made_shot', points: 2, made: true, turnover: false },
    provenance: {
      provider: 'pbpstats', sourceGameId: '0022500001', sourcePossessionId: '7',
      retrievedAt: '2026-01-01T00:00:00Z', movementAvailable: false,
      fieldOrigins: {}, rawReference: {},
    },
  }
}

function templateFrame() {
  const frame = makeStructuredPlay().initialFrame
  const player = frame.players[0] as PlayerState
  frame.players = Array.from({ length: 10 }, (_, index) => ({
    ...structuredClone(player),
    player: { ...player.player, id: `template-${index}`, teamId: index < 5 ? 'team-1' : 'team-2' },
    teamSide: index < 5 ? 'offense' as const : 'defense' as const,
    position: { x: 55 + index, y: 5 + index * 3 },
  }))
  return frame
}

describe('manual historical reconstruction', () => {
  it('loads source players but creates no historical paths or actions', () => {
    const result = createManualReconstruction(possession(), templateFrame())
    expect(result.routes).toEqual([])
    expect(result.actions).toEqual([])
    expect(result.initialFrame.ball.state).toBe('loose')
    expect(result.initialFrame.players[0]?.player.name).toBe('10 Player 1')
    expect(result.initialFrame.metadata.reconstructionOrigin).toBe('manual_reconstruction')
    expect(result.initialFrame.metadata.historicalMovementAvailable).toBe(false)
    expect(result.initialFrame.metadata.courtPlacementOrigin).toBe('editor_placeholder_not_historical')
  })

  it('refuses to imply a complete lineup when source lineup data is missing', () => {
    const source = possession()
    source.defensiveLineup = []
    expect(() => createManualReconstruction(source, templateFrame()))
      .toThrow(IncompleteHistoricalLineupError)
  })
})
