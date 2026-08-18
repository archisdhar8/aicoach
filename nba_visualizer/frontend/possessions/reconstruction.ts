import type {
  PlayerState,
  PlayDefinition,
  PossessionPlayer,
  RealPossession,
  SimulationFrame,
  TeamSide,
} from '@/frontend/domain/models'

export class IncompleteHistoricalLineupError extends Error {}

function replaceSide(
  players: PlayerState[],
  side: TeamSide,
  lineup: PossessionPlayer[],
): PlayerState[] {
  let index = 0
  return players.map((state) => {
    if (state.teamSide !== side) return state
    const sourcePlayer = lineup[index++]
    if (sourcePlayer === undefined) return state
    return {
      ...state,
      player: {
        id: sourcePlayer.id,
        teamId: sourcePlayer.teamId ?? state.player.teamId,
        name: sourcePlayer.displayName ?? `NBA player ${sourcePlayer.externalId}`,
        externalId: sourcePlayer.externalId,
        source: 'historical public play-by-play',
      },
    }
  })
}

export function createManualReconstruction(
  possession: RealPossession,
  template: SimulationFrame,
): PlayDefinition {
  if (possession.offensiveLineup.length !== 5 || possession.defensiveLineup.length !== 5) {
    throw new IncompleteHistoricalLineupError(
      'A complete 5-on-5 lineup is required before this possession can be reconstructed.',
    )
  }
  const now = new Date().toISOString()
  const players = replaceSide(
    replaceSide(structuredClone(template.players), 'offense', possession.offensiveLineup),
    'defense',
    possession.defensiveLineup,
  )
  const offenseTeamId = possession.offensiveLineup[0]?.teamId
    ?? template.possession.offenseTeamId
  const defenseTeamId = possession.defensiveLineup[0]?.teamId
    ?? template.possession.defenseTeamId
  const firstObservedActor = possession.events.find((event) => (
    event.playerExternalId !== null && event.playerExternalId !== undefined
    && possession.offensiveLineup.some((player) => player.externalId === event.playerExternalId)
  ))?.playerExternalId
  const handler = players.find((state) => (
    state.teamSide === 'offense' && state.player.externalId === firstObservedActor
  ))
  const ball = handler === undefined
    ? { state: 'loose' as const, position: { ...template.ball.position }, heightFeet: 0 }
    : {
        state: 'possessed' as const,
        playerId: handler.player.id,
        position: { x: handler.position.x + 0.5, y: handler.position.y + 2.5 },
        heightFeet: 3.5,
      }

  return {
    id: crypto.randomUUID(),
    name: `Manual reconstruction · Q${possession.period} ${possession.startClock ?? ''}`.trim(),
    initialFrame: {
      ...structuredClone(template),
      players,
      ball,
      currentActions: [],
      possession: { ...template.possession, offenseTeamId, defenseTeamId },
      metadata: {
        ...template.metadata,
        realPossessionId: possession.id,
        sourceGameId: possession.gameExternalId,
        sourcePossessionId: possession.provenance.sourcePossessionId,
        reconstructionOrigin: 'manual_reconstruction',
        historicalMovementAvailable: false,
        courtPlacementOrigin: 'editor_placeholder_not_historical',
        ballOwnershipOrigin: handler === undefined
          ? 'unavailable_loose_ball'
          : 'derived_from_first_observed_offensive_event',
      },
    },
    routes: [],
    actions: [],
    createdAt: now,
    updatedAt: now,
  }
}
