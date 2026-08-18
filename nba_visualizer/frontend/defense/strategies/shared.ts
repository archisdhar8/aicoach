import type { CourtPosition, Coverage } from '@/frontend/domain/models'
import { offensivePositionAt } from '../events'
import { travelTime } from '../geometry'
import type {
  CoverageContext,
  DefensiveInstruction,
  DefensiveResponse,
  ScreenInteractionEvent,
} from '../types'

export function emptyResponse(
  coverage: Coverage,
  context: CoverageContext,
): DefensiveResponse {
  return {
    coverage,
    supported: true,
    events: context.events,
    assignments: context.assignments.map((assignment) => ({ ...assignment })),
    instructions: [],
    exposedOffensivePlayerIds: [],
  }
}

export function firstScreen(context: CoverageContext): ScreenInteractionEvent | undefined {
  return context.events[0]
}

export function defenderPosition(context: CoverageContext, defenderId: string): CourtPosition {
  return context.play.initialFrame.players.find(
    (state) => state.player.id === defenderId,
  )?.position ?? { x: 47, y: 25 }
}

export function moveInstruction(
  context: CoverageContext,
  defenderId: string,
  targetOffensivePlayerId: string | undefined,
  startTime: number,
  target: CourtPosition,
  state: string,
  speed: number = context.parameters.defenderSpeedFeetPerSecond,
): DefensiveInstruction {
  const source = defenderPosition(context, defenderId)
  return {
    defenderId,
    targetOffensivePlayerId,
    waypoints: [
      { timeSeconds: startTime, position: { ...source }, state: 'reacting' },
      {
        timeSeconds: startTime + travelTime(source, target, speed),
        position: { ...target },
        state,
      },
    ],
  }
}

export function handlerTarget(context: CoverageContext, event: ScreenInteractionEvent): CourtPosition {
  return offensivePositionAt(
    context.play,
    event.handlerId,
    event.timeSeconds + context.parameters.reactionDelaySeconds + 0.7,
  )
}
