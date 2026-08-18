import type { CoverageContext, CoverageStrategy, DefensiveResponse } from '../types'
import { emptyResponse, firstScreen, handlerTarget, moveInstruction } from './shared'

export class IceCoverageStrategy implements CoverageStrategy {
  readonly coverage = 'ice' as const

  generateResponse(context: CoverageContext): DefensiveResponse {
    const response = emptyResponse(this.coverage, context)
    const event = firstScreen(context)
    if (event === undefined || event.side === 'middle') {
      return {
        ...response,
        supported: false,
        unsupportedReason: event === undefined
          ? 'ICE requires a detected side pick-and-roll screen interaction.'
          : 'ICE currently supports side pick-and-roll geometry only.',
      }
    }
    const handler = handlerTarget(context, event)
    const forceDirection = event.side === 'right' ? -1 : 1
    const start = event.timeSeconds + context.parameters.reactionDelaySeconds
    response.instructions = [
      moveInstruction(
        context,
        event.pointOfAttackDefenderId,
        event.handlerId,
        start,
        { x: handler.x - 1, y: handler.y - forceDirection * 2 },
        'icing_away_from_middle',
      ),
      moveInstruction(
        context,
        event.screenerDefenderId,
        event.screenerId,
        start,
        { x: handler.x + context.parameters.helpDistanceFeet, y: handler.y + forceDirection * 2 },
        'ice_containing_baseline',
      ),
    ]
    return response
  }
}
