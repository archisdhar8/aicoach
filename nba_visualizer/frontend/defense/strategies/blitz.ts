import type { CoverageContext, CoverageStrategy, DefensiveResponse } from '../types'
import { emptyResponse, firstScreen, handlerTarget, moveInstruction } from './shared'

export class BlitzCoverageStrategy implements CoverageStrategy {
  readonly coverage = 'blitz' as const

  generateResponse(context: CoverageContext): DefensiveResponse {
    const response = emptyResponse(this.coverage, context)
    const event = firstScreen(context)
    if (event === undefined) return response
    const target = handlerTarget(context, event)
    const start = event.timeSeconds + context.parameters.reactionDelaySeconds
    response.instructions = [
      moveInstruction(
        context,
        event.pointOfAttackDefenderId,
        event.handlerId,
        start,
        { x: target.x - 0.8, y: target.y - 1.2 },
        'blitzing_handler',
      ),
      moveInstruction(
        context,
        event.screenerDefenderId,
        event.handlerId,
        start,
        { x: target.x + 0.8, y: target.y + 1.2 },
        'blitzing_handler',
      ),
    ]
    response.exposedOffensivePlayerIds = [event.screenerId]
    return response
  }
}
