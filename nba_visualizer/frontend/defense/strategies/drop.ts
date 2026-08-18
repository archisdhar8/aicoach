import { pointToward, RIM_POSITION } from '../geometry'
import type { CoverageContext, CoverageStrategy, DefensiveResponse } from '../types'
import { emptyResponse, firstScreen, handlerTarget, moveInstruction } from './shared'

export class DropCoverageStrategy implements CoverageStrategy {
  readonly coverage = 'drop' as const

  generateResponse(context: CoverageContext): DefensiveResponse {
    const response = emptyResponse(this.coverage, context)
    const event = firstScreen(context)
    if (event === undefined) return response
    const handler = handlerTarget(context, event)
    const start = event.timeSeconds + context.parameters.reactionDelaySeconds
    const bigTarget = pointToward(RIM_POSITION, handler, context.parameters.dropDepthFeet)
    response.instructions = [
      moveInstruction(
        context,
        event.pointOfAttackDefenderId,
        event.handlerId,
        start + context.parameters.screenNavigationDelaySeconds,
        handler,
        'recovering_to_handler',
        context.parameters.recoverySpeedFeetPerSecond,
      ),
      moveInstruction(
        context,
        event.screenerDefenderId,
        event.screenerId,
        start,
        bigTarget,
        'drop_protecting_rim',
      ),
    ]
    return response
  }
}
