import { offensivePositionAt } from '../events'
import { pointToward } from '../geometry'
import type { CoverageContext, CoverageStrategy, DefensiveResponse } from '../types'
import { defenderPosition, emptyResponse, firstScreen, handlerTarget, moveInstruction } from './shared'

export class HedgeCoverageStrategy implements CoverageStrategy {
  readonly coverage = 'hedge' as const

  generateResponse(context: CoverageContext): DefensiveResponse {
    const response = emptyResponse(this.coverage, context)
    const event = firstScreen(context)
    if (event === undefined) return response
    const start = event.timeSeconds + context.parameters.reactionDelaySeconds
    const handler = handlerTarget(context, event)
    const bigStart = defenderPosition(context, event.screenerDefenderId)
    const contain = pointToward(handler, bigStart, context.parameters.helpDistanceFeet)
    const recover = offensivePositionAt(context.play, event.screenerId, start + 1.2)
    const big = moveInstruction(
      context,
      event.screenerDefenderId,
      event.screenerId,
      start,
      contain,
      'hedging_handler',
    )
    big.waypoints.push({
      timeSeconds: Math.max(big.waypoints.at(-1)?.timeSeconds ?? start, start + 0.55)
        + Math.hypot(recover.x - contain.x, recover.y - contain.y)
          / context.parameters.recoverySpeedFeetPerSecond,
      position: recover,
      state: 'recovering_to_screener',
    })
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
      big,
    ]
    return response
  }
}
