import { offensivePositionAt } from '../events'
import type { CoverageContext, CoverageStrategy, DefensiveResponse } from '../types'
import { emptyResponse, firstScreen, handlerTarget, moveInstruction } from './shared'

export class SwitchCoverageStrategy implements CoverageStrategy {
  readonly coverage = 'switch' as const

  generateResponse(context: CoverageContext): DefensiveResponse {
    const response = emptyResponse(this.coverage, context)
    const event = firstScreen(context)
    if (event === undefined) return response
    const switchTime = event.timeSeconds
      + context.parameters.reactionDelaySeconds
      + context.parameters.screenNavigationDelaySeconds
    response.assignments = response.assignments.map((assignment) => (
      assignment.defenderId === event.pointOfAttackDefenderId
        || assignment.defenderId === event.screenerDefenderId
        ? { ...assignment, endTime: switchTime }
        : assignment
    ))
    response.assignments.push(
      {
        defenderId: event.pointOfAttackDefenderId,
        offensivePlayerId: event.screenerId,
        startTime: switchTime,
        reason: 'switch',
      },
      {
        defenderId: event.screenerDefenderId,
        offensivePlayerId: event.handlerId,
        startTime: switchTime,
        reason: 'switch',
      },
    )
    response.instructions = [
      moveInstruction(
        context,
        event.pointOfAttackDefenderId,
        event.screenerId,
        switchTime,
        offensivePositionAt(context.play, event.screenerId, switchTime + 0.7),
        'switched_to_screener',
      ),
      moveInstruction(
        context,
        event.screenerDefenderId,
        event.handlerId,
        switchTime,
        handlerTarget(context, event),
        'switched_to_handler',
      ),
    ]
    return response
  }
}
