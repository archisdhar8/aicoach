import type { CourtPosition, PlayAction, PlayDefinition } from '@/frontend/domain/models'
import { actionEndTime } from '@/frontend/domain/models'
import type { DefensiveAssignment, ScreenInteractionEvent } from './types'

export function initialAssignments(play: PlayDefinition): DefensiveAssignment[] {
  const offense = play.initialFrame.players.filter((state) => state.teamSide === 'offense')
  const defense = play.initialFrame.players.filter((state) => state.teamSide === 'defense')
  return defense.slice(0, offense.length).map((defender, index) => ({
    defenderId: defender.player.id,
    offensivePlayerId: offense[index].player.id,
    startTime: 0,
    reason: 'initial_matchup',
  }))
}

export function detectScreenInteractions(
  play: PlayDefinition,
  assignments: DefensiveAssignment[],
): ScreenInteractionEvent[] {
  const screens = play.actions.filter((action) => action.actionType === 'screen')
  const dribbles = play.actions.filter((action) => action.actionType === 'dribble')
  const events: ScreenInteractionEvent[] = []
  for (const screen of screens) {
    const handlerAction = dribbles.find(
      (dribble) => dribble.startTime < actionEndTime(screen)
        && actionEndTime(dribble) > screen.startTime,
    )
    if (handlerAction === undefined || screen.targetPlayerId == null) continue
    const pointOfAttack = assignmentAt(assignments, screen.targetPlayerId, screen.startTime)
    if (pointOfAttack?.offensivePlayerId !== handlerAction.playerId) continue
    const screenerDefender = assignments.find(
      (assignment) => assignment.offensivePlayerId === screen.playerId
        && assignment.startTime <= screen.startTime
        && (assignment.endTime === undefined || screen.startTime < assignment.endTime),
    )
    if (screenerDefender === undefined) continue
    events.push({
      type: 'screen_interaction',
      screenActionId: screen.id,
      timeSeconds: screen.startTime,
      handlerId: handlerAction.playerId,
      screenerId: screen.playerId,
      pointOfAttackDefenderId: screen.targetPlayerId,
      screenerDefenderId: screenerDefender.defenderId,
      location: { ...screen.screenLocation },
      side: screen.screenLocation.y < 15 ? 'right' : screen.screenLocation.y > 35 ? 'left' : 'middle',
    })
  }
  return events.sort((left, right) => left.timeSeconds - right.timeSeconds)
}

export function assignmentAt(
  assignments: DefensiveAssignment[],
  defenderId: string,
  timeSeconds: number,
): DefensiveAssignment | undefined {
  return assignments
    .filter((assignment) => assignment.defenderId === defenderId
      && assignment.startTime <= timeSeconds
      && (assignment.endTime === undefined || timeSeconds < assignment.endTime))
    .sort((left, right) => right.startTime - left.startTime)[0]
}

export function offensivePositionAt(
  play: PlayDefinition,
  playerId: string,
  timeSeconds: number,
): CourtPosition {
  const initial = play.initialFrame.players.find((state) => state.player.id === playerId)?.position
  if (initial === undefined) return { x: 47, y: 25 }
  const actions = play.actions
    .filter((action) => action.playerId === playerId)
    .sort((left, right) => left.startTime - right.startTime)
  let position = { ...initial }
  for (const action of actions) {
    if (timeSeconds < action.startTime) return position
    if (timeSeconds < actionEndTime(action)) return positionDuring(action, timeSeconds)
    position = positionAfter(action, position)
  }
  return position
}

function positionDuring(action: PlayAction, timeSeconds: number): CourtPosition {
  const progress = (timeSeconds - action.startTime) / action.duration
  if (action.actionType === 'move' || action.actionType === 'cut' || action.actionType === 'dribble') {
    const points = [action.source, ...action.waypoints, action.target]
    const scaled = Math.min(Math.max(progress, 0), 1) * (points.length - 1)
    const index = Math.min(Math.floor(scaled), points.length - 2)
    const segment = scaled - index
    return {
      x: points[index].x + (points[index + 1].x - points[index].x) * segment,
      y: points[index].y + (points[index + 1].y - points[index].y) * segment,
    }
  }
  if (action.actionType === 'screen') return { ...action.screenLocation }
  return { ...action.source }
}

function positionAfter(action: PlayAction, fallback: CourtPosition): CourtPosition {
  if (action.actionType === 'move' || action.actionType === 'cut' || action.actionType === 'dribble') {
    return { ...action.target }
  }
  if (action.actionType === 'screen') return { ...action.screenLocation }
  return fallback
}
