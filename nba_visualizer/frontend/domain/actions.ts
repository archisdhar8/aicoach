import type {
  CourtPosition,
  PlayDefinition,
} from './models'
import { actionEndTime } from './models'

export interface ActionValidationError {
  code: 'invalid_player' | 'invalid_target' | 'invalid_timing' | 'overlap' | 'possession'
  message: string
  actionId?: string
}

export class PlayValidationError extends Error {
  constructor(readonly errors: ActionValidationError[]) {
    super(errors.map((error) => error.message).join('; '))
    this.name = 'PlayValidationError'
  }
}

export function validatePlayDefinition(play: PlayDefinition): ActionValidationError[] {
  const errors: ActionValidationError[] = []
  const players = new Set(play.initialFrame.players.map((state) => state.player.id))

  for (const action of play.actions) {
    if (!players.has(action.playerId)) {
      errors.push({
        code: 'invalid_player',
        actionId: action.id,
        message: `Action ${action.id} references a player outside this play.`,
      })
    }
    if (action.duration <= 0 || action.startTime < 0) {
      errors.push({
        code: 'invalid_timing',
        actionId: action.id,
        message: `${action.actionType.toUpperCase()} must have a non-negative start and positive duration.`,
      })
    }
    if (!isCourtPosition(action.source)) {
      errors.push({ code: 'invalid_target', actionId: action.id, message: 'Action source is outside the court.' })
    }
    if ('target' in action && !isCourtPosition(action.target)) {
      errors.push({ code: 'invalid_target', actionId: action.id, message: 'Action target is outside the court.' })
    }
    if (action.actionType === 'screen' && !isCourtPosition(action.screenLocation)) {
      errors.push({ code: 'invalid_target', actionId: action.id, message: 'Screen location is outside the court.' })
    }
    if ('targetPlayerId' in action && action.targetPlayerId != null && !players.has(action.targetPlayerId)) {
      errors.push({
        code: 'invalid_target',
        actionId: action.id,
        message: `${action.actionType.toUpperCase()} references a target player outside this play.`,
      })
    }
    if (action.actionType === 'pass' && action.targetPlayerId === action.playerId) {
      errors.push({ code: 'invalid_target', actionId: action.id, message: 'A player cannot pass to themself.' })
    }
  }

  for (const playerId of players) {
    const actions = play.actions
      .filter((action) => action.playerId === playerId)
      .sort((left, right) => left.startTime - right.startTime)
    for (let index = 1; index < actions.length; index += 1) {
      const previous = actions[index - 1]
      const current = actions[index]
      if (current.startTime < actionEndTime(previous)) {
        errors.push({
          code: 'overlap',
          actionId: current.id,
          message: `${playerName(play, playerId)} cannot execute ${previous.actionType.toUpperCase()} and ${current.actionType.toUpperCase()} at the same time.`,
        })
      }
    }
  }

  let possessor = play.initialFrame.ball.state === 'possessed'
    ? play.initialFrame.ball.playerId
    : null
  let ballAvailableAt = 0
  const ballActions = play.actions
    .filter((action) => action.actionType === 'pass' || action.actionType === 'shoot' || action.actionType === 'dribble')
    .sort((left, right) => left.startTime - right.startTime)
  for (const action of ballActions) {
    if (action.startTime < ballAvailableAt || action.playerId !== possessor) {
      errors.push({
        code: 'possession',
        actionId: action.id,
        message: `${playerName(play, action.playerId)} cannot ${action.actionType.toUpperCase()} without possessing the ball.`,
      })
    }
    if (action.actionType === 'pass') {
      possessor = action.targetPlayerId
      ballAvailableAt = actionEndTime(action)
    } else if (action.actionType === 'shoot') {
      possessor = null
      ballAvailableAt = actionEndTime(action)
    }
  }
  return deduplicate(errors)
}

export function assertValidPlay(play: PlayDefinition): void {
  const errors = validatePlayDefinition(play)
  if (errors.length > 0) throw new PlayValidationError(errors)
}

function playerName(play: PlayDefinition, playerId: string): string {
  return play.initialFrame.players.find((state) => state.player.id === playerId)?.player.name ?? playerId
}

function isCourtPosition(position: CourtPosition): boolean {
  return Number.isFinite(position.x)
    && Number.isFinite(position.y)
    && position.x >= 47
    && position.x <= 94
    && position.y >= 0
    && position.y <= 50
}

function deduplicate(errors: ActionValidationError[]): ActionValidationError[] {
  const seen = new Set<string>()
  return errors.filter((error) => {
    const key = `${error.code}:${error.actionId ?? ''}:${error.message}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}
