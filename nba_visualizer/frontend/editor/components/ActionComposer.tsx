import { useState, type FormEvent } from 'react'
import type {
  ActionType,
  CourtPosition,
  PlayAction,
  PlayDefinition,
} from '@/frontend/domain/models'

interface ActionComposerProps {
  play: PlayDefinition
  selectedPlayerId: string | null
  onAdd: (action: PlayAction) => void
}

const ACTIONS: Array<{ value: ActionType; label: string }> = [
  { value: 'move', label: 'MOVE' },
  { value: 'cut', label: 'CUT' },
  { value: 'dribble', label: 'DRIBBLE' },
  { value: 'screen', label: 'SCREEN' },
  { value: 'pass', label: 'PASS' },
  { value: 'shoot', label: 'SHOOT' },
  { value: 'hold', label: 'HOLD' },
]

export function ActionComposer({ play, selectedPlayerId, onAdd }: ActionComposerProps) {
  const [actionType, setActionType] = useState<ActionType>('move')
  const [startTime, setStartTime] = useState(0)
  const [duration, setDuration] = useState(1)
  const [targetX, setTargetX] = useState(78)
  const [targetY, setTargetY] = useState(25)
  const [targetPlayerId, setTargetPlayerId] = useState('')
  const [orientationDegrees, setOrientationDegrees] = useState(90)
  const [result, setResult] = useState<'made' | 'missed'>('made')

  const selected = play.initialFrame.players.find((state) => state.player.id === selectedPlayerId)
  const otherPlayers = play.initialFrame.players.filter((state) => state.player.id !== selectedPlayerId)
  const resolvedTargetPlayerId = otherPlayers.some((state) => state.player.id === targetPlayerId)
    ? targetPlayerId
    : otherPlayers[0]?.player.id ?? ''

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault()
    if (selected === undefined) return
    const source = { ...selected.position }
    const target = clampPosition({ x: targetX, y: targetY })
    const base = {
      id: crypto.randomUUID(),
      playerId: selected.player.id,
      startTime,
      duration,
      source,
      metadata: {},
    }
    let action: PlayAction
    if (actionType === 'move' || actionType === 'cut' || actionType === 'dribble') {
      action = { ...base, actionType, target, waypoints: [] }
    } else if (actionType === 'screen') {
      action = {
        ...base,
        actionType,
        screenLocation: target,
        orientationDegrees,
        targetPlayerId: targetPlayerId || null,
      }
    } else if (actionType === 'pass') {
      const receiver = play.initialFrame.players.find((state) => state.player.id === resolvedTargetPlayerId)
      if (receiver === undefined) return
      action = {
        ...base,
        actionType,
        targetPlayerId: resolvedTargetPlayerId,
        target: { ...receiver.position },
      }
    } else if (actionType === 'shoot') {
      action = {
        ...base,
        actionType,
        target: { x: 88.75, y: 25 },
        deterministicResult: result,
      }
    } else {
      action = { ...base, actionType }
    }
    onAdd(action)
  }

  return (
    <form className="action-composer" onSubmit={submit}>
      <div className="section-heading">
        <div>
          <span>Structured action</span>
          <h2>Add basketball action</h2>
        </div>
        <span className="action-player">{selected?.player.name ?? 'Select a player'}</span>
      </div>

      <label>
        Action
        <select value={actionType} onChange={(event) => setActionType(event.target.value as ActionType)}>
          {ACTIONS.map((action) => <option key={action.value} value={action.value}>{action.label}</option>)}
        </select>
      </label>
      <div className="field-row">
        <label>
          Start
          <input min="0" step="0.1" type="number" value={startTime} onChange={(event) => setStartTime(Number(event.target.value))} />
        </label>
        <label>
          Duration
          <input min="0.1" step="0.1" type="number" value={duration} onChange={(event) => setDuration(Number(event.target.value))} />
        </label>
      </div>

      {(actionType === 'move' || actionType === 'cut' || actionType === 'dribble' || actionType === 'screen') && (
        <div className="field-row">
          <label>
            Target x
            <input min="47" max="94" step="0.5" type="number" value={targetX} onChange={(event) => setTargetX(Number(event.target.value))} />
          </label>
          <label>
            Target y
            <input min="0" max="50" step="0.5" type="number" value={targetY} onChange={(event) => setTargetY(Number(event.target.value))} />
          </label>
        </div>
      )}

      {(actionType === 'pass' || actionType === 'screen') && (
        <label>
          {actionType === 'pass' ? 'Receiver' : 'Screen target (optional)'}
          <select value={actionType === 'screen' ? targetPlayerId : resolvedTargetPlayerId} onChange={(event) => setTargetPlayerId(event.target.value)}>
            {actionType === 'screen' && <option value="">No known target</option>}
            {otherPlayers.map((state) => (
              <option key={state.player.id} value={state.player.id}>
                {state.player.name} · {state.teamSide}
              </option>
            ))}
          </select>
        </label>
      )}

      {actionType === 'screen' && (
        <label>
          Orientation
          <input min="0" max="359" step="5" type="number" value={orientationDegrees} onChange={(event) => setOrientationDegrees(Number(event.target.value))} />
        </label>
      )}

      {actionType === 'shoot' && (
        <label>
          Demo result
          <select value={result} onChange={(event) => setResult(event.target.value as 'made' | 'missed')}>
            <option value="made">Made</option>
            <option value="missed">Missed</option>
          </select>
        </label>
      )}

      <button className="primary-action" disabled={selected === undefined} type="submit">
        Add {actionType.toUpperCase()}
      </button>
    </form>
  )
}

function clampPosition(position: CourtPosition): CourtPosition {
  return {
    x: Math.min(Math.max(position.x, 47), 94),
    y: Math.min(Math.max(position.y, 0), 50),
  }
}
