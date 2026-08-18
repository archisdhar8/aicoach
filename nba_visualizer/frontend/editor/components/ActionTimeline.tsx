import type { PlayAction, PlayerState } from '@/frontend/domain/models'
import { actionEndTime } from '@/frontend/domain/models'

interface ActionTimelineProps {
  actions: PlayAction[]
  players: PlayerState[]
  onDelete: (actionId: string) => void
}

export function ActionTimeline({ actions, players, onDelete }: ActionTimelineProps) {
  const names = new Map(players.map((state) => [state.player.id, state.player.name]))
  return (
    <section className="timeline-panel" aria-label="Play action timeline">
      <div className="section-heading">
        <div>
          <span>Possession timeline</span>
          <h2>{actions.length} actions</h2>
        </div>
      </div>
      {actions.length === 0 ? (
        <p className="empty-timeline">Draw a Phase 1 route or add a structured action.</p>
      ) : (
        <div className="timeline-scroll">
          <table>
            <thead>
              <tr><th>Action</th><th>Player</th><th>Start</th><th>End</th><th><span className="sr-only">Delete</span></th></tr>
            </thead>
            <tbody>
              {[...actions].sort((left, right) => left.startTime - right.startTime).map((action) => (
                <tr key={action.id}>
                  <td>
                    <span className={`action-chip action-${action.actionType}`}>{action.actionType}</span>
                    {action.actionType === 'shoot' && <small className="shot-action-summary">
                      {action.deterministicResult === 'made'
                        ? `Made · +${action.metadata.pointValue ?? 2}`
                        : 'Miss · no score'}
                      {typeof action.metadata.makeProbabilityHeuristic === 'number'
                        ? ` · ${(action.metadata.makeProbabilityHeuristic * 100).toFixed(0)}% player-aware heuristic`
                        : ''}
                    </small>}
                  </td>
                  <td>{names.get(action.playerId) ?? 'Unknown'}</td>
                  <td>{action.startTime.toFixed(1)}s</td>
                  <td>{actionEndTime(action).toFixed(1)}s</td>
                  <td><button aria-label={`Delete ${action.actionType} action`} className="delete-action" onClick={() => onDelete(action.id)} type="button">×</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
