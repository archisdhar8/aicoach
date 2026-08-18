import { useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import type {
  CourtPosition,
  PlayAction,
  PlayerRoute,
  SimulationFrame,
} from '@/frontend/domain/models'
import { actionEndTime } from '@/frontend/domain/models'
import { PlayerMarker } from '@/frontend/players/components/PlayerMarker'
import { clientPointToHalfCourt, halfCourtToSvg } from '@/frontend/court/rendering/coordinates'
import type { DefensiveDebugState } from '@/frontend/defense/engine'
import type { DefensiveResponse } from '@/frontend/defense/engine'
import type { AnalyticsOverlay, FrameAnalytics } from '@/frontend/analytics/types'
import { routeLength } from '@/frontend/editor/whiteboard'
import { shotPointValue } from '@/frontend/analytics/shotOutcome'

interface BasketballCourtProps {
  frame: SimulationFrame
  routes: PlayerRoute[]
  actions: PlayAction[]
  selectedPlayerId: string | null
  editingDisabled: boolean
  futureBallOwnerId: string | null
  onPlayerTap: (playerId: string) => void
  onGestureComplete: (playerId: string, positions: CourtPosition[]) => void
  onShoot: () => void
  defensiveDebug: DefensiveDebugState | null
  defensivePreview: DefensiveResponse | null
  defensiveDebugEnabled: boolean
  analyticsOverlays: AnalyticsOverlay[]
}

type Gesture = { playerId: string; positions: CourtPosition[] }

export function BasketballCourt({
  frame,
  routes,
  actions,
  selectedPlayerId,
  editingDisabled,
  futureBallOwnerId,
  onPlayerTap,
  onGestureComplete,
  onShoot,
  defensiveDebug,
  defensivePreview,
  defensiveDebugEnabled,
  analyticsOverlays,
}: BasketballCourtProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const gestureRef = useRef<Gesture | null>(null)
  const [gesturePreview, setGesturePreview] = useState<Gesture | null>(null)
  const ballPoint = halfCourtToSvg(frame.ball.position)
  const completedShot = actions
    .filter((action): action is Extract<PlayAction, { actionType: 'shoot' }> => (
      action.actionType === 'shoot' && actionEndTime(action) <= frame.timestampSeconds + 0.001
    ))
    .sort((left, right) => actionEndTime(right) - actionEndTime(left))[0]
  const orderedPlayers = [
    ...frame.players.filter((state) => state.player.id !== selectedPlayerId && state.teamSide === 'defense'),
    ...frame.players.filter((state) => state.player.id !== selectedPlayerId && state.teamSide === 'offense'),
    ...frame.players.filter((state) => state.player.id === selectedPlayerId),
  ]

  function playerPointerDown(playerId: string, event: ReactPointerEvent<SVGGElement>): void {
    event.preventDefault()
    event.stopPropagation()
    if (editingDisabled || svgRef.current === null) return
    svgRef.current.setPointerCapture(event.pointerId)
    const player = frame.players.find((state) => state.player.id === playerId)
    if (player !== undefined) {
      const gesture = { playerId, positions: [{ ...player.position }] }
      gestureRef.current = gesture
      setGesturePreview(gesture)
    }
  }

  function courtPointerDown(event: ReactPointerEvent<SVGSVGElement>): void {
    if (editingDisabled || selectedPlayerId === null) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    const player = frame.players.find((state) => state.player.id === selectedPlayerId)
    if (player === undefined) return
    const gesture = {
      playerId: selectedPlayerId,
      positions: [{ ...player.position }, pointerPosition(event)],
    }
    gestureRef.current = gesture
    setGesturePreview(gesture)
  }

  function pointerMove(event: ReactPointerEvent<SVGSVGElement>): void {
    const gesture = gestureRef.current
    if (gesture === null || editingDisabled) return
    event.preventDefault()
    const position = pointerPosition(event)
    const positions = [...gesture.positions, position]
    gestureRef.current = { ...gesture, positions }
    setGesturePreview({ ...gesture, positions })
  }

  function finishGesture(event: ReactPointerEvent<SVGSVGElement>): void {
    const gesture = gestureRef.current
    if (gesture !== null && event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    gestureRef.current = null
    setGesturePreview(null)
    if (gesture === null) return
    if (routeLength(gesture.positions) < 0.55) onPlayerTap(gesture.playerId)
    else onGestureComplete(gesture.playerId, gesture.positions)
  }

  function pointerPosition(event: ReactPointerEvent): CourtPosition {
    const svg = svgRef.current
    if (svg === null) return { x: 47, y: 0 }
    return clientPointToHalfCourt(
      { x: event.clientX, y: event.clientY },
      svg.getBoundingClientRect(),
    )
  }

  return (
    <svg
      aria-label={`Basketball half-court at ${frame.timestampSeconds.toFixed(1)} seconds`}
      className="court-svg whiteboard-court"
      data-testid="basketball-court"
      onPointerCancel={finishGesture}
      onPointerDown={courtPointerDown}
      onPointerMove={pointerMove}
      onPointerUp={finishGesture}
      ref={svgRef}
      role="img"
      viewBox="0 0 47 50"
    >
      <defs>
        <pattern id="wood" width="5" height="50" patternUnits="userSpaceOnUse">
          <rect width="5" height="50" fill="#c98c51" />
          <line className="court-wood-line" x1="5" x2="5" y1="0" y2="50" />
        </pattern>
        <marker id="route-arrow" markerHeight="4" markerWidth="4" orient="auto" refX="3" refY="2">
          <path d="M 0 0 L 4 2 L 0 4 z" />
        </marker>
        <marker id="action-arrow" markerHeight="4" markerWidth="4" orient="auto" refX="3" refY="2">
          <path className="action-arrow-head" d="M 0 0 L 4 2 L 0 4 z" />
        </marker>
      </defs>
      <rect width="47" height="50" fill="url(#wood)" />
      <rect className="court-line" x=".4" y=".4" width="46.2" height="49.2" />
      <line className="court-line half-court-boundary" x1=".5" x2=".5" y1=".5" y2="49.5" />
      <path className="court-line" d="M .5 19 A 6 6 0 0 1 .5 31" />

      <rect className="paint-fill" x="28" y="17" width="19" height="16" />
      <rect className="court-line" x="28" y="17" width="18.6" height="16" />
      <line className="court-line free-throw-line" x1="28" x2="28" y1="17" y2="33" />
      <path className="court-line free-throw-circle" d="M 28 19 A 6 6 0 0 0 28 31" />
      <path className="court-dash" d="M 28 19 A 6 6 0 0 1 28 31" />

      <line className="court-line backboard" x1="41.75" x2="41.75" y1="22" y2="28" />
      <circle className="court-line hoop" cx="41.75" cy="25" r=".75" />
      <g className="basket-net" aria-label="Basket and net">
        <circle cx="41.75" cy="25" r=".62" />
        <circle cx="41.75" cy="25" r=".34" />
        <path d="M 41.2 24.7 L 42.3 25.3 M 41.2 25.3 L 42.3 24.7 M 41.75 24.38 L 41.75 25.62 M 41.13 25 L 42.37 25" />
      </g>
      <path className="court-line restricted-area" d="M 41.75 21 A 4 4 0 0 0 41.75 29" />
      <path className="court-line three-point" d="M 47 3 L 33 3 A 23.75 23.75 0 0 0 33 47 L 47 47" />

      <g className="route-layer" aria-label="Player routes">
        {routes.map((route) => {
          const points = route.points.map((point) => halfCourtToSvg(point.position))
          const path = points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ')
          return (
            <path
              className={`player-route${route.playerId === selectedPlayerId ? ' selected' : ''}`}
              d={path}
              data-route-player-id={route.playerId}
              key={route.playerId}
              markerEnd="url(#route-arrow)"
            />
          )
        })}
      </g>

      <g className="structured-action-layer" aria-label="Structured basketball actions">
        {actions.filter((action) => action.metadata.phase1Route !== true).map((action) => (
          <ActionMark
            action={action}
            active={frame.currentActions.some((current) => current.id === action.id)}
            key={action.id}
          />
        ))}
      </g>

      {!editingDisabled && defensivePreview && (
        <g className="defensive-preview-layer" aria-label="Automatic defensive response preview">
          {defensivePreview.instructions.map((instruction) => (
            <path
              d={instruction.waypoints.map((waypoint, index) => {
                const point = halfCourtToSvg(waypoint.position)
                return `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
              }).join(' ')}
              data-defender-preview-id={instruction.defenderId}
              key={instruction.defenderId}
            />
          ))}
        </g>
      )}

      {gesturePreview && gesturePreview.positions.length > 1 && (
        <path
          className={`gesture-preview${gesturePreview.playerId === futureBallOwnerId ? ' ball-handler' : ''}`}
          d={gesturePreview.positions.map((position, index) => {
            const point = halfCourtToSvg(position)
            return `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
          }).join(' ')}
          markerEnd="url(#action-arrow)"
        />
      )}

      {completedShot && (
        <g className={`shot-result-feedback ${completedShot.deterministicResult}`} aria-label={
          completedShot.deterministicResult === 'made'
            ? `Made ${shotMetadataPointValue(completedShot)} point shot`
            : 'Missed shot'
        }>
          <circle cx="38.2" cy="19.5" r="3.2" />
          <text x="38.2" y="19.3">
            {completedShot.deterministicResult === 'made'
              ? `+${shotMetadataPointValue(completedShot)}`
              : 'MISS'}
          </text>
          <text className="shot-result-detail" x="38.2" y="21.2">
            {completedShot.deterministicResult === 'made' ? 'MADE' : 'NO SCORE'}
          </text>
        </g>
      )}

      {frame.analytics && (
        <SpatialAnalyticsOverlay
          analytics={frame.analytics}
          defensiveDebug={defensiveDebug}
          enabled={analyticsOverlays}
          frame={frame}
        />
      )}
      {defensiveDebugEnabled && defensiveDebug && (
        <DefensiveDebugOverlay debug={defensiveDebug} frame={frame} />
      )}

      {orderedPlayers.map((state) => (
        <PlayerMarker
          interactive={!editingDisabled}
          key={state.player.id}
          onKeyDown={(event) => {
            if (!editingDisabled && (event.key === 'Enter' || event.key === ' ')) {
              event.preventDefault()
              onPlayerTap(state.player.id)
            }
          }}
          onPointerDown={(event) => playerPointerDown(state.player.id, event)}
          selected={state.player.id === selectedPlayerId}
          state={state}
        />
      ))}
      <circle
        aria-label="Shoot at rim"
        className="rim-hit-target"
        cx="41.75"
        cy="25"
        onPointerDown={(event) => {
          event.preventDefault()
          event.stopPropagation()
          if (!editingDisabled) onShoot()
        }}
        r="2.4"
        role="button"
        tabIndex={editingDisabled ? -1 : 0}
      >
        <title>Click the rim to shoot</title>
      </circle>
      <circle className="ball-marker" cx={ballPoint.x} cy={ballPoint.y} r="1.05">
        <title>Ball</title>
      </circle>
    </svg>
  )
}

function SpatialAnalyticsOverlay({
  analytics,
  defensiveDebug,
  enabled,
  frame,
}: {
  analytics: FrameAnalytics
  defensiveDebug: DefensiveDebugState | null
  enabled: AnalyticsOverlay[]
  frame: SimulationFrame
}) {
  const playerPosition = (playerId: string) => frame.players.find(
    (state) => state.player.id === playerId,
  )?.position
  return (
    <g className="spatial-analytics-layer" aria-label="Spatial analytics overlay">
      {enabled.includes('spacing') && analytics.spacing.pairDistances.map((pair) => {
        const first = playerPosition(pair.firstPlayerId)
        const second = playerPosition(pair.secondPlayerId)
        if (first === undefined || second === undefined) return null
        const from = halfCourtToSvg(first)
        const to = halfCourtToSvg(second)
        return <line className="analytics-spacing" key={`${pair.firstPlayerId}-${pair.secondPlayerId}`} x1={from.x} x2={to.x} y1={from.y} y2={to.y}><title>{pair.distanceFeet.toFixed(1)} ft spacing</title></line>
      })}
      {enabled.includes('passing_lanes') && analytics.passingLanes.map((lane) => {
        const from = halfCourtToSvg(lane.source)
        const to = halfCourtToSvg(lane.target)
        return <line className={`analytics-pass risk-${lane.riskBand}`} data-risk-score={lane.interceptionRiskScore} key={lane.targetPlayerId} x1={from.x} x2={to.x} y1={from.y} y2={to.y}><title>Pass risk {lane.interceptionRiskScore.toFixed(0)}/100 · clearance {lane.minimumDefenderDistanceFeet?.toFixed(1) ?? 'n/a'} ft</title></line>
      })}
      {enabled.includes('defender_distance') && analytics.nearestDefenders.map((evaluation) => {
        if (evaluation.defenderId === null) return null
        const offense = playerPosition(evaluation.offensivePlayerId)
        const defense = playerPosition(evaluation.defenderId)
        if (offense === undefined || defense === undefined) return null
        const from = halfCourtToSvg(offense)
        const to = halfCourtToSvg(defense)
        const middle = { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 }
        return (
          <g key={evaluation.offensivePlayerId}>
            <line className="analytics-defender-distance" x1={from.x} x2={to.x} y1={from.y} y2={to.y} />
            <text className="analytics-distance-label" x={middle.x} y={middle.y}>{evaluation.distanceFeet?.toFixed(1)}</text>
          </g>
        )
      })}
      {enabled.includes('driving_lanes') && analytics.driveLane && (
        <polygon
          className="analytics-drive-lane"
          data-openness-score={analytics.driveLane.opennessScore}
          points={corridorPolygon(
            analytics.driveLane.source,
            analytics.driveLane.target,
            analytics.driveLane.corridorWidthFeet,
          ).map(halfCourtToSvg).map((point) => `${point.x},${point.y}`).join(' ')}
        >
          <title>Drive openness {analytics.driveLane.opennessScore.toFixed(0)}/100</title>
        </polygon>
      )}
      {enabled.includes('shot_openness') && analytics.shotOpenness.map((shot) => {
        const point = halfCourtToSvg(shot.location)
        return (
          <circle
            className="analytics-shot-openness"
            data-openness-score={shot.opennessScore}
            cx={point.x}
            cy={point.y}
            key={shot.offensivePlayerId}
            r={1.1 + shot.opennessScore / 100 * 1.5}
          >
            <title>Heuristic shot openness {shot.opennessScore.toFixed(0)}/100 · {shot.shotDistanceFeet.toFixed(1)} ft</title>
          </circle>
        )
      })}
      {enabled.includes('matchups') && defensiveDebug?.assignments.map((assignment) => {
        const defender = playerPosition(assignment.defenderId)
        const offense = playerPosition(assignment.offensivePlayerId)
        if (defender === undefined || offense === undefined) return null
        const from = halfCourtToSvg(defender)
        const to = halfCourtToSvg(offense)
        return <line className="analytics-matchup" key={assignment.defenderId} x1={from.x} x2={to.x} y1={from.y} y2={to.y} />
      })}
    </g>
  )
}

function corridorPolygon(
  source: CourtPosition,
  target: CourtPosition,
  widthFeet: number,
): CourtPosition[] {
  const dx = target.x - source.x
  const dy = target.y - source.y
  const length = Math.max(Math.hypot(dx, dy), 0.001)
  const offset = { x: -dy / length * widthFeet, y: dx / length * widthFeet }
  return [
    { x: source.x + offset.x, y: source.y + offset.y },
    { x: target.x + offset.x, y: target.y + offset.y },
    { x: target.x - offset.x, y: target.y - offset.y },
    { x: source.x - offset.x, y: source.y - offset.y },
  ]
}

function DefensiveDebugOverlay({
  debug,
  frame,
}: {
  debug: DefensiveDebugState
  frame: SimulationFrame
}) {
  const playerPosition = (playerId: string) => frame.players.find(
    (state) => state.player.id === playerId,
  )?.position
  return (
    <g className="defensive-debug-layer" aria-label="Defensive debug overlay">
      {debug.assignments.map((assignment) => {
        const defender = playerPosition(assignment.defenderId)
        const offense = playerPosition(assignment.offensivePlayerId)
        if (defender === undefined || offense === undefined) return null
        const from = halfCourtToSvg(defender)
        const to = halfCourtToSvg(offense)
        return <line className="debug-assignment" key={`${assignment.defenderId}-${assignment.offensivePlayerId}`} x1={from.x} x2={to.x} y1={from.y} y2={to.y} />
      })}
      {debug.targets.map((target) => {
        const defender = playerPosition(target.defenderId)
        if (defender === undefined) return null
        const from = halfCourtToSvg(defender)
        const to = halfCourtToSvg(target.position)
        return (
          <g key={target.defenderId}>
            <line className="debug-target" x1={from.x} x2={to.x} y1={from.y} y2={to.y} />
            <circle className="debug-target-point" cx={to.x} cy={to.y} r=".7"><title>{target.state}</title></circle>
          </g>
        )
      })}
      {debug.screenEvents.map((event) => {
        const point = halfCourtToSvg(event.location)
        return <circle className="debug-screen-event" cx={point.x} cy={point.y} key={event.screenActionId} r="2.8" />
      })}
      <text className="debug-state-label" x="2" y="4">{debug.coverage.toUpperCase()} · {debug.coverageState}</text>
    </g>
  )
}

function ActionMark({ action, active }: { action: PlayAction; active: boolean }) {
  const source = halfCourtToSvg(action.source)
  const stateClass = active ? ' active' : ''
  if (action.actionType === 'screen') {
    const location = halfCourtToSvg(action.screenLocation)
    return (
      <g
        className={`screen-mark${stateClass}`}
        data-action-id={action.id}
        data-action-type="screen"
        transform={`translate(${location.x} ${location.y}) rotate(${action.orientationDegrees})`}
      >
        <line x1="-2.2" x2="2.2" y1="0" y2="0" />
        <line x1="-2.2" x2="-2.2" y1="-.9" y2=".9" />
        <line x1="2.2" x2="2.2" y1="-.9" y2=".9" />
        <title>Screen at {action.startTime.toFixed(1)} seconds</title>
      </g>
    )
  }
  if (action.actionType === 'hold') {
    return (
      <circle
        className={`hold-mark${stateClass}`}
        cx={source.x}
        cy={source.y}
        data-action-id={action.id}
        r="2.4"
      />
    )
  }
  const target = halfCourtToSvg(action.target)
  const waypoints = action.actionType === 'move' || action.actionType === 'cut' || action.actionType === 'dribble'
    ? action.waypoints.map(halfCourtToSvg)
    : []
  const path = [source, ...waypoints, target]
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`)
    .join(' ')
  return (
    <path
      className={`structured-action action-${action.actionType}${stateClass}`}
      d={path}
      data-action-id={action.id}
      data-action-type={action.actionType}
      markerEnd="url(#action-arrow)"
    />
  )
}

function shotMetadataPointValue(action: Extract<PlayAction, { actionType: 'shoot' }>): 2 | 3 {
  return action.metadata.pointValue === 3 || action.metadata.pointValue === 2
    ? action.metadata.pointValue
    : shotPointValue(action.source)
}
