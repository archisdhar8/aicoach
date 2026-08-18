import type { KeyboardEventHandler, PointerEventHandler } from 'react'
import type { PlayerState } from '@/frontend/domain/models'
import { halfCourtToSvg } from '@/frontend/court/rendering/coordinates'

interface PlayerMarkerProps {
  state: PlayerState
  selected: boolean
  interactive: boolean
  onKeyDown: KeyboardEventHandler<SVGGElement>
  onPointerDown: PointerEventHandler<SVGGElement>
}

export function PlayerMarker({ state, selected, interactive, onKeyDown, onPointerDown }: PlayerMarkerProps) {
  const point = halfCourtToSvg(state.position)
  const fill = state.teamSide === 'offense' ? '#e85b26' : '#265db8'
  const shortName = playerMarkerName(state.player.name)

  return (
    <g
      aria-label={`${state.player.name}, number ${state.player.jerseyNumber}, ${state.teamSide}`}
      className={`player-marker${selected ? ' selected' : ''}`}
      data-player-id={state.player.id}
      data-player-x={state.position.x.toFixed(3)}
      data-player-y={state.position.y.toFixed(3)}
      onKeyDown={onKeyDown}
      onPointerDown={onPointerDown}
      role="button"
      style={{ cursor: interactive ? 'grab' : 'default' }}
      tabIndex={interactive ? 0 : -1}
      transform={`translate(${point.x} ${point.y})`}
    >
      {selected && <circle className="selection-ring" r="4" />}
      <circle r="2.8" fill={fill} />
      <text className="player-number" y="-.45">{state.player.jerseyNumber ?? '—'}</text>
      <text className="player-name" y="1.35">{shortName}</text>
      <title>{`${state.player.name} · ${state.teamSide}`}</title>
    </g>
  )
}

function playerMarkerName(name: string): string {
  const parts = name.trim().split(/\s+/)
  return (parts.at(-1) ?? name).toUpperCase().slice(0, 8)
}
