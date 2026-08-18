'use client'

import { useEffect, useState } from 'react'
import { nbaApi } from '@/frontend/data/apiClient'
import type {
  NBAPlayerData,
  NBATeamData,
  PlayerState,
  TeamSide,
} from '@/frontend/domain/models'

const SEASON = currentNbaSeason()

interface LineupSelectorProps {
  disabled: boolean
  players: PlayerState[]
  onApply: (side: TeamSide, team: NBATeamData, players: NBAPlayerData[]) => void
}

interface SidePickerProps {
  label: string
  side: TeamSide
  teams: NBATeamData[]
  disabled: boolean
  onApply: LineupSelectorProps['onApply']
}

export function LineupSelector({ disabled, players, onApply }: LineupSelectorProps) {
  const [teams, setTeams] = useState<NBATeamData[]>([])
  const [status, setStatus] = useState<'loading' | 'ready' | 'unavailable'>('loading')

  useEffect(() => {
    const controller = new AbortController()
    nbaApi.listTeams(controller.signal)
      .then((response) => {
        setTeams(response.teams)
        setStatus('ready')
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        setStatus('unavailable')
      })
    return () => controller.abort()
  }, [])

  return (
    <section className="lineup-selector" aria-label="NBA lineups">
      <div className="lineup-heading">
        <div>
          <strong>Choose the teams</strong>
          <span>Select a team to load current-roster players and the best available starting-five estimate</span>
        </div>
        <span className={`data-availability ${status}`}>
          {status === 'ready' ? `${teams.length} teams available` : status === 'loading' ? 'Loading cache…' : 'NBA data unavailable'}
        </span>
      </div>
      {status === 'unavailable' ? (
        <p className="data-unavailable">No cached NBA directory is available. Demo players and play editing remain fully usable.</p>
      ) : (
        <div className="lineup-sides">
          <SidePicker disabled={disabled || status !== 'ready'} label="Offense" onApply={onApply} side="offense" teams={teams} />
          <SidePicker disabled={disabled || status !== 'ready'} label="Defense" onApply={onApply} side="defense" teams={teams} />
        </div>
      )}
      <span className="sr-only">{players.length} players currently on court</span>
    </section>
  )
}

function SidePicker({ label, side, teams, disabled, onApply }: SidePickerProps) {
  const [teamId, setTeamId] = useState('')
  const [team, setTeam] = useState<NBATeamData | null>(null)
  const [roster, setRoster] = useState<NBAPlayerData[]>([])
  const [selectedIds, setSelectedIds] = useState<string[]>(['', '', '', '', ''])
  const [message, setMessage] = useState('Choose a team')

  async function loadRoster(nextTeamId: string): Promise<void> {
    setTeamId(nextTeamId)
    setRoster([])
    setSelectedIds(['', '', '', '', ''])
    const selectedTeam = teams.find((item) => item.id === nextTeamId) ?? null
    setTeam(selectedTeam)
    if (selectedTeam === null) {
      setMessage('Choose a team')
      return
    }
    setMessage('Loading cached roster…')
    try {
      const rosterResponse = await nbaApi.getRoster(nextTeamId, SEASON, true)
      setRoster(rosterResponse.players)
      try {
        const lineupResponse = await nbaApi.getPreferredLineup(nextTeamId, SEASON, true)
        const preferredPlayers = lineupResponse.players
        setSelectedIds(preferredPlayers.map((player) => player.id))
        setMessage(`Current numbers + latest starters · ${lineupResponse.cacheStatus.replace('_', ' ')}`)
        if (preferredPlayers.length === 5) onApply(side, selectedTeam, preferredPlayers)
      } catch {
        setSelectedIds(['', '', '', '', ''])
        setMessage('Starting-five data unavailable · choose five manually')
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Roster unavailable')
    }
  }

  const selectedPlayers = selectedIds
    .map((id) => roster.find((player) => player.id === id))
    .filter((player): player is NBAPlayerData => player !== undefined)
  const canApply = team !== null && selectedPlayers.length === 5 && new Set(selectedIds).size === 5

  return (
    <fieldset className="lineup-side" disabled={disabled}>
      <legend>{label}</legend>
      <label>
        Team
        <select aria-label={`${label} NBA team`} onChange={(event) => void loadRoster(event.target.value)} value={teamId}>
          <option value="">Choose team</option>
          {teams.map((item) => <option key={item.id} value={item.id}>{item.fullName}</option>)}
        </select>
      </label>
      <div className="lineup-footer">
        <span>{message}</span>
      </div>
      {selectedPlayers.length > 0 && <div className="lineup-preview" aria-label={`${label} starting five`}>
        {selectedPlayers.map((player) => (
          <span key={player.id}>
            <b>#{player.jerseyNumber ?? '—'}</b> {player.lastName || player.displayName}
          </span>
        ))}
      </div>}
      {roster.length > 0 && <details className="lineup-edit">
        <summary>Adjust starting five</summary>
        <div className="lineup-slots">
          {selectedIds.map((id, index) => (
            <label key={`${side}-${index}`}>
              {index + 1}
              <select
                aria-label={`${label} player ${index + 1}`}
                disabled={roster.length === 0}
                onChange={(event) => setSelectedIds((current) => current.map((value, slot) => slot === index ? event.target.value : value))}
                value={id}
              >
                <option value="">Select player</option>
                {roster.map((player) => (
                  <option disabled={selectedIds.includes(player.id) && player.id !== id} key={player.id} value={player.id}>
                    #{player.jerseyNumber ?? '—'} · {player.displayName}{player.position ? ` · ${player.position}` : ''}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
        <button className="primary-action" disabled={!canApply} onClick={() => team && onApply(side, team, selectedPlayers)} type="button">Apply changes</button>
      </details>}
    </fieldset>
  )
}

function currentNbaSeason(now = new Date()): string {
  const year = now.getUTCFullYear()
  // NBA.com publishes upcoming-season rosters during July free agency, months
  // before regular-season games exist. Use that roster so offseason departures
  // are removed immediately; lineup ranking can still use prior-season roles.
  const seasonStartYear = now.getUTCMonth() >= 6 ? year : year - 1
  return `${seasonStartYear}-${String(seasonStartYear + 1).slice(-2)}`
}
