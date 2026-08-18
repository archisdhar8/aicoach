'use client'

import { useEffect, useMemo, useState } from 'react'
import { nbaApi } from '@/frontend/data/apiClient'
import type {
  NBAGameData,
  NBATeamData,
  RealPossession,
} from '@/frontend/domain/models'

interface PossessionBrowserProps {
  disabled: boolean
  onReconstruct: (possession: RealPossession) => void
}

export function PossessionBrowser({ disabled, onReconstruct }: PossessionBrowserProps) {
  const [teams, setTeams] = useState<NBATeamData[]>([])
  const [games, setGames] = useState<NBAGameData[]>([])
  const [possessions, setPossessions] = useState<RealPossession[]>([])
  const [selected, setSelected] = useState<RealPossession | null>(null)
  const [teamId, setTeamId] = useState('')
  const [season, setSeason] = useState('2025-26')
  const [gameDate, setGameDate] = useState('')
  const [gameId, setGameId] = useState('')
  const [quarter, setQuarter] = useState('all')
  const [outcome, setOutcome] = useState('all')
  const [player, setPlayer] = useState('')
  const [shotType, setShotType] = useState('')
  const [status, setStatus] = useState('Choose a team, season, or date to find games.')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    nbaApi.listTeams(controller.signal)
      .then((response) => setTeams(response.teams))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        setStatus('NBA directory unavailable. Cached editor data remains usable.')
      })
    return () => controller.abort()
  }, [])

  const filtered = useMemo(() => possessions.filter((possession) => {
    if (quarter !== 'all' && possession.period !== Number(quarter)) return false
    if (outcome === 'made' && possession.result.made !== true) return false
    if (outcome === 'missed' && possession.result.made !== false) return false
    if (outcome === 'turnover' && !possession.result.turnover) return false
    const playerQuery = player.trim().toLowerCase()
    if (playerQuery && ![
      ...possession.offensiveLineup,
      ...possession.defensiveLineup,
    ].some((item) => `${item.displayName ?? ''} ${item.externalId}`.toLowerCase().includes(playerQuery))) return false
    const shotQuery = shotType.trim().toLowerCase()
    if (shotQuery && !possession.events.some((event) => (
      event.shotType ?? ''
    ).toLowerCase().includes(shotQuery))) return false
    return true
  }), [outcome, player, possessions, quarter, shotType])

  async function findGames(): Promise<void> {
    setBusy(true)
    setSelected(null)
    setPossessions([])
    try {
      const response = await nbaApi.listGames({
        season: season || undefined,
        teamId: teamId || undefined,
        gameDate: gameDate || undefined,
      })
      setGames(response.games)
      setGameId(response.games[0]?.id ?? '')
      setStatus(response.games.length === 0
        ? 'No cached or available games matched these filters.'
        : `${response.games.length} games · ${response.cacheStatus.replace('_', ' ')}`)
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Games are unavailable.')
    } finally {
      setBusy(false)
    }
  }

  async function loadPossessions(): Promise<void> {
    if (!gameId) return
    setBusy(true)
    setSelected(null)
    try {
      const response = await nbaApi.listPossessions(gameId)
      setPossessions(response.possessions)
      setSelected(response.possessions[0] ?? null)
      setStatus(`${response.possessions.length} possessions · ${response.cacheStatus.replace('_', ' ')}`)
    } catch (error) {
      setPossessions([])
      setStatus(error instanceof Error ? error.message : 'Possessions are unavailable.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="possession-browser" aria-label="Real NBA possessions">
      <div className="lineup-heading">
        <div>
          <strong>Real NBA possessions</strong>
          <span>Public event data; no tracking-coordinate reconstruction</span>
        </div>
        <span className="data-availability ready">{status}</span>
      </div>
      <div className="possession-search">
        <label>Team<select aria-label="Possession team" onChange={(event) => setTeamId(event.target.value)} value={teamId}>
          <option value="">All teams</option>
          {teams.map((team) => <option key={team.id} value={team.id}>{team.fullName}</option>)}
        </select></label>
        <label>Season<input aria-label="Possession season" onChange={(event) => setSeason(event.target.value)} value={season} /></label>
        <label>Date<input aria-label="Possession game date" onChange={(event) => setGameDate(event.target.value)} type="date" value={gameDate} /></label>
        <button className="primary-action" disabled={busy || disabled} onClick={() => void findGames()} type="button">Find games</button>
        <label>Game<select aria-label="NBA game" onChange={(event) => setGameId(event.target.value)} value={gameId}>
          <option value="">Choose game</option>
          {games.map((game) => <option key={game.id} value={game.id}>{game.gameDate} · {game.externalId} · {game.status ?? 'NBA game'}</option>)}
        </select></label>
        <button disabled={!gameId || busy || disabled} onClick={() => void loadPossessions()} type="button">Load possessions</button>
      </div>

      {possessions.length > 0 && <>
        <div className="possession-filters" aria-label="Possession filters">
          <label>Quarter<select onChange={(event) => setQuarter(event.target.value)} value={quarter}><option value="all">All</option>{[1, 2, 3, 4].map((value) => <option key={value}>{value}</option>)}</select></label>
          <label>Result<select onChange={(event) => setOutcome(event.target.value)} value={outcome}><option value="all">All</option><option value="made">Made</option><option value="missed">Missed</option><option value="turnover">Turnover</option></select></label>
          <label>Player<input onChange={(event) => setPlayer(event.target.value)} placeholder="Name or NBA ID" value={player} /></label>
          <label>Shot type<input onChange={(event) => setShotType(event.target.value)} placeholder="3PT, layup…" value={shotType} /></label>
        </div>
        <div className="possession-results">
          <div className="possession-list" aria-label="Possession list">
            {filtered.map((possession) => <button
              aria-pressed={selected?.id === possession.id}
              className={selected?.id === possession.id ? 'active' : ''}
              key={possession.id}
              onClick={() => setSelected(possession)}
              type="button"
            >
              <strong>Q{possession.period} · {possession.startClock ?? 'clock unavailable'}</strong>
              <span>{possession.result.resultType.replaceAll('_', ' ')} · {possession.result.points} pts</span>
            </button>)}
            {filtered.length === 0 && <p>No possessions match these filters.</p>}
          </div>
          {selected && <PossessionDetails disabled={disabled} onReconstruct={onReconstruct} possession={selected} />}
        </div>
      </>}
    </section>
  )
}

function PossessionDetails({
  disabled,
  onReconstruct,
  possession,
}: {
  disabled: boolean
  onReconstruct: (possession: RealPossession) => void
  possession: RealPossession
}) {
  const completeLineups = possession.offensiveLineup.length === 5
    && possession.defensiveLineup.length === 5
  return <article className="possession-details">
    <div className="provenance-row">
      <span>{possession.provenance.provider}</span>
      <span>source #{possession.provenance.sourcePossessionId}</span>
      <span>observed + derived fields labeled</span>
    </div>
    <div className="real-lineups">
      <Lineup label="Offense" players={possession.offensiveLineup} />
      <Lineup label="Defense" players={possession.defensiveLineup} />
    </div>
    <ol className="event-list">
      {possession.events.map((event) => <li key={event.sourceEventId}>
        <time>{event.clock ?? '—'}</time>
        <span>{event.description ?? event.eventType}</span>
        {(event.shotX !== null && event.shotX !== undefined) && <small>shot ({event.shotX}, {event.shotY ?? '—'}) · {event.shotType ?? 'type unavailable'}</small>}
      </li>)}
    </ol>
    <div className="movement-warning" role="note">
      <strong>Movement unavailable from public data.</strong>
      <span>Create/reconstruct routes manually. Court placements are editor placeholders, never historical tracking.</span>
    </div>
    {!completeLineups && <p className="data-unavailable">A complete 5-on-5 lineup was not available, so reconstruction is disabled instead of inventing players.</p>}
    <button className="primary-action" disabled={disabled || !completeLineups} onClick={() => onReconstruct(possession)} type="button">Reconstruct manually</button>
  </article>
}

function Lineup({ label, players }: { label: string; players: RealPossession['offensiveLineup'] }) {
  return <div><strong>{label}</strong><span>{players.length > 0 ? players.map((player) => player.displayName ?? `NBA ${player.externalId}`).join(' · ') : 'Unavailable'}</span></div>
}
