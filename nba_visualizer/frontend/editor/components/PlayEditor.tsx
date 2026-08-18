'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { BasketballCourt } from '@/frontend/court/components/BasketballCourt'
import { SimulationClock, type PlaybackRate } from '@/frontend/animation/playback'
import {
  sampleSimulation,
  simulatePlay,
  type SimulationResult,
} from '@/frontend/animation/simulation'
import { validatePlayDefinition } from '@/frontend/domain/actions'
import { PlaybackControls } from '@/frontend/controls/components/PlaybackControls'
import { DefensiveParameterControls } from '@/frontend/controls/components/DefensiveParameterControls'
import { AnalyticsControls } from '@/frontend/controls/components/AnalyticsControls'
import { nbaApi } from '@/frontend/data/apiClient'
import { ActionComposer } from './ActionComposer'
import { ActionTimeline } from './ActionTimeline'
import { LineupSelector } from '@/frontend/players/components/LineupSelector'
import { PossessionBrowser } from '@/frontend/possessions/components/PossessionBrowser'
import {
  createManualReconstruction,
  IncompleteHistoricalLineupError,
} from '@/frontend/possessions/reconstruction'
import {
  defensiveDebugAt,
  DEFAULT_DEFENSIVE_PARAMETERS,
  type DefenseMode,
  type DefensiveParameters,
} from '@/frontend/defense/engine'
import type { AnalyticsOverlay } from '@/frontend/analytics/types'
import { PickAndRollComparison } from '@/frontend/analytics/components/PickAndRollComparison'
import { evaluatePlayerAwareShot } from '@/frontend/analytics/shotOutcome'
import type {
  CourtPosition,
  Coverage,
  NBAPlayerData,
  NBATeamData,
  PlayAction,
  PlayDefinition,
  RealPossession,
  SimulationFrame,
  TeamSide,
} from '@/frontend/domain/models'
import {
  ballAvailableAt,
  distance,
  futureBallOwner,
  inferredMovementType,
  nearestTarget,
  playerAvailableAt,
  playerPositionAfterActions,
  routeLength,
  shouldReposition,
  simplifyGesture,
  WHITEBOARD_RIM,
} from '@/frontend/editor/whiteboard'

type ApiStatus = 'loading' | 'connected' | 'error' | 'saving'
type SaveMessage = { kind: 'success' | 'error'; text: string } | null

export function PlayEditor() {
  const [plays, setPlays] = useState<PlayDefinition[]>([])
  const [play, setPlay] = useState<PlayDefinition | null>(null)
  const [exampleFrame, setExampleFrame] = useState<SimulationFrame | null>(null)
  const [simulation, setSimulation] = useState<SimulationResult | null>(null)
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null)
  const [apiStatus, setApiStatus] = useState<ApiStatus>('loading')
  const [saveMessage, setSaveMessage] = useState<SaveMessage>(null)
  const [sourcePossession, setSourcePossession] = useState<RealPossession | null>(null)
  const [attemptErrors, setAttemptErrors] = useState<string[]>([])
  const [currentTimeSeconds, setCurrentTimeSeconds] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackRate, setPlaybackRate] = useState<PlaybackRate>(1)
  const [defenseMode, setDefenseMode] = useState<DefenseMode>('coverage')
  const [debugDefense, setDebugDefense] = useState(false)
  const [defensiveParameters, setDefensiveParameters] = useState<DefensiveParameters>({
    ...DEFAULT_DEFENSIVE_PARAMETERS,
  })
  const [analyticsOverlays, setAnalyticsOverlays] = useState<AnalyticsOverlay[]>([])
  const [analysisOpen, setAnalysisOpen] = useState(false)
  const [screenCandidateId, setScreenCandidateId] = useState<string | null>(null)
  const [formationMode, setFormationMode] = useState(true)
  const [screenTargeting, setScreenTargeting] = useState(false)
  const clockRef = useRef<SimulationClock | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    Promise.all([
      nbaApi.getExampleFrame(controller.signal),
      nbaApi.listPlays(controller.signal),
    ])
      .then(([frame, savedPlays]) => {
        setExampleFrame(frame)
        setPlays(savedPlays)
        const initialPlay = createBlankPlay(frame)
        setPlay(initialPlay)
        setSelectedPlayerId(initialPlay.initialFrame.players[0]?.player.id ?? null)
        setApiStatus('connected')
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        setApiStatus('error')
      })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!isPlaying) return
    let animationFrameId = 0
    const advance = (realTimeMs: number) => {
      const clock = clockRef.current
      if (clock === null) return
      setCurrentTimeSeconds(clock.advance(realTimeMs))
      if (clock.isPlaying) {
        animationFrameId = requestAnimationFrame(advance)
      } else {
        setIsPlaying(false)
      }
    }
    animationFrameId = requestAnimationFrame(advance)
    return () => cancelAnimationFrame(animationFrameId)
  }, [isPlaying])

  const validationErrors = useMemo(
    () => play === null ? [] : validatePlayDefinition(play),
    [play],
  )
  const previewSimulation = useMemo(
    () => play === null || validationErrors.length > 0
      ? null
      : simulatePlay(play, { defenseMode, defensiveParameters }),
    [play, validationErrors.length, defenseMode, defensiveParameters],
  )

  function resetSimulationOutput(): void {
    clockRef.current = null
    setSimulation(null)
    setCurrentTimeSeconds(0)
    setIsPlaying(false)
  }

  function loadPlay(nextPlay: PlayDefinition): void {
    resetSimulationOutput()
    setAttemptErrors([])
    setSaveMessage(null)
    setPlay(structuredClone(nextPlay))
    setSourcePossession(null)
    setSelectedPlayerId(nextPlay.initialFrame.players[0]?.player.id ?? null)
    setFormationMode(nextPlay.actions.length === 0)
  }

  function updatePlay(updater: (current: PlayDefinition) => PlayDefinition): void {
    if (isPlaying) return
    setPlay((current) => current === null ? null : updater(current))
    setSaveMessage(null)
    setAttemptErrors([])
    resetSimulationOutput()
  }

  function movePlayer(playerId: string, position: CourtPosition): void {
    updatePlay((current) => {
      const initialPlayer = current.initialFrame.players.find((state) => state.player.id === playerId)
      if (initialPlayer === undefined) return current
      const ball = current.initialFrame.ball.state === 'possessed'
        && current.initialFrame.ball.playerId === playerId
        ? {
            ...current.initialFrame.ball,
            position: { x: position.x + 0.5, y: position.y + 2.5 },
          }
        : current.initialFrame.ball
      return {
        ...current,
        initialFrame: {
          ...current.initialFrame,
          ball,
          players: current.initialFrame.players.map((state) => (
            state.player.id === playerId ? { ...state, position } : state
          )),
          metadata: sourcePossession === null ? current.initialFrame.metadata : {
            ...current.initialFrame.metadata,
            manualPlacementEdited: true,
          },
        },
        routes: current.routes.filter((route) => route.playerId !== playerId),
        actions: current.actions.filter(
          (action) => !(action.playerId === playerId && action.metadata.phase1Route === true),
        ),
      }
    })
  }

  function handlePlayerTap(playerId: string): void {
    if (play === null) return
    const tapped = play.initialFrame.players.find((state) => state.player.id === playerId)
    const ownerId = futureBallOwner(play)
    const owner = play.initialFrame.players.find((state) => state.player.id === ownerId)
    if (screenTargeting) {
      if (tapped?.teamSide === 'offense' && playerId !== ownerId) {
        setSelectedPlayerId(playerId)
        setScreenTargeting(false)
        setScreenForPlayer(playerId)
      } else {
        setAttemptErrors(['Choose an off-ball offensive player to set the screen.'])
      }
      return
    }
    if (!formationMode && (
      selectedPlayerId === ownerId
      && ownerId !== null
      && playerId !== ownerId
      && tapped?.teamSide === 'offense'
      && owner?.teamSide === 'offense'
    )) {
      createPass(ownerId, playerId)
      return
    }
    setSelectedPlayerId(playerId)
    setScreenCandidateId(null)
  }

  function completeWhiteboardGesture(playerId: string, rawPositions: CourtPosition[]): void {
    if (play === null) return
    const player = play.initialFrame.players.find((state) => state.player.id === playerId)
    if (player === undefined) return
    const positions = simplifyGesture(rawPositions)
    const target = positions.at(-1)
    if (target === undefined) return
    setSelectedPlayerId(playerId)
    setScreenCandidateId(null)

    if (formationMode || shouldReposition(positions)) {
      movePlayer(playerId, target)
      return
    }
    if (player.teamSide === 'defense') {
      setAttemptErrors(['Defender movement is automatic. Use a short drag to adjust the starting matchup.'])
      return
    }

    const ownerId = futureBallOwner(play)
    const receiver = ownerId === playerId
      ? nearestTarget(play.initialFrame.players, target, playerId, 'offense')
      : null
    if (receiver !== null) {
      createPass(playerId, receiver.player.id)
      return
    }

    const actionType = inferredMovementType(play, playerId)
    if (actionType === null) return
    updatePlay((current) => {
      const source = playerPositionAfterActions(
        current.actions,
        playerId,
        player.position,
      )
      const normalizedPositions = [source, ...positions.slice(1)]
      const startTime = actionType === 'dribble'
        ? Math.max(
            playerAvailableAt(current.actions, playerId),
            ballAvailableAt(current.actions),
          )
        : playerAvailableAt(current.actions, playerId)
      const speed = actionType === 'dribble' ? 7 : 10
      const action: PlayAction = {
        id: crypto.randomUUID(),
        actionType,
        playerId,
        startTime,
        duration: Math.max(routeLength(normalizedPositions) / speed, 0.35),
        source,
        target: { ...target },
        waypoints: normalizedPositions.slice(1, -1).map((point) => ({ ...point })),
        metadata: gestureMetadata(actionType),
      }
      return { ...current, actions: [...current.actions, action] }
    })

    const nearbyDefender = nearestTarget(
      play.initialFrame.players,
      target,
      playerId,
      'defense',
      5.5,
    )
    if (actionType === 'cut' && nearbyDefender !== null) setScreenCandidateId(playerId)
  }

  function createPass(passerId: string, receiverId: string): void {
    if (play === null || futureBallOwner(play) !== passerId) {
      setAttemptErrors(['Only the player with the ball can pass.'])
      return
    }
    updatePlay((current) => {
      const passer = current.initialFrame.players.find((state) => state.player.id === passerId)
      const receiver = current.initialFrame.players.find((state) => state.player.id === receiverId)
      if (passer?.teamSide !== 'offense' || receiver?.teamSide !== 'offense') return current
      const startTime = Math.max(
        playerAvailableAt(current.actions, passerId),
        ballAvailableAt(current.actions),
      )
      const source = playerPositionAfterActions(current.actions, passerId, passer.position)
      const target = playerPositionAfterActions(current.actions, receiverId, receiver.position)
      const action: PlayAction = {
        id: crypto.randomUUID(),
        actionType: 'pass',
        playerId: passerId,
        targetPlayerId: receiverId,
        startTime,
        duration: Math.max(distance(source, target) / 32, 0.28),
        source,
        target,
        metadata: gestureMetadata('pass'),
      }
      return { ...current, actions: [...current.actions, action] }
    })
    setSelectedPlayerId(receiverId)
    setScreenCandidateId(null)
  }

  function shootAtRim(): void {
    if (formationMode) {
      setAttemptErrors(['Finish positioning the players before adding a shot.'])
      return
    }
    if (play === null || selectedPlayerId === null || futureBallOwner(play) !== selectedPlayerId) {
      setAttemptErrors(['Select the player with the ball, then click the rim to shoot.'])
      return
    }
    updatePlay((current) => {
      const shooter = current.initialFrame.players.find(
        (state) => state.player.id === selectedPlayerId && state.teamSide === 'offense',
      )
      if (shooter === undefined) return current
      const startTime = Math.max(
        playerAvailableAt(current.actions, selectedPlayerId),
        ballAvailableAt(current.actions),
      )
      const source = playerPositionAfterActions(current.actions, selectedPlayerId, shooter.position)
      const actionId = crypto.randomUUID()
      let defensivePlayers = current.initialFrame.players.filter((state) => state.teamSide === 'defense')
      try {
        const beforeShot = simulatePlay(current, { defenseMode, defensiveParameters })
        defensivePlayers = sampleSimulation(beforeShot, startTime).players.filter(
          (state) => state.teamSide === 'defense',
        )
      } catch {
        // Initial defensive positions remain a valid fallback for an incomplete draft.
      }
      const outcome = evaluatePlayerAwareShot(
        actionId,
        shooter.player,
        source,
        defensivePlayers,
      )
      const action: PlayAction = {
        id: actionId,
        actionType: 'shoot',
        playerId: selectedPlayerId,
        startTime,
        duration: 0.8,
        source,
        target: WHITEBOARD_RIM,
        deterministicResult: outcome.result,
        metadata: {
          ...gestureMetadata('shoot'),
          outcomeMethod: 'distance_and_defender_proximity_heuristic',
          pointValue: outcome.pointValue,
          shotDistanceFeet: Number(outcome.shotDistanceFeet.toFixed(2)),
          nearestDefenderDistanceFeet: outcome.nearestDefenderDistanceFeet === null
            ? null
            : Number(outcome.nearestDefenderDistanceFeet.toFixed(2)),
          makeProbabilityHeuristic: Number(outcome.makeProbabilityHeuristic.toFixed(3)),
          deterministicRoll: Number(outcome.deterministicRoll.toFixed(3)),
          shotProfileSource: outcome.profileSource,
          shotProfileSeason: outcome.profileSeason ?? 'unavailable',
          shotProfileAttempts: outcome.profileAttempts,
          playerBaselineProbability: Number(outcome.playerBaselineProbability.toFixed(3)),
          distanceAdjustment: Number(outcome.distanceAdjustment.toFixed(3)),
          pressureAdjustment: Number(outcome.pressureAdjustment.toFixed(3)),
        },
      }
      return { ...current, actions: [...current.actions, action] }
    })
  }

  function setScreenForPlayer(screenerId: string): void {
    if (play === null || futureBallOwner(play) === screenerId) return
    updatePlay((current) => {
      const screener = current.initialFrame.players.find(
        (state) => state.player.id === screenerId && state.teamSide === 'offense',
      )
      if (screener === undefined) return current
      const location = playerPositionAfterActions(current.actions, screenerId, screener.position)
      const handlerId = futureBallOwner(current)
      const offense = current.initialFrame.players.filter((state) => state.teamSide === 'offense')
      const defense = current.initialFrame.players.filter((state) => state.teamSide === 'defense')
      const handlerIndex = offense.findIndex((state) => state.player.id === handlerId)
      const assignedOnBallDefender = handlerIndex >= 0 ? defense[handlerIndex] : undefined
      const nearestDefender = nearestTarget(
        current.initialFrame.players,
        location,
        screenerId,
        'defense',
        7,
      )
      const targetDefender = assignedOnBallDefender ?? nearestDefender ?? undefined
      const handler = offense.find((state) => state.player.id === handlerId)
      const orientationDegrees = handler === undefined
        ? 90
        : Math.atan2(location.y - handler.position.y, location.x - handler.position.x)
          * 180 / Math.PI + 90
      const startTime = playerAvailableAt(current.actions, screenerId)
      const action: PlayAction = {
        id: crypto.randomUUID(),
        actionType: 'screen',
        playerId: screenerId,
        startTime,
        duration: 1.2,
        source: location,
        screenLocation: location,
        orientationDegrees,
        targetPlayerId: targetDefender?.player.id ?? null,
        metadata: {
          ...gestureMetadata('screen'),
          targetBallHandlerId: handlerId,
          matchupTargeted: assignedOnBallDefender !== undefined,
        },
      }
      return { ...current, actions: [...current.actions, action] }
    })
    setScreenCandidateId(null)
    setSelectedPlayerId(futureBallOwner(play))
  }

  function gestureMetadata(actionType: string): Record<string, string | boolean> {
    return {
      whiteboardGesture: true,
      inferredAction: actionType,
      ...(sourcePossession === null ? {} : {
        provenance: 'manual_reconstruction',
        realPossessionId: sourcePossession.id,
      }),
    }
  }

  function addAction(action: PlayAction): void {
    if (play === null) return
    const next = { ...play, actions: [...play.actions, action] }
    const errors = validatePlayDefinition(next)
    if (errors.length > 0) {
      setAttemptErrors(errors.map((error) => error.message))
      return
    }
    updatePlay(() => next)
  }

  function deleteAction(actionId: string): void {
    updatePlay((current) => {
      const deleted = current.actions.find((action) => action.id === actionId)
      return {
        ...current,
        actions: current.actions.filter((action) => action.id !== actionId),
        routes: deleted?.metadata.phase1Route === true
          ? current.routes.filter((route) => route.playerId !== deleted.playerId)
          : current.routes,
      }
    })
  }

  function undoLastAction(): void {
    updatePlay((current) => {
      const last = current.actions.at(-1)
      if (last === undefined) return current
      return {
        ...current,
        actions: current.actions.slice(0, -1),
        routes: last.metadata.phase1Route === true
          ? current.routes.filter((route) => route.playerId !== last.playerId)
          : current.routes,
      }
    })
  }

  function clearPlayActions(): void {
    updatePlay((current) => ({
      ...current,
      routes: [],
      actions: [],
    }))
  }

  function changeCoverage(coverage: Coverage): void {
    updatePlay((current) => ({
      ...current,
      initialFrame: {
        ...current.initialFrame,
        possession: { ...current.initialFrame.possession, coverage },
      },
    }))
  }

  function changeDefenseMode(mode: DefenseMode): void {
    setDefenseMode(mode)
    resetSimulationOutput()
  }

  function changeDefensiveParameters(parameters: DefensiveParameters): void {
    setDefensiveParameters(parameters)
    resetSimulationOutput()
  }

  function replaceLineup(
    side: TeamSide,
    team: NBATeamData,
    lineup: NBAPlayerData[],
  ): void {
    if (lineup.length !== 5) return
    const firstPlayerId = lineup[0]?.id ?? null
    updatePlay((current) => {
      let lineupIndex = 0
      let nextBall = current.initialFrame.ball
      const players = current.initialFrame.players.map((state) => {
        if (state.teamSide !== side) return state
        const replacement = lineup[lineupIndex++]
        if (replacement === undefined) return state
        if (nextBall.state === 'possessed' && nextBall.playerId === state.player.id) {
          nextBall = { ...nextBall, playerId: replacement.id }
        }
        return {
          ...state,
          player: {
            id: replacement.id,
            teamId: team.id,
            name: replacement.displayName,
            jerseyNumber: replacement.jerseyNumber,
            position: replacement.position,
            height: replacement.height,
            externalId: replacement.externalId,
            source: replacement.source,
            shootingProfile: replacement.shootingProfile,
          },
        }
      })
      return {
        ...current,
        routes: [],
        actions: [],
        initialFrame: {
          ...current.initialFrame,
          players,
          ball: nextBall,
          possession: {
            ...current.initialFrame.possession,
            offenseTeamId: side === 'offense' ? team.id : current.initialFrame.possession.offenseTeamId,
            defenseTeamId: side === 'defense' ? team.id : current.initialFrame.possession.defenseTeamId,
          },
          metadata: {
            ...current.initialFrame.metadata,
            [`${side}TeamExternalId`]: team.externalId,
            lineupSource: 'nba.com cache',
          },
        },
      }
    })
    setSelectedPlayerId(firstPlayerId)
    setFormationMode(true)
  }

  function ensureSimulation(): SimulationResult | null {
    if (simulation !== null) return simulation
    if (play === null || (play.routes.length === 0 && play.actions.length === 0)) return null
    const errors = validatePlayDefinition(play)
    if (errors.length > 0) {
      setAttemptErrors(errors.map((error) => error.message))
      return null
    }
    const nextSimulation = previewSimulation
      ?? simulatePlay(play, { defenseMode, defensiveParameters })
    const clock = new SimulationClock(nextSimulation.durationSeconds)
    clock.setPlaybackRate(playbackRate, performance.now())
    clockRef.current = clock
    setSimulation(nextSimulation)
    return nextSimulation
  }

  function togglePlayback(): void {
    const result = ensureSimulation()
    if (result === null) return
    const clock = clockRef.current
    if (clock === null) return
    const now = performance.now()
    if (clock.isPlaying) {
      clock.pause(now)
      setCurrentTimeSeconds(clock.timeSeconds)
      setIsPlaying(false)
    } else {
      if (clock.timeSeconds >= result.durationSeconds - 0.001) {
        clock.restart()
        setCurrentTimeSeconds(0)
      }
      clock.play(now)
      setCurrentTimeSeconds(clock.timeSeconds)
      setIsPlaying(clock.isPlaying)
    }
  }

  function giveBallToSelectedPlayer(): void {
    if (selectedPlayerId === null) return
    updatePlay((current) => {
      const selected = current.initialFrame.players.find(
        (state) => state.player.id === selectedPlayerId && state.teamSide === 'offense',
      )
      if (selected === undefined) return current
      return {
        ...current,
        initialFrame: {
          ...current.initialFrame,
          ball: {
            state: 'possessed',
            playerId: selectedPlayerId,
            position: { x: selected.position.x + 0.5, y: selected.position.y + 2.5 },
            heightFeet: 3.5,
          },
          metadata: {
            ...current.initialFrame.metadata,
            ballOwnershipOrigin: sourcePossession === null
              ? 'user_selected'
              : 'manual_reconstruction',
          },
        },
      }
    })
  }

  function restart(): void {
    const result = ensureSimulation()
    if (result === null || clockRef.current === null) return
    clockRef.current.restart()
    setCurrentTimeSeconds(0)
    setIsPlaying(false)
  }

  function seek(timeSeconds: number): void {
    const result = ensureSimulation()
    if (result === null || clockRef.current === null) return
    clockRef.current.seek(timeSeconds, performance.now())
    setCurrentTimeSeconds(clockRef.current.timeSeconds)
  }

  function changePlaybackRate(rate: PlaybackRate): void {
    setPlaybackRate(rate)
    clockRef.current?.setPlaybackRate(rate, performance.now())
  }

  function toggleAnalyticsOverlay(overlay: AnalyticsOverlay): void {
    setAnalyticsOverlays((current) => current.includes(overlay)
      ? current.filter((item) => item !== overlay)
      : [...current, overlay])
  }

  function newPlay(): void {
    const frame = exampleFrame ?? play?.initialFrame
    if (frame === undefined || frame === null) return
    const next = createBlankPlay(frame)
    setPlays((current) => [next, ...current])
    loadPlay(next)
  }

  function beginReconstruction(possession: RealPossession): void {
    const frame = exampleFrame ?? play?.initialFrame
    if (frame === undefined || frame === null) return
    try {
      const next = createManualReconstruction(possession, frame)
      resetSimulationOutput()
      setAttemptErrors([])
      setSaveMessage(null)
      setSourcePossession(possession)
      setPlay(next)
      setSelectedPlayerId(next.initialFrame.players[0]?.player.id ?? null)
      document.querySelector('.court-panel')?.scrollIntoView({ behavior: 'smooth' })
    } catch (error) {
      setSaveMessage({
        kind: 'error',
        text: error instanceof IncompleteHistoricalLineupError
          ? error.message
          : 'Could not prepare this possession for reconstruction.',
      })
    }
  }

  async function saveReconstruction(): Promise<void> {
    if (play === null || sourcePossession === null) return
    if (validationErrors.length > 0) {
      setAttemptErrors(validationErrors.map((error) => error.message))
      return
    }
    setApiStatus('saving')
    try {
      await nbaApi.saveReconstruction(sourcePossession.id, {
        ...play,
        updatedAt: new Date().toISOString(),
      })
      setSaveMessage({ kind: 'success', text: 'Manual reconstruction saved separately from source history.' })
      setApiStatus('connected')
    } catch (error) {
      setSaveMessage({ kind: 'error', text: error instanceof Error ? error.message : 'Could not save reconstruction.' })
      setApiStatus('error')
    }
  }

  async function savePlay(): Promise<void> {
    if (play === null) return
    if (validationErrors.length > 0) {
      setAttemptErrors(validationErrors.map((error) => error.message))
      return
    }
    setApiStatus('saving')
    setSaveMessage(null)
    const next = { ...play, updatedAt: new Date().toISOString() }
    try {
      const saved = await nbaApi.savePlay(next)
      setPlay(saved)
      setPlays((current) => [saved, ...current.filter((item) => item.id !== saved.id)])
      setSaveMessage({ kind: 'success', text: 'Structured play saved.' })
      setApiStatus('connected')
    } catch (error) {
      setSaveMessage({ kind: 'error', text: error instanceof Error ? error.message : 'Could not save play.' })
      setApiStatus('error')
    }
  }

  async function duplicatePlay(): Promise<void> {
    if (play === null) return
    try {
      if (!plays.some((item) => item.id === play.id)) await savePlay()
      const duplicate = await nbaApi.duplicatePlay(play.id)
      setPlays((current) => [duplicate, ...current])
      loadPlay(duplicate)
    } catch (error) {
      setSaveMessage({ kind: 'error', text: error instanceof Error ? error.message : 'Could not duplicate play.' })
    }
  }

  async function deletePlay(): Promise<void> {
    if (play === null || !window.confirm(`Delete “${play.name}”?`)) return
    try {
      if (plays.some((item) => item.id === play.id)) await nbaApi.deletePlay(play.id)
      const remaining = plays.filter((item) => item.id !== play.id)
      setPlays(remaining)
      if (remaining[0] !== undefined) loadPlay(remaining[0])
      else if (exampleFrame !== null) {
        const blank = createBlankPlay(exampleFrame)
        setPlays([blank])
        loadPlay(blank)
      }
    } catch (error) {
      setSaveMessage({ kind: 'error', text: error instanceof Error ? error.message : 'Could not delete play.' })
    }
  }

  const selectedPlayer = play?.initialFrame.players.find(
    (state) => state.player.id === selectedPlayerId,
  )
  const durationSeconds = simulation?.durationSeconds ?? play?.actions.reduce(
    (duration, action) => Math.max(duration, action.startTime + action.duration),
    play.routes.reduce(
      (duration, route) => Math.max(duration, route.points.at(-1)?.timeSeconds ?? 0),
      0,
    ),
  ) ?? 0
  const activeSimulation = simulation ?? previewSimulation
  const displayedFrame = play === null
    ? null
    : activeSimulation === null
      ? play.initialFrame
      : sampleSimulation(activeSimulation, currentTimeSeconds)
  const defensiveResponse = activeSimulation?.defensiveResponse ?? null
  const defensiveDebug = defensiveDebugAt(
    simulation?.defensiveResponse ?? defensiveResponse,
    currentTimeSeconds,
  )
  const visibleErrors = [...new Set([...attemptErrors, ...validationErrors.map((error) => error.message)])]
  const futureOwnerId = play === null ? null : futureBallOwner(play)
  const futureOwner = play?.initialFrame.players.find((state) => state.player.id === futureOwnerId)
  const latestShot = play?.actions
    .filter((action): action is Extract<PlayAction, { actionType: 'shoot' }> => action.actionType === 'shoot')
    .at(-1)
  const latestShooter = play?.initialFrame.players.find(
    (state) => state.player.id === latestShot?.playerId,
  )

  return (
    <section className="visualizer" aria-label="Interactive play visualizer">
      {play && (
        <LineupSelector
          disabled={isPlaying}
          onApply={replaceLineup}
          players={play.initialFrame.players}
        />
      )}
      <header className="whiteboard-toolbar">
        <div className="whiteboard-name">
          <label htmlFor="play-name">Play name</label>
          <input
            id="play-name"
            aria-label="Play name"
            maxLength={120}
            value={play?.name ?? ''}
            onChange={(event) => updatePlay((current) => ({ ...current, name: event.target.value }))}
          />
        </div>
        <div className="whiteboard-actions">
          <button disabled={isPlaying || (play?.actions.length ?? 0) === 0} onClick={undoLastAction} type="button">Undo</button>
          <button disabled={isPlaying || (play?.actions.length ?? 0) === 0} onClick={clearPlayActions} type="button">Clear</button>
          <button aria-expanded={analysisOpen} onClick={() => setAnalysisOpen((open) => !open)} type="button">Analyze</button>
        </div>
        <label className="compact-coverage">
          <span>Defense</span>
          <select
            aria-label="Defensive coverage"
            disabled={isPlaying}
            value={defenseMode === 'offense_only' ? 'none' : play?.initialFrame.possession.coverage ?? 'drop'}
            onChange={(event) => {
              if (event.target.value === 'none') changeDefenseMode('offense_only')
              else {
                changeDefenseMode('coverage')
                changeCoverage(event.target.value as Coverage)
              }
            }}
          >
            <option value="none">None</option>
            <option value="drop">Drop</option>
            <option value="switch">Switch</option>
            <option value="hedge">Hedge</option>
            <option value="blitz">Blitz</option>
            <option value="ice">ICE</option>
          </select>
        </label>
      </header>

      {sourcePossession && <div className="reconstruction-banner" role="status">
        <div>
          <strong>Manual reconstruction · source possession #{sourcePossession.provenance.sourcePossessionId}</strong>
          <span>Players and events come from {sourcePossession.provenance.provider}. Current court positions are labeled editor placeholders; no movement was inferred.</span>
        </div>
        <button className="primary-action" disabled={apiStatus === 'saving'} onClick={() => void saveReconstruction()} type="button">Save reconstruction separately</button>
      </div>}

      {visibleErrors.length > 0 && (
        <div className="validation-banner" role="alert">
          <strong>Play needs attention</strong>
          <ul>{visibleErrors.map((error) => <li key={error}>{error}</li>)}</ul>
        </div>
      )}

      <div className="whiteboard-layout">
        <div className="court-panel">
          <div className="panel-header">
            <div>
              <strong>Draw the possession</strong>
              <span>Drag a player to draw. Click a teammate to pass. Click the rim to shoot; distance and defender pressure determine the demo result.</span>
            </div>
            <span className="ball-owner-pill">Ball · {futureOwner?.player.name ?? 'loose'}</span>
          </div>
          <div className="court-wrap">
            <div className="court-mode-guide">
              <strong>{formationMode ? 'Set starting spots' : 'Draw the play'}</strong>
              <span>{formationMode
                ? 'Drag any player anywhere on the court. Movement will not create an action yet.'
                : 'Ball handler draw = dribble · off-ball draw = cut · teammate click = pass · select off-ball player = screen'}</span>
              {!formationMode && <button
                aria-pressed={screenTargeting}
                className={`screen-tool${screenTargeting ? ' active' : ''}`}
                disabled={isPlaying}
                onClick={() => setScreenTargeting((active) => !active)}
                type="button"
              >{screenTargeting ? 'Click screener…' : 'Add screen'}</button>}
              <button
                className={formationMode ? 'finish-formation' : ''}
                disabled={isPlaying}
                onClick={() => {
                  setScreenTargeting(false)
                  setFormationMode((active) => !active)
                }}
                type="button"
              >{formationMode ? 'Done positioning →' : 'Edit starting spots'}</button>
            </div>
            {screenTargeting && play && (
              <div aria-label="Choose screener" className="screen-picker" role="group">
                <strong>Who sets it?</strong>
                <span>Choose an off-ball player. Their current spot becomes the screen location.</span>
                {play.initialFrame.players
                  .filter((state) => state.teamSide === 'offense' && state.player.id !== futureOwnerId)
                  .map((state) => (
                    <button
                      aria-label={`Use ${state.player.name} as screener`}
                      disabled={isPlaying}
                      key={state.player.id}
                      onClick={() => {
                        setSelectedPlayerId(state.player.id)
                        setScreenTargeting(false)
                        setScreenForPlayer(state.player.id)
                      }}
                      type="button"
                    >#{state.player.jerseyNumber} {shortPlayerName(state.player.name)}</button>
                  ))}
              </div>
            )}
            {displayedFrame && play && (
              <BasketballCourt
                actions={play.actions}
                analyticsOverlays={analyticsOverlays}
                defensiveDebug={defensiveDebug}
                defensiveDebugEnabled={debugDefense}
                defensivePreview={defensiveResponse}
                editingDisabled={isPlaying}
                frame={displayedFrame}
                futureBallOwnerId={futureOwnerId}
                onGestureComplete={completeWhiteboardGesture}
                onPlayerTap={handlePlayerTap}
                onShoot={shootAtRim}
                routes={play.routes}
                selectedPlayerId={selectedPlayerId}
              />
            )}
            {apiStatus === 'loading' && <p className="loading-message">Loading structured plays…</p>}
            {apiStatus === 'error' && play === null && <p className="error-message">Start the FastAPI backend on port 8000 to load and save plays.</p>}
          </div>
          {latestShot && typeof latestShot.metadata.makeProbabilityHeuristic === 'number' && (
            <div className="shot-model-summary" role="status">
              <strong>{latestShooter?.player.name ?? 'Shooter'} · {(latestShot.metadata.makeProbabilityHeuristic * 100).toFixed(0)}% make estimate</strong>
              <span>
                {latestShot.metadata.shotProfileSource === 'nba_season_totals'
                  ? `${latestShot.metadata.shotProfileSeason} NBA totals · ${latestShot.metadata.shotProfileAttempts} relevant attempts`
                  : 'League fallback · player shooting sample unavailable'}
                {' · '}{Number(latestShot.metadata.shotDistanceFeet).toFixed(1)} ft
                {' · '}{latestShot.metadata.nearestDefenderDistanceFeet === null
                  ? 'no nearby defender'
                  : `${Number(latestShot.metadata.nearestDefenderDistanceFeet).toFixed(1)} ft defender distance`}
              </span>
              <small>Deterministic teaching estimate—not a tracking-derived or calibrated expected field-goal model.</small>
            </div>
          )}
          <PlaybackControls
            canPlay={(play?.actions.length ?? 0) > 0 || (play?.routes.length ?? 0) > 0}
            currentTimeSeconds={currentTimeSeconds}
            durationSeconds={durationSeconds}
            isPlaying={isPlaying}
            onPlaybackRateChange={changePlaybackRate}
            onPlayPause={togglePlayback}
            onRestart={restart}
            onSeek={seek}
            playbackRate={playbackRate}
          />
        </div>

        {selectedPlayer && <aside className="contextual-inspector">
          <section className="selected-player-card">
            <span>Selected</span>
            <h2>{selectedPlayer?.player.name ?? 'None'}</h2>
            {selectedPlayer && (
              <p data-testid="selected-player-position">
                #{selectedPlayer.player.jerseyNumber} · {selectedPlayer.player.position} · {selectedPlayer.teamSide}
              </p>
            )}
            {futureOwnerId === selectedPlayer.player.id && <span className="ball-state-label">Has the ball next</span>}
            {selectedPlayer?.teamSide === 'offense'
              && futureOwnerId !== selectedPlayer.player.id
              && <button className="give-ball-button" disabled={isPlaying} onClick={giveBallToSelectedPlayer} type="button">Give ball to this player</button>}
            {selectedPlayer.teamSide === 'offense' && futureOwnerId !== selectedPlayer.player.id && (
              <button
                className={screenCandidateId === selectedPlayer.player.id ? 'screen-suggestion active' : 'screen-suggestion'}
                disabled={isPlaying}
                onClick={() => setScreenForPlayer(selectedPlayer.player.id)}
                type="button"
              >Set screen for ball handler</button>
            )}
            {selectedPlayer.teamSide === 'offense' && futureOwnerId !== selectedPlayer.player.id && (
              <small className="screen-help">Move this player near the on-ball defender, then set the screen. Coverage reacts automatically.</small>
            )}
            <button disabled={isPlaying || !play?.actions.some((action) => action.playerId === selectedPlayer.player.id)} onClick={() => {
              const last = play?.actions.filter((action) => action.playerId === selectedPlayer.player.id).at(-1)
              if (last) deleteAction(last.id)
            }} type="button">Delete last action</button>
          </section>
        </aside>}
      </div>

      {analysisOpen && <aside className="analysis-drawer">
        <AnalyticsControls
          analytics={displayedFrame?.analytics}
          enabled={analyticsOverlays}
          onToggle={toggleAnalyticsOverlay}
        />
        {play?.name === 'High pick-and-roll' && (
          <PickAndRollComparison play={play} parameters={defensiveParameters} />
        )}
      </aside>}

      {play && <details className="timing-drawer">
        <summary>Edit timing & actions</summary>
        <ActionTimeline actions={play.actions} onDelete={deleteAction} players={play.initialFrame.players} />
      </details>}

      <details className="advanced-drawer">
        <summary>Advanced</summary>
        <div className="advanced-drawer-content">
          <section className="play-library">
            <div className="play-identity">
              <label htmlFor="play-select">Saved play</label>
              <select id="play-select" value={sourcePossession === null ? play?.id ?? '' : ''} onChange={(event) => {
                const selected = plays.find((item) => item.id === event.target.value)
                if (selected !== undefined) loadPlay(selected)
              }}>
                {sourcePossession && <option value="">Manual historical reconstruction</option>}
                {plays.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
            </div>
            <div className="library-actions">
              <button onClick={newPlay} type="button">New</button>
              <button className="primary-action" disabled={play === null || apiStatus === 'saving'} onClick={() => void (sourcePossession ? saveReconstruction() : savePlay())} type="button">{apiStatus === 'saving' ? 'Saving…' : 'Save'}</button>
              <button disabled={play === null || sourcePossession !== null} onClick={() => void duplicatePlay()} type="button">Duplicate</button>
              <button className="danger-action" disabled={play === null || sourcePossession !== null} onClick={() => void deletePlay()} type="button">Delete</button>
            </div>
            {saveMessage && <span className={`save-message ${saveMessage.kind}`}>{saveMessage.text}</span>}
          </section>
          <PossessionBrowser disabled={isPlaying} onReconstruct={beginReconstruction} />
          {play && <details className="advanced-action-editor">
            <summary>Manual action form</summary>
            <ActionComposer onAdd={addAction} play={play} selectedPlayerId={selectedPlayerId} />
          </details>}
          <label className="debug-toggle">
            <input checked={debugDefense} onChange={(event) => setDebugDefense(event.target.checked)} type="checkbox" />
            Defensive debug overlays
          </label>
          <DefensiveParameterControls onChange={changeDefensiveParameters} value={defensiveParameters} />
          {defensiveResponse && <p className={`coverage-status${defensiveResponse.supported ? '' : ' unsupported'}`}>
            {defensiveResponse.supported
              ? `${defensiveResponse.coverage.toUpperCase()} · ${defensiveResponse.events.length} detected screen event${defensiveResponse.events.length === 1 ? '' : 's'}`
              : defensiveResponse.unsupportedReason}
          </p>}
        </div>
      </details>
    </section>
  )
}

function createBlankPlay(frame: SimulationFrame): PlayDefinition {
  const now = new Date().toISOString()
  const snapshot = structuredClone(frame)
  const pointGuard = snapshot.players.find((state) => state.teamSide === 'offense')
  if (pointGuard !== undefined) {
    snapshot.ball = {
      state: 'possessed',
      playerId: pointGuard.player.id,
      position: { x: pointGuard.position.x + 0.5, y: pointGuard.position.y + 1.5 },
      heightFeet: 3.5,
    }
  }
  snapshot.timestampSeconds = 0
  snapshot.currentActions = []
  return {
    id: crypto.randomUUID(),
    name: 'Untitled play',
    initialFrame: snapshot,
    routes: [],
    actions: [],
    createdAt: now,
    updatedAt: now,
  }
}

function shortPlayerName(name: string): string {
  return name.trim().split(/\s+/).at(-1) ?? name
}
