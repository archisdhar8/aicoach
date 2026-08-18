import type {
  GameListResponse,
  PlayDefinition,
  PossessionListResponse,
  PossessionReconstruction,
  PreferredLineupResponse,
  RosterResponse,
  SimulationFrame,
  TeamListResponse,
} from '@/frontend/domain/models'

const API_BASE_URL = process.env.NEXT_PUBLIC_NBA_API_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
    this.name = 'ApiError'
  }
}

export interface NbaApi {
  getExampleFrame(signal?: AbortSignal): Promise<SimulationFrame>
  listPlays(signal?: AbortSignal): Promise<PlayDefinition[]>
  savePlay(play: PlayDefinition): Promise<PlayDefinition>
  duplicatePlay(playId: string): Promise<PlayDefinition>
  deletePlay(playId: string): Promise<void>
  listTeams(signal?: AbortSignal): Promise<TeamListResponse>
  getRoster(teamId: string, season: string, refresh?: boolean, signal?: AbortSignal): Promise<RosterResponse>
  getPreferredLineup(teamId: string, season: string, refresh?: boolean, signal?: AbortSignal): Promise<PreferredLineupResponse>
  listGames(filters: { season?: string; teamId?: string; gameDate?: string }, signal?: AbortSignal): Promise<GameListResponse>
  listPossessions(gameId: string, signal?: AbortSignal): Promise<PossessionListResponse>
  saveReconstruction(possessionId: string, play: PlayDefinition): Promise<PossessionReconstruction>
}

export class HttpNbaApi implements NbaApi {
  constructor(private readonly baseUrl: string = API_BASE_URL) {}

  async getExampleFrame(signal?: AbortSignal): Promise<SimulationFrame> {
    return this.request<SimulationFrame>('/api/v1/simulation/example', { signal })
  }

  async listPlays(signal?: AbortSignal): Promise<PlayDefinition[]> {
    return this.request<PlayDefinition[]>('/api/v1/plays', { signal })
  }

  async savePlay(play: PlayDefinition): Promise<PlayDefinition> {
    return this.request<PlayDefinition>('/api/v1/plays', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(play),
    })
  }

  async duplicatePlay(playId: string): Promise<PlayDefinition> {
    return this.request<PlayDefinition>(`/api/v1/plays/${playId}/duplicate`, { method: 'POST' })
  }

  async deletePlay(playId: string): Promise<void> {
    await this.request<void>(`/api/v1/plays/${playId}`, { method: 'DELETE' })
  }

  async listTeams(signal?: AbortSignal): Promise<TeamListResponse> {
    return this.request<TeamListResponse>('/api/v1/nba/teams', { signal })
  }

  async getRoster(
    teamId: string,
    season: string,
    refresh = false,
    signal?: AbortSignal,
  ): Promise<RosterResponse> {
    const query = new URLSearchParams({ season, refresh: String(refresh) })
    return this.request<RosterResponse>(`/api/v1/nba/teams/${teamId}/roster?${query}`, { signal })
  }

  async getPreferredLineup(
    teamId: string,
    season: string,
    refresh = false,
    signal?: AbortSignal,
  ): Promise<PreferredLineupResponse> {
    const query = new URLSearchParams({ season, refresh: String(refresh) })
    return this.request<PreferredLineupResponse>(
      `/api/v1/nba/teams/${teamId}/preferred-lineup?${query}`,
      { signal },
    )
  }

  async listGames(
    filters: { season?: string; teamId?: string; gameDate?: string },
    signal?: AbortSignal,
  ): Promise<GameListResponse> {
    const query = new URLSearchParams()
    if (filters.season) query.set('season', filters.season)
    if (filters.teamId) query.set('team_id', filters.teamId)
    if (filters.gameDate) query.set('game_date', filters.gameDate)
    return this.request<GameListResponse>(`/api/v1/nba/games?${query}`, { signal })
  }

  async listPossessions(gameId: string, signal?: AbortSignal): Promise<PossessionListResponse> {
    return this.request<PossessionListResponse>(`/api/v1/nba/games/${gameId}/possessions`, { signal })
  }

  async saveReconstruction(
    possessionId: string,
    play: PlayDefinition,
  ): Promise<PossessionReconstruction> {
    return this.request<PossessionReconstruction>(
      `/api/v1/real-possessions/${possessionId}/reconstructions`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ play }),
      },
    )
  }

  private async request<T>(path: string, init: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, init)
    if (!response.ok) {
      let message = `NBA API returned ${response.status}`
      try {
        const payload = await response.json() as {
          detail?: string | { message?: string } | Array<{ msg?: string }>
        }
        if (typeof payload.detail === 'string') message = payload.detail
        if (payload.detail !== null && typeof payload.detail === 'object' && !Array.isArray(payload.detail)) {
          message = payload.detail.message ?? message
        }
        if (Array.isArray(payload.detail)) {
          message = payload.detail.map((item) => item.msg ?? 'Invalid play').join('; ')
        }
      } catch {
        // Keep the status-based fallback.
      }
      throw new ApiError(message, response.status)
    }
    if (response.status === 204) return undefined as T
    return response.json() as Promise<T>
  }
}

export const nbaApi: NbaApi = new HttpNbaApi()
export const simulationApi = nbaApi
