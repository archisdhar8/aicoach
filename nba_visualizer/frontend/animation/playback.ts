export type PlaybackRate = 0.5 | 1 | 2

export class SimulationClock {
  private simulationTimeSeconds = 0
  private lastRealTimeMs: number | null = null
  private playing = false
  private rate: PlaybackRate = 1

  constructor(readonly durationSeconds: number) {
    if (!Number.isFinite(durationSeconds) || durationSeconds < 0) {
      throw new Error('simulation duration must be finite and non-negative')
    }
  }

  get timeSeconds(): number { return this.simulationTimeSeconds }
  get isPlaying(): boolean { return this.playing }
  get playbackRate(): PlaybackRate { return this.rate }

  play(realTimeMs: number): void {
    if (this.durationSeconds === 0) return
    if (this.simulationTimeSeconds >= this.durationSeconds) this.simulationTimeSeconds = 0
    this.lastRealTimeMs = realTimeMs
    this.playing = true
  }

  pause(realTimeMs: number): void {
    this.advance(realTimeMs)
    this.playing = false
    this.lastRealTimeMs = null
  }

  advance(realTimeMs: number): number {
    if (!this.playing) return this.simulationTimeSeconds
    if (this.lastRealTimeMs === null) this.lastRealTimeMs = realTimeMs
    const elapsedSeconds = Math.max(0, realTimeMs - this.lastRealTimeMs) / 1000
    this.simulationTimeSeconds = Math.min(
      this.durationSeconds,
      this.simulationTimeSeconds + elapsedSeconds * this.rate,
    )
    this.lastRealTimeMs = realTimeMs
    if (this.simulationTimeSeconds >= this.durationSeconds) {
      this.playing = false
      this.lastRealTimeMs = null
    }
    return this.simulationTimeSeconds
  }

  seek(timeSeconds: number, realTimeMs: number): void {
    this.simulationTimeSeconds = Math.min(Math.max(timeSeconds, 0), this.durationSeconds)
    this.lastRealTimeMs = this.playing ? realTimeMs : null
  }

  restart(): void {
    this.simulationTimeSeconds = 0
    this.playing = false
    this.lastRealTimeMs = null
  }

  setPlaybackRate(rate: PlaybackRate, realTimeMs: number): void {
    this.advance(realTimeMs)
    this.rate = rate
    this.lastRealTimeMs = this.playing ? realTimeMs : null
  }
}
