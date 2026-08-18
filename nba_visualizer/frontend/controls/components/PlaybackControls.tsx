import type { PlaybackRate } from '@/frontend/animation/playback'

interface PlaybackControlsProps {
  currentTimeSeconds: number
  durationSeconds: number
  isPlaying: boolean
  playbackRate: PlaybackRate
  canPlay: boolean
  onPlayPause: () => void
  onRestart: () => void
  onSeek: (timeSeconds: number) => void
  onPlaybackRateChange: (rate: PlaybackRate) => void
}

export function PlaybackControls({
  currentTimeSeconds,
  durationSeconds,
  isPlaying,
  playbackRate,
  canPlay,
  onPlayPause,
  onRestart,
  onSeek,
  onPlaybackRateChange,
}: PlaybackControlsProps) {
  return (
    <div className="playback-controls">
      <div className="transport-row">
        <button disabled={!canPlay} onClick={onPlayPause} type="button">
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button disabled={!canPlay} onClick={onRestart} type="button">Restart</button>
        <output aria-live="polite" data-testid="simulation-time">
          {formatTime(currentTimeSeconds)} / {formatTime(durationSeconds)}
        </output>
      </div>
      <label className="timeline-label">
        <span>Possession timeline</span>
        <input
          aria-label="Possession timeline"
          disabled={!canPlay}
          max={Math.max(durationSeconds, 0.01)}
          min="0"
          onInput={(event) => onSeek(Number(event.currentTarget.value))}
          step="0.01"
          type="range"
          value={Math.min(currentTimeSeconds, Math.max(durationSeconds, 0.01))}
        />
      </label>
      <div className="speed-row" role="group" aria-label="Playback speed">
        {([0.5, 1, 2] as PlaybackRate[]).map((rate) => (
          <button
            aria-pressed={playbackRate === rate}
            className={playbackRate === rate ? 'active' : ''}
            key={rate}
            onClick={() => onPlaybackRateChange(rate)}
            type="button"
          >
            {rate}×
          </button>
        ))}
      </div>
    </div>
  )
}

function formatTime(timeSeconds: number): string {
  return `${timeSeconds.toFixed(2)}s`
}
