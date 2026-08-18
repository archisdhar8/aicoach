import type { AnalyticsOverlay, FrameAnalytics } from '@/frontend/analytics/types'

const OVERLAYS: Array<{ value: AnalyticsOverlay; label: string }> = [
  { value: 'passing_lanes', label: 'Passing lanes' },
  { value: 'defender_distance', label: 'Defender distance' },
  { value: 'driving_lanes', label: 'Driving lanes' },
  { value: 'spacing', label: 'Spacing' },
  { value: 'shot_openness', label: 'Shot openness' },
  { value: 'matchups', label: 'Matchups' },
]

interface AnalyticsControlsProps {
  enabled: AnalyticsOverlay[]
  analytics?: FrameAnalytics
  onToggle: (overlay: AnalyticsOverlay) => void
}

export function AnalyticsControls({ enabled, analytics, onToggle }: AnalyticsControlsProps) {
  return (
    <section className="analytics-controls">
      <h2>Spatial analytics</h2>
      <p>Deterministic geometry and heuristic scores—not expected shooting percentage.</p>
      <div className="analytics-toggle-grid" role="group" aria-label="Analytics overlays">
        {OVERLAYS.map((overlay) => (
          <button
            aria-pressed={enabled.includes(overlay.value)}
            className={enabled.includes(overlay.value) ? 'active' : ''}
            key={overlay.value}
            onClick={() => onToggle(overlay.value)}
            type="button"
          >
            {overlay.label}
          </button>
        ))}
      </div>
      {analytics && (
        <div className="analytics-summary" role="status">
          <span>Analytics {analytics.sampledAtSeconds.toFixed(1)}s</span>
          <span>{analytics.computationDurationMs.toFixed(2)} ms</span>
        </div>
      )}
    </section>
  )
}
