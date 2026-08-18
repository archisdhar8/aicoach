import type { Coverage } from '@/frontend/domain/models'
import type { DefenseMode, DefensiveResponse } from '@/frontend/defense/engine'

const COVERAGES: Coverage[] = ['drop', 'switch', 'hedge', 'blitz', 'ice']

interface CoverageSelectorProps {
  value: Coverage
  onChange: (coverage: Coverage) => void
  defenseMode: DefenseMode
  onDefenseModeChange: (mode: DefenseMode) => void
  debugEnabled: boolean
  onDebugChange: (enabled: boolean) => void
  response: DefensiveResponse | null
}

export function CoverageSelector({
  value,
  onChange,
  defenseMode,
  onDefenseModeChange,
  debugEnabled,
  onDebugChange,
  response,
}: CoverageSelectorProps) {
  return (
    <div className="coverage-controls">
      <div className="coverage-grid" role="group" aria-label="Defensive coverage">
        <button
          className={`coverage-button${defenseMode === 'offense_only' ? ' active' : ''}`}
          onClick={() => onDefenseModeChange('offense_only')}
          type="button"
        >
          OFFENSE ONLY
        </button>
        {COVERAGES.map((coverage) => (
        <button
          className={`coverage-button${defenseMode === 'coverage' && coverage === value ? ' active' : ''}`}
          key={coverage}
          onClick={() => {
            onDefenseModeChange('coverage')
            onChange(coverage)
          }}
          type="button"
        >
          {coverage.toUpperCase()}
        </button>
        ))}
      </div>
      {defenseMode === 'coverage' && response && (
        <div className={`coverage-status${response.supported ? '' : ' unsupported'}`} role="status">
          {response.supported
            ? `${response.coverage.toUpperCase()} · matchup tracking active · ${response.events.length} screen event${response.events.length === 1 ? '' : 's'}`
            : response.unsupportedReason}
        </div>
      )}
      <label className="debug-toggle">
        <input
          checked={debugEnabled}
          onChange={(event) => onDebugChange(event.target.checked)}
          type="checkbox"
        />
        Defensive debug overlays
      </label>
    </div>
  )
}
