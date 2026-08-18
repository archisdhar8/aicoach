import type { DefensiveParameters } from '@/frontend/defense/engine'

interface DefensiveParameterControlsProps {
  value: DefensiveParameters
  onChange: (parameters: DefensiveParameters) => void
}

const FIELDS: Array<{
  key: keyof DefensiveParameters
  label: string
  min: number
  max: number
  step: number
}> = [
  { key: 'defenderSpeedFeetPerSecond', label: 'Defender speed (ft/s)', min: 5, max: 25, step: 0.5 },
  { key: 'maximumAccelerationFeetPerSecondSquared', label: 'Max acceleration (ft/s²)', min: 5, max: 50, step: 1 },
  { key: 'defensiveCushionFeet', label: 'Defensive cushion (ft)', min: 0.5, max: 8, step: 0.5 },
  { key: 'reactionDelaySeconds', label: 'Reaction delay (s)', min: 0, max: 2, step: 0.05 },
  { key: 'screenNavigationDelaySeconds', label: 'Screen delay (s)', min: 0, max: 2, step: 0.05 },
  { key: 'recoverySpeedFeetPerSecond', label: 'Recovery speed (ft/s)', min: 5, max: 25, step: 0.5 },
  { key: 'dropDepthFeet', label: 'Drop depth (ft)', min: 1, max: 15, step: 0.5 },
  { key: 'helpDistanceFeet', label: 'Help distance (ft)', min: 1, max: 12, step: 0.5 },
]

export function DefensiveParameterControls({
  value,
  onChange,
}: DefensiveParameterControlsProps) {
  return (
    <details className="defensive-parameters">
      <summary>Behavior parameters</summary>
      <div className="parameter-grid">
        {FIELDS.map((field) => (
          <label key={field.key}>
            {field.label}
            <input
              max={field.max}
              min={field.min}
              onChange={(event) => onChange({
                ...value,
                [field.key]: Number(event.target.value),
              })}
              step={field.step}
              type="number"
              value={value[field.key]}
            />
          </label>
        ))}
      </div>
      <p>Teaching defaults only; not empirically calibrated.</p>
    </details>
  )
}
