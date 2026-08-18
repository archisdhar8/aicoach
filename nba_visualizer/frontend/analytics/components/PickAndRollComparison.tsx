import { useMemo } from 'react'
import { sampleSimulation, simulatePlay } from '@/frontend/animation/simulation'
import type { PlayDefinition } from '@/frontend/domain/models'
import type { DefensiveParameters } from '@/frontend/defense/engine'

interface PickAndRollComparisonProps {
  play: PlayDefinition
  parameters: DefensiveParameters
}

export function PickAndRollComparison({ play, parameters }: PickAndRollComparisonProps) {
  const comparison = useMemo(() => {
    const sampleCoverage = (coverage: 'drop' | 'blitz') => {
      const definition = structuredClone(play)
      definition.initialFrame.possession.coverage = coverage
      const result = simulatePlay(definition, { defensiveParameters: parameters })
      const frame = sampleSimulation(result, Math.min(1.9, result.durationSeconds))
      const screenerId = definition.actions.find(
        (action) => action.actionType === 'screen',
      )?.playerId
      const screenerShot = frame.analytics?.shotOpenness.find(
        (evaluation) => evaluation.offensivePlayerId === screenerId,
      )
      const screenerPass = frame.analytics?.passingLanes.find(
        (evaluation) => evaluation.targetPlayerId === screenerId,
      )
      return {
        drive: frame.analytics?.driveLane?.opennessScore ?? 0,
        screenerShot: screenerShot?.opennessScore ?? 0,
        passRisk: screenerPass?.interceptionRiskScore ?? 0,
        committedToHandler: result.defensiveResponse?.instructions.filter(
          (instruction) => instruction.targetOffensivePlayerId
            === frame.analytics?.driveLane?.handlerId,
        ).length ?? 0,
      }
    }
    return { drop: sampleCoverage('drop'), blitz: sampleCoverage('blitz') }
  }, [play, parameters])

  return (
    <section className="coverage-comparison" aria-label="Drop and Blitz analytics comparison">
      <div>
        <strong>Same high P&amp;R · 1.9s</strong>
        <span>Why openings differ</span>
      </div>
      <table>
        <thead><tr><th>Coverage</th><th>Drive</th><th>Roll shot</th><th>Pass risk</th><th>At handler</th></tr></thead>
        <tbody>
          <ComparisonRow label="Drop" values={comparison.drop} />
          <ComparisonRow label="Blitz" values={comparison.blitz} />
        </tbody>
      </table>
      <p>Scores are geometric heuristics from the generated frame, not outcome probabilities.</p>
    </section>
  )
}

function ComparisonRow({
  label,
  values,
}: {
  label: string
  values: { drive: number; screenerShot: number; passRisk: number; committedToHandler: number }
}) {
  return (
    <tr>
      <th>{label}</th>
      <td>{values.drive.toFixed(0)}</td>
      <td>{values.screenerShot.toFixed(0)}</td>
      <td>{values.passRisk.toFixed(0)}</td>
      <td>{values.committedToHandler}</td>
    </tr>
  )
}
