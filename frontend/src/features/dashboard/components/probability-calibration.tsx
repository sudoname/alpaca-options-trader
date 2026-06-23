import {
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ORA } from '@/lib/oracle/theme'
import { fmtNum, isUsable } from '@/lib/oracle/format'
import { useCalibrationPop, useProbability } from '@/lib/oracle/hooks'
import { ChartCard, tooltipStyle } from './chart-card'
import { ChartGuard } from './chart-guard'

export function ProbabilityCalibration() {
  const pop = useCalibrationPop()
  const prob = useProbability()

  const buckets = Array.isArray(pop.data?.buckets) ? pop.data!.buckets! : []
  const obs = buckets
    .map((b) => ({
      x: Number(b.predicted ?? b.mid ?? b.p ?? b.bucket),
      y: Number(b.realized ?? b.actual ?? b.win_rate),
    }))
    .filter((d) => !Number.isNaN(d.x) && !Number.isNaN(d.y))
    .sort((a, b) => a.x - b.x)

  const p = prob.data
  const footer = isUsable(p) ? (
    <>
      Brier <b>{fmtNum(p?.brier)}</b> · baseline <b>{fmtNum(p?.baseline_brier)}</b>{' '}
      · skill <b>{fmtNum(p?.skill)}</b> · n={p?.sample_size ?? '—'}
    </>
  ) : null

  return (
    <ChartCard title='Probability Calibration' footer={footer}>
      <ChartGuard
        data={pop.data}
        isLoading={pop.isLoading}
        empty={!pop.isLoading && obs.length === 0}
        minHeight={260}
      >
        <ResponsiveContainer width='100%' height={260}>
          <ComposedChart
            data={obs}
            margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
          >
            <XAxis
              type='number'
              dataKey='x'
              domain={[0, 1]}
              tick={{ fill: ORA.axis, fontSize: 11 }}
              tickFormatter={(v) => Number(v).toFixed(1)}
              label={{ value: 'predicted', position: 'insideBottom', offset: -2, fill: ORA.axis, fontSize: 11 }}
            />
            <YAxis
              type='number'
              domain={[0, 1]}
              tick={{ fill: ORA.axis, fontSize: 11 }}
              tickFormatter={(v) => Number(v).toFixed(1)}
            />
            <Tooltip {...tooltipStyle} />
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              stroke={ORA.muted}
              strokeDasharray='4 4'
            />
            <Line
              type='monotone'
              dataKey='y'
              stroke={ORA.accent}
              strokeWidth={2}
              dot={{ fill: ORA.accent, r: 3 }}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartGuard>
    </ChartCard>
  )
}
