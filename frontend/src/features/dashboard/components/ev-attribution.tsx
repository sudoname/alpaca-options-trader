import {
  Bar,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ORA } from '@/lib/oracle/theme'
import { type EvBucketRaw } from '@/lib/oracle/types'
import { useEvAttribution } from '@/lib/oracle/hooks'
import { ChartCard, tooltipStyle } from './chart-card'
import { ChartGuard } from './chart-guard'

// The API returns ev_buckets as a dict {label: stats}; normalize to an array.
function normalizeBuckets(
  raw: Record<string, EvBucketRaw> | EvBucketRaw[] | undefined
): EvBucketRaw[] {
  if (!raw) return []
  if (Array.isArray(raw)) return raw
  if (typeof raw === 'object') {
    return Object.entries(raw).map(([label, b]) => ({ label, ...b }))
  }
  return []
}

export function EvAttribution() {
  const { data, isLoading } = useEvAttribution()
  const buckets = normalizeBuckets(data?.ev_buckets ?? data?.buckets)
  const rows = buckets.map((b) => ({
    label: String(b.label ?? b.bucket ?? b.range ?? ''),
    win_rate: Number(b.win_rate ?? b.winrate),
    profit_factor: Number(b.profit_factor ?? b.pf),
  }))

  return (
    <ChartCard title='EV Attribution'>
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && rows.length === 0}
        minHeight={280}
      >
        <ResponsiveContainer width='100%' height={280}>
          <ComposedChart
            data={rows}
            margin={{ top: 8, right: 8, bottom: 8, left: 0 }}
          >
            <XAxis dataKey='label' tick={{ fill: ORA.axis, fontSize: 11 }} />
            <YAxis
              yAxisId='left'
              domain={[0, 1]}
              tick={{ fill: ORA.axis, fontSize: 11 }}
              tickFormatter={(v) => Number(v).toFixed(1)}
            />
            <YAxis
              yAxisId='right'
              orientation='right'
              tick={{ fill: ORA.axis, fontSize: 11 }}
            />
            <Tooltip {...tooltipStyle} />
            <Bar
              yAxisId='left'
              dataKey='win_rate'
              name='win rate'
              fill={ORA.accent}
              radius={[3, 3, 0, 0]}
              isAnimationActive={false}
            />
            <Line
              yAxisId='right'
              type='monotone'
              dataKey='profit_factor'
              name='profit factor'
              stroke={ORA.green}
              strokeWidth={2}
              dot={{ fill: ORA.green, r: 3 }}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartGuard>
    </ChartCard>
  )
}
