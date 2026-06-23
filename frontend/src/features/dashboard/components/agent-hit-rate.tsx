import {
  Bar,
  BarChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ORA } from '@/lib/oracle/theme'
import { useAgents } from '@/lib/oracle/hooks'
import { ChartCard, tooltipStyle } from './chart-card'
import { ChartGuard } from './chart-guard'

export function AgentHitRate() {
  const { data, isLoading } = useAgents()
  const rows = Array.isArray(data?.agents)
    ? data!.agents!.map((a) => ({
        agent: String(a.agent),
        hit_rate: Number(a.hit_rate),
      }))
    : []
  const base = data?.base_win_rate
  const height = Math.max(260, rows.length * 34)

  return (
    <ChartCard title='Agent Hit-Rate'>
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && rows.length === 0}
        minHeight={260}
      >
        <ResponsiveContainer width='100%' height={height}>
          <BarChart
            layout='vertical'
            data={rows}
            margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
          >
            <XAxis
              type='number'
              domain={[0, 1]}
              tick={{ fill: ORA.axis, fontSize: 11 }}
              tickFormatter={(v) => Number(v).toFixed(1)}
            />
            <YAxis
              type='category'
              dataKey='agent'
              width={110}
              tick={{ fill: ORA.axis, fontSize: 11 }}
            />
            <Tooltip {...tooltipStyle} />
            {base != null ? (
              <ReferenceLine
                x={Number(base)}
                stroke={ORA.amber}
                strokeDasharray='4 4'
              />
            ) : null}
            <Bar
              dataKey='hit_rate'
              fill={ORA.accent}
              radius={[0, 3, 3, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </ChartGuard>
    </ChartCard>
  )
}
