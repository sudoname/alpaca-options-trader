import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ORA } from '@/lib/oracle/theme'
import { useFeatureImportance } from '@/lib/oracle/hooks'
import { ChartCard, tooltipStyle } from './chart-card'
import { ChartGuard } from './chart-guard'

export function FeatureImportance() {
  const { data, isLoading } = useFeatureImportance()
  const rows = Array.isArray(data?.features)
    ? data!.features!.map((f) => ({
        agent: String(f.agent),
        importance: Number(f.importance),
      }))
    : []
  const height = Math.max(260, rows.length * 34)

  return (
    <ChartCard title='Feature Importance'>
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
            <XAxis type='number' tick={{ fill: ORA.axis, fontSize: 11 }} />
            <YAxis
              type='category'
              dataKey='agent'
              width={110}
              tick={{ fill: ORA.axis, fontSize: 11 }}
            />
            <Tooltip {...tooltipStyle} />
            <Bar
              dataKey='importance'
              fill={ORA.green}
              radius={[0, 3, 3, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </ChartGuard>
    </ChartCard>
  )
}
