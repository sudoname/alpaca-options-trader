import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ORA } from '@/lib/oracle/theme'
import { fmtNum, isUsable } from '@/lib/oracle/format'
import { useWeights } from '@/lib/oracle/hooks'
import { ChartCard, tooltipStyle } from './chart-card'
import { ChartGuard } from './chart-guard'

export function AdaptiveWeights() {
  const { data, isLoading } = useWeights()
  const cur = data?.current || {}
  const rows = Object.keys(cur).map((k) => ({ agent: k, weight: Number(cur[k]) }))

  const footer = isUsable(data) ? (
    <>
      snapshots <b>{data?.snapshots ?? 0}</b> · drift{' '}
      <b>{fmtNum(data?.drift, 3)}</b>
    </>
  ) : null

  return (
    <ChartCard title='Adaptive Weights' footer={footer}>
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && rows.length === 0}
        minHeight={260}
      >
        <ResponsiveContainer width='100%' height={260}>
          <BarChart
            data={rows}
            margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
          >
            <XAxis dataKey='agent' tick={{ fill: ORA.axis, fontSize: 11 }} />
            <YAxis tick={{ fill: ORA.axis, fontSize: 11 }} />
            <Tooltip {...tooltipStyle} />
            <Bar
              dataKey='weight'
              fill={ORA.accent}
              radius={[3, 3, 0, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </ChartGuard>
    </ChartCard>
  )
}
