import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ORA } from '@/lib/oracle/theme'
import { fmtNum, fmtPct } from '@/lib/oracle/format'
import { useEpisodes } from '@/lib/oracle/hooks'
import { ChartCard, tooltipStyle } from './chart-card'
import { ChartGuard } from './chart-guard'

export function RlEpisodes() {
  const { data, isLoading } = useEpisodes()
  const counts = data?.chosen_action_counts || {}
  const rows = Object.keys(counts).map((k) => ({
    action: k,
    count: Number(counts[k]),
  }))
  const s = data?.stats || {}

  const footer = rows.length ? (
    <>
      total <b>{s.total ?? 0}</b> · completed <b>{s.completed ?? 0}</b> · win
      rate <b>{fmtPct(s.win_rate)}</b> · mean net{' '}
      <b>{fmtNum(s.mean_net_pnl_pct, 2)}%</b>
    </>
  ) : null

  return (
    <ChartCard title='RL Episodes' footer={footer}>
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
            <XAxis dataKey='action' tick={{ fill: ORA.axis, fontSize: 11 }} />
            <YAxis tick={{ fill: ORA.axis, fontSize: 11 }} allowDecimals={false} />
            <Tooltip {...tooltipStyle} />
            <Bar
              dataKey='count'
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
