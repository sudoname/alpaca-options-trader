import { useState } from 'react'
import { type AgentContribution } from '@/lib/oracle/types'
import { fmtNum, fmtPct, signClass } from '@/lib/oracle/format'
import { useAgentContributions } from '@/lib/oracle/hooks'
import { cn } from '@/lib/utils'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { ChartCard } from './chart-card'
import { ChartGuard } from './chart-guard'

type Col = {
  label: string
  num?: boolean
  render: (r: AgentContribution) => React.ReactNode
  sort: (r: AgentContribution) => number | string | null
}

function numOrNull(v: number | null | undefined): number | null {
  return v == null || Number.isNaN(v) ? null : Number(v)
}

// net_pnl means arrive in percent units already (30 == +30%).
function rawPct(v: number | null | undefined): string {
  return v == null || Number.isNaN(v) ? '—' : `${Number(v).toFixed(1)}%`
}

function hitWithN(
  hit: number | null | undefined,
  n: number | null | undefined
): React.ReactNode {
  if (hit == null && !n) return '—'
  return (
    <span>
      {hit == null ? '—' : fmtPct(hit, 0)}
      <span className='text-muted-foreground'> ({n ?? 0})</span>
    </span>
  )
}

const columns: Col[] = [
  {
    label: 'Agent',
    render: (r) => String(r.agent ?? '—'),
    sort: (r) => r.agent ?? null,
  },
  {
    label: 'N',
    num: true,
    render: (r) => r.n ?? '—',
    sort: (r) => numOrNull(r.n),
  },
  {
    label: 'Agree %',
    num: true,
    render: (r) =>
      r.agreement_rate == null ? '—' : fmtPct(r.agreement_rate, 0),
    sort: (r) => numOrNull(r.agreement_rate),
  },
  {
    label: 'Hit (agree)',
    num: true,
    render: (r) =>
      r.hit_rate_when_agreeing == null
        ? '—'
        : fmtPct(r.hit_rate_when_agreeing, 0),
    sort: (r) => numOrNull(r.hit_rate_when_agreeing),
  },
  {
    label: 'Hit (bull)',
    num: true,
    render: (r) => hitWithN(r.hit_rate_when_bullish, r.n_bullish),
    sort: (r) => numOrNull(r.hit_rate_when_bullish),
  },
  {
    label: 'Hit (bear)',
    num: true,
    render: (r) => hitWithN(r.hit_rate_when_bearish, r.n_bearish),
    sort: (r) => numOrNull(r.hit_rate_when_bearish),
  },
  {
    label: 'Net (wins)',
    num: true,
    render: (r) => (
      <span className={signClass(r.mean_net_win)}>
        {rawPct(r.mean_net_win)}
      </span>
    ),
    sort: (r) => numOrNull(r.mean_net_win),
  },
  {
    label: 'Net (losses)',
    num: true,
    render: (r) => (
      <span className={signClass(r.mean_net_loss)}>
        {rawPct(r.mean_net_loss)}
      </span>
    ),
    sort: (r) => numOrNull(r.mean_net_loss),
  },
  {
    label: 'Conf',
    num: true,
    render: (r) => fmtNum(r.mean_conf, 2),
    sort: (r) => numOrNull(r.mean_conf),
  },
]

function cmp(a: number | string | null, b: number | string | null): number {
  if (a == null && b == null) return 0
  if (a == null) return 1 // blanks sort last
  if (b == null) return -1
  if (typeof a === 'number' && typeof b === 'number') return a - b
  return String(a).localeCompare(String(b))
}

export function AgentContributions() {
  const { data, isLoading } = useAgentContributions()
  const rows = Array.isArray(data?.agents) ? data!.agents! : []
  // Default: Agree % descending, so the agents that actually vote with the
  // taken direction surface first.
  const [sort, setSort] = useState<{ idx: number; dir: 1 | -1 }>({
    idx: 2,
    dir: -1,
  })

  const display = rows
    .slice()
    .sort((a, b) => cmp(columns[sort.idx].sort(a), columns[sort.idx].sort(b)) * sort.dir)

  const onSort = (idx: number) =>
    setSort((prev) =>
      prev.idx === idx
        ? { idx, dir: (prev.dir * -1) as 1 | -1 }
        : { idx, dir: -1 }
    )

  return (
    <ChartCard
      title='Agent Contributions'
      footer={
        data?.n_trades != null
          ? `scored on ${data.n_trades.toLocaleString()} resolved trades (${(
              data.n_with_votes ?? 0
            ).toLocaleString()} with votes) · agree = vote sign matches traded direction · hit = trade won`
          : 'per-agent vote quality on resolved trades'
      }
    >
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && rows.length === 0}
        minHeight={120}
      >
        <div className='overflow-x-auto'>
          <Table>
            <TableHeader>
              <TableRow>
                {columns.map((c, i) => {
                  const active = sort.idx === i
                  const arrow = active ? (sort.dir > 0 ? ' ▲' : ' ▼') : ''
                  return (
                    <TableHead
                      key={c.label}
                      onClick={() => onSort(i)}
                      className={cn(
                        'cursor-pointer select-none',
                        c.num && 'text-end'
                      )}
                    >
                      {c.label}
                      {arrow}
                    </TableHead>
                  )
                })}
              </TableRow>
            </TableHeader>
            <TableBody>
              {display.map((r, i) => (
                <TableRow key={r.agent ?? i}>
                  {columns.map((c) => (
                    <TableCell
                      key={c.label}
                      className={cn('tabular-nums', c.num && 'text-end')}
                    >
                      {c.render(r)}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </ChartGuard>
    </ChartCard>
  )
}
