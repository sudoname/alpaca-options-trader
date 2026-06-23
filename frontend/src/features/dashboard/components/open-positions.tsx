import { useState } from 'react'
import { type Position } from '@/lib/oracle/types'
import { fmtMoney, fmtNum, fmtPct, signClass } from '@/lib/oracle/format'
import { usePositions } from '@/lib/oracle/hooks'
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
  render: (r: Position) => React.ReactNode
  sort: (r: Position) => number | string | null
}

function numOrNull(v: number | null | undefined): number | null {
  return v == null || Number.isNaN(v) ? null : Number(v)
}

const columns: Col[] = [
  {
    label: 'Symbol',
    render: (r) => String(r.symbol ?? '—'),
    sort: (r) => r.symbol ?? null,
  },
  {
    label: 'Underlying',
    render: (r) => String(r.underlying ?? '—'),
    sort: (r) => r.underlying ?? null,
  },
  {
    label: 'Qty',
    num: true,
    render: (r) => r.quantity ?? '—',
    sort: (r) => numOrNull(r.quantity),
  },
  {
    label: 'Entry',
    num: true,
    render: (r) => fmtNum(r.entry_price, 2),
    sort: (r) => numOrNull(r.entry_price),
  },
  {
    label: 'Current',
    num: true,
    render: (r) => fmtNum(r.current_price, 2),
    sort: (r) => numOrNull(r.current_price),
  },
  {
    label: 'P/L $',
    num: true,
    render: (r) => (
      <span className={signClass(r.unrealized_pl)}>
        {fmtMoney(r.unrealized_pl)}
      </span>
    ),
    sort: (r) => numOrNull(r.unrealized_pl),
  },
  {
    label: 'P/L %',
    num: true,
    render: (r) => (
      <span className={signClass(r.unrealized_plpc)}>
        {fmtPct(r.unrealized_plpc, 1)}
      </span>
    ),
    sort: (r) => numOrNull(r.unrealized_plpc),
  },
  {
    label: 'Opened',
    render: (r) => String(r.entry_time ?? '—'),
    sort: (r) => r.entry_time ?? null,
  },
  {
    label: 'EV',
    num: true,
    render: (r) => fmtNum(r.expected_value, 2),
    sort: (r) => numOrNull(r.expected_value),
  },
  {
    label: 'PoP',
    num: true,
    render: (r) =>
      r.probability_of_profit != null
        ? fmtPct(r.probability_of_profit, 0)
        : '—',
    sort: (r) => numOrNull(r.probability_of_profit),
  },
]

function cmp(a: number | string | null, b: number | string | null): number {
  if (a == null && b == null) return 0
  if (a == null) return 1 // blanks sort last
  if (b == null) return -1
  if (typeof a === 'number' && typeof b === 'number') return a - b
  return String(a).localeCompare(String(b))
}

export function OpenPositions() {
  const { data, isLoading } = usePositions()
  const rows = Array.isArray(data?.positions) ? data!.positions! : []
  const [sort, setSort] = useState<{ idx: number; dir: 1 | -1 } | null>(null)

  let display = rows
  if (sort) {
    const col = columns[sort.idx]
    display = rows
      .slice()
      .sort((a, b) => cmp(col.sort(a), col.sort(b)) * sort.dir)
  }

  const onSort = (idx: number) =>
    setSort((prev) =>
      prev && prev.idx === idx
        ? { idx, dir: (prev.dir * -1) as 1 | -1 }
        : { idx, dir: 1 }
    )

  return (
    <ChartCard title='Open Positions'>
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
                  const active = sort?.idx === i
                  const arrow = active ? (sort!.dir > 0 ? ' ▲' : ' ▼') : ''
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
                <TableRow key={i}>
                  {columns.map((c) => (
                    <TableCell
                      key={c.label}
                      className={cn(c.num && 'text-end')}
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
