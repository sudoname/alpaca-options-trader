import { type PutTimeStopBoundary } from '@/lib/oracle/types'
import { fmtNum, signClass } from '@/lib/oracle/format'
import { usePutTimeStop } from '@/lib/oracle/hooks'
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

// Values from this endpoint are already in PERCENT units (e.g. 6.0 == 6%),
// unlike fmtPct which multiplies a fraction by 100. Format without scaling.
function pct(v: number | null | undefined, dp = 1): string {
  return v == null || Number.isNaN(v) ? '—' : `${Number(v).toFixed(dp)}%`
}

export function PutTimeStop() {
  const { data, isLoading } = usePutTimeStop()
  const rows = Array.isArray(data?.boundaries) ? data!.boundaries! : []
  const cap = data?.cap_days ?? 5
  const mean = data?.mean_shadow_delta_pct
  const helped = data?.count_helped ?? 0
  const hurt = data?.count_hurt ?? 0

  return (
    <ChartCard
      title={`Put ${cap}-Day Stop (shadow)`}
      footer='Forward-test only: records the measured day-N option mark for open puts. Never exits, sizes, or alters a trade. shadow_delta = day-N P/L − actual exit P/L (positive ⇒ the cap would have helped).'
    >
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && rows.length === 0}
        minHeight={120}
      >
        <div className='mb-3 grid grid-cols-2 gap-2 sm:grid-cols-4'>
          <Metric label='Boundaries' value={String(data?.n_boundaries ?? 0)} />
          <Metric label='Resolved' value={String(data?.n_resolved ?? 0)} />
          <Metric
            label='Mean Δ'
            value={pct(mean)}
            className={signClass(mean)}
          />
          <Metric label='Helped / Hurt' value={`${helped} / ${hurt}`} />
        </div>

        <div className='overflow-x-auto'>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Underlying</TableHead>
                <TableHead>Opened</TableHead>
                <TableHead className='text-end'>Entry</TableHead>
                <TableHead className='text-end'>Day-{cap} Mark</TableHead>
                <TableHead className='text-end'>Day-{cap} P/L</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className='text-end'>Exit P/L</TableHead>
                <TableHead className='text-end'>Δ Shadow</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r: PutTimeStopBoundary, i: number) => (
                <TableRow key={i}>
                  <TableCell>{String(r.underlying ?? r.symbol ?? '—')}</TableCell>
                  <TableCell>{String(r.entry_time ?? '—').slice(0, 10)}</TableCell>
                  <TableCell className='text-end'>
                    {fmtNum(r.entry_price, 2)}
                  </TableCell>
                  <TableCell className='text-end'>
                    {fmtNum(r.boundary_mark, 2)}
                  </TableCell>
                  <TableCell
                    className={cn('text-end', signClass(r.boundary_pnl_pct))}
                  >
                    {pct(r.boundary_pnl_pct)}
                  </TableCell>
                  <TableCell>{r.resolved ? 'resolved' : 'open'}</TableCell>
                  <TableCell
                    className={cn('text-end', signClass(r.actual_exit_pnl_pct))}
                  >
                    {r.resolved ? pct(r.actual_exit_pnl_pct) : '—'}
                  </TableCell>
                  <TableCell
                    className={cn('text-end', signClass(r.shadow_delta_pct))}
                  >
                    {r.resolved ? pct(r.shadow_delta_pct) : '—'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </ChartGuard>
    </ChartCard>
  )
}

function Metric({
  label,
  value,
  className,
}: {
  label: string
  value: string
  className?: string
}) {
  return (
    <div className='rounded-md border p-2'>
      <div className='text-muted-foreground text-xs tracking-wide uppercase'>
        {label}
      </div>
      <div className={cn('mt-1 text-lg font-semibold', className)}>{value}</div>
    </div>
  )
}
