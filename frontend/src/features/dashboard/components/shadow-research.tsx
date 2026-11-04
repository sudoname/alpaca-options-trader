import {
  type ShadowCandidateRow,
  type ShadowVariantStats,
} from '@/lib/oracle/types'
import { fmtNum, signClass } from '@/lib/oracle/format'
import { useShadowResearch } from '@/lib/oracle/hooks'
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

// Values from this endpoint are already in PERCENT units (e.g. 1.2 == 1.2%),
// unlike fmtPct which multiplies a fraction by 100. Format without scaling.
function pct(v: number | null | undefined, dp = 2): string {
  return v == null || Number.isNaN(v) ? '—' : `${Number(v).toFixed(dp)}%`
}

// hit_rate IS a fraction (0..1) — scale it once here.
function rate(v: number | null | undefined): string {
  return v == null || Number.isNaN(v) ? '—' : `${(Number(v) * 100).toFixed(1)}%`
}

export function ShadowResearch() {
  const { data, isLoading } = useShadowResearch()
  const variants = Array.isArray(data?.variants) ? data!.variants! : []
  const recent = Array.isArray(data?.recent) ? data!.recent! : []

  return (
    <ChartCard
      title='Shadow Strategies (observe-only)'
      footer='Research only: shadow strategies log would-be trades, never place orders. Outcomes are simulated underlying-move counterfactuals — direction edge, not option P/L.'
    >
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && variants.length === 0 && recent.length === 0}
        minHeight={120}
      >
        {variants.map((v: ShadowVariantStats) => (
          <div key={v.variant ?? '?'} className='mb-3'>
            <div className='text-muted-foreground mb-1 text-xs font-medium tracking-wide uppercase'>
              {v.variant ?? '?'}
            </div>
            <div className='grid grid-cols-2 gap-2 sm:grid-cols-5'>
              <Metric label='Candidates' value={String(v.n_candidates ?? 0)} />
              <Metric label='Accepted' value={String(v.n_accepted ?? 0)} />
              <Metric label='Resolved' value={String(v.n_resolved ?? 0)} />
              <Metric label='Hit Rate' value={rate(v.hit_rate)} />
              <Metric
                label='Avg P/L'
                value={pct(v.avg_pnl_pct)}
                className={signClass(v.avg_pnl_pct)}
              />
            </div>
          </div>
        ))}

        <div className='overflow-x-auto'>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Time</TableHead>
                <TableHead>Variant</TableHead>
                <TableHead>Underlying</TableHead>
                <TableHead>Dir</TableHead>
                <TableHead className='text-end'>Score</TableHead>
                <TableHead className='text-end'>Price</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className='text-end'>CF P/L</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recent.map((r: ShadowCandidateRow, i: number) => (
                <TableRow key={i}>
                  <TableCell>
                    {String(r.at ?? '—').slice(0, 16).replace('T', ' ')}
                  </TableCell>
                  <TableCell>{String(r.variant ?? '—')}</TableCell>
                  <TableCell>{String(r.underlying ?? '—')}</TableCell>
                  <TableCell>{String(r.direction ?? '—')}</TableCell>
                  <TableCell className='text-end'>
                    {fmtNum(r.score, 2)}
                  </TableCell>
                  <TableCell className='text-end'>
                    {fmtNum(r.underlying_price, 2)}
                  </TableCell>
                  <TableCell>
                    {r.accepted ? 'accepted' : (r.reject_code ?? 'rejected')}
                  </TableCell>
                  <TableCell className={cn('text-end', signClass(r.pnl_pct))}>
                    {pct(r.pnl_pct)}
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
