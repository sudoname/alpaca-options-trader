import {
  type ShadowPnlBook,
  type ShadowPnlClosedTrade,
  type ShadowPnlOpenTrade,
} from '@/lib/oracle/types'
import { fmtMoney, fmtPct, signClass } from '@/lib/oracle/format'
import { useShadowPnl } from '@/lib/oracle/hooks'
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

const BOOK_LABEL: Record<string, string> = {
  dir: 'Direction (calls/puts)',
  vol: 'Volatility (straddles)',
}

export function ShadowTraderPnl() {
  const { data, isLoading } = useShadowPnl()
  const books = data?.books ?? {}
  const bookNames = Object.keys(books)
  const openTrades = Array.isArray(data?.open_trades) ? data!.open_trades! : []
  const lastTrades = Array.isArray(data?.last_trades) ? data!.last_trades! : []

  return (
    <ChartCard
      title='Shadow Options Trader (SPY ML, paper)'
      footer={
        <>
          Paper-only shadow books fed by the SPY 10-min ML monitor: direction
          book buys ATM calls/puts on confident leans (expected ~coin flip);
          vol book buys ATM straddles when predicted move beats option-implied
          move. Exits: dynamic take-profit / trailing stop / 10-min horizon.
          {data?.updated_at ? ` Updated ${data.updated_at} ET.` : ''}
          {data?.stale ? ' (STALE — pusher offline?)' : ''}
        </>
      }
    >
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && bookNames.length === 0}
        minHeight={120}
      >
        <div className='grid grid-cols-2 gap-2 sm:grid-cols-4'>
          <Metric
            label="Today's P/L"
            value={fmtMoney(data?.today_pnl)}
            className={signClass(data?.today_pnl)}
          />
          <Metric
            label='Total P/L'
            value={fmtMoney(data?.total_pnl)}
            className={signClass(data?.total_pnl)}
          />
          <Metric
            label='Open Trades'
            value={String(openTrades.length)}
          />
          <Metric
            label='Freshness'
            value={
              data?.age_s == null
                ? '—'
                : data.age_s < 90
                  ? 'live'
                  : `${Math.round(data.age_s / 60)}m ago`
            }
            className={data?.stale ? 'text-[#f85149]' : ''}
          />
        </div>

        {bookNames.map((name) => {
          const b: ShadowPnlBook = books[name] ?? {}
          return (
            <div key={name} className='mt-3'>
              <div className='text-muted-foreground mb-1 text-xs font-medium tracking-wide uppercase'>
                {BOOK_LABEL[name] ?? name}
              </div>
              <div className='grid grid-cols-2 gap-2 sm:grid-cols-5'>
                <Metric
                  label='Today'
                  value={fmtMoney(b.today_pnl)}
                  className={signClass(b.today_pnl)}
                />
                <Metric
                  label='Total'
                  value={fmtMoney(b.total_pnl)}
                  className={signClass(b.total_pnl)}
                />
                <Metric
                  label='Trades'
                  value={`${b.trades ?? 0} (${b.wins ?? 0}W)`}
                />
                <Metric label='Open' value={String(b.open ?? 0)} />
                <Metric label='Skipped' value={String(b.skipped ?? 0)} />
              </div>
            </div>
          )
        })}

        {openTrades.length > 0 && (
          <div className='mt-3 overflow-x-auto'>
            <div className='text-muted-foreground mb-1 text-xs font-medium tracking-wide uppercase'>
              Open
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Book</TableHead>
                  <TableHead>Entry</TableHead>
                  <TableHead>Legs</TableHead>
                  <TableHead className='text-end'>Cost</TableHead>
                  <TableHead className='text-end'>Mid</TableHead>
                  <TableHead className='text-end'>Peak</TableHead>
                  <TableHead className='text-end'>TP</TableHead>
                  <TableHead className='text-end'>Trail</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {openTrades.map((o: ShadowPnlOpenTrade, i: number) => (
                  <TableRow key={i}>
                    <TableCell>{o.book ?? '—'}</TableCell>
                    <TableCell>
                      {String(o.entry_time ?? '—').slice(11, 16)}
                    </TableCell>
                    <TableCell>{(o.legs ?? []).join(' + ') || '—'}</TableCell>
                    <TableCell className='text-end'>
                      {fmtMoney(o.entry_cost)}
                    </TableCell>
                    <TableCell className='text-end'>
                      {fmtMoney(o.last_mid)}
                    </TableCell>
                    <TableCell className='text-end'>
                      {fmtMoney(o.peak)}
                    </TableCell>
                    <TableCell className='text-end'>
                      {o.tp_gain == null ? '—' : `+${fmtPct(o.tp_gain, 0)}`}
                    </TableCell>
                    <TableCell className='text-end'>
                      {fmtPct(o.trail_pct, 1)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}

        {lastTrades.length > 0 && (
          <div className='mt-3 overflow-x-auto'>
            <div className='text-muted-foreground mb-1 text-xs font-medium tracking-wide uppercase'>
              Recent Closes
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Book</TableHead>
                  <TableHead>Day</TableHead>
                  <TableHead>Entry</TableHead>
                  <TableHead>Exit</TableHead>
                  <TableHead>Reason</TableHead>
                  <TableHead>Legs</TableHead>
                  <TableHead className='text-end'>P/L</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {lastTrades.map((r: ShadowPnlClosedTrade, i: number) => (
                  <TableRow key={i}>
                    <TableCell>{r.book ?? '—'}</TableCell>
                    <TableCell>{r.day ?? '—'}</TableCell>
                    <TableCell>
                      {String(r.entry_time ?? '—').slice(11, 16)}
                    </TableCell>
                    <TableCell>
                      {String(r.exit_time ?? '—').slice(11, 16)}
                    </TableCell>
                    <TableCell>{r.exit_reason ?? '—'}</TableCell>
                    <TableCell>{(r.legs ?? []).join(' + ') || '—'}</TableCell>
                    <TableCell className={cn('text-end', signClass(r.pnl))}>
                      {fmtMoney(r.pnl)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
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
