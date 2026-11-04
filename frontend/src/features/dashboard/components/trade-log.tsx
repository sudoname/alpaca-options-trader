import { type TradeLogRow } from '@/lib/oracle/types'
import { fmtMoney, fmtNum, signClass } from '@/lib/oracle/format'
import { useTradeLog } from '@/lib/oracle/hooks'
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

const EVENT_STYLES: Record<string, string> = {
  OPEN: 'border-[#ffb000]/40 bg-[#ffb000]/10 text-[#ffb000]',
  CLOSE: 'border-[#3fb950]/40 bg-[#3fb950]/10 text-[#3fb950]',
}

// pnl_percent is already in percent units (e.g. -12.5 == -12.5%).
function pct(v: number | null | undefined): string {
  return v == null || Number.isNaN(v) ? '—' : `${Number(v).toFixed(1)}%`
}

export function TradeLog() {
  const { data, isLoading } = useTradeLog()
  const rows = Array.isArray(data?.rows) ? data!.rows! : []

  return (
    <ChartCard
      title='Trade Log'
      footer='Executed fills from the live single-leg bot — newest first.'
    >
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && rows.length === 0}
        minHeight={120}
      >
        <div className='max-h-96 overflow-x-auto overflow-y-auto'>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Time</TableHead>
                <TableHead>Event</TableHead>
                <TableHead>Symbol</TableHead>
                <TableHead>Underlying</TableHead>
                <TableHead>Type</TableHead>
                <TableHead className='text-end'>Qty</TableHead>
                <TableHead className='text-end'>Price</TableHead>
                <TableHead className='text-end'>P/L $</TableHead>
                <TableHead className='text-end'>P/L %</TableHead>
                <TableHead>Outcome</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r: TradeLogRow, i: number) => (
                <TableRow key={i}>
                  <TableCell className='font-mono text-xs tabular-nums'>
                    {String(r.at ?? '—').slice(0, 16).replace('T', ' ')}
                  </TableCell>
                  <TableCell>
                    <span
                      className={cn(
                        'inline-block border px-1.5 py-0.5 font-mono text-[10px] font-medium tracking-[0.1em] uppercase',
                        EVENT_STYLES[String(r.event ?? '')] ?? ''
                      )}
                    >
                      {String(r.event ?? '—')}
                    </span>
                  </TableCell>
                  <TableCell className='font-mono text-xs'>
                    {String(r.symbol ?? '—')}
                  </TableCell>
                  <TableCell>{String(r.underlying ?? '—')}</TableCell>
                  <TableCell>{String(r.opt_type ?? '—')}</TableCell>
                  <TableCell className='text-end tabular-nums'>
                    {r.qty == null ? '—' : fmtNum(r.qty, 0)}
                  </TableCell>
                  <TableCell className='text-end tabular-nums'>
                    {r.price == null ? '—' : fmtNum(r.price, 2)}
                  </TableCell>
                  <TableCell
                    className={cn('text-end tabular-nums', signClass(r.pnl))}
                  >
                    {fmtMoney(r.pnl)}
                  </TableCell>
                  <TableCell
                    className={cn(
                      'text-end tabular-nums',
                      signClass(r.pnl_percent)
                    )}
                  >
                    {pct(r.pnl_percent)}
                  </TableCell>
                  <TableCell>{String(r.outcome ?? '—')}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </ChartGuard>
    </ChartCard>
  )
}
