import { fmtMoney, fmtPct, signClass } from '@/lib/oracle/format'
import { useRegimePerformance } from '@/lib/oracle/hooks'
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

export function RegimePerformance() {
  const { data, isLoading } = useRegimePerformance()
  const rows = Array.isArray(data?.regimes) ? data!.regimes! : []

  return (
    <ChartCard title='Regime Performance'>
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && rows.length === 0}
        minHeight={120}
      >
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Regime</TableHead>
              <TableHead className='text-end'>Trades</TableHead>
              <TableHead className='text-end'>Win rate</TableHead>
              <TableHead className='text-end'>Avg P/L</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r, i) => {
              const avg = r.average_pnl ?? r.avg_pnl
              return (
                <TableRow key={i}>
                  <TableCell>{String(r.regime ?? r.label ?? '—')}</TableCell>
                  <TableCell className='text-end'>
                    {r.trades ?? r.n ?? '—'}
                  </TableCell>
                  <TableCell className='text-end'>{fmtPct(r.win_rate)}</TableCell>
                  <TableCell className={`text-end ${signClass(avg)}`}>
                    {fmtMoney(avg)}
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </ChartGuard>
    </ChartCard>
  )
}
