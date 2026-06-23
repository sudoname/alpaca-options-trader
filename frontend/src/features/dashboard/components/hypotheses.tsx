import { fmtNum, fmtPct } from '@/lib/oracle/format'
import { useHypotheses } from '@/lib/oracle/hooks'
import { Badge } from '@/components/ui/badge'
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

function ConclusionBadge({ conclusion }: { conclusion?: string }) {
  const s = String(conclusion || '').toUpperCase()
  if (s.includes('CONFIRM') || s.includes('SUPPORT')) {
    return (
      <Badge className='border-transparent bg-[#3fb950]/15 text-[#3fb950]'>
        {s}
      </Badge>
    )
  }
  if (s.includes('REJECT') || s.includes('REFUTE')) {
    return <Badge variant='destructive'>{s}</Badge>
  }
  return <Badge variant='secondary'>{s || '—'}</Badge>
}

export function Hypotheses() {
  const { data, isLoading } = useHypotheses()
  const rows = Array.isArray(data?.hypotheses) ? data!.hypotheses! : []

  return (
    <ChartCard title='Hypotheses'>
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && rows.length === 0}
        minHeight={120}
      >
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Hypothesis</TableHead>
              <TableHead>Conclusion</TableHead>
              <TableHead className='text-end'>Confidence</TableHead>
              <TableHead className='text-end'>WR A/B</TableHead>
              <TableHead className='text-end'>Effect</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r, i) => (
              <TableRow key={i}>
                <TableCell>{String(r.hypothesis_name ?? '—')}</TableCell>
                <TableCell>
                  <ConclusionBadge conclusion={r.conclusion} />
                </TableCell>
                <TableCell className='text-end'>
                  {fmtNum(r.confidence, 2)}
                </TableCell>
                <TableCell className='text-end'>
                  {fmtPct(r.win_rate_a, 0)} / {fmtPct(r.win_rate_b, 0)}
                </TableCell>
                <TableCell className='text-end'>
                  {fmtNum(r.effect_size, 2)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </ChartGuard>
    </ChartCard>
  )
}
