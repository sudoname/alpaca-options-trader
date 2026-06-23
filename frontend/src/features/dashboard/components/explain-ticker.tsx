import { useState, type FormEvent } from 'react'
import { fmtNum, fmtPct, isUsable } from '@/lib/oracle/format'
import { useExplain } from '@/lib/oracle/hooks'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { ChartCard } from './chart-card'

const TICKER_RE = /^[A-Z.]{1,8}$/

function ProbPill({ label, value }: { label: string; value?: number | null }) {
  return (
    <div className='bg-muted/40 flex min-w-[88px] flex-col items-center rounded-md px-3 py-2'>
      <div className='text-xl font-bold'>{fmtPct(value, 0)}</div>
      <div className='text-muted-foreground text-xs'>{label}</div>
    </div>
  )
}

export function ExplainTicker() {
  const [input, setInput] = useState('')
  const [ticker, setTicker] = useState<string | null>(null)
  const [invalid, setInvalid] = useState(false)

  const { data, isFetching } = useExplain(ticker)

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    const clean = input.trim().toUpperCase()
    if (!TICKER_RE.test(clean)) {
      setInvalid(true)
      setTicker(null)
      return
    }
    setInvalid(false)
    setTicker(clean)
  }

  const p = data?.probability || {}
  const votes = Array.isArray(data?.votes) ? data!.votes! : []
  const ex =
    data?.explanation && typeof data.explanation === 'object'
      ? data.explanation
      : {}
  const summary =
    ('summary_str' in ex ? ex.summary_str : undefined) ||
    data?.summary_str ||
    ''
  const reasons =
    'top_reasons' in ex && Array.isArray(ex.top_reasons) ? ex.top_reasons : []

  return (
    <ChartCard title='Explain a Ticker'>
      <form onSubmit={onSubmit} className='flex items-center gap-2'>
        <Input
          value={input}
          maxLength={8}
          placeholder='e.g. AAPL'
          onChange={(e) => setInput(e.target.value)}
          className='max-w-[160px] uppercase'
        />
        <Button type='submit'>Explain</Button>
      </form>

      <div className='mt-4'>
        {invalid ? (
          <Badge variant='destructive'>invalid ticker</Badge>
        ) : !ticker ? (
          <span className='text-muted-foreground text-sm'>
            Enter a ticker to see the agent breakdown.
          </span>
        ) : isFetching && !data ? (
          <span className='text-muted-foreground text-sm'>…</span>
        ) : !isUsable(data) ? (
          <Badge variant={data?.verdict === 'ERROR' ? 'destructive' : 'secondary'}>
            {data?.verdict === 'ERROR' ? 'error' : 'insufficient data'}
          </Badge>
        ) : (
          <div className='space-y-4'>
            <div className='flex flex-wrap gap-3'>
              <ProbPill label='P(call)' value={p.call ?? p.p_call} />
              <ProbPill label='P(put)' value={p.put ?? p.p_put} />
              <ProbPill label='P(no-trade)' value={p.no_trade ?? p.p_no_trade} />
            </div>

            {votes.length ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Agent</TableHead>
                    <TableHead className='text-end'>Bull</TableHead>
                    <TableHead className='text-end'>Bear</TableHead>
                    <TableHead className='text-end'>Conf</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {votes.map((v, i) => (
                    <TableRow key={i}>
                      <TableCell>{String(v.agent ?? v.name ?? '—')}</TableCell>
                      <TableCell className='text-end'>
                        {fmtNum(v.bullish_score ?? v.bull, 2)}
                      </TableCell>
                      <TableCell className='text-end'>
                        {fmtNum(v.bearish_score ?? v.bear, 2)}
                      </TableCell>
                      <TableCell className='text-end'>
                        {fmtNum(v.confidence, 2)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : null}

            {summary ? (
              <p className='text-muted-foreground text-sm'>{String(summary)}</p>
            ) : null}
            {reasons.length ? (
              <ul className='text-muted-foreground list-disc space-y-1 ps-5 text-sm'>
                {reasons.map((r, i) => (
                  <li key={i}>{String(r)}</li>
                ))}
              </ul>
            ) : null}
          </div>
        )}
      </div>
    </ChartCard>
  )
}
