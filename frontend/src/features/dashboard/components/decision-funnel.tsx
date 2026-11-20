import { useDecisionFunnel } from '@/lib/oracle/hooks'
import { ChartCard } from './chart-card'
import { ChartGuard } from './chart-guard'

const STAGE_LABELS: Record<string, string> = {
  decisions: 'Decisions scanned',
  traded: 'Traded (CALL/PUT)',
  blocked: 'Blocked (SKIP)',
  cap_skipped: 'Cap-skipped',
}

const STAGE_COLORS: Record<string, string> = {
  decisions: '#8a8a8a',
  traded: '#ffb000',
  blocked: '#f85149',
  cap_skipped: '#58a6ff',
}

export function DecisionFunnel() {
  const { data, isLoading } = useDecisionFunnel()
  const stages = Array.isArray(data?.funnel) ? data!.funnel! : []
  const reasons = Array.isArray(data?.block_reasons)
    ? data!.block_reasons!
    : []
  const total = stages.find((s) => s.stage === 'decisions')?.count ?? 0
  const maxReason = reasons.reduce((m, r) => Math.max(m, r.count ?? 0), 0)

  return (
    <ChartCard
      title='Decision Funnel'
      footer='Every scan decision from the RL episode store — traded vs blocked, and why.'
    >
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && stages.length === 0}
        minHeight={120}
      >
        <div className='grid gap-6 md:grid-cols-2'>
          <div className='space-y-2'>
            {stages.map((s) => {
              const count = s.count ?? 0
              const frac = total > 0 ? count / total : 0
              const key = String(s.stage ?? '')
              return (
                <div key={key}>
                  <div className='flex items-baseline justify-between font-mono text-xs'>
                    <span className='text-muted-foreground tracking-[0.1em] uppercase'>
                      {STAGE_LABELS[key] ?? key}
                    </span>
                    <span className='tabular-nums'>
                      {count.toLocaleString()}
                      <span className='text-muted-foreground'>
                        {' '}
                        ({(frac * 100).toFixed(1)}%)
                      </span>
                    </span>
                  </div>
                  <div className='bg-muted/40 mt-1 h-2 w-full'>
                    <div
                      className='h-2'
                      style={{
                        width: `${Math.max(frac * 100, count > 0 ? 0.5 : 0)}%`,
                        background: STAGE_COLORS[key] ?? '#ffb000',
                      }}
                    />
                  </div>
                </div>
              )
            })}
            <div className='text-muted-foreground pt-2 font-mono text-[11px]'>
              completed {data?.completed?.toLocaleString() ?? '—'} · open{' '}
              {data?.open?.toLocaleString() ?? '—'}
            </div>
          </div>

          <div>
            <div className='text-muted-foreground mb-2 font-mono text-xs tracking-[0.1em] uppercase'>
              Top block reasons
            </div>
            {reasons.length === 0 ? (
              <div className='text-muted-foreground font-mono text-xs'>—</div>
            ) : (
              <div className='space-y-1.5'>
                {reasons.map((r, i) => {
                  const count = r.count ?? 0
                  const frac = maxReason > 0 ? count / maxReason : 0
                  return (
                    <div key={i} className='font-mono text-xs'>
                      <div className='flex items-baseline justify-between gap-2'>
                        <span className='truncate'>{r.prefix ?? '—'}</span>
                        <span className='text-muted-foreground tabular-nums'>
                          {count.toLocaleString()}
                        </span>
                      </div>
                      <div className='bg-muted/40 mt-0.5 h-1 w-full'>
                        <div
                          className='h-1 bg-[#f85149]/70'
                          style={{ width: `${frac * 100}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </ChartGuard>
    </ChartCard>
  )
}
