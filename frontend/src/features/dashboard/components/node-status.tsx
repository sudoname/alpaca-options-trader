import { type NodeHealth } from '@/lib/oracle/types'
import { useNodeStatus } from '@/lib/oracle/hooks'
import { cn } from '@/lib/utils'
import { ChartCard } from './chart-card'
import { ChartGuard } from './chart-guard'

const HEALTH_STYLES: Record<string, string> = {
  online: 'border-[#3fb950]/40 bg-[#3fb950]/10 text-[#3fb950]',
  stale: 'border-[#ffb000]/40 bg-[#ffb000]/10 text-[#ffb000]',
  offline: 'border-[#f85149]/40 bg-[#f85149]/10 text-[#f85149]',
}

function HealthTag({ status }: { status?: NodeHealth }) {
  const s = status ?? 'offline'
  return (
    <span
      className={cn(
        'flex items-center gap-1.5 border px-2 py-0.5 font-mono text-[10px] font-medium tracking-[0.15em] uppercase',
        HEALTH_STYLES[s] ?? HEALTH_STYLES.offline
      )}
    >
      <span aria-hidden className='size-1.5 animate-pulse bg-current' />
      {s}
    </span>
  )
}

function age(seconds: number | null | undefined): string {
  if (seconds == null || Number.isNaN(seconds)) return '—'
  const s = Math.max(0, Math.round(seconds))
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.round(s / 60)}m ago`
  return `${(s / 3600).toFixed(1)}h ago`
}

function NodeRow({
  name,
  status,
  facts,
}: {
  name: string
  status?: NodeHealth
  facts: Array<{ label: string; value: string }>
}) {
  return (
    <div className='border-border flex flex-col gap-2 border p-3'>
      <div className='flex items-center justify-between gap-2'>
        <span className='font-mono text-xs font-medium tracking-[0.18em] uppercase'>
          {name}
        </span>
        <HealthTag status={status} />
      </div>
      <div className='grid grid-cols-2 gap-x-4 gap-y-1'>
        {facts.map((f) => (
          <div
            key={f.label}
            className='flex items-baseline justify-between gap-2'
          >
            <span className='text-muted-foreground font-mono text-[10px] tracking-[0.1em] uppercase'>
              {f.label}
            </span>
            <span className='font-mono text-xs tabular-nums'>{f.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function NodeStatus() {
  const { data, isLoading } = useNodeStatus()
  const sched = data?.scheduler
  const trades = data?.trades
  const ledger = data?.ledger

  return (
    <ChartCard title='Node Status'>
      <ChartGuard data={data} isLoading={isLoading} minHeight={100}>
        <div className='grid grid-cols-1 gap-3 md:grid-cols-3'>
          <NodeRow
            name='Scheduler'
            status={sched?.status}
            facts={[
              { label: 'Mode', value: String(sched?.mode ?? '—') },
              {
                label: 'Market',
                value:
                  sched?.is_open == null
                    ? '—'
                    : sched.is_open
                      ? 'OPEN'
                      : 'CLOSED',
              },
              { label: 'Heartbeat', value: age(sched?.heartbeat_age_s) },
              {
                label: 'Scan',
                value:
                  sched?.scan_interval_min == null
                    ? '—'
                    : `${sched.scan_interval_min}m`,
              },
              {
                label: 'Watching',
                value:
                  sched?.symbols_watched == null
                    ? '—'
                    : `${sched.symbols_watched} syms`,
              },
            ]}
          />
          <NodeRow
            name='Trades'
            status={trades?.status}
            facts={[
              {
                label: 'Today',
                value:
                  sched?.trades_today == null
                    ? '—'
                    : `${sched.trades_today}${
                        sched.max_new_trades_per_day == null
                          ? ''
                          : ` / ${sched.max_new_trades_per_day} max`
                      }`,
              },
              {
                label: 'Open',
                value:
                  trades?.open_count == null ? '—' : String(trades.open_count),
              },
              {
                label: 'PDT left',
                value:
                  sched?.pdt_remaining == null
                    ? '—'
                    : String(sched.pdt_remaining),
              },
              {
                label: 'Can trade',
                value:
                  sched?.pdt_can_trade == null
                    ? '—'
                    : sched.pdt_can_trade
                      ? 'YES'
                      : 'NO',
              },
              {
                label: 'Last fill',
                value: trades?.last_trade_at
                  ? String(trades.last_trade_at).slice(0, 16).replace('T', ' ')
                  : '—',
              },
            ]}
          />
          <NodeRow
            name='Ledger'
            status={ledger?.status}
            facts={[
              {
                label: 'Snapshots',
                value:
                  ledger?.snapshots == null
                    ? '—'
                    : ledger.snapshots.toLocaleString(),
              },
              {
                label: 'Events',
                value:
                  ledger?.events == null ? '—' : ledger.events.toLocaleString(),
              },
            ]}
          />
        </div>
      </ChartGuard>
    </ChartCard>
  )
}
