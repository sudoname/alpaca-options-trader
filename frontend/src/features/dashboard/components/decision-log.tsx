import { Fragment, useState } from 'react'
import {
  type AgentVote,
  type DecisionRow,
  type DecisionScope,
} from '@/lib/oracle/types'
import { fmtNum, signClass } from '@/lib/oracle/format'
import { useDecisions } from '@/lib/oracle/hooks'
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

const SCOPES: { value: DecisionScope; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'traded', label: 'Traded' },
  { value: 'blocked', label: 'Blocked' },
]

const ACTION_STYLES: Record<string, string> = {
  CALL: 'border-[#ffb000]/40 bg-[#ffb000]/10 text-[#ffb000]',
  PUT: 'border-[#f85149]/40 bg-[#f85149]/10 text-[#f85149]',
  SKIP: 'border-muted-foreground/30 bg-muted/30 text-muted-foreground',
}

// net_pnl_pct arrives in percent units (e.g. -12.5 == -12.5%).
function rawPct(v: number | null | undefined): string {
  return v == null || Number.isNaN(v) ? '—' : `${Number(v).toFixed(1)}%`
}

function ActionBadge({ action }: { action: string | null | undefined }) {
  const a = String(action ?? '—')
  return (
    <span
      className={cn(
        'inline-block border px-1.5 py-0.5 font-mono text-[10px] font-medium tracking-[0.1em] uppercase',
        ACTION_STYLES[a] ?? 'border-muted-foreground/30 text-muted-foreground'
      )}
    >
      {a}
    </span>
  )
}

function VoteBar({ v }: { v: AgentVote }) {
  const net = v.net == null || Number.isNaN(v.net) ? 0 : Number(v.net)
  const frac = Math.min(Math.abs(net), 1)
  const color = net > 0 ? '#3fb950' : net < 0 ? '#f85149' : '#8a8a8a'
  return (
    <div className='flex items-center gap-2 font-mono text-[11px]'>
      <span className='text-muted-foreground w-44 shrink-0 truncate'>
        {v.agent ?? '—'}
      </span>
      <div className='bg-muted/40 relative h-2 w-40 shrink-0'>
        {/* center line */}
        <div className='bg-foreground/20 absolute top-0 left-1/2 h-2 w-px' />
        <div
          className='absolute top-0 h-2'
          style={{
            width: `${(frac / 2) * 100}%`,
            background: color,
            left: net >= 0 ? '50%' : `${50 - (frac / 2) * 100}%`,
          }}
        />
      </div>
      <span className='tabular-nums' style={{ color }}>
        {net >= 0 ? '+' : ''}
        {net.toFixed(2)}
      </span>
      <span className='text-muted-foreground tabular-nums'>
        bull {fmtNum(v.bull, 2)} · bear {fmtNum(v.bear, 2)} · conf{' '}
        {fmtNum(v.conf, 2)}
      </span>
    </div>
  )
}

function ExpandedDetail({ r }: { r: DecisionRow }) {
  const votes = Array.isArray(r.agent_votes) ? r.agent_votes : null
  const g = r.gate
  return (
    <div className='space-y-3 px-2 py-3'>
      <div className='text-muted-foreground font-mono text-[11px]'>
        {r.symbol ?? '—'} · strat {r.strat ?? '—'} · qty {r.qty ?? '—'} ·
        conviction {fmtNum(r.conviction, 2)} · direction {r.direction ?? '—'} ·
        regime conf {fmtNum(r.regime_confidence, 2)}
      </div>
      {g != null ? (
        <div className='font-mono text-[11px]'>
          <span className='text-muted-foreground'>gate </span>
          <span className={g.veto ? 'text-[#f85149]' : 'text-[#3fb950]'}>
            {g.veto ? 'VETO' : 'pass'}
          </span>
          <span className='text-muted-foreground'>
            {' '}
            · q {fmtNum(g.q, 3)} · visits {g.visits ?? '—'} · conf{' '}
            {fmtNum(g.confidence, 2)}
            {g.reason ? ` · ${g.reason}` : ''}
          </span>
        </div>
      ) : null}
      {r.block_reason_full ? (
        <div className='border-[#f85149]/30 bg-[#f85149]/5 border px-2 py-1.5 font-mono text-[11px] break-words'>
          <span className='text-[#f85149]'>blocked: </span>
          {r.block_reason_full}
        </div>
      ) : null}
      {votes && votes.length > 0 ? (
        <div className='space-y-1'>
          <div className='text-muted-foreground font-mono text-[10px] tracking-[0.15em] uppercase'>
            Agent votes (net, bearish ← → bullish)
          </div>
          {votes.map((v, i) => (
            <VoteBar key={v.agent ?? i} v={v} />
          ))}
        </div>
      ) : (
        <div className='text-muted-foreground font-mono text-[11px]'>
          no agent votes recorded for this decision
        </div>
      )}
    </div>
  )
}

export function DecisionLog() {
  const [scope, setScope] = useState<DecisionScope>('all')
  const { data, isLoading } = useDecisions(scope)
  const rows = Array.isArray(data?.rows) ? data!.rows! : []
  const [openId, setOpenId] = useState<string | null>(null)

  const toggle = (id: string | null | undefined) => {
    if (!id) return
    setOpenId((prev) => (prev === id ? null : id))
  }

  return (
    <ChartCard
      title='Decision Log'
      footer={
        data?.total != null
          ? `${rows.length} of ${data.total.toLocaleString()} decisions (newest first) · SKIP P/L is counterfactual (CF) — what the trade would have done, not realized money · click a row for agent votes`
          : 'click a row for per-agent votes and gate detail'
      }
    >
      <div className='mb-3 flex items-center gap-1'>
        {SCOPES.map((s) => (
          <button
            key={s.value}
            type='button'
            onClick={() => {
              setScope(s.value)
              setOpenId(null)
            }}
            className={cn(
              'border px-2 py-0.5 font-mono text-[11px] tracking-[0.1em] uppercase transition-colors',
              scope === s.value
                ? 'border-[#ffb000]/60 bg-[#ffb000]/10 text-[#ffb000]'
                : 'border-border text-muted-foreground hover:text-foreground'
            )}
          >
            {s.label}
          </button>
        ))}
      </div>
      <ChartGuard
        data={data}
        isLoading={isLoading}
        empty={!isLoading && rows.length === 0}
        minHeight={160}
      >
        <div className='max-h-[32rem] overflow-x-auto overflow-y-auto'>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Time</TableHead>
                <TableHead>Underlying</TableHead>
                <TableHead>Rule</TableHead>
                <TableHead>Chosen</TableHead>
                <TableHead>Mode</TableHead>
                <TableHead>Outcome</TableHead>
                <TableHead className='text-end'>P/L %</TableHead>
                <TableHead className='text-end'>Hold d</TableHead>
                <TableHead>Regime</TableHead>
                <TableHead className='w-6' />
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r, i) => {
                const id = r.decision_id ?? `row-${i}`
                const expanded = openId === id
                return (
                  <Fragment key={id}>
                    <TableRow
                      onClick={() => toggle(id)}
                      className='cursor-pointer'
                    >
                      <TableCell className='font-mono text-xs tabular-nums'>
                        {String(r.as_of ?? '—').slice(0, 16).replace('T', ' ')}
                      </TableCell>
                      <TableCell>{String(r.underlying ?? '—')}</TableCell>
                      <TableCell className='font-mono text-xs'>
                        {String(r.rule_action ?? '—')}
                        <span className='text-muted-foreground tabular-nums'>
                          {r.rule_confidence == null
                            ? ''
                            : ` ${Number(r.rule_confidence).toFixed(0)}`}
                        </span>
                      </TableCell>
                      <TableCell>
                        <ActionBadge action={r.chosen_action} />
                      </TableCell>
                      <TableCell className='font-mono text-xs'>
                        {String(r.mode ?? '—')}
                      </TableCell>
                      <TableCell className='font-mono text-xs'>
                        {String(r.outcome ?? '—')}
                      </TableCell>
                      <TableCell
                        className={cn(
                          'text-end font-mono text-xs tabular-nums',
                          signClass(r.net_pnl_pct)
                        )}
                      >
                        {rawPct(r.net_pnl_pct)}
                        {r.pnl_is_counterfactual && r.net_pnl_pct != null ? (
                          <span className='text-muted-foreground'> CF</span>
                        ) : null}
                      </TableCell>
                      <TableCell className='text-end font-mono text-xs tabular-nums'>
                        {r.hold_days ?? '—'}
                      </TableCell>
                      <TableCell className='font-mono text-xs'>
                        {String(r.regime_label ?? '—')}
                      </TableCell>
                      <TableCell className='text-muted-foreground font-mono text-xs'>
                        {expanded ? '▾' : '▸'}
                      </TableCell>
                    </TableRow>
                    {expanded ? (
                      <TableRow className='bg-muted/20 hover:bg-muted/20'>
                        <TableCell colSpan={10}>
                          <ExpandedDetail r={r} />
                        </TableCell>
                      </TableRow>
                    ) : null}
                  </Fragment>
                )
              })}
            </TableBody>
          </Table>
        </div>
      </ChartGuard>
    </ChartCard>
  )
}
