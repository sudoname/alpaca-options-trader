import { type NodeHealth, type Oracle4Gate } from '@/lib/oracle/types'
import { useOracle4 } from '@/lib/oracle/hooks'
import { cn } from '@/lib/utils'
import { ChartCard } from './chart-card'
import { ChartGuard } from './chart-guard'

const HEALTH_STYLES: Record<string, string> = {
  online: 'border-[#3fb950]/40 bg-[#3fb950]/10 text-[#3fb950]',
  stale: 'border-[#ffb000]/40 bg-[#ffb000]/10 text-[#ffb000]',
  offline: 'border-[#f85149]/40 bg-[#f85149]/10 text-[#f85149]',
}

// Promotion verdicts + gate statuses reuse the same three-color palette.
const VERDICT_STYLES: Record<string, string> = {
  ELIGIBLE_PENDING_HUMAN_REVIEW: HEALTH_STYLES.online,
  DO_NOT_PROMOTE: HEALTH_STYLES.offline,
  INSUFFICIENT_EVIDENCE: HEALTH_STYLES.stale,
}

const GATE_STYLES: Record<string, string> = {
  PASS: HEALTH_STYLES.online,
  FAIL: HEALTH_STYLES.offline,
  INSUFFICIENT: HEALTH_STYLES.stale,
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

function VerdictTag({ verdict }: { verdict?: string | null }) {
  if (!verdict) return null
  return (
    <span
      className={cn(
        'border px-2 py-0.5 font-mono text-[10px] font-medium tracking-[0.15em] uppercase',
        VERDICT_STYLES[verdict] ?? HEALTH_STYLES.stale
      )}
    >
      {verdict.split('_').join(' ')}
    </span>
  )
}

function SectionRow({
  name,
  status,
  tag,
  facts,
}: {
  name: string
  status?: NodeHealth
  tag?: React.ReactNode
  facts: Array<{ label: string; value: string }>
}) {
  return (
    <div className='border-border flex flex-col gap-2 border p-3'>
      <div className='flex items-center justify-between gap-2'>
        <span className='font-mono text-xs font-medium tracking-[0.18em] uppercase'>
          {name}
        </span>
        <div className='flex items-center gap-2'>
          {tag}
          <HealthTag status={status} />
        </div>
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

function GatesStrip({ gates }: { gates?: Oracle4Gate[] }) {
  if (!gates?.length) return null
  return (
    <div className='flex flex-wrap gap-1.5'>
      {gates.map((g, i) => (
        <span
          key={`${g.name ?? 'gate'}-${i}`}
          title={`${g.name ?? ''}: observed ${g.observed ?? '—'} | ${g.requirement ?? ''}`}
          className={cn(
            'border px-1.5 py-0.5 font-mono text-[9px] tracking-[0.1em] uppercase',
            GATE_STYLES[String(g.status ?? '')] ?? HEALTH_STYLES.stale
          )}
        >
          {g.name ?? '—'}
        </span>
      ))}
    </div>
  )
}

function num(v: number | null | undefined, digits = 3): string {
  return v == null || Number.isNaN(v) ? '—' : v.toFixed(digits)
}

function ts(v: string | null | undefined): string {
  return v ? String(v).slice(0, 16).replace('T', ' ') : '—'
}

export function Oracle4Status() {
  const { data, isLoading } = useOracle4()
  const reg = data?.registry
  const promo = data?.promotion
  const training = data?.training
  const shadow = data?.shadow

  const champion = reg?.champion_id ?? null
  const approved =
    reg?.models?.filter((m) => m.approval_state === 'approved').length ?? null

  return (
    <ChartCard
      title='Oracle4 Research'
      footer='Challenger pipeline — observe-only; promotion always requires a human'
    >
      <ChartGuard data={data} isLoading={isLoading} minHeight={120}>
        <div className='flex flex-col gap-3'>
          <div className='grid grid-cols-1 gap-3 md:grid-cols-2'>
            <SectionRow
              name='Registry'
              status={reg?.status}
              facts={[
                { label: 'Champion', value: champion ?? 'none' },
                {
                  label: 'Models',
                  value: reg?.n_models == null ? '—' : String(reg.n_models),
                },
                {
                  label: 'Approved',
                  value: approved == null || reg?.n_models == null
                    ? '—'
                    : `${approved} / ${reg.n_models}`,
                },
                {
                  label: 'Latest',
                  value: reg?.models?.length
                    ? String(reg.models[reg.models.length - 1].model_id ?? '—')
                    : '—',
                },
              ]}
            />
            <SectionRow
              name='Shadow'
              status={shadow?.status}
              facts={[
                {
                  label: 'Scans',
                  value:
                    shadow?.n_scans == null ? '—' : String(shadow.n_scans),
                },
                {
                  label: 'Trade / abstain',
                  value:
                    shadow?.n_scans == null
                      ? '—'
                      : `${shadow.n_trade ?? 0} / ${shadow.n_abstain ?? 0}`,
                },
                { label: 'Last scan', value: ts(shadow?.last_scan_utc) },
                { label: 'Orders sent', value: '0 (by design)' },
              ]}
            />
          </div>

          {training?.models?.length ? (
            <div className='border-border flex flex-col gap-2 border p-3'>
              <div className='flex items-center justify-between gap-2'>
                <span className='font-mono text-xs font-medium tracking-[0.18em] uppercase'>
                  Trained models (OOF, purged walk-forward)
                </span>
                <HealthTag status={training?.status} />
              </div>
              <div className='overflow-x-auto'>
                <table className='w-full font-mono text-xs'>
                  <thead>
                    <tr className='text-muted-foreground text-[10px] tracking-[0.1em] uppercase'>
                      <th className='pr-3 text-left'>Model</th>
                      <th className='pr-3 text-left'>Device</th>
                      <th className='pr-3 text-right'>N</th>
                      <th className='pr-3 text-right'>Acc</th>
                      <th className='pr-3 text-right'>LogLoss</th>
                      <th className='pr-3 text-right'>Brier</th>
                      <th className='pr-3 text-right'>ECE</th>
                      <th className='pr-3 text-right'>Base acc</th>
                      <th className='text-left'>Fingerprint</th>
                    </tr>
                  </thead>
                  <tbody>
                    {training.models.map((m, i) => (
                      <tr key={`${m.file ?? m.model ?? 'm'}-${i}`}>
                        <td className='pr-3 text-left'>
                          {m.model ?? '—'}
                          {m.horizon ? ` ${m.horizon}` : ''}
                        </td>
                        <td className='pr-3 text-left'>{m.device ?? '—'}</td>
                        <td className='pr-3 text-right tabular-nums'>
                          {m.n == null ? '—' : m.n.toLocaleString()}
                        </td>
                        <td className='pr-3 text-right tabular-nums'>
                          {num(m.accuracy)}
                        </td>
                        <td className='pr-3 text-right tabular-nums'>
                          {num(m.log_loss)}
                        </td>
                        <td className='pr-3 text-right tabular-nums'>
                          {num(m.brier)}
                        </td>
                        <td className='pr-3 text-right tabular-nums'>
                          {num(m.ece)}
                        </td>
                        <td className='pr-3 text-right tabular-nums'>
                          {num(m.baseline_accuracy)}
                        </td>
                        <td className='text-muted-foreground text-left'>
                          {m.fingerprint12 ?? '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}

          <div className='border-border flex flex-col gap-2 border p-3'>
            <div className='flex items-center justify-between gap-2'>
              <span className='font-mono text-xs font-medium tracking-[0.18em] uppercase'>
                Promotion
              </span>
              <div className='flex items-center gap-2'>
                <VerdictTag verdict={promo?.verdict} />
                <HealthTag status={promo?.status} />
              </div>
            </div>
            <div className='grid grid-cols-2 gap-x-4 gap-y-1 md:grid-cols-4'>
              {[
                { label: 'Variant', value: promo?.variant ?? '—' },
                { label: 'Baseline', value: promo?.baseline ?? '—' },
                {
                  label: 'Gates P/F/I',
                  value:
                    promo?.n_pass == null
                      ? '—'
                      : `${promo.n_pass} / ${promo.n_fail ?? 0} / ${promo.n_insufficient ?? 0}`,
                },
                { label: 'Generated', value: ts(promo?.generated_utc) },
              ].map((f) => (
                <div
                  key={f.label}
                  className='flex items-baseline justify-between gap-2'
                >
                  <span className='text-muted-foreground font-mono text-[10px] tracking-[0.1em] uppercase'>
                    {f.label}
                  </span>
                  <span className='font-mono text-xs tabular-nums'>
                    {f.value}
                  </span>
                </div>
              ))}
            </div>
            {promo?.verdict_reason ? (
              <p className='text-muted-foreground font-mono text-[10px]'>
                {promo.verdict_reason}
              </p>
            ) : null}
            <GatesStrip gates={promo?.gates} />
          </div>

          {data?.hint ? (
            <p className='text-muted-foreground font-mono text-[10px] tracking-[0.1em] uppercase'>
              {data.hint}
            </p>
          ) : null}
        </div>
      </ChartGuard>
    </ChartCard>
  )
}
