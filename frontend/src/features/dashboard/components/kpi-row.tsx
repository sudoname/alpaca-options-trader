import { fgColor } from '@/lib/oracle/theme'
import { fmtMoney, fmtPct, isUsable } from '@/lib/oracle/format'
import { useKpis, usePositions, useSentiment } from '@/lib/oracle/hooks'
import { StatCard } from './stat-card'

function GreenRedValue({
  green,
  red,
}: {
  green: number | null | undefined
  red: number | null | undefined
}) {
  return (
    <span>
      <span className='text-[#3fb950]'>{fmtMoney(green)}</span>
      <span className='text-muted-foreground'> / </span>
      <span className='text-[#f85149]'>{fmtMoney(red)}</span>
    </span>
  )
}

function greenRedShare(
  green: number | null | undefined,
  red: number | null | undefined
): { g: number; r: number } | null {
  const g = Math.abs(green || 0)
  const r = Math.abs(red || 0)
  const tot = g + r
  if (tot <= 0) return null
  return { g: g / tot, r: r / tot }
}

function GreenRedSub({
  green,
  red,
}: {
  green: number | null | undefined
  red: number | null | undefined
}) {
  const sh = greenRedShare(green, red)
  if (!sh) return <span className='text-muted-foreground'>—</span>
  return (
    <span>
      <span className='text-[#3fb950]'>{(sh.g * 100).toFixed(0)}%</span>
      <span className='text-muted-foreground'> / </span>
      <span className='text-[#f85149]'>{(sh.r * 100).toFixed(0)}%</span>
    </span>
  )
}

const DASH = '—'

export function KpiRow() {
  const kpis = useKpis()
  const positions = usePositions()
  const sentiment = useSentiment()

  const k = kpis.data
  const kUsable = isUsable(k)
  const p = positions.data
  const pUsable = isUsable(p)
  const s = sentiment.data
  const sUsable = isUsable(s) && s?.score != null

  // OPEN green/red needs live marks; CLOSED comes from realized KPIs.
  const openReady = pUsable && !!p?.marks_available
  const openFallback =
    p && p.marks_available === false ? 'no live marks' : DASH
  const closedReady = kUsable

  return (
    <div className='grid grid-cols-2 gap-4 md:grid-cols-4'>
      <StatCard
        label='Realized P/L'
        value={kUsable ? fmtMoney(k?.realized_total) : DASH}
        sign={kUsable ? k?.realized_total : null}
      />
      <StatCard
        label="Today's P/L"
        value={kUsable ? fmtMoney(k?.today_realized) : DASH}
        sign={kUsable ? k?.today_realized : null}
      />
      <StatCard
        label='Win rate'
        value={kUsable && k?.win_rate != null ? fmtPct(k.win_rate) : DASH}
      />
      <StatCard
        label='Open positions'
        value={kUsable && k?.open_positions != null ? k.open_positions : DASH}
      />
      <StatCard
        label='Closed trades'
        value={kUsable && k?.closed_trades != null ? k.closed_trades : DASH}
      />
      <StatCard
        label='Open Green / Red'
        value={
          openReady ? (
            <GreenRedValue green={p?.green_sum} red={p?.red_sum} />
          ) : (
            <span className='text-muted-foreground'>{openFallback}</span>
          )
        }
        sub={
          openReady ? (
            <GreenRedSub green={p?.green_sum} red={p?.red_sum} />
          ) : null
        }
      />
      <StatCard
        label='Closed Green / Red'
        value={
          closedReady ? (
            <GreenRedValue
              green={k?.closed_green_sum}
              red={k?.closed_red_sum}
            />
          ) : (
            <span className='text-muted-foreground'>{DASH}</span>
          )
        }
        sub={
          closedReady ? (
            <GreenRedSub green={k?.closed_green_sum} red={k?.closed_red_sum} />
          ) : null
        }
      />
      <StatCard
        label='Fear & Greed'
        value={sUsable ? Math.round(Number(s?.score)) : DASH}
        valueColor={sUsable ? fgColor(Number(s?.score)) : undefined}
        sub={sUsable ? s?.classification || '' : null}
      />
    </div>
  )
}
