import { type ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { Card, CardContent } from '@/components/ui/card'
import { HudCorners } from './chart-card'

type StatCardProps = {
  label: string
  value: ReactNode
  /** Sign for coloring the value: positive -> green, negative -> red. */
  sign?: number | null
  sub?: ReactNode
  /** Explicit color override (e.g. Fear & Greed band color). */
  valueColor?: string
}

export function StatCard({
  label,
  value,
  sign,
  sub,
  valueColor,
}: StatCardProps) {
  const signClass =
    valueColor == null && typeof sign === 'number'
      ? sign > 0
        ? 'text-[#3fb950]'
        : sign < 0
          ? 'text-[#f85149]'
          : ''
      : ''
  return (
    <Card className='relative gap-0 py-0'>
      <HudCorners />
      <CardContent className='px-4 py-3'>
        <div className='text-muted-foreground font-mono text-[10px] font-medium tracking-[0.18em] uppercase'>
          {label}
        </div>
        <div
          className={cn('mt-1 font-mono text-2xl font-bold tabular-nums', signClass)}
          style={valueColor ? { color: valueColor } : undefined}
        >
          {value}
        </div>
        {sub != null && sub !== '' ? (
          <div className='text-muted-foreground mt-1 text-xs'>{sub}</div>
        ) : null}
      </CardContent>
    </Card>
  )
}
