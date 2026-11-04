import { type ReactNode } from 'react'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'

// HUD corner brackets: four hairline L-marks on the card corners.
export function HudCorners() {
  return (
    <span aria-hidden className='pointer-events-none absolute inset-0'>
      <span className='border-foreground/40 absolute -top-px -left-px size-2 border-t border-l' />
      <span className='border-foreground/40 absolute -top-px -right-px size-2 border-t border-r' />
      <span className='border-foreground/40 absolute -bottom-px -left-px size-2 border-b border-l' />
      <span className='border-foreground/40 absolute -right-px -bottom-px size-2 border-r border-b' />
    </span>
  )
}

type ChartCardProps = {
  title: string
  children: ReactNode
  footer?: ReactNode
}

export function ChartCard({ title, children, footer }: ChartCardProps) {
  return (
    <Card className='relative gap-4 py-4'>
      <HudCorners />
      <CardHeader>
        <CardTitle className='hud-index text-muted-foreground font-mono text-xs font-medium tracking-[0.18em] uppercase'>
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {children}
        {footer != null && footer !== '' ? (
          <div className='text-muted-foreground mt-3 font-mono text-[11px]'>
            {footer}
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}

// Shared dark tooltip style for Recharts.
export const tooltipStyle = {
  contentStyle: {
    background: '#0e0e0e',
    border: '1px solid #1f1f1f',
    borderRadius: 0,
    color: '#e6e6e6',
    fontSize: 11,
    fontFamily: "'JetBrains Mono', ui-monospace, monospace",
  },
  labelStyle: { color: '#8a8a8a' },
  itemStyle: { color: '#e6e6e6' },
} as const
