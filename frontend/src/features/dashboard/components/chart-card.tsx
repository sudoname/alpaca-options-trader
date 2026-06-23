import { type ReactNode } from 'react'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'

type ChartCardProps = {
  title: string
  children: ReactNode
  footer?: ReactNode
}

export function ChartCard({ title, children, footer }: ChartCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {children}
        {footer != null && footer !== '' ? (
          <div className='text-muted-foreground mt-3 text-xs'>{footer}</div>
        ) : null}
      </CardContent>
    </Card>
  )
}

// Shared dark tooltip style for Recharts.
export const tooltipStyle = {
  contentStyle: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 8,
    color: '#c9d1d9',
    fontSize: 12,
  },
  labelStyle: { color: '#8b949e' },
  itemStyle: { color: '#c9d1d9' },
} as const
