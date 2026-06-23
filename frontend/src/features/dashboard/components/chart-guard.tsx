import { type ReactNode } from 'react'
import { type ApiEnvelope } from '@/lib/oracle/types'
import { badgeLabel, isUsable } from '@/lib/oracle/format'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

type ChartGuardProps = {
  data: ApiEnvelope | undefined
  isLoading?: boolean
  /** Treat a usable-but-empty payload as "no data". */
  empty?: boolean
  minHeight?: number
  children: ReactNode
}

// Preserves the legacy degradation UX: render a badge placeholder when a
// widget's source returns INSUFFICIENT_DATA / ERROR (or has no rows), else the
// chart/table.
export function ChartGuard({
  data,
  isLoading,
  empty,
  minHeight = 120,
  children,
}: ChartGuardProps) {
  const center = 'flex items-center justify-center'
  if (isLoading && !data) {
    return (
      <div className={cn(center)} style={{ minHeight }}>
        <Skeleton className='h-24 w-full' />
      </div>
    )
  }

  if (!isUsable(data)) {
    const label = badgeLabel(data)
    const isError = data?.verdict === 'ERROR' || !data
    return (
      <div className={cn(center)} style={{ minHeight }}>
        <Badge variant={isError ? 'destructive' : 'secondary'}>{label}</Badge>
      </div>
    )
  }

  if (empty) {
    return (
      <div className={cn(center)} style={{ minHeight }}>
        <Badge variant='secondary'>no data</Badge>
      </div>
    )
  }

  return <>{children}</>
}
