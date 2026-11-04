import { RefreshCw } from 'lucide-react'
import { useDashboardStatus } from '@/lib/oracle/hooks'
import { useRefreshStore } from '@/lib/oracle/refresh-store'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'

const STATUS_STYLES: Record<string, string> = {
  live: 'border-[#3fb950]/40 bg-[#3fb950]/10 text-[#3fb950]',
  refreshing: 'border-[#ffb000]/40 bg-[#ffb000]/10 text-[#ffb000]',
  error: 'border-[#f85149]/40 bg-[#f85149]/10 text-[#f85149]',
}

export function DashboardToolbar() {
  const { status, lastUpdated, refresh, isFetching } = useDashboardStatus()
  const autoRefresh = useRefreshStore((s) => s.autoRefresh)
  const setAutoRefresh = useRefreshStore((s) => s.setAutoRefresh)

  return (
    <div className='flex flex-wrap items-center gap-3'>
      <span
        className={cn(
          'flex items-center gap-1.5 border px-2 py-0.5 font-mono text-[10px] font-medium tracking-[0.15em] uppercase',
          STATUS_STYLES[status]
        )}
      >
        <span aria-hidden className='size-1.5 animate-pulse bg-current' />
        {status}
      </span>
      {lastUpdated ? (
        <span className='text-muted-foreground font-mono text-[10px] tracking-[0.1em] uppercase'>
          updated {lastUpdated.toLocaleTimeString()}
        </span>
      ) : null}
      <div className='flex items-center gap-2'>
        <Switch
          id='autorefresh'
          checked={autoRefresh}
          onCheckedChange={setAutoRefresh}
        />
        <Label
          htmlFor='autorefresh'
          className='text-muted-foreground font-mono text-[10px] tracking-[0.1em] uppercase'
        >
          Auto-refresh
        </Label>
      </div>
      <Button
        variant='outline'
        size='sm'
        onClick={() => refresh()}
        disabled={isFetching}
      >
        <RefreshCw className={cn('size-4', isFetching && 'animate-spin')} />
        Refresh
      </Button>
    </div>
  )
}
