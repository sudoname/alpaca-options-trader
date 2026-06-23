import { RefreshCw } from 'lucide-react'
import { useDashboardStatus } from '@/lib/oracle/hooks'
import { useRefreshStore } from '@/lib/oracle/refresh-store'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'

const STATUS_STYLES: Record<string, string> = {
  live: 'bg-[#3fb950]/15 text-[#3fb950]',
  refreshing: 'bg-[#58a6ff]/15 text-[#58a6ff]',
  error: 'bg-[#f85149]/15 text-[#f85149]',
}

export function DashboardToolbar() {
  const { status, lastUpdated, refresh, isFetching } = useDashboardStatus()
  const autoRefresh = useRefreshStore((s) => s.autoRefresh)
  const setAutoRefresh = useRefreshStore((s) => s.setAutoRefresh)

  return (
    <div className='flex flex-wrap items-center gap-3'>
      <span
        className={cn(
          'rounded-full px-2.5 py-0.5 text-xs font-medium',
          STATUS_STYLES[status]
        )}
      >
        {status}
      </span>
      {lastUpdated ? (
        <span className='text-muted-foreground text-xs'>
          updated {lastUpdated.toLocaleTimeString()}
        </span>
      ) : null}
      <div className='flex items-center gap-2'>
        <Switch
          id='autorefresh'
          checked={autoRefresh}
          onCheckedChange={setAutoRefresh}
        />
        <Label htmlFor='autorefresh' className='text-muted-foreground text-xs'>
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
