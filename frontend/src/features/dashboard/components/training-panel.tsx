import { useEffect, useRef, useState } from 'react'
import { Play } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useStartTraining, useTrainStatus } from '@/lib/oracle/actions-hooks'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ChartCard } from './chart-card'

const MODEL_LABELS: Record<string, string> = {
  mlp: 'MLP (neural)',
  gbm: 'GBM (boosted trees)',
  logistic: 'Logistic (linear)',
}

const STATE_STYLES: Record<string, string> = {
  idle: 'text-muted-foreground border-border',
  running: 'text-[#ffb000] border-[#ffb000]/40',
  done: 'text-[#3fb950] border-[#3fb950]/40',
  failed: 'text-[#f85149] border-[#f85149]/40',
  interrupted: 'text-[#f85149] border-[#f85149]/40',
}

const GOVERNANCE_NOTE =
  'Produces an UNAPPROVED challenger only. Approval and champion promotion remain human, CLI-only.'

export function TrainingPanel() {
  const { data: status } = useTrainStatus()
  const start = useStartTraining()
  const [model, setModel] = useState('mlp')

  const enabled = status?.enabled === true
  const state = status?.state ?? 'idle'
  const running = state === 'running'
  const steps = status?.steps?.length
    ? status.steps
    : ['snapshot', 'dataset', 'train', 'register']
  const models = status?.allowed_models?.length
    ? status.allowed_models
    : ['mlp', 'gbm', 'logistic']

  // Keep the log pinned to the bottom while a job streams output.
  const logRef = useRef<HTMLPreElement>(null)
  const logTail = status?.log_tail ?? ''
  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [logTail])

  return (
    <ChartCard title='Model Training' footer={GOVERNANCE_NOTE}>
      {!enabled ? (
        <p className='text-muted-foreground font-mono text-xs'>
          Training is disabled on this server (DASHBOARD_ENABLE_TRAINING is
          off).
        </p>
      ) : (
        <div className='space-y-4'>
          <div className='flex flex-wrap items-center gap-3'>
            <Select value={model} onValueChange={setModel} disabled={running}>
              <SelectTrigger size='sm' className='w-48 font-mono text-xs'>
                <SelectValue placeholder='Model' />
              </SelectTrigger>
              <SelectContent>
                {models.map((m) => (
                  <SelectItem key={m} value={m} className='font-mono text-xs'>
                    {MODEL_LABELS[m] ?? m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              size='sm'
              variant='outline'
              className='font-mono text-xs tracking-[0.1em] uppercase'
              disabled={running || start.isPending}
              onClick={() => start.mutate(model)}
            >
              <Play className='size-3.5' />
              {running ? 'Running…' : 'Start training'}
            </Button>
            <span
              className={cn(
                'border px-2 py-0.5 font-mono text-[10px] tracking-[0.15em] uppercase',
                STATE_STYLES[state] ?? STATE_STYLES.idle
              )}
            >
              {state}
            </span>
            {status?.job_id && (
              <span className='text-muted-foreground font-mono text-[10px]'>
                {status.job_id}
              </span>
            )}
          </div>

          <div className='flex flex-wrap items-center gap-2'>
            {steps.map((s, i) => {
              const activeIdx = status?.step ? steps.indexOf(status.step) : -1
              const isActive = running && s === status?.step
              const isDone =
                state === 'done' || (activeIdx >= 0 && i < activeIdx)
              return (
                <span
                  key={s}
                  className={cn(
                    'font-mono text-[10px] tracking-[0.15em] uppercase',
                    isActive
                      ? 'text-[#ffb000]'
                      : isDone
                        ? 'text-[#3fb950]'
                        : 'text-muted-foreground'
                  )}
                >
                  {i > 0 && <span className='text-muted-foreground'>→ </span>}
                  {s}
                </span>
              )
            })}
          </div>

          {state === 'failed' && status?.error && (
            <p className='font-mono text-xs text-[#f85149]'>{status.error}</p>
          )}
          {state === 'interrupted' && (
            <p className='font-mono text-xs text-[#f85149]'>
              {status?.error ?? 'Job interrupted by a server restart.'}
            </p>
          )}

          {logTail && (
            <pre
              ref={logRef}
              className='border-border bg-background/60 max-h-64 overflow-auto border p-2 font-mono text-[10px] leading-relaxed whitespace-pre-wrap'
            >
              {logTail}
            </pre>
          )}
        </div>
      )}
    </ChartCard>
  )
}
