import { useEffect, useRef } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import {
  fetchTrainStatus,
  startTraining,
  type TrainStatus,
} from './actions-api'

// Deliberately NOT under the ['oracle'] key: the training poll must not flip
// the read-only status pill to "refreshing" every few seconds.
const TRAIN_STATUS_KEY = ['actions', 'train-status'] as const

const POLL_RUNNING_MS = 3_000
const POLL_IDLE_MS = 15_000
const POLL_DISABLED_MS = 60_000

export function useTrainStatus() {
  const queryClient = useQueryClient()
  const query = useQuery<TrainStatus>({
    queryKey: TRAIN_STATUS_KEY,
    queryFn: fetchTrainStatus,
    refetchInterval: (q) => {
      const data = q.state.data
      if (!data || !data.enabled) return POLL_DISABLED_MS
      return data.state === 'running' ? POLL_RUNNING_MS : POLL_IDLE_MS
    },
    refetchIntervalInBackground: false,
  })

  // Toast + refresh the Oracle4 card when a run we watched finishes.
  const prevState = useRef<TrainStatus['state'] | null>(null)
  const state = query.data?.state ?? null
  useEffect(() => {
    if (prevState.current === 'running' && state && state !== 'running') {
      if (state === 'done') {
        toast.success('Training finished', {
          description:
            'Registered as an UNAPPROVED challenger. Promotion stays human, CLI-only.',
        })
      } else {
        toast.error(`Training ${state}`, {
          description: query.data?.error ?? undefined,
        })
      }
      queryClient.invalidateQueries({ queryKey: ['oracle', 'oracle4'] })
    }
    prevState.current = state
  }, [state, query.data?.error, queryClient])

  return query
}

export function useStartTraining() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (model: string) => startTraining(model),
    onSuccess: (res) => {
      if (res.ok) {
        toast.success('Training started', { description: `Job ${res.job_id}` })
      } else if (res.status === 409) {
        toast.warning('A training job is already running')
      } else {
        toast.error('Could not start training', {
          description: res.error,
        })
      }
      queryClient.invalidateQueries({ queryKey: TRAIN_STATUS_KEY })
    },
  })
}
