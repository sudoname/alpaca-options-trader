// Opt-in training actions client. The /api/actions/* blueprint only exists
// when DASHBOARD_ENABLE_TRAINING is on; when it is off the Flask SPA fallback
// answers these paths with HTML, so any non-JSON response is mapped to a
// "disabled" status instead of throwing.

export type TrainState = 'idle' | 'running' | 'done' | 'failed' | 'interrupted'

export interface TrainStatus {
  enabled: boolean
  state: TrainState
  job_id: string | null
  model: string | null
  step: string | null
  steps: string[]
  started_at: string | null
  finished_at: string | null
  returncode: number | null
  error: string | null
  artifacts: Record<string, unknown>
  log_tail: string
  allowed_models: string[]
}

export const DISABLED_STATUS: TrainStatus = {
  enabled: false,
  state: 'idle',
  job_id: null,
  model: null,
  step: null,
  steps: [],
  started_at: null,
  finished_at: null,
  returncode: null,
  error: null,
  artifacts: {},
  log_tail: '',
  allowed_models: [],
}

export async function fetchTrainStatus(): Promise<TrainStatus> {
  try {
    const res = await fetch('/api/actions/train/status', {
      headers: { Accept: 'application/json' },
    })
    const ct = res.headers.get('content-type') ?? ''
    if (!res.ok || !ct.includes('application/json')) return DISABLED_STATUS
    const json = (await res.json()) as Partial<TrainStatus>
    if (typeof json?.state !== 'string') return DISABLED_STATUS
    return { ...DISABLED_STATUS, ...json, enabled: json.enabled !== false }
  } catch {
    return DISABLED_STATUS
  }
}

export interface StartTrainResult {
  ok: boolean
  status: number
  job_id?: string
  error?: string
}

export async function startTraining(model: string): Promise<StartTrainResult> {
  try {
    const res = await fetch('/api/actions/train', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({ model }),
    })
    const ct = res.headers.get('content-type') ?? ''
    const json = ct.includes('application/json')
      ? ((await res.json()) as { job_id?: string; error?: string })
      : null
    if (res.status === 202 && json?.job_id) {
      return { ok: true, status: 202, job_id: json.job_id }
    }
    return {
      ok: false,
      status: res.status,
      error: json?.error ?? `HTTP ${res.status}`,
    }
  } catch (e) {
    return { ok: false, status: 0, error: String(e) }
  }
}
