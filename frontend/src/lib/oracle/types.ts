// TypeScript shapes for the read-only Oracle JSON API. Field aliases mirror the
// tolerant reads in the legacy dashboard/app.js (e.g. `predicted ?? mid ?? p`).

export type Verdict = 'OK' | 'INSUFFICIENT_DATA' | 'ERROR' | string

export type ApiEnvelope = {
  verdict?: Verdict
  error?: string
}

export type Kpis = ApiEnvelope & {
  realized_total?: number | null
  today_realized?: number | null
  win_rate?: number | null
  open_positions?: number | null
  closed_trades?: number | null
  closed_green_sum?: number | null
  closed_red_sum?: number | null
}

export type Position = {
  symbol?: string
  underlying?: string
  quantity?: number | null
  entry_price?: number | null
  current_price?: number | null
  unrealized_pl?: number | null
  unrealized_plpc?: number | null
  entry_time?: string
  expected_value?: number | null
  probability_of_profit?: number | null
}

export type Positions = ApiEnvelope & {
  positions?: Position[]
  green_sum?: number | null
  red_sum?: number | null
  marks_available?: boolean
}

export type EpisodeStats = {
  total?: number
  completed?: number
  win_rate?: number | null
  mean_net_pnl_pct?: number | null
}

export type Episodes = ApiEnvelope & {
  chosen_action_counts?: Record<string, number>
  stats?: EpisodeStats
}

export type Regime = ApiEnvelope & {
  confidence?: number | null
  label?: string
  reasons?: string[]
}

export type SentimentComponent = {
  name?: string
  available?: boolean
  score?: number | null
}

export type Sentiment = ApiEnvelope & {
  score?: number | null
  classification?: string
  source?: string
  cnn_score?: number | null
  custom_score?: number | null
  from_cache?: boolean
  components?: SentimentComponent[]
}

export type Probability = ApiEnvelope & {
  brier?: number | null
  baseline_brier?: number | null
  skill?: number | null
  sample_size?: number | null
}

export type CalibrationBucket = {
  predicted?: number
  mid?: number
  p?: number
  bucket?: number
  realized?: number
  actual?: number
  win_rate?: number
}

export type CalibrationPop = ApiEnvelope & {
  buckets?: CalibrationBucket[]
}

export type Agent = {
  agent?: string
  hit_rate?: number | null
}

export type Agents = ApiEnvelope & {
  agents?: Agent[]
  base_win_rate?: number | null
}

export type Feature = {
  agent?: string
  importance?: number | null
}

export type FeatureImportance = ApiEnvelope & {
  features?: Feature[]
}

export type Weights = ApiEnvelope & {
  current?: Record<string, number>
  snapshots?: number | null
  drift?: number | null
}

export type EvBucketRaw = {
  label?: string
  bucket?: string
  range?: string
  win_rate?: number
  winrate?: number
  profit_factor?: number
  pf?: number
}

export type EvAttribution = ApiEnvelope & {
  ev_buckets?: Record<string, EvBucketRaw> | EvBucketRaw[]
  buckets?: Record<string, EvBucketRaw> | EvBucketRaw[]
}

export type RegimePerfRow = {
  regime?: string
  label?: string
  trades?: number
  n?: number
  win_rate?: number | null
  average_pnl?: number | null
  avg_pnl?: number | null
}

export type RegimePerformance = ApiEnvelope & {
  regimes?: RegimePerfRow[]
}

export type HypothesisRow = {
  hypothesis_name?: string
  conclusion?: string
  confidence?: number | null
  win_rate_a?: number | null
  win_rate_b?: number | null
  effect_size?: number | null
}

export type Hypotheses = ApiEnvelope & {
  hypotheses?: HypothesisRow[]
}

export type ExplainVote = {
  agent?: string
  name?: string
  bullish_score?: number | null
  bull?: number | null
  bearish_score?: number | null
  bear?: number | null
  confidence?: number | null
}

export type Explain = ApiEnvelope & {
  probability?: {
    call?: number | null
    p_call?: number | null
    put?: number | null
    p_put?: number | null
    no_trade?: number | null
    p_no_trade?: number | null
  }
  votes?: ExplainVote[]
  explanation?: { summary_str?: string; top_reasons?: string[] } | string
  summary_str?: string
}
