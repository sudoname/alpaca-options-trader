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

export type PutTimeStopBoundary = {
  symbol?: string
  underlying?: string
  entry_time?: string
  entry_price?: number | null
  cap_days?: number | null
  hold_days_at_boundary?: number | null
  boundary_mark?: number | null
  boundary_pnl_pct?: number | null
  resolved?: boolean
  actual_exit_pnl_pct?: number | null
  shadow_delta_pct?: number | null
  held_extra_days?: number | null
}

export type PutTimeStop = ApiEnvelope & {
  cap_days?: number | null
  n_boundaries?: number | null
  n_resolved?: number | null
  mean_shadow_delta_pct?: number | null
  median_shadow_delta_pct?: number | null
  count_helped?: number | null
  count_hurt?: number | null
  boundaries?: PutTimeStopBoundary[]
}

export type ShadowVariantStats = {
  variant?: string
  n_candidates?: number | null
  n_accepted?: number | null
  n_resolved?: number | null
  hit_rate?: number | null
  avg_pnl_pct?: number | null
  median_pnl_pct?: number | null
  reject_codes?: Record<string, number>
  by_direction?: Record<
    string,
    { n?: number | null; hit_rate?: number | null; avg_pnl_pct?: number | null }
  >
}

export type ShadowCandidateRow = {
  at?: string
  variant?: string
  underlying?: string
  direction?: string
  score?: number | null
  underlying_price?: number | null
  accepted?: boolean
  reject_code?: string | null
  pnl_pct?: number | null
}

export type ShadowResearch = ApiEnvelope & {
  n_candidates?: number | null
  n_accepted?: number | null
  variants?: ShadowVariantStats[]
  recent?: ShadowCandidateRow[]
}

export type NodeHealth = 'online' | 'stale' | 'offline' | string

export type NodeStatus = ApiEnvelope & {
  scheduler?: {
    status?: NodeHealth
    mode?: string | null
    is_open?: boolean | null
    heartbeat_age_s?: number | null
    trades_today?: number | null
    max_new_trades_per_day?: number | null
    scan_interval_min?: number | null
    symbols_watched?: number | null
    pdt_remaining?: number | null
    pdt_can_trade?: boolean | null
  }
  trades?: {
    status?: NodeHealth
    last_trade_at?: string | null
    open_count?: number | null
  }
  ledger?: {
    status?: NodeHealth
    snapshots?: number | null
    events?: number | null
  }
}

export type TradeLogRow = {
  at?: string | null
  event?: 'OPEN' | 'CLOSE' | string
  symbol?: string | null
  underlying?: string | null
  opt_type?: string | null
  qty?: number | null
  price?: number | null
  pnl?: number | null
  pnl_percent?: number | null
  outcome?: string | null
  regime?: string | null
}

export type TradeLog = ApiEnvelope & {
  rows?: TradeLogRow[]
  count?: number | null
  total?: number | null
}

// Shadow options trader (spy-ml monitor -> Alpaca paper): shadow_pnl.json
export type ShadowPnlBook = {
  today_pnl?: number | null
  total_pnl?: number | null
  trades?: number | null
  wins?: number | null
  open?: number | null
  skipped?: number | null
}

export type ShadowPnlOpenTrade = {
  book?: string
  entry_time?: string | null
  legs?: string[]
  entry_cost?: number | null
  last_mid?: number | null
  peak?: number | null
  tp_gain?: number | null
  trail_pct?: number | null
}

export type ShadowPnlClosedTrade = {
  book?: string
  day?: string | null
  pnl?: number | null
  exit_reason?: string | null
  entry_time?: string | null
  exit_time?: string | null
  legs?: string[]
}

export type ShadowPnl = ApiEnvelope & {
  updated_at?: string | null
  today_pnl?: number | null
  total_pnl?: number | null
  books?: Record<string, ShadowPnlBook>
  open_trades?: ShadowPnlOpenTrade[]
  last_trades?: ShadowPnlClosedTrade[]
  age_s?: number | null
  stale?: boolean
}

// Oracle4 research pipeline (registry / training / promotion / shadow): /api/oracle4
export type Oracle4RegistryModel = {
  model_id?: string | null
  kind?: string | null
  role?: string | null
  approval_state?: string | null
  approved_by?: string | null
  registered_utc?: string | null
  checksum12?: string | null
}

export type Oracle4Gate = {
  name?: string | null
  status?: string | null
  observed?: string | number | null
  requirement?: string | null
}

export type Oracle4TrainedModel = {
  file?: string | null
  model?: string | null
  horizon?: string | null
  label_type?: string | null
  device?: string | null
  fingerprint12?: string | null
  dataset_hash12?: string | null
  n?: number | null
  accuracy?: number | null
  log_loss?: number | null
  brier?: number | null
  ece?: number | null
  baseline_accuracy?: number | null
  baseline_log_loss?: number | null
  n_folds?: number | null
  holdout?: string[] | null
}

export type Oracle4 = ApiEnvelope & {
  hint?: string | null
  registry?: {
    status?: NodeHealth
    champion_id?: string | null
    n_models?: number | null
    models?: Oracle4RegistryModel[]
  }
  promotion?: {
    status?: NodeHealth
    file?: string | null
    verdict?: string | null
    verdict_reason?: string | null
    variant?: string | null
    baseline?: string | null
    generated_utc?: string | null
    n_pass?: number | null
    n_fail?: number | null
    n_insufficient?: number | null
    gates?: Oracle4Gate[]
  }
  training?: {
    status?: NodeHealth
    models?: Oracle4TrainedModel[]
  }
  shadow?: {
    status?: NodeHealth
    n_scans?: number | null
    n_trade?: number | null
    n_abstain?: number | null
    last_scan_utc?: string | null
    last_scan_age_s?: number | null
    candidates_by_book?: Record<string, number> | null
    paired_groups?: number | null
    outcome_states?: Record<string, number> | null
  }
  forward?: {
    status?: NodeHealth
    books?: Oracle4ForwardBook[]
  }
}

export type Oracle4ForwardBook = {
  file?: string | null
  book?: string | null
  model_id?: string | null
  generated_utc?: string | null
  window?: { start?: string | null; end?: string | null } | null
  forward_sessions?: number | null
  resolved_opportunities?: number | null
  actual_fills?: number | null
  live_fill_based?: boolean | null
  resolution_rate?: number | null
  n_unresolved?: number | null
  validation?: string | null
  validation_errors?: number | null
}
