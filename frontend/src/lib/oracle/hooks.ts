import { useEffect, useRef, useState } from 'react'
import {
  useIsFetching,
  useQuery,
  useQueryClient,
  type UseQueryResult,
} from '@tanstack/react-query'
import { apiGet } from './api'
import { REFRESH_MS, useRefreshStore } from './refresh-store'
import {
  type Agents,
  type ApiEnvelope,
  type CalibrationPop,
  type EvAttribution,
  type Episodes,
  type Explain,
  type FeatureImportance,
  type Hypotheses,
  type Kpis,
  type Positions,
  type Probability,
  type Regime,
  type RegimePerformance,
  type Sentiment,
  type Weights,
} from './types'

const ORACLE_KEY = ['oracle'] as const

function useOracleQuery<T extends ApiEnvelope>(
  key: string,
  path: string
): UseQueryResult<T> {
  const autoRefresh = useRefreshStore((s) => s.autoRefresh)
  return useQuery<T>({
    queryKey: [...ORACLE_KEY, key],
    queryFn: () => apiGet<T>(path),
    refetchInterval: autoRefresh ? REFRESH_MS : false,
    refetchIntervalInBackground: false,
  })
}

export const useKpis = () => useOracleQuery<Kpis>('kpis', 'single-leg/kpis')
export const usePositions = () =>
  useOracleQuery<Positions>('positions', 'single-leg/positions')
export const useEpisodes = () =>
  useOracleQuery<Episodes>('episodes', 'single-leg/episodes')
export const useRegime = () => useOracleQuery<Regime>('regime', 'regime')
export const useSentiment = () =>
  useOracleQuery<Sentiment>('sentiment', 'sentiment')
export const useProbability = () =>
  useOracleQuery<Probability>('probability', 'probability')
export const useCalibrationPop = () =>
  useOracleQuery<CalibrationPop>('calibration-pop', 'calibration/pop')
export const useAgents = () => useOracleQuery<Agents>('agents', 'agents')
export const useFeatureImportance = () =>
  useOracleQuery<FeatureImportance>('feature-importance', 'feature-importance')
export const useWeights = () => useOracleQuery<Weights>('weights', 'weights')
export const useEvAttribution = () =>
  useOracleQuery<EvAttribution>('ev-attribution', 'ev-attribution')
export const useRegimePerformance = () =>
  useOracleQuery<RegimePerformance>('regime-performance', 'regime-performance')
export const useHypotheses = () =>
  useOracleQuery<Hypotheses>('hypotheses', 'hypotheses')

// Explain-a-Ticker: lazy, enabled only when a validated ticker is provided.
export function useExplain(ticker: string | null) {
  return useQuery<Explain>({
    queryKey: [...ORACLE_KEY, 'explain', ticker],
    queryFn: () => apiGet<Explain>('explain/' + encodeURIComponent(ticker!)),
    enabled: !!ticker,
    refetchInterval: false,
  })
}

export type DashboardStatus = 'live' | 'refreshing' | 'error'

// Aggregate status pill + last-updated time across all oracle queries.
export function useDashboardStatus() {
  const queryClient = useQueryClient()
  const fetching = useIsFetching({ queryKey: ORACLE_KEY })
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const prevFetching = useRef(0)

  // Stamp the time when a refresh cycle settles (fetching -> idle).
  useEffect(() => {
    if (prevFetching.current > 0 && fetching === 0) {
      setLastUpdated(new Date())
    }
    prevFetching.current = fetching
  }, [fetching])

  const hasError = queryClient
    .getQueryCache()
    .findAll({ queryKey: ORACLE_KEY })
    .some((q) => {
      const data = q.state.data as ApiEnvelope | undefined
      return data?.verdict === 'ERROR'
    })

  const status: DashboardStatus =
    fetching > 0 ? 'refreshing' : hasError ? 'error' : 'live'

  const refresh = () =>
    queryClient.invalidateQueries({ queryKey: ORACLE_KEY })

  return { status, lastUpdated, refresh, isFetching: fetching > 0 }
}
