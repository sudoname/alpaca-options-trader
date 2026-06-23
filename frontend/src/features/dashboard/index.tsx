import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ThemeSwitch } from '@/components/theme-switch'
import { AdaptiveWeights } from './components/adaptive-weights'
import { AgentHitRate } from './components/agent-hit-rate'
import { DashboardToolbar } from './components/dashboard-toolbar'
import { EvAttribution } from './components/ev-attribution'
import { ExplainTicker } from './components/explain-ticker'
import { FeatureImportance } from './components/feature-importance'
import { Hypotheses } from './components/hypotheses'
import { KpiRow } from './components/kpi-row'
import { OpenPositions } from './components/open-positions'
import { ProbabilityCalibration } from './components/probability-calibration'
import { RegimeCard } from './components/regime-card'
import { RegimePerformance } from './components/regime-performance'
import { RlEpisodes } from './components/rl-episodes'
import { SentimentCard } from './components/sentiment-card'

export function Dashboard() {
  return (
    <>
      <Header>
        <h1 className='text-lg font-semibold tracking-tight'>Oracle</h1>
        <div className='ms-auto flex items-center gap-3'>
          <DashboardToolbar />
          <ThemeSwitch />
        </div>
      </Header>

      <Main>
        <KpiRow />

        <div className='mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2'>
          <RegimeCard />
          <SentimentCard />
          <ProbabilityCalibration />
          <AgentHitRate />
          <FeatureImportance />
          <AdaptiveWeights />
          <EvAttribution />
          <RlEpisodes />
          <RegimePerformance />
          <Hypotheses />
        </div>

        <div className='mt-4'>
          <OpenPositions />
        </div>

        <div className='mt-4'>
          <ExplainTicker />
        </div>

        <p className='text-muted-foreground mt-6 text-center text-xs'>
          Read-only analytics. This page cannot place, size, gate, or close any
          trade.
        </p>
      </Main>
    </>
  )
}
