import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
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
import { PutTimeStop } from './components/put-time-stop'
import { RegimeCard } from './components/regime-card'
import { RegimePerformance } from './components/regime-performance'
import { RlEpisodes } from './components/rl-episodes'
import { SentimentCard } from './components/sentiment-card'
import { ShadowResearch } from './components/shadow-research'

export function Dashboard() {
  return (
    <>
      <Header>
        <h1 className='font-mono text-sm font-semibold tracking-[0.25em] uppercase'>
          Oracle
          <span
            aria-hidden
            className='ms-2 inline-block size-1.5 animate-pulse bg-[#ffb000] align-middle'
          />
        </h1>
        <div className='ms-auto flex items-center gap-3'>
          <DashboardToolbar />
        </div>
      </Header>

      <Main className='hud-sections'>
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
          <PutTimeStop />
        </div>

        <div className='mt-4'>
          <ShadowResearch />
        </div>

        <div className='mt-4'>
          <ExplainTicker />
        </div>

        <p className='text-muted-foreground mt-6 text-center font-mono text-[10px] tracking-[0.15em] uppercase'>
          Read-only analytics — this page cannot place, size, gate, or close
          any trade
        </p>
      </Main>
    </>
  )
}
