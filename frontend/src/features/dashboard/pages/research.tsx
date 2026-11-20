import { AdaptiveWeights } from '../components/adaptive-weights'
import { AgentHitRate } from '../components/agent-hit-rate'
import { EvAttribution } from '../components/ev-attribution'
import { ExplainTicker } from '../components/explain-ticker'
import { FeatureImportance } from '../components/feature-importance'
import { Hypotheses } from '../components/hypotheses'
import { ProbabilityCalibration } from '../components/probability-calibration'
import { RegimePerformance } from '../components/regime-performance'
import { RlEpisodes } from '../components/rl-episodes'
import { ShadowResearch } from '../components/shadow-research'
import { DashboardShell } from '../dashboard-shell'

export function ResearchPage() {
  return (
    <DashboardShell>
      <div className='grid grid-cols-1 gap-4 lg:grid-cols-2'>
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
        <ShadowResearch />
      </div>
      <div className='mt-4'>
        <ExplainTicker />
      </div>
    </DashboardShell>
  )
}
