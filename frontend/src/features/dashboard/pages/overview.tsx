import { KpiRow } from '../components/kpi-row'
import { NodeStatus } from '../components/node-status'
import { OpenPositions } from '../components/open-positions'
import { RegimeCard } from '../components/regime-card'
import { SentimentCard } from '../components/sentiment-card'
import { DashboardShell } from '../dashboard-shell'

export function OverviewPage() {
  return (
    <DashboardShell>
      <NodeStatus />
      <div className='mt-4'>
        <KpiRow />
      </div>
      <div className='mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2'>
        <RegimeCard />
        <SentimentCard />
      </div>
      <div className='mt-4'>
        <OpenPositions />
      </div>
    </DashboardShell>
  )
}
