import { AgentContributions } from '../components/agent-contributions'
import { DecisionFunnel } from '../components/decision-funnel'
import { DecisionLog } from '../components/decision-log'
import { DashboardShell } from '../dashboard-shell'

export function DecisionsPage() {
  return (
    <DashboardShell>
      <DecisionFunnel />
      <div className='mt-4'>
        <AgentContributions />
      </div>
      <div className='mt-4'>
        <DecisionLog />
      </div>
    </DashboardShell>
  )
}
