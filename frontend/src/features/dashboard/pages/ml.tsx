import { Oracle4Status } from '../components/oracle4-status'
import { TrainingPanel } from '../components/training-panel'
import { DashboardShell } from '../dashboard-shell'

export function MlPage() {
  return (
    <DashboardShell>
      <TrainingPanel />
      <div className='mt-4'>
        <Oracle4Status />
      </div>
    </DashboardShell>
  )
}
