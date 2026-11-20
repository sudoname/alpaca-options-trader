import { PutTimeStop } from '../components/put-time-stop'
import { ShadowTraderPnl } from '../components/shadow-trader-pnl'
import { TradeLog } from '../components/trade-log'
import { DashboardShell } from '../dashboard-shell'

export function TradingPage() {
  return (
    <DashboardShell>
      <TradeLog />
      <div className='mt-4'>
        <PutTimeStop />
      </div>
      <div className='mt-4'>
        <ShadowTraderPnl />
      </div>
    </DashboardShell>
  )
}
