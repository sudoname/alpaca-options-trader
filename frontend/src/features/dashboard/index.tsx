import { OverviewPage } from './pages/overview'

// The root route renders the Overview section; other sections live in
// ./pages and are mounted by their own routes (/trading, /research, /ml).
export function Dashboard() {
  return <OverviewPage />
}
