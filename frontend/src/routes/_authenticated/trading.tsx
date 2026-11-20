import { createFileRoute } from '@tanstack/react-router'
import { TradingPage } from '@/features/dashboard/pages/trading'

export const Route = createFileRoute('/_authenticated/trading')({
  component: TradingPage,
})
