import { createFileRoute } from '@tanstack/react-router'
import { MlPage } from '@/features/dashboard/pages/ml'

export const Route = createFileRoute('/_authenticated/ml')({
  component: MlPage,
})
