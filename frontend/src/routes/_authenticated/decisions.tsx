import { createFileRoute } from '@tanstack/react-router'
import { DecisionsPage } from '@/features/dashboard/pages/decisions'

export const Route = createFileRoute('/_authenticated/decisions')({
  component: DecisionsPage,
})
