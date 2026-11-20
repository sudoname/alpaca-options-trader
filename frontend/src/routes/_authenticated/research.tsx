import { createFileRoute } from '@tanstack/react-router'
import { ResearchPage } from '@/features/dashboard/pages/research'

export const Route = createFileRoute('/_authenticated/research')({
  component: ResearchPage,
})
