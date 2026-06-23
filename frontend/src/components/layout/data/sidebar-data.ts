import { LayoutDashboard } from 'lucide-react'
import { Logo } from '@/assets/logo'
import { type SidebarData } from '../types'

export const sidebarData: SidebarData = {
  user: {
    name: 'Oracle',
    email: 'read-only analytics',
    avatar: '/images/shadcn.png',
  },
  teams: [
    {
      name: 'Oracle',
      logo: Logo,
      plan: 'Single-Leg Intraday',
    },
  ],
  navGroups: [
    {
      title: 'General',
      items: [
        {
          title: 'Dashboard',
          url: '/',
          icon: LayoutDashboard,
        },
      ],
    },
  ],
}
