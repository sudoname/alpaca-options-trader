import { type ReactNode } from 'react'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { DashboardToolbar } from './components/dashboard-toolbar'
import { SectionNav } from './components/section-nav'

export function DashboardShell({ children }: { children: ReactNode }) {
  return (
    <>
      <Header>
        <h1 className='font-mono text-sm font-semibold tracking-[0.25em] uppercase'>
          Oracle
          <span
            aria-hidden
            className='ms-2 inline-block size-1.5 animate-pulse bg-[#ffb000] align-middle'
          />
        </h1>
        <SectionNav />
        <div className='ms-auto flex items-center gap-3'>
          <DashboardToolbar />
        </div>
      </Header>

      <Main className='hud-sections'>
        {children}
        <p className='text-muted-foreground mt-6 text-center font-mono text-[10px] tracking-[0.15em] uppercase'>
          Read-only analytics — this page cannot place, size, gate, or close
          any trade
        </p>
      </Main>
    </>
  )
}
