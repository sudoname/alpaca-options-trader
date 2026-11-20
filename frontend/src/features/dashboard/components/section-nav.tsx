import { Link } from '@tanstack/react-router'

const SECTIONS = [
  { to: '/', label: 'Overview', exact: true },
  { to: '/trading', label: 'Trading', exact: false },
  { to: '/decisions', label: 'Decisions', exact: false },
  { to: '/research', label: 'Research', exact: false },
  { to: '/ml', label: 'ML', exact: false },
] as const

export function SectionNav() {
  return (
    <nav
      aria-label='Dashboard sections'
      className='flex items-center gap-1 overflow-x-auto'
    >
      {SECTIONS.map((s) => (
        <Link
          key={s.to}
          to={s.to}
          activeOptions={{ exact: s.exact }}
          className='text-muted-foreground hover:text-foreground border-b-2 border-transparent px-2 py-1 font-mono text-[11px] tracking-[0.15em] uppercase transition-colors'
          activeProps={{
            className:
              'text-foreground border-b-2 border-[#ffb000] px-2 py-1 font-mono text-[11px] tracking-[0.15em] uppercase',
          }}
        >
          {s.label}
        </Link>
      ))}
    </nav>
  )
}
