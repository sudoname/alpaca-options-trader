// Shared palette + Fear & Greed bands (tactical HUD theme).
export const ORA = {
  accent: '#ffb000',
  green: '#3fb950',
  red: '#f85149',
  amber: '#d29922',
  grid: '#1f1f1f',
  muted: '#8a8a8a',
  text: '#e6e6e6',
  axis: '#8a8a8a',
  mono: "'JetBrains Mono', ui-monospace, monospace",
} as const

export type FgBand = { lo: number; hi: number; color: string }

export const FG_BANDS: FgBand[] = [
  { lo: 0, hi: 25, color: '#f85149' }, // Extreme Fear
  { lo: 25, hi: 45, color: '#d29922' }, // Fear
  { lo: 45, hi: 55, color: '#8a8a8a' }, // Neutral
  { lo: 55, hi: 75, color: '#56b870' }, // Greed
  { lo: 75, hi: 100, color: '#3fb950' }, // Extreme Greed
]

export function fgColor(score: number): string {
  for (const b of FG_BANDS) if (score >= b.lo && score <= b.hi) return b.color
  return ORA.accent
}
