// Shared palette + Fear & Greed bands (mirrors dashboard/app.js).
export const ORA = {
  accent: '#58a6ff',
  green: '#3fb950',
  red: '#f85149',
  amber: '#d29922',
  grid: '#30363d',
  muted: '#8b949e',
  text: '#c9d1d9',
  axis: '#8b949e',
} as const

export type FgBand = { lo: number; hi: number; color: string }

export const FG_BANDS: FgBand[] = [
  { lo: 0, hi: 25, color: '#f85149' }, // Extreme Fear
  { lo: 25, hi: 45, color: '#d29922' }, // Fear
  { lo: 45, hi: 55, color: '#8b949e' }, // Neutral
  { lo: 55, hi: 75, color: '#56b870' }, // Greed
  { lo: 75, hi: 100, color: '#3fb950' }, // Extreme Greed
]

export function fgColor(score: number): string {
  for (const b of FG_BANDS) if (score >= b.lo && score <= b.hi) return b.color
  return ORA.accent
}
