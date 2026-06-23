// Formatters + verdict helpers, faithful ports of dashboard/app.js.
import { type ApiEnvelope } from './types'

export function isUsable(d: ApiEnvelope | undefined | null): boolean {
  return !!d && d.verdict !== 'INSUFFICIENT_DATA' && d.verdict !== 'ERROR'
}

export type BadgeKind = 'error' | 'insufficient' | null

export function badgeKind(d: ApiEnvelope | undefined | null): BadgeKind {
  if (!d) return 'error'
  if (d.verdict === 'ERROR') return 'error'
  if (d.verdict === 'INSUFFICIENT_DATA') return 'insufficient'
  return null
}

export function badgeLabel(d: ApiEnvelope | undefined | null): string {
  const kind = badgeKind(d)
  if (kind === 'error') return d ? 'error' : 'no data'
  if (kind === 'insufficient') return 'insufficient data'
  return ''
}

function isNum(v: unknown): v is number {
  return typeof v === 'number' && !Number.isNaN(v)
}

export function fmtMoney(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—'
  const s = v < 0 ? '-' : ''
  return `${s}$${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}`
}

export function fmtPct(v: number | null | undefined, dp = 1): string {
  return v == null || Number.isNaN(v) ? '—' : (v * 100).toFixed(dp) + '%'
}

export function fmtNum(v: number | null | undefined, dp = 3): string {
  return v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(dp)
}

// Sign class for colored cells: positive -> green, negative -> red.
export function signClass(v: number | null | undefined): string {
  if (!isNum(v)) return ''
  if (v > 0) return 'text-[#3fb950]'
  if (v < 0) return 'text-[#f85149]'
  return ''
}
