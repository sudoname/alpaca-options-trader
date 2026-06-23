import { ORA } from '@/lib/oracle/theme'
import { useRegime } from '@/lib/oracle/hooks'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { ChartGuard } from './chart-guard'
import { Gauge } from './gauge'

export function RegimeCard() {
  const { data, isLoading } = useRegime()
  const conf = Number(data?.confidence || 0)
  const label = String(data?.label || '—').replace(/_/g, ' ')
  const reasons = Array.isArray(data?.reasons) ? data!.reasons! : []

  return (
    <Card>
      <CardHeader>
        <CardTitle>Market Regime</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartGuard data={data} isLoading={isLoading} minHeight={180}>
          <Gauge
            value={conf * 100}
            color={ORA.accent}
            valueText={`${(conf * 100).toFixed(0)}%`}
          />
          <div className='mt-1 text-center'>
            <div
              className='text-lg font-semibold'
              style={{ color: ORA.accent }}
            >
              {label}
            </div>
            <div className='text-muted-foreground text-xs'>market regime</div>
          </div>
          {reasons.length ? (
            <ul className='text-muted-foreground mt-3 list-disc space-y-1 ps-5 text-sm'>
              {reasons.map((r, i) => (
                <li key={i}>{String(r)}</li>
              ))}
            </ul>
          ) : null}
        </ChartGuard>
      </CardContent>
    </Card>
  )
}
