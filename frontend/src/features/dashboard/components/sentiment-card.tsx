import { FG_BANDS, fgColor } from '@/lib/oracle/theme'
import { useSentiment } from '@/lib/oracle/hooks'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { ChartGuard } from './chart-guard'
import { Gauge } from './gauge'

export function SentimentCard() {
  const { data, isLoading } = useSentiment()
  const noScore = data?.score == null
  const score = Number(data?.score)
  const cls = data?.classification || '—'
  const col = fgColor(score)

  const comps = Array.isArray(data?.components)
    ? data!.components!.filter((c) => c && c.available && c.score != null)
    : []

  const srcParts: string[] = []
  if (data?.source) {
    srcParts.push(`source: ${data.source}`)
    if (data.cnn_score != null) srcParts.push(`CNN ${Math.round(data.cnn_score)}`)
    if (data.custom_score != null)
      srcParts.push(`custom ${Math.round(data.custom_score)}`)
    if (data.from_cache) srcParts.push('cached')
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Market Sentiment</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartGuard
          data={data}
          isLoading={isLoading}
          empty={!isLoading && noScore}
          minHeight={180}
        >
          <Gauge
            value={score}
            bands={FG_BANDS}
            valueText={`${Math.round(score)}`}
            valueColor={col}
          />
          <div className='mt-1 text-center'>
            <div className='text-lg font-semibold' style={{ color: col }}>
              {cls}
            </div>
            <div className='text-muted-foreground text-xs'>fear &amp; greed</div>
          </div>
          {srcParts.length ? (
            <div className='text-muted-foreground mt-3 text-xs'>
              {srcParts.join(' · ')}
            </div>
          ) : null}
          {comps.length ? (
            <ul className='mt-2 space-y-1 text-sm'>
              {comps.map((c, i) => (
                <li key={i}>
                  <span className='text-muted-foreground'>
                    {String(c.name).replace(/_/g, ' ')}:{' '}
                  </span>
                  <span
                    className='font-semibold'
                    style={{ color: fgColor(Number(c.score)) }}
                  >
                    {Math.round(Number(c.score))}
                  </span>
                </li>
              ))}
            </ul>
          ) : null}
        </ChartGuard>
      </CardContent>
    </Card>
  )
}
