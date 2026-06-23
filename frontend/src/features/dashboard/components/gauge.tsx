import { ORA, type FgBand } from '@/lib/oracle/theme'

type GaugeProps = {
  /** 0..100 */
  value: number
  /** When provided, draws colored band segments (Fear & Greed). */
  bands?: FgBand[]
  /** Arc color for the simple (single-value) gauge. */
  color?: string
  valueText: string
  valueColor?: string
}

const VB_W = 220
const VB_H = 130
const CX = 110
const CY = 110
const R = 88
const SW = 16

function polar(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = (angleDeg * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy - r * Math.sin(rad) }
}

// Half-circle arc (left -> right over the top) for fractions f0..f1 in [0,1].
function arcPath(f0: number, f1: number) {
  const a0 = 180 - 180 * Math.min(Math.max(f0, 0), 1)
  const a1 = 180 - 180 * Math.min(Math.max(f1, 0), 1)
  const start = polar(CX, CY, R, a0)
  const end = polar(CX, CY, R, a1)
  const largeArc = a0 - a1 > 180 ? 1 : 0
  return `M ${start.x} ${start.y} A ${R} ${R} 0 ${largeArc} 1 ${end.x} ${end.y}`
}

export function Gauge({
  value,
  bands,
  color = ORA.accent,
  valueText,
  valueColor = ORA.text,
}: GaugeProps) {
  const v = Math.min(Math.max(value, 0), 100)
  const frac = v / 100

  return (
    <div className='flex flex-col items-center'>
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className='w-full max-w-[260px]'
        role='img'
      >
        {/* track */}
        <path
          d={arcPath(0, 1)}
          fill='none'
          stroke='#1c2330'
          strokeWidth={SW}
          strokeLinecap='round'
        />
        {bands ? (
          // Banded variant: one colored segment per Fear & Greed band.
          bands.map((b) => (
            <path
              key={`${b.lo}-${b.hi}`}
              d={arcPath(b.lo / 100, b.hi / 100)}
              fill='none'
              stroke={b.color}
              strokeWidth={SW}
            />
          ))
        ) : (
          // Simple variant: a single accent arc up to the value.
          <path
            d={arcPath(0, frac)}
            fill='none'
            stroke={color}
            strokeWidth={SW}
            strokeLinecap='round'
          />
        )}
        {bands ? (
          // Needle pointing at the score.
          (() => {
            const tip = polar(CX, CY, R - SW / 2 - 2, 180 - 180 * frac)
            return (
              <>
                <line
                  x1={CX}
                  y1={CY}
                  x2={tip.x}
                  y2={tip.y}
                  stroke={ORA.text}
                  strokeWidth={3}
                  strokeLinecap='round'
                />
                <circle cx={CX} cy={CY} r={5} fill={ORA.text} />
              </>
            )
          })()
        ) : null}
        <text
          x={CX}
          y={CY - 14}
          textAnchor='middle'
          fontSize='30'
          fontWeight='700'
          fill={valueColor}
        >
          {valueText}
        </text>
      </svg>
    </div>
  )
}
