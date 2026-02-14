import TimeChart from './TimeChart.jsx';

/**
 * Hero metric card — the single most important number on the page.
 * Large monospace value with cursor state indicator.
 *
 * @param {Object} props
 * @param {*} props.value - Primary metric value
 * @param {string} props.label - Card label (shown via data-label)
 * @param {string} [props.unit] - Unit suffix
 * @param {string} [props.delta] - Delta/change text
 * @param {boolean} [props.warning] - Warning state (orange border + text)
 * @param {boolean} [props.loading] - Loading state (cursor-working)
 * @param {Array} [props.sparkData] - uPlot data array for sparkline [timestamps[], values[]]
 * @param {string} [props.sparkColor] - CSS color for sparkline (default: var(--accent))
 */
export default function HeroCard({ value, label, unit, delta, warning, loading, sparkData, sparkColor }) {
  const cursorClass = loading ? 'cursor-working' : 'cursor-active';

  return (
    <div
      class={`t-frame ${cursorClass}`}
      data-label={label}
      style={warning ? 'border-left: 3px solid var(--status-warning);' : ''}
    >
      <div class="flex items-baseline gap-2" style="justify-content: space-between;">
        <div class="flex items-baseline gap-2">
          <span
            class="data-mono"
            style={`font-size: var(--type-hero); font-weight: 600; color: ${warning ? 'var(--status-warning)' : 'var(--accent)'}; line-height: 1;`}
          >
            {value ?? '\u2014'}
          </span>
          {unit && (
            <span
              class="data-mono"
              style="font-size: var(--type-headline); color: var(--text-tertiary);"
            >
              {unit}
            </span>
          )}
        </div>
        {sparkData && sparkData.length > 1 && sparkData[0].length > 1 && (
          <div style="width: 80px; height: 32px; flex-shrink: 0;">
            <TimeChart
              data={sparkData}
              series={[{ label: label || 'trend', color: sparkColor || 'var(--accent)', width: 1.5 }]}
              compact
            />
          </div>
        )}
      </div>
      {delta && (
        <div
          style="font-size: var(--type-label); color: var(--text-secondary); margin-top: 8px; font-family: var(--font-mono);"
        >
          {delta}
        </div>
      )}
    </div>
  );
}
