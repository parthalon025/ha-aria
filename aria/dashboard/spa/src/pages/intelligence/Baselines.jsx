import { Section, Callout } from './utils.jsx';

const DAYS_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const DAY_ABBR = { Monday: 'Mon', Tuesday: 'Tue', Wednesday: 'Wed', Thursday: 'Thu', Friday: 'Fri', Saturday: 'Sat', Sunday: 'Sun' };

const METRICS = [
  { key: 'power_watts', label: 'Power', unit: 'W', color: 'var(--accent)', isNegative: false },
  { key: 'lights_on', label: 'Lights', unit: '', color: 'var(--accent)', isNegative: false },
  { key: 'devices_home', label: 'Devices', unit: '', color: 'var(--accent)', isNegative: false },
  { key: 'unavailable', label: 'Unavail', unit: '', color: 'var(--status-error)', isNegative: true },
];

function computeRanges(baselines) {
  const ranges = {};
  for (const m of METRICS) {
    let min = Infinity;
    let max = -Infinity;
    for (const day of DAYS_ORDER) {
      const b = baselines[day];
      if (!b) continue;
      const val = b[m.key]?.mean;
      if (val === null || val === undefined) continue;
      if (val < min) min = val;
      if (val > max) max = val;
    }
    ranges[m.key] = { min: min === Infinity ? 0 : min, max: max === -Infinity ? 0 : max };
  }
  return ranges;
}

function intensity(value, min, max) {
  if (max === min) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function formatValue(value, metric) {
  if (value === null || value === undefined) return '\u2014';
  if (metric.key === 'power_watts') return Math.round(value).toLocaleString();
  return Math.round(value).toString();
}

function HeatCell({ value, std, metric, range }) {
  const hasMean = value !== null && value !== undefined;
  const i = hasMean ? intensity(value, range.min, range.max) : 0;
  const displayVal = formatValue(value, metric);
  const title = hasMean && std !== null && std !== undefined
    ? `${metric.label}: ${displayVal}${metric.unit ? ' ' + metric.unit : ''} \u00B1 ${Math.round(std * 10) / 10}`
    : metric.label;

  return (
    <div
      class="relative flex items-center justify-center rounded"
      style={`min-height: 2.5rem; min-width: 3.5rem;`}
      title={title}
    >
      {/* Color layer */}
      <div
        class="absolute inset-0 rounded"
        style={`background: ${hasMean ? metric.color : 'var(--bg-surface-raised)'}; opacity: ${hasMean ? 0.12 + i * 0.55 : 0.15};`}
      />
      {/* Value */}
      <span
        class="relative z-10 font-mono"
        style={`font-size: var(--type-micro); color: var(--text-primary);`}
      >
        {displayVal}
      </span>
    </div>
  );
}

export function Baselines({ baselines }) {
  if (!baselines || Object.keys(baselines).length === 0) {
    return (
      <Section
        title="Baselines"
        subtitle="This is what 'normal' looks like for each day. Deviations from these numbers trigger anomaly detection."
      >
        <Callout>No baselines yet. The first baseline is calculated after the first daily snapshot.</Callout>
      </Section>
    );
  }

  const today = new Date().toLocaleDateString('en-US', { weekday: 'long' });
  const ranges = computeRanges(baselines);

  return (
    <Section
      title="Baselines"
      subtitle="This is 'normal' for each day of the week. The system flags deviations from these averages. More samples = tighter predictions."
      summary={Object.keys(baselines).length + " days"}
    >
      <p class="text-xs" style="color: var(--text-tertiary)">Each cell shows what's typical for that day and metric. Darker color = higher value. Hover any cell to see the exact number with its margin of error. The 'n' column is how many readings were averaged.</p>
      <div class="t-frame" data-label="baselines">
        {/* Heatmap grid */}
        <div
          style={`
            display: grid;
            grid-template-columns: auto repeat(${METRICS.length}, 1fr) auto;
            gap: 0.25rem;
            padding: 0.5rem;
          `}
        >
          {/* Header row: empty corner, metric labels, samples header */}
          <div />
          {METRICS.map(m => (
            <div
              key={m.key}
              class="text-center text-xs uppercase"
              style="color: var(--text-tertiary); padding-bottom: 0.25rem;"
            >
              {m.label}
            </div>
          ))}
          <div
            class="text-center text-xs uppercase"
            style="color: var(--text-tertiary); padding-bottom: 0.25rem; padding-left: 0.5rem;"
          >
            n
          </div>

          {/* Data rows */}
          {DAYS_ORDER.map(day => {
            const b = baselines[day];
            const isToday = day === today;

            return [
              // Day label
              <div
                key={day + '-label'}
                class="flex items-center text-xs font-medium pr-2"
                style={`
                  color: ${isToday ? 'var(--accent)' : 'var(--text-secondary)'};
                  ${isToday ? 'border-left: 2px solid var(--accent); padding-left: 0.375rem;' : 'padding-left: calc(2px + 0.375rem);'}
                `}
              >
                {DAY_ABBR[day]}
              </div>,

              // Metric cells
              ...METRICS.map(m => (
                <HeatCell
                  key={day + '-' + m.key}
                  value={b?.[m.key]?.mean}
                  std={b?.[m.key]?.stddev}
                  metric={m}
                  range={ranges[m.key]}
                />
              )),

              // Sample count
              <div
                key={day + '-samples'}
                class="flex items-center justify-center text-xs"
                style="color: var(--text-tertiary); padding-left: 0.5rem;"
              >
                {b?.sample_count ?? '\u2014'}
              </div>,
            ];
          })}
        </div>
        {/* Legend */}
        <div class="flex flex-wrap items-center gap-3 mt-2 pt-2" style="border-top: 1px solid var(--border-subtle); font-size: var(--type-micro); color: var(--text-tertiary); padding: 0 0.5rem 0.5rem;">
          <div class="flex items-center gap-1">
            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 2px; background: var(--accent); opacity: 0.2;" />
            <span>Low</span>
          </div>
          <div class="flex items-center gap-1">
            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 2px; background: var(--accent); opacity: 0.65;" />
            <span>High</span>
          </div>
          <div class="flex items-center gap-1">
            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 2px; background: var(--status-error); opacity: 0.65;" />
            <span>High unavail (bad)</span>
          </div>
          <span style="color: var(--accent);">\u25C0 today</span>
        </div>
      </div>
    </Section>
  );
}
