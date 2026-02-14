import { Section, Callout } from './utils.jsx';
import TimeChart from '../../components/TimeChart.jsx';

function toUPlotData(records, timeKey, metric) {
  const timestamps = records.map(r => {
    if (timeKey === 'date') {
      return Math.floor(new Date(r.date).getTime() / 1000);
    } else {
      const now = new Date();
      return Math.floor(new Date(now.getFullYear(), now.getMonth(), now.getDate(), r.hour).getTime() / 1000);
    }
  });
  const values = records.map(r => r[metric] ?? null);
  return [timestamps, values];
}

function MetricChart({ label, data, color, height = 80 }) {
  return (
    <div>
      <div class="text-xs font-bold uppercase" style={{ color: 'var(--text-tertiary)' }}>{label}</div>
      <TimeChart
        data={data}
        series={[{ label, color }]}
        height={height}
      />
    </div>
  );
}

export function TrendsOverTime({ trendData, intradayTrend }) {
  const hasTrend = trendData && trendData.length > 0;
  const hasIntraday = intradayTrend && intradayTrend.length > 0;

  if (!hasTrend && !hasIntraday) {
    return (
      <Section
        title="Trends Over Time"
        subtitle="Spot when something changed — a new device, a routine shift, or a problem building."
      >
        <Callout>No trend data yet. Daily snapshots are collected each night at 11:30 PM.</Callout>
      </Section>
    );
  }

  // Detect notable changes in daily trend
  let trendNote = null;
  if (hasTrend && trendData.length >= 2) {
    const last = trendData[trendData.length - 1];
    const prev = trendData[trendData.length - 2];
    const changes = [];
    if (last.power_watts != null && prev.power_watts != null) {
      const d = last.power_watts - prev.power_watts;
      if (Math.abs(d) > 50) changes.push(`Power ${d > 0 ? 'up' : 'down'} ${Math.abs(Math.round(d))}W vs yesterday`);
    }
    if (last.unavailable != null && prev.unavailable != null) {
      const d = last.unavailable - prev.unavailable;
      if (d > 10) changes.push(`${d} more entities unavailable than yesterday — check your network`);
    }
    if (changes.length > 0) trendNote = changes.join('. ') + '.';
  }

  return (
    <Section
      title="Trends Over Time"
      subtitle="Spot when something changed — a new device, a routine shift, or a problem building."
    >
      {trendNote && <Callout>{trendNote}</Callout>}
      <p class="text-xs" style={{ color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
        Each chart tracks one measurement over time. Power is your home's total electricity usage.
        Lights is how many are on. Unavailable means devices that stopped reporting — a spike
        usually means a network issue.
      </p>
      <div class="t-frame" data-label="trends">
        {hasTrend && (
          <div class="space-y-2">
            <div class="text-xs font-bold uppercase" style={{ color: 'var(--text-tertiary)' }}>Daily (30d)</div>
            <MetricChart
              label="Power (W)"
              data={toUPlotData(trendData, 'date', 'power_watts')}
              color="var(--accent)"
            />
            <MetricChart
              label="Lights On"
              data={toUPlotData(trendData, 'date', 'lights_on')}
              color="var(--accent-warm)"
            />
            <MetricChart
              label="Unavailable"
              data={toUPlotData(trendData, 'date', 'unavailable')}
              color="var(--status-error)"
            />
          </div>
        )}
        {hasIntraday && (
          <div class="space-y-2">
            <div class="text-xs font-bold uppercase" style={{ color: 'var(--text-tertiary)' }}>Today (Intraday)</div>
            <MetricChart
              label="Power (W)"
              data={toUPlotData(intradayTrend, 'hour', 'power_watts')}
              color="var(--accent-dim)"
            />
            <MetricChart
              label="Unavailable"
              data={toUPlotData(intradayTrend, 'hour', 'unavailable')}
              color="var(--status-error)"
            />
          </div>
        )}
      </div>
    </Section>
  );
}
