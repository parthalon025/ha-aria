import { Section, Callout } from './utils.jsx';

function BarChart({ data, dataKey, label, color }) {
  if (!data || data.length === 0) return null;
  const values = data.map(d => d[dataKey]).filter(v => v != null);
  const max = Math.max(...values, 1);

  return (
    <div class="space-y-1">
      <div class="text-xs font-medium text-gray-600">{label}</div>
      <div class="flex items-end gap-1 h-12">
        {data.map((d, i) => {
          const val = d[dataKey];
          if (val == null) return <div key={i} class="flex-1" />;
          const height = Math.max((val / max) * 100, 4);
          return (
            <div
              key={i}
              class="flex-1 rounded-t transition-all"
              style={{ height: `${height}%`, backgroundColor: color, minWidth: '4px' }}
              title={`${d.date || 'h' + d.hour}: ${val}`}
            />
          );
        })}
      </div>
      <div class="flex justify-between text-[10px] text-gray-400">
        <span>{data[0]?.date || ('h' + data[0]?.hour)}</span>
        {data.length > 1 && <span>{data[data.length - 1]?.date || ('h' + data[data.length - 1]?.hour)}</span>}
      </div>
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
        subtitle="Spot when something changed \u2014 a new device, a routine shift, or a problem building."
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
      if (d > 10) changes.push(`${d} more entities unavailable than yesterday \u2014 check your network`);
    }
    if (changes.length > 0) trendNote = changes.join('. ') + '.';
  }

  return (
    <Section
      title="Trends Over Time"
      subtitle="Spot when something changed \u2014 a new device, a routine shift, or a problem building. Each bar is one day."
    >
      {trendNote && <Callout color="amber">{trendNote}</Callout>}
      <div class="bg-white rounded-lg shadow-sm p-4 space-y-4">
        {hasTrend && (
          <div class="space-y-3">
            <div class="text-xs font-bold text-gray-500 uppercase">Daily</div>
            <BarChart data={trendData} dataKey="power_watts" label="Power (W) \u2014 total household draw" color="#3b82f6" />
            <BarChart data={trendData} dataKey="lights_on" label="Lights On \u2014 how many at snapshot time" color="#f59e0b" />
            <BarChart data={trendData} dataKey="unavailable" label="Unavailable \u2014 entities not responding (should be low)" color="#ef4444" />
          </div>
        )}
        {hasIntraday && (
          <div class="space-y-3">
            <div class="text-xs font-bold text-gray-500 uppercase">Today (Intraday)</div>
            <BarChart data={intradayTrend} dataKey="power_watts" label="Power (W)" color="#6366f1" />
            <BarChart data={intradayTrend} dataKey="unavailable" label="Unavailable" color="#f43f5e" />
          </div>
        )}
      </div>
    </Section>
  );
}
