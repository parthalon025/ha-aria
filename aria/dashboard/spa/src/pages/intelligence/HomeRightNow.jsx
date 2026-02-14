import { Section } from './utils.jsx';
import { SIGNIFICANCE_PCT, UNAVAILABLE_WARNING_THRESHOLD } from '../../constants.js';

export function HomeRightNow({ intraday, baselines }) {
  if (!intraday || intraday.length === 0) return null;
  const latest = intraday[intraday.length - 1];

  // Compare to baseline if available
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long' });
  const baseline = baselines && baselines[today];

  function compareToBaseline(key, val) {
    if (val == null || !baseline || !baseline[key]) return null;
    const mean = baseline[key].mean;
    if (mean == null) return null;
    const diff = val - mean;
    const pct = mean > 0 ? Math.round((diff / mean) * 100) : 0;
    if (Math.abs(pct) < SIGNIFICANCE_PCT) return { text: 'typical', style: 'color: var(--text-tertiary)' };
    if (pct > 0) return { text: `+${pct}% vs ${today}`, style: 'color: var(--status-warning)' };
    return { text: `${pct}% vs ${today}`, style: 'color: var(--accent)' };
  }

  const items = [
    { label: 'Power (W)', value: latest.power_watts != null ? latest.power_watts : '\u2014', note: compareToBaseline('power_watts', latest.power_watts) },
    { label: 'Lights On', value: latest.lights_on != null ? latest.lights_on : '\u2014', note: compareToBaseline('lights_on', latest.lights_on) },
    { label: 'Devices Home', value: latest.devices_home != null ? latest.devices_home : '\u2014' },
    { label: 'Unavailable', value: latest.unavailable != null ? latest.unavailable : '\u2014', warning: (latest.unavailable || 0) > UNAVAILABLE_WARNING_THRESHOLD, note: (latest.unavailable || 0) > UNAVAILABLE_WARNING_THRESHOLD ? { text: 'High \u2014 check devices', style: 'color: var(--status-warning)' } : null },
  ];

  const subtitle = baseline
    ? `Latest snapshot compared to your typical ${today}. Updated every 4 hours (more often when home is active).`
    : 'Latest snapshot of your home. Comparisons to baseline will appear after 7 days of data.';

  return (
    <Section title="Home Right Now" subtitle={subtitle} summary={items.length + " metrics"}>
      <div class="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {items.map((item, i) => (
          <div key={i} class="t-frame p-4" data-label={item.label} style={item.warning ? 'border: 2px solid var(--status-warning)' : ''}>
            <div class="text-2xl font-bold" style={item.warning ? 'color: var(--status-warning)' : 'color: var(--accent)'}>
              {item.value}
            </div>
            <div class="text-sm mt-1" style="color: var(--text-tertiary)">{item.label}</div>
            {item.note && (
              <div class="text-xs mt-1" style={item.note.style}>{item.note.text}</div>
            )}
          </div>
        ))}
      </div>
    </Section>
  );
}
