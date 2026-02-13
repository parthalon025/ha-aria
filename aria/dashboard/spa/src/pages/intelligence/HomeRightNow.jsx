import { Section } from './utils.jsx';

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
    if (Math.abs(pct) < 10) return { text: 'typical', color: 'text-gray-400' };
    if (pct > 0) return { text: `+${pct}% vs ${today}`, color: 'text-amber-500' };
    return { text: `${pct}% vs ${today}`, color: 'text-blue-500' };
  }

  const items = [
    { label: 'Power (W)', value: latest.power_watts != null ? latest.power_watts : '\u2014', note: compareToBaseline('power_watts', latest.power_watts) },
    { label: 'Lights On', value: latest.lights_on != null ? latest.lights_on : '\u2014', note: compareToBaseline('lights_on', latest.lights_on) },
    { label: 'Devices Home', value: latest.devices_home != null ? latest.devices_home : '\u2014' },
    { label: 'Unavailable', value: latest.unavailable != null ? latest.unavailable : '\u2014', warning: (latest.unavailable || 0) > 100, note: (latest.unavailable || 0) > 100 ? { text: 'High \u2014 check devices', color: 'text-amber-600' } : null },
  ];

  const subtitle = baseline
    ? `Latest snapshot compared to your typical ${today}. Updated every 4 hours (more often when home is active).`
    : 'Latest snapshot of your home. Comparisons to baseline will appear after 7 days of data.';

  return (
    <Section title="Home Right Now" subtitle={subtitle}>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        {items.map((item, i) => (
          <div key={i} class={`bg-white rounded-lg shadow-sm p-4 ${item.warning ? 'border-2 border-amber-500' : ''}`}>
            <div class={`text-2xl font-bold ${item.warning ? 'text-amber-500' : 'text-blue-500'}`}>
              {item.value}
            </div>
            <div class="text-sm text-gray-500 mt-1">{item.label}</div>
            {item.note && (
              <div class={`text-xs mt-1 ${item.note.color}`}>{item.note.text}</div>
            )}
          </div>
        ))}
      </div>
    </Section>
  );
}
