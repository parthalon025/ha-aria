import { Section, Callout } from './utils.jsx';

const DAYS_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

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

  return (
    <Section
      title="Baselines"
      subtitle="This is 'normal' for each day of the week. The system flags deviations from these averages. More samples = tighter predictions."
    >
      <div class="bg-white rounded-lg shadow-sm overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-gray-200 text-left text-xs text-gray-500 uppercase">
              <th class="px-4 py-2">Day</th>
              <th class="px-4 py-2">Samples</th>
              <th class="px-4 py-2">Power (W)</th>
              <th class="px-4 py-2">Lights</th>
              <th class="px-4 py-2">Devices</th>
              <th class="px-4 py-2">Unavail</th>
            </tr>
          </thead>
          <tbody>
            {DAYS_ORDER.map(day => {
              const b = baselines[day];
              const isToday = day === today;
              if (!b) {
                return (
                  <tr key={day} class="border-b border-gray-100 text-gray-300">
                    <td class="px-4 py-2">{day}{isToday ? ' (today)' : ''}</td>
                    <td class="px-4 py-2" colSpan="5">no data</td>
                  </tr>
                );
              }
              return (
                <tr key={day} class={`border-b border-gray-100 ${isToday ? 'bg-blue-50' : ''}`}>
                  <td class="px-4 py-2 font-medium text-gray-700">{day}{isToday ? ' (today)' : ''}</td>
                  <td class="px-4 py-2">{b.sample_count}</td>
                  <td class="px-4 py-2">{b.power_watts?.mean != null ? Math.round(b.power_watts.mean * 10) / 10 : '\u2014'}</td>
                  <td class="px-4 py-2">{b.lights_on?.mean != null ? Math.round(b.lights_on.mean) : '\u2014'}</td>
                  <td class="px-4 py-2">{b.devices_home?.mean != null ? Math.round(b.devices_home.mean) : '\u2014'}</td>
                  <td class="px-4 py-2">{b.unavailable?.mean != null ? Math.round(b.unavailable.mean) : '\u2014'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Section>
  );
}
