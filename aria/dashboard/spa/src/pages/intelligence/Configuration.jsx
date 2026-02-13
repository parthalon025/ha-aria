import { Section } from './utils.jsx';

export function Configuration({ config }) {
  if (!config) return null;

  const featureGroups = {};
  if (config.feature_config) {
    for (const [key, val] of Object.entries(config.feature_config)) {
      const group = key.replace(/_features?$/, '').replace(/_/g, ' ');
      featureGroups[group] = val;
    }
  }

  return (
    <Section
      title="Configuration"
      subtitle="Current engine settings. Edit ~/ha-logs/intelligence/feature_config.json to change."
    >
      <details class="bg-white rounded-lg shadow-sm">
        <summary class="px-4 py-3 cursor-pointer text-sm font-medium text-gray-700 hover:bg-gray-50">
          Show configuration details
        </summary>
        <div class="px-4 pb-4 space-y-4">
          <div>
            <div class="text-xs font-bold text-gray-500 uppercase mb-1">ML Weight Schedule</div>
            <table class="text-xs w-full">
              <thead>
                <tr class="text-left text-gray-500"><th class="py-1">Days of Data</th><th>ML Weight</th></tr>
              </thead>
              <tbody>
                {config.ml_weight_schedule && Object.entries(config.ml_weight_schedule).map(([range, weight]) => (
                  <tr key={range}>
                    <td class="py-1">{range}</td>
                    <td>{(weight * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div>
            <div class="text-xs font-bold text-gray-500 uppercase mb-1">Anomaly Threshold</div>
            <p class="text-sm text-gray-700">{config.anomaly_threshold} standard deviations from baseline triggers an anomaly flag.</p>
          </div>

          <div>
            <div class="text-xs font-bold text-gray-500 uppercase mb-1">Feature Toggles</div>
            <div class="flex flex-wrap gap-2">
              {Object.entries(featureGroups).map(([name, enabled]) => (
                <span
                  key={name}
                  class={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${
                    enabled ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-400'
                  }`}
                >
                  {name}: {enabled ? 'on' : 'off'}
                </span>
              ))}
            </div>
          </div>
        </div>
      </details>
    </Section>
  );
}
