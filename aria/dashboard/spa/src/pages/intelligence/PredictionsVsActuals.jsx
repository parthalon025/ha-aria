import { Section, Callout, confidenceColor } from './utils.jsx';

export function PredictionsVsActuals({ predictions, intradayTrend }) {
  if (!predictions || !predictions.target_date) {
    return (
      <Section
        title="Predictions vs Actuals"
        subtitle="Once active, large deltas here mean something unusual is happening \u2014 worth investigating."
      >
        <Callout>Predictions need at least 7 days of data. The system is still learning what "normal" looks like for each day of the week.</Callout>
      </Section>
    );
  }

  const latest = intradayTrend && intradayTrend.length > 0 ? intradayTrend[intradayTrend.length - 1] : {};
  const metrics = ['power_watts', 'lights_on', 'devices_home', 'unavailable', 'useful_events'];

  // Find biggest delta for callout
  let biggestDelta = null;
  metrics.forEach(m => {
    const pred = predictions[m] || {};
    const actual = latest[m];
    if (actual != null && pred.predicted != null && pred.predicted > 0) {
      const pct = Math.abs((actual - pred.predicted) / pred.predicted * 100);
      if (pct > 30 && (!biggestDelta || pct > biggestDelta.pct)) {
        biggestDelta = { metric: m.replace(/_/g, ' '), pct: Math.round(pct), actual, predicted: pred.predicted };
      }
    }
  });

  return (
    <Section
      title="Predictions vs Actuals"
      subtitle="Large deltas mean something unusual is happening. Small deltas mean the system understands your patterns."
    >
      {biggestDelta && (
        <Callout color="amber">
          {biggestDelta.metric} is {biggestDelta.pct}% off prediction ({biggestDelta.actual} actual vs {biggestDelta.predicted} predicted). Worth a look?
        </Callout>
      )}
      <div class="bg-white rounded-lg shadow-sm overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-gray-200 text-left text-xs text-gray-500 uppercase">
              <th class="px-4 py-2">Metric</th>
              <th class="px-4 py-2">Predicted</th>
              <th class="px-4 py-2">Actual</th>
              <th class="px-4 py-2">Delta</th>
              <th class="px-4 py-2">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map(m => {
              const pred = predictions[m] || {};
              const actual = latest[m];
              const delta = actual != null && pred.predicted != null
                ? Math.round((actual - pred.predicted) * 10) / 10
                : null;
              const bigDelta = delta != null && pred.predicted > 0 && Math.abs(delta / pred.predicted) > 0.3;
              return (
                <tr key={m} class={`border-b border-gray-100 ${bigDelta ? 'bg-amber-50' : ''}`}>
                  <td class="px-4 py-2 font-medium text-gray-700">{m.replace(/_/g, ' ')}</td>
                  <td class="px-4 py-2">{pred.predicted != null ? pred.predicted : '\u2014'}</td>
                  <td class="px-4 py-2">{actual != null ? actual : '\u2014'}</td>
                  <td class="px-4 py-2">{delta != null ? (delta >= 0 ? '+' : '') + delta : '\u2014'}</td>
                  <td class="px-4 py-2">
                    <span class={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${confidenceColor(pred.confidence)}`}>
                      {pred.confidence || 'n/a'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Section>
  );
}
