import { Section, Callout } from './utils.jsx';

export function DriftStatus({ drift }) {
  if (!drift || drift.days_analyzed === 0) {
    return (
      <Section
        title="Drift Detection"
        subtitle="Concept drift means your home's patterns have changed — maybe a new schedule, seasonal shift, or new devices. ARIA uses Page-Hinkley and ADWIN algorithms to detect this automatically."
      >
        <Callout>Drift detection data is not yet available. It will populate after ARIA runs its drift check (aria check-drift) with enough prediction history.</Callout>
      </Section>
    );
  }

  const metrics = Object.keys(drift.rolling_mae || {});

  return (
    <Section
      title="Drift Detection"
      subtitle="Concept drift means your home's patterns have changed — maybe a new schedule, seasonal shift, or new devices. ARIA uses Page-Hinkley and ADWIN algorithms to detect this and retrains when needed."
      summary={drift.needs_retrain ? "drift detected" : "stable"}
    >
      <div class="space-y-3">
        {/* Overall status */}
        <div class="t-frame p-4" data-label="status">
          <div class="flex items-center gap-3">
            <span class="inline-block w-3 h-3 rounded-full" style={`background: ${drift.needs_retrain ? 'var(--status-warning)' : 'var(--status-healthy)'}`} />
            <span class="text-sm font-bold" style={`color: ${drift.needs_retrain ? 'var(--status-warning)' : 'var(--status-healthy)'}`}>
              {drift.needs_retrain ? 'Drift Detected' : 'Stable'}
            </span>
            <span class="text-xs" style="color: var(--text-tertiary)">{drift.days_analyzed} days analyzed</span>
          </div>
          {drift.reason && drift.reason !== 'no data' && (
            <p class="text-xs mt-2" style="color: var(--text-secondary)">{drift.reason}</p>
          )}
        </div>

        {/* Per-metric cards */}
        {metrics.length > 0 && (
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {metrics.map(metric => {
              const isDrifted = (drift.drifted_metrics || []).includes(metric);
              const rolling = drift.rolling_mae[metric];
              const current = drift.current_mae?.[metric];
              const thresh = drift.threshold?.[metric];
              const ph = drift.page_hinkley?.[metric];

              return (
                <div key={metric} class="t-frame p-3" data-label={metric.replace(/_/g, ' ')}
                  style={isDrifted ? 'border-left: 3px solid var(--status-warning)' : ''}>
                  <div class="flex items-center justify-between mb-2">
                    <span class="text-xs font-bold" style="color: var(--text-primary)">{metric.replace(/_/g, ' ')}</span>
                    <span class="inline-block w-2 h-2 rounded-full" style={`background: ${isDrifted ? 'var(--status-warning)' : 'var(--status-healthy)'}`} />
                  </div>
                  <div class="space-y-1 text-xs" style="color: var(--text-secondary)">
                    <div class="flex justify-between">
                      <span style="color: var(--text-tertiary)">Rolling MAE</span>
                      <span class="data-mono">{rolling ?? '—'}</span>
                    </div>
                    {current !== null && current !== undefined && (
                      <div class="flex justify-between">
                        <span style="color: var(--text-tertiary)">Current MAE</span>
                        <span class="data-mono" style={current > (thresh || Infinity) ? 'color: var(--status-warning)' : ''}>{current}</span>
                      </div>
                    )}
                    {thresh !== null && thresh !== undefined && (
                      <div class="flex justify-between">
                        <span style="color: var(--text-tertiary)">Threshold</span>
                        <span class="data-mono">{thresh}</span>
                      </div>
                    )}
                    {ph && ph.drift_detected && (
                      <div class="text-xs mt-1" style="color: var(--status-warning)">
                        Page-Hinkley: drift at point {ph.drift_point} (score: {ph.drift_score})
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Legend */}
        <div class="flex items-center gap-4 text-xs pt-2" style="color: var(--text-tertiary)">
          <div class="flex items-center gap-1">
            <span class="inline-block w-2 h-2 rounded-full" style="background: var(--status-healthy)" />
            <span>Stable</span>
          </div>
          <div class="flex items-center gap-1">
            <span class="inline-block w-2 h-2 rounded-full" style="background: var(--status-warning)" />
            <span>Drift detected</span>
          </div>
        </div>
      </div>
    </Section>
  );
}
