import { Section, Callout, relativeTime } from './utils.jsx';

export function AnomalyAlerts({ anomalies }) {
  if (!anomalies) {
    return (
      <Section
        title="Anomaly Detection"
        subtitle="Anomaly detection flags things that look unusual compared to your home's normal patterns — like high power draw at 3 AM or unexpected device activity."
      >
        <Callout>Anomaly detection data is not yet available. It will appear after the first ML training cycle.</Callout>
      </Section>
    );
  }

  const items = anomalies.anomalies || [];
  const ae = anomalies.autoencoder;
  const isof = anomalies.isolation_forest;

  return (
    <Section
      title="Anomaly Detection"
      subtitle="Anomaly detection flags things that look unusual compared to your home's normal patterns — like high power draw at 3 AM or unexpected device activity."
      summary={items.length > 0 ? `${items.length} anomal${items.length === 1 ? 'y' : 'ies'}` : 'clear'}
    >
      <div class="space-y-3">
        {/* Method status badges */}
        <div class="flex items-center gap-3">
          {isof && (
            <span class="text-xs font-medium rounded-full px-2.5 py-0.5"
              style={`background: ${isof.contamination ? 'var(--accent-glow)' : 'var(--bg-inset)'}; color: ${isof.contamination ? 'var(--accent)' : 'var(--text-tertiary)'}`}>
              Isolation Forest{isof.contamination ? ` (${(isof.contamination * 100).toFixed(1)}%)` : ''}
            </span>
          )}
          {ae && (
            <span class="text-xs font-medium rounded-full px-2.5 py-0.5"
              style={`background: ${ae.enabled ? 'var(--accent-glow)' : 'var(--bg-inset)'}; color: ${ae.enabled ? 'var(--accent)' : 'var(--text-tertiary)'}`}>
              Autoencoder {ae.enabled ? 'active' : 'inactive'}
            </span>
          )}
        </div>

        {/* Anomaly list or empty state */}
        {items.length === 0 ? (
          <Callout>No anomalies detected. This means your home's behavior matches its normal patterns.</Callout>
        ) : (
          <div class="space-y-2">
            {items.map((a, i) => (
              <div key={i} class="t-frame p-3" data-label="anomaly"
                style="border-left: 3px solid var(--status-warning)"
                {...(a.severity === 'critical' || (a.score !== null && a.score !== undefined && a.score < -0.5) ? { 'data-sh-effect': 'threat-pulse' } : {})}>
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2">
                    <span class="inline-block w-2 h-2 rounded-full" style="background: var(--status-warning)" />
                    <span class="text-xs font-bold" style="color: var(--text-primary)">{a.entity_id || a.metric || 'unknown'}</span>
                  </div>
                  <span class="text-xs" style="color: var(--text-tertiary)">{relativeTime(a.timestamp)}</span>
                </div>
                <div class="flex items-center gap-4 mt-1 text-xs" style="color: var(--text-secondary)">
                  <span>Score: <span class="data-mono" style="color: var(--status-warning)">{typeof (a.score ?? a.severity) === 'number' ? (a.score ?? a.severity).toFixed(3) : (a.score ?? a.severity)}</span></span>
                  <span class="text-xs rounded-full px-2 py-0.5" style="background: var(--bg-inset); color: var(--text-tertiary)">{a.type}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Legend */}
        <div class="flex items-center gap-4 text-xs pt-2" style="color: var(--text-tertiary)">
          <div class="flex items-center gap-1">
            <span class="inline-block w-2 h-2 rounded-full" style="background: var(--status-warning)" />
            <span>Anomalous</span>
          </div>
          <span>Lower scores = more unusual</span>
        </div>
      </div>
    </Section>
  );
}
