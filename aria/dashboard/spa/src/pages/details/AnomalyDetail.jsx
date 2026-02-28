/**
 * AnomalyDetail — Detail renderer for anomaly entities.
 * Three-section layout: Summary (severity + stats), Explanation (SHAP bars), History.
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';
import { relativeTime } from '../intelligence/utils.jsx';

/** Staleness threshold: 30 minutes in ms. */
const STALE_MS = 30 * 60 * 1000;

function severityStyle(severity) {
  if (severity === 'critical') return { color: 'var(--sh-threat)', cls: 'sh-threat-pulse' };
  if (severity === 'warning') return { color: 'var(--status-warning)', cls: '' };
  return { color: 'var(--text-primary)', cls: '' };
}

export default function AnomalyDetail({ id, type: _type }) {
  const [anomaly, setAnomaly] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      fetchJson('/api/ml/anomalies'),
      fetchJson('/api/anomalies/explain').catch(err => { console.warn('Optional fetch failed:', err.message); return null; }),
    ])
      .then(([anomalies, explain]) => {
        const list = Array.isArray(anomalies) ? anomalies : (anomalies?.anomalies || []);
        const match = list.find((a) => a.entity_id === id || a.id === id);
        setAnomaly(match || null);
        setExplanation(explain);
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!anomaly) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No anomaly found for entity: {id}
        </p>
      </div>
    );
  }

  const severity = anomaly.severity || 'info';
  const sv = severityStyle(severity);
  const score = anomaly.score !== null && anomaly.score !== undefined ? anomaly.score.toFixed(2) : '\u2014';
  const detectedAt = anomaly.detected_at || anomaly.timestamp;
  const isStale = detectedAt && (Date.now() - new Date(detectedAt).getTime()) > STALE_MS;

  // SHAP attributions from explanation data
  const shapData = explanation?.shap_values || explanation?.feature_attributions || null;
  const pathTrace = explanation?.path_trace || explanation?.trace || null;

  const statsItems = [
    { label: 'Entity', value: anomaly.entity_id || id },
    { label: 'Score', value: score },
    { label: 'Detected', value: relativeTime(detectedAt) },
  ];
  if (anomaly.area) {
    statsItems.push({ label: 'Area', value: anomaly.area });
  }

  // Find max attribution for scaling bars
  const shapEntries = shapData
    ? Object.entries(shapData).sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
    : [];
  const maxAttr = shapEntries.length > 0
    ? Math.max(...shapEntries.map(([, v]) => Math.abs(v)))
    : 1;

  return (
    <div class={`space-y-6 ${isStale ? 'sh-frozen' : ''}`}>
      {/* Summary */}
      <div class={`t-frame ${sv.cls}`} data-label="summary">
        <div style="margin-bottom: 16px;">
          <span
            class="data-mono"
            style={`font-size: var(--type-hero); font-weight: 600; color: ${sv.color}; line-height: 1; text-transform: uppercase;`}
          >
            {severity}
          </span>
        </div>
        <StatsGrid items={statsItems} />
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        {shapEntries.length > 0 ? (
          <div class="space-y-2">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Feature Attributions
            </span>
            {shapEntries.map(([feature, value]) => {
              const pct = (Math.abs(value) / maxAttr) * 100;
              const barColor = value >= 0 ? 'var(--accent)' : 'var(--accent-purple)';
              return (
                <div key={feature} class="flex items-center gap-2" style="font-family: var(--font-mono); font-size: var(--type-label);">
                  <span style="color: var(--text-secondary); min-width: 120px; flex-shrink: 0;">{feature}</span>
                  <div style="flex: 1; height: 8px; background: var(--bg-inset); border-radius: 4px; overflow: hidden;">
                    <div style={`width: ${pct}%; height: 100%; background: ${barColor}; border-radius: 4px;`} />
                  </div>
                  <span style={`color: ${barColor}; min-width: 50px; text-align: right;`}>{value.toFixed(3)}</span>
                </div>
              );
            })}
          </div>
        ) : pathTrace && Array.isArray(pathTrace) ? (
          <div class="space-y-1">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Path Trace
            </span>
            <ol style="margin: 0; padding-left: 20px;">
              {pathTrace.map((step, idx) => (
                <li key={idx} style="font-size: var(--type-label); color: var(--text-secondary); font-family: var(--font-mono); margin-bottom: 4px;">
                  {typeof step === 'string' ? step : step.description || JSON.stringify(step)}
                </li>
              ))}
            </ol>
          </div>
        ) : (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No explanation data available
          </p>
        )}
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        {detectedAt ? (
          <p style="color: var(--text-secondary); font-family: var(--font-mono); font-size: var(--type-label);">
            Anomaly detected {relativeTime(detectedAt)}
            {isStale && (
              <span style="color: var(--sh-frozen); margin-left: 8px;">(stale)</span>
            )}
          </p>
        ) : (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No historical data available
          </p>
        )}
      </div>
    </div>
  );
}
