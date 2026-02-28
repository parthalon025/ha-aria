/**
 * CapabilityDetail — Detail renderer for capability registry entries.
 * Three-section layout: Summary (name + status), Explanation (deps + features), History (health).
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';
import { relativeTime } from '../intelligence/utils.jsx';

function capStatusColor(status) {
  if (status === 'active') return 'background: var(--status-healthy-glow); color: var(--status-healthy);';
  if (status === 'archived') return 'background: var(--status-error-glow); color: var(--status-error);';
  return 'background: var(--bg-inset); color: var(--text-secondary);';
}

export default function CapabilityDetail({ id, type: _type }) {
  const [capability, setCapability] = useState(null);
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      fetchJson(`/api/capabilities/registry/${encodeURIComponent(id)}`),
      fetchJson('/api/capabilities/history').catch(err => { console.warn('Optional fetch failed:', err.message); return null; }),
    ])
      .then(([capResult, histResult]) => {
        const data = capResult?.data || capResult || null;
        setCapability(data);
        setHistory(histResult?.data || histResult || null);
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!capability) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No capability found for: {id}
        </p>
      </div>
    );
  }

  const status = capability.status || 'candidate';
  const statsItems = [
    { label: 'Name', value: capability.name || id },
    { label: 'Status', value: status },
  ];
  if (capability.layer) {
    statsItems.push({ label: 'Layer', value: capability.layer });
  }
  if (capability.can_predict !== null && capability.can_predict !== undefined) {
    statsItems.push({ label: 'Can Predict', value: capability.can_predict ? 'Yes' : 'No' });
  }

  const dependencies = capability.dependencies || capability.deps || [];
  const features = capability.features || capability.feature_list || [];
  const health = capability.health || {};

  // History entries from capabilities/history endpoint
  const histEntries = history
    ? (Array.isArray(history) ? history : (history.entries || history[id] || []))
    : [];
  const filteredHist = Array.isArray(histEntries)
    ? histEntries.filter((entry) => entry.capability === id || entry.name === id || !entry.capability)
    : [];

  return (
    <div class="space-y-6">
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        <div style="margin-bottom: 12px;">
          <span
            style={`display: inline-block; padding: 2px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: var(--type-label); text-transform: uppercase; ${capStatusColor(status)}`}
          >
            {status}
          </span>
        </div>
        <StatsGrid items={statsItems} />
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        {capability.description && (
          <p style="color: var(--text-secondary); font-family: var(--font-mono); font-size: var(--type-label); margin-bottom: 12px;">
            {capability.description}
          </p>
        )}

        {dependencies.length > 0 && (
          <div class="space-y-1" style="margin-bottom: 12px;">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Dependencies
            </span>
            {dependencies.map((dep, idx) => (
              <div key={idx} style="font-family: var(--font-mono); font-size: var(--type-label); color: var(--text-secondary);">
                {typeof dep === 'string' ? dep : (dep.name || JSON.stringify(dep))}
              </div>
            ))}
          </div>
        )}

        {features.length > 0 && (
          <div class="space-y-1">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Features
            </span>
            {features.map((feat, idx) => (
              <div key={idx} style="font-family: var(--font-mono); font-size: var(--type-label); color: var(--text-secondary);">
                {typeof feat === 'string' ? feat : (feat.name || JSON.stringify(feat))}
              </div>
            ))}
          </div>
        )}

        {dependencies.length === 0 && features.length === 0 && !capability.description && (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No additional details available
          </p>
        )}
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        {health.last_seen && (
          <div class="space-y-1" style="margin-bottom: 12px;">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Health
            </span>
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Last Seen</span>
              <span style="color: var(--text-secondary);">{relativeTime(health.last_seen)}</span>
            </div>
            {health.error_count !== null && health.error_count !== undefined && (
              <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary);">Error Count</span>
                <span style={`color: ${health.error_count > 0 ? 'var(--status-error)' : 'var(--text-secondary)'};`}>
                  {health.error_count}
                </span>
              </div>
            )}
          </div>
        )}

        {filteredHist.length > 0 ? (
          <div class="space-y-1">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Discovery History
            </span>
            {filteredHist.map((entry, idx) => (
              <div key={idx} class="flex gap-3" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary); min-width: 60px; flex-shrink: 0;">
                  {relativeTime(entry.timestamp || entry.time)}
                </span>
                <span style="color: var(--text-secondary);">
                  {entry.event || entry.action || 'update'}
                </span>
                <span style="color: var(--text-tertiary); flex: 1; text-align: right;">
                  {entry.detail || ''}
                </span>
              </div>
            ))}
          </div>
        ) : !health.last_seen ? (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No historical data available
          </p>
        ) : null}
      </div>
    </div>
  );
}
