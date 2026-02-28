/**
 * ConfigDetail — Detail renderer for configuration entries.
 * Three-section layout: Summary (key + value), Explanation (constraints), History (changes).
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';
import { relativeTime } from '../intelligence/utils.jsx';

export default function ConfigDetail({ id, type: _type }) {
  const [config, setConfig] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      fetchJson(`/api/config/${encodeURIComponent(id)}`),
      fetchJson('/api/config-history').catch(err => { console.warn('Optional fetch failed:', err.message); return null; }),
    ])
      .then(([configResult, histResult]) => {
        const data = configResult?.data || configResult || null;
        setConfig(data);

        if (histResult) {
          const hData = histResult?.data || histResult || {};
          const allHist = Array.isArray(hData) ? hData : (hData.entries || hData.changes || hData[id] || []);
          const filtered = Array.isArray(allHist)
            ? allHist.filter((entry) => entry.key === id || entry.config_key === id || !entry.key)
            : [];
          filtered.sort((a, b) => {
            const ta = new Date(a.timestamp || a.changed_at || 0).getTime();
            const tb = new Date(b.timestamp || b.changed_at || 0).getTime();
            return tb - ta;
          });
          setHistory(filtered);
        }
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!config) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No config found for: {id}
        </p>
      </div>
    );
  }

  const currentValue = config.value ?? config.current_value ?? '\u2014';
  const defaultValue = config.default ?? config.default_value ?? null;
  const isDifferent = defaultValue !== null && defaultValue !== undefined && String(currentValue) !== String(defaultValue);

  const statsItems = [
    { label: 'Key', value: config.key || id },
    { label: 'Value', value: String(currentValue) },
  ];
  if (defaultValue !== null && defaultValue !== undefined) {
    statsItems.push({ label: 'Default', value: String(defaultValue) });
  }
  if (config.value_type || config.type) {
    statsItems.push({ label: 'Type', value: config.value_type || config.type });
  }
  if (config.category) {
    statsItems.push({ label: 'Category', value: config.category });
  }

  return (
    <div class="space-y-6">
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        {isDifferent && (
          <div style="margin-bottom: 12px;">
            <span
              style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: var(--type-label); background: var(--status-warning-glow, var(--bg-inset)); color: var(--status-warning);"
            >
              modified from default
            </span>
          </div>
        )}
        <StatsGrid items={statsItems} />
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        {config.description && (
          <p style="color: var(--text-secondary); font-family: var(--font-mono); font-size: var(--type-label); margin-bottom: 12px;">
            {config.description}
          </p>
        )}

        <div class="space-y-1">
          {config.min !== null && config.min !== undefined && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Min</span>
              <span style="color: var(--text-secondary);">{String(config.min)}</span>
            </div>
          )}
          {config.max !== null && config.max !== undefined && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Max</span>
              <span style="color: var(--text-secondary);">{String(config.max)}</span>
            </div>
          )}
          {isDifferent && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Differs from Default</span>
              <span style="color: var(--status-warning);">Yes</span>
            </div>
          )}
        </div>

        {!config.description && (config.min === null || config.min === undefined) && (config.max === null || config.max === undefined) && !isDifferent && (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No additional details available
          </p>
        )}
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        {history.length > 0 ? (
          <div class="space-y-2">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Change History ({history.length})
            </span>
            {history.map((entry, idx) => (
              <div key={idx} style="font-family: var(--font-mono); font-size: var(--type-label); padding: 4px 0; border-bottom: 1px solid var(--border-subtle);">
                <div class="flex gap-3">
                  <span style="color: var(--text-tertiary); min-width: 60px; flex-shrink: 0;">
                    {relativeTime(entry.timestamp || entry.changed_at)}
                  </span>
                  <span style="color: var(--text-secondary); flex: 1;">
                    {entry.old_value !== null && entry.old_value !== undefined && (
                      <span>{String(entry.old_value)} → </span>
                    )}
                    {String(entry.new_value ?? entry.value ?? '\u2014')}
                  </span>
                  {entry.changed_by && (
                    <span style="color: var(--text-tertiary);">{entry.changed_by}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No change history available
          </p>
        )}
      </div>
    </div>
  );
}
