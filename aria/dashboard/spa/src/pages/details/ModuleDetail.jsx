/**
 * ModuleDetail — Detail renderer for ARIA modules.
 * Three-section layout: Summary (status + uptime), Explanation (description + deps), History (errors + events).
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';
import { relativeTime, durationSince } from '../intelligence/utils.jsx';

function moduleStatusColor(status) {
  if (status === 'running') return 'background: var(--status-healthy-glow); color: var(--status-healthy);';
  if (status === 'error') return 'background: var(--status-error-glow); color: var(--status-error);';
  return 'background: var(--bg-inset); color: var(--text-secondary);';
}

export default function ModuleDetail({ id, type: _type }) {
  const [moduleData, setModuleData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    fetchJson(`/api/modules/${encodeURIComponent(id)}`)
      .then((result) => {
        const data = result?.data || result || null;
        setModuleData(data);
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!moduleData) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No module found for: {id}
        </p>
      </div>
    );
  }

  const status = moduleData.status || 'unknown';
  const statsItems = [
    { label: 'Module', value: moduleData.name || id },
    { label: 'Status', value: status },
  ];
  if (moduleData.started_at || moduleData.startup_time) {
    statsItems.push({ label: 'Uptime', value: durationSince(moduleData.started_at || moduleData.startup_time) });
  }
  if (moduleData.event_count !== null && moduleData.event_count !== undefined) {
    statsItems.push({ label: 'Events', value: String(moduleData.event_count) });
  }

  const cacheCategories = moduleData.cache_categories || moduleData.categories || [];
  const dependencies = moduleData.dependencies || moduleData.deps || [];

  return (
    <div class="space-y-6">
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        <div style="margin-bottom: 12px;">
          <span
            style={`display: inline-block; padding: 2px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: var(--type-label); text-transform: uppercase; ${moduleStatusColor(status)}`}
          >
            {status}
          </span>
        </div>
        <StatsGrid items={statsItems} />
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        {moduleData.description && (
          <p style="color: var(--text-secondary); font-family: var(--font-mono); font-size: var(--type-label); margin-bottom: 12px;">
            {moduleData.description}
          </p>
        )}

        {cacheCategories.length > 0 && (
          <div class="space-y-1" style="margin-bottom: 12px;">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Cache Categories
            </span>
            {cacheCategories.map((cat, idx) => (
              <div key={idx} style="font-family: var(--font-mono); font-size: var(--type-label); color: var(--text-secondary);">
                {typeof cat === 'string' ? cat : (cat.name || JSON.stringify(cat))}
              </div>
            ))}
          </div>
        )}

        {dependencies.length > 0 && (
          <div class="space-y-1">
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

        {!moduleData.description && cacheCategories.length === 0 && dependencies.length === 0 && (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No additional details available
          </p>
        )}
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        {moduleData.last_error && (
          <div class="space-y-1" style="margin-bottom: 12px;">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Last Error
            </span>
            <p style="color: var(--status-error); font-family: var(--font-mono); font-size: var(--type-label); word-break: break-word;">
              {typeof moduleData.last_error === 'object' ? JSON.stringify(moduleData.last_error) : String(moduleData.last_error)}
            </p>
            {moduleData.last_error_at && (
              <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
                {relativeTime(moduleData.last_error_at)}
              </p>
            )}
          </div>
        )}

        {moduleData.started_at && (
          <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
            <span style="color: var(--text-tertiary);">Started</span>
            <span style="color: var(--text-secondary);">{relativeTime(moduleData.started_at)}</span>
          </div>
        )}

        {!moduleData.last_error && !moduleData.started_at && (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No historical data available
          </p>
        )}
      </div>
    </div>
  );
}
