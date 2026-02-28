/**
 * BaselineDetail — Detail renderer for baseline metrics.
 * Three-section layout: Summary (current vs baseline), Explanation (meaning + deviation), History (limited).
 * The id is the metric name.
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';

export default function BaselineDetail({ id, type: _type }) {
  const [baseline, setBaseline] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    fetchJson('/api/cache/intelligence')
      .then((result) => {
        const data = result?.data || result || {};
        const baselines = data.baselines || data.baseline_metrics || {};
        let found = null;

        if (baselines[id]) {
          found = { metric: id, ...baselines[id] };
        } else if (Array.isArray(baselines)) {
          const match = baselines.find(
            (b) => b.metric === id || b.name === id || b.id === id
          );
          if (match) found = match;
        }

        setBaseline(found);
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!baseline) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No baseline found for: {id}
        </p>
      </div>
    );
  }

  const current = baseline.current ?? baseline.current_value ?? null;
  const baselineVal = baseline.baseline ?? baseline.baseline_value ?? baseline.average ?? null;
  let deviation = baseline.deviation ?? baseline.deviation_pct ?? null;

  // Calculate deviation if not provided
  if ((deviation === null || deviation === undefined) && current !== null && current !== undefined && baselineVal !== null && baselineVal !== undefined && baselineVal !== 0) {
    deviation = ((current - baselineVal) / Math.abs(baselineVal)) * 100;
  }

  const isAbove = deviation !== null && deviation !== undefined && deviation > 0;
  const deviationColor = deviation !== null && deviation !== undefined
    ? (isAbove ? 'var(--status-warning)' : 'var(--accent)')
    : 'var(--text-secondary)';

  const statsItems = [
    { label: 'Metric', value: baseline.metric || id },
  ];
  if (current !== null && current !== undefined) {
    statsItems.push({ label: 'Current', value: typeof current === 'number' ? current.toFixed(2) : String(current) });
  }
  if (baselineVal !== null && baselineVal !== undefined) {
    statsItems.push({ label: 'Baseline', value: typeof baselineVal === 'number' ? baselineVal.toFixed(2) : String(baselineVal) });
  }
  if (deviation !== null && deviation !== undefined) {
    statsItems.push({ label: 'Deviation', value: `${deviation >= 0 ? '+' : ''}${deviation.toFixed(1)}%` });
  }

  return (
    <div class="space-y-6">
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        {deviation !== null && deviation !== undefined && (
          <div style="margin-bottom: 12px;">
            <span
              class="data-mono"
              style={`font-size: var(--type-hero); font-weight: 600; line-height: 1; color: ${deviationColor};`}
            >
              {deviation >= 0 ? '+' : ''}{deviation.toFixed(1)}%
            </span>
            <span
              style={`display: inline-block; margin-left: 12px; padding: 2px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: var(--type-label); background: var(--bg-inset); color: ${deviationColor};`}
            >
              {isAbove ? 'above baseline' : 'below baseline'}
            </span>
          </div>
        )}
        <StatsGrid items={statsItems} />
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        <div class="space-y-1">
          <p style="color: var(--text-secondary); font-family: var(--font-mono); font-size: var(--type-label); margin-bottom: 8px;">
            {baseline.description || `This baseline tracks the historical average for "${id}".`}
          </p>

          {current !== null && current !== undefined && baselineVal !== null && baselineVal !== undefined && (
            <p style="color: var(--text-secondary); font-family: var(--font-mono); font-size: var(--type-label);">
              Current value ({typeof current === 'number' ? current.toFixed(2) : current}) is{' '}
              <span style={`color: ${deviationColor};`}>
                {isAbove ? 'above' : 'below'}
              </span>{' '}
              the baseline ({typeof baselineVal === 'number' ? baselineVal.toFixed(2) : baselineVal}).
              {deviation !== null && deviation !== undefined && Math.abs(deviation) > 20 && (
                <span style={`color: ${deviationColor};`}> This is a significant deviation.</span>
              )}
            </p>
          )}

          {baseline.significance !== null && baseline.significance !== undefined && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label); margin-top: 8px;">
              <span style="color: var(--text-tertiary);">Significance</span>
              <span style="color: var(--text-secondary);">{String(baseline.significance)}</span>
            </div>
          )}
          {baseline.sample_size !== null && baseline.sample_size !== undefined && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Sample Size</span>
              <span style="color: var(--text-secondary);">{String(baseline.sample_size)}</span>
            </div>
          )}
        </div>
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          Baseline computed from historical averages
        </p>
      </div>
    </div>
  );
}
