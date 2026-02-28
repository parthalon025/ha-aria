/**
 * DriftDetail — Detail renderer for ML drift detection data.
 * Three-section layout: Summary (overall status), Explanation (detector rows), History (timestamps).
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';
import { relativeTime } from '../intelligence/utils.jsx';

function driftStatusColor(status) {
  if (status === 'stable') return 'color: var(--status-healthy);';
  if (status === 'critical') return 'color: var(--sh-threat);';
  return 'color: var(--status-warning);';
}

export default function DriftDetail({ id, type: _type }) {
  const [driftData, setDriftData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    fetchJson('/api/ml/drift')
      .then((result) => {
        const data = result?.data || result || {};
        // Drift data could be keyed by model id or be a flat object
        if (data[id]) {
          setDriftData(data[id]);
        } else if (data.status || data.detectors) {
          // Single drift result (not keyed)
          setDriftData(data);
        } else {
          // Try to find in an array
          const list = Array.isArray(data) ? data : [];
          const match = list.find((d) => d.id === id || d.model === id || d.name === id);
          setDriftData(match || data || null);
        }
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!driftData) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No drift data found for: {id}
        </p>
      </div>
    );
  }

  const overallStatus = driftData.status || driftData.drift_status || 'unknown';
  const method = driftData.detection_method || driftData.method || null;

  const statsItems = [
    { label: 'Status', value: overallStatus },
  ];
  if (method) {
    statsItems.push({ label: 'Method', value: method });
  }
  if (driftData.last_checked || driftData.timestamp) {
    statsItems.push({ label: 'Last Checked', value: relativeTime(driftData.last_checked || driftData.timestamp) });
  }

  // Detectors: page_hinkley, adwin, rolling_mae, etc.
  const detectors = driftData.detectors || {};
  const detectorEntries = Object.entries(detectors);

  // Also check for top-level detector values
  const pageHinkley = driftData.page_hinkley || detectors.page_hinkley || null;
  const adwin = driftData.adwin || detectors.adwin || null;
  const rollingMae = driftData.rolling_mae || detectors.rolling_mae || null;

  // Build detector rows if no nested detectors object
  const detectorRows = detectorEntries.length > 0
    ? detectorEntries
    : [
        ...(pageHinkley ? [['page_hinkley', pageHinkley]] : []),
        ...(adwin ? [['adwin', adwin]] : []),
        ...(rollingMae ? [['rolling_mae', rollingMae]] : []),
      ];

  // History / trend
  const trend = driftData.trend || driftData.history || [];
  const detectionTimestamps = driftData.detection_timestamps || [];

  return (
    <div class="space-y-6">
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        <div style="margin-bottom: 12px;">
          <span
            class="data-mono"
            style={`font-size: var(--type-hero); font-weight: 600; line-height: 1; text-transform: uppercase; ${driftStatusColor(overallStatus)}`}
          >
            {overallStatus}
          </span>
        </div>
        <StatsGrid items={statsItems} />
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        {detectorRows.length > 0 ? (
          <div class="space-y-2">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Detectors
            </span>
            {detectorRows.map(([name, det]) => {
              const detObj = typeof det === 'object' ? det : { score: det };
              const score = detObj.score ?? detObj.value ?? null;
              const threshold = detObj.threshold ?? null;
              const triggered = detObj.triggered ?? detObj.is_triggered ?? ((score !== null && score !== undefined) && (threshold !== null && threshold !== undefined) ? score > threshold : null);
              return (
                <div key={name} style="font-family: var(--font-mono); font-size: var(--type-label); padding: 8px 0; border-bottom: 1px solid var(--border-subtle);">
                  <div class="flex justify-between items-center">
                    <span style="color: var(--text-secondary); font-weight: 500;">{name}</span>
                    {triggered !== null && triggered !== undefined && (
                      <span
                        style={`padding: 1px 6px; border-radius: 3px; font-size: var(--type-label); ${
                          triggered
                            ? 'background: var(--status-error-glow); color: var(--status-error);'
                            : 'background: var(--status-healthy-glow); color: var(--status-healthy);'
                        }`}
                      >
                        {triggered ? 'triggered' : 'ok'}
                      </span>
                    )}
                  </div>
                  <div class="flex gap-4" style="margin-top: 4px;">
                    {score !== null && score !== undefined && (
                      <span style="color: var(--text-tertiary);">
                        score: <span style="color: var(--text-secondary);">{Number(score).toFixed(4)}</span>
                      </span>
                    )}
                    {threshold !== null && threshold !== undefined && (
                      <span style="color: var(--text-tertiary);">
                        threshold: <span style="color: var(--text-secondary);">{Number(threshold).toFixed(4)}</span>
                      </span>
                    )}
                  </div>
                  {detObj.window_size !== null && detObj.window_size !== undefined && (
                    <div style="margin-top: 2px; color: var(--text-tertiary);">
                      window: <span style="color: var(--text-secondary);">{detObj.window_size}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No detector data available
          </p>
        )}
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        {Array.isArray(trend) && trend.length > 0 ? (
          <div class="space-y-1">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Drift Trend
            </span>
            {trend.map((entry, idx) => (
              <div key={idx} class="flex gap-3" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary); min-width: 60px; flex-shrink: 0;">
                  {relativeTime(entry.timestamp || entry.time)}
                </span>
                <span style={driftStatusColor(entry.status || 'unknown')}>
                  {entry.status || 'check'}
                </span>
                {entry.score !== null && entry.score !== undefined && (
                  <span style="color: var(--text-tertiary); flex: 1; text-align: right;">
                    score: {Number(entry.score).toFixed(4)}
                  </span>
                )}
              </div>
            ))}
          </div>
        ) : detectionTimestamps.length > 0 ? (
          <div class="space-y-1">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Detection Timestamps
            </span>
            {detectionTimestamps.map((ts, idx) => (
              <div key={idx} style="font-family: var(--font-mono); font-size: var(--type-label); color: var(--text-secondary);">
                {relativeTime(ts)}
              </div>
            ))}
          </div>
        ) : (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No drift history available
          </p>
        )}
      </div>
    </div>
  );
}
