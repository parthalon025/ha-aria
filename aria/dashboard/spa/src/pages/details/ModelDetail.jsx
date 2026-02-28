/**
 * ModelDetail — Detail renderer for ML models.
 * Three-section layout: Summary (type + accuracy), Explanation (features + params), History (training + drift).
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';
import { relativeTime } from '../intelligence/utils.jsx';

export default function ModelDetail({ id, type: _type }) {
  const [model, setModel] = useState(null);
  const [drift, setDrift] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      fetchJson('/api/ml/models'),
      fetchJson('/api/ml/drift').catch(err => { console.warn('Optional fetch failed:', err.message); return null; }),
    ])
      .then(([modelsResult, driftResult]) => {
        const data = modelsResult?.data || modelsResult || {};
        const list = Array.isArray(data) ? data : (data.models || []);

        // Try matching by id, name, or model_type
        const match = list.find(
          (m) => m.id === id || m.name === id || m.model_type === id || String(m.id) === id
        );
        // If no direct match, try the data as a dict keyed by model name
        if (!match && !Array.isArray(data) && data[id]) {
          setModel(data[id]);
        } else {
          setModel(match || null);
        }

        setDrift(driftResult?.data || driftResult || null);
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!model) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No model found for: {id}
        </p>
      </div>
    );
  }

  const accuracy = model.accuracy ?? model.score ?? model.metric ?? null;
  const statsItems = [
    { label: 'Model', value: model.name || model.model_type || id },
    { label: 'Status', value: model.status || 'unknown' },
  ];
  if (accuracy !== null && accuracy !== undefined) {
    statsItems.push({ label: 'Accuracy', value: `${(accuracy * 100).toFixed(1)}%` });
  }
  if (model.last_trained || model.trained_at) {
    statsItems.push({ label: 'Last Trained', value: relativeTime(model.last_trained || model.trained_at) });
  }

  // Feature importance
  const featureImportance = model.feature_importance || model.feature_importances || null;
  const fiEntries = featureImportance
    ? Object.entries(featureImportance).sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
    : [];
  const maxFi = fiEntries.length > 0
    ? Math.max(...fiEntries.map(([, v]) => Math.abs(v)))
    : 1;

  // Hyperparameters
  const hyperparams = model.hyperparameters || model.params || model.config || null;
  const hpEntries = hyperparams ? Object.entries(hyperparams) : [];

  // Training history
  const trainingHistory = model.training_history || model.history || [];

  // Drift for this model
  const modelDrift = drift
    ? (drift[id] || drift[model.name] || drift[model.model_type] || null)
    : null;

  return (
    <div class="space-y-6">
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        <StatsGrid items={statsItems} />
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        {fiEntries.length > 0 ? (
          <div class="space-y-2" style="margin-bottom: 16px;">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Feature Importance
            </span>
            {fiEntries.map(([feature, value]) => {
              const pct = (Math.abs(value) / maxFi) * 100;
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
        ) : null}

        {hpEntries.length > 0 ? (
          <div class="space-y-1">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Hyperparameters
            </span>
            {hpEntries.map(([key, val]) => (
              <div key={key} class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary);">{key}</span>
                <span style="color: var(--text-secondary); max-width: 60%; text-align: right; word-break: break-word;">
                  {typeof val === 'object' ? JSON.stringify(val) : String(val ?? '\u2014')}
                </span>
              </div>
            ))}
          </div>
        ) : null}

        {fiEntries.length === 0 && hpEntries.length === 0 && (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No model details available
          </p>
        )}
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        {modelDrift && (
          <div class="space-y-1" style="margin-bottom: 12px;">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Drift Status
            </span>
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Status</span>
              <span style={`color: ${modelDrift.status === 'stable' ? 'var(--status-healthy)' : 'var(--status-warning)'};`}>
                {modelDrift.status || 'unknown'}
              </span>
            </div>
            {modelDrift.score !== null && modelDrift.score !== undefined && (
              <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary);">Drift Score</span>
                <span style="color: var(--text-secondary);">{Number(modelDrift.score).toFixed(4)}</span>
              </div>
            )}
          </div>
        )}

        {Array.isArray(trainingHistory) && trainingHistory.length > 0 ? (
          <div class="space-y-1">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Training History ({trainingHistory.length})
            </span>
            {trainingHistory.map((entry, idx) => (
              <div key={idx} class="flex gap-3" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary); min-width: 60px; flex-shrink: 0;">
                  {relativeTime(entry.timestamp || entry.trained_at)}
                </span>
                <span style="color: var(--text-secondary);">
                  {entry.event || 'trained'}
                </span>
                {entry.accuracy !== null && entry.accuracy !== undefined && (
                  <span style="color: var(--text-tertiary); flex: 1; text-align: right;">
                    accuracy: {(entry.accuracy * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            ))}
          </div>
        ) : !modelDrift ? (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No training history available
          </p>
        ) : null}
      </div>
    </div>
  );
}
