/**
 * ML Engine page — surfaces feature selection, model health, and training history.
 * Fetches from /api/ml/features, /api/ml/models, /api/ml/drift.
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../api.js';
import PageBanner from '../components/PageBanner.jsx';
import HeroCard from '../components/HeroCard.jsx';
import CollapsibleSection from '../components/CollapsibleSection.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function Callout({ children }) {
  return (
    <div class="t-callout p-3 text-sm">
      {children}
    </div>
  );
}

function Badge({ label, color }) {
  return (
    <span
      class="data-mono"
      style={`display: inline-block; font-size: var(--type-label); padding: 2px 8px; border-radius: var(--radius); background: ${color}; color: var(--bg-base); font-weight: 600;`}
    >
      {label}
    </span>
  );
}

function interpretationColor(interpretation) {
  switch (interpretation) {
    case 'stable': return 'var(--status-healthy)';
    case 'behavioral_drift': return 'var(--status-warning)';
    case 'meta_learner_error': return 'var(--status-error)';
    case 'meta_learner_improvement': return 'var(--accent)';
    default: return 'var(--text-tertiary)';
  }
}

function formatDate(ts) {
  if (!ts) return '\u2014';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return '\u2014';
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

function formatPct(val) {
  if (val == null) return '\u2014';
  return `${(val * 100).toFixed(1)}%`;
}

// ─── Section A: Feature Selection ─────────────────────────────────────────────

function FeatureSelection({ features, loading }) {
  const selected = features?.selected || [];
  const total = features?.total || 0;
  const method = features?.method || 'none';
  const lastComputed = features?.last_computed;
  const isEmpty = selected.length === 0;

  const methodLabels = {
    mrmr: 'mRMR',
    importance: 'Importance',
    none: 'None',
  };

  return (
    <CollapsibleSection
      title="Feature Selection"
      subtitle="mRMR-ranked input signals"
      summary={isEmpty ? 'not yet computed' : `${selected.length} of ${total}`}
      defaultOpen={true}
      loading={loading}
    >
      {isEmpty ? (
        <Callout>Feature selection activates after the first ML training cycle.</Callout>
      ) : (
        <div class="space-y-4">
          <HeroCard
            value={`${selected.length} of ${total}`}
            label="Selected Features"
          />

          <div class="flex items-center gap-3 flex-wrap" style="margin-top: 12px;">
            <Badge label={methodLabels[method] || method} color="var(--accent)" />
            {lastComputed && (
              <span class="data-mono" style="font-size: var(--type-label); color: var(--text-tertiary);">
                computed {formatDate(lastComputed)}
              </span>
            )}
          </div>

          <div class="t-frame" data-label="Ranked Features">
            <ol style="list-style: none; padding: 0; margin: 0;">
              {selected.map((name, i) => (
                <li
                  key={name}
                  style={`display: flex; align-items: center; gap: 8px; padding: 6px 0; font-family: var(--font-mono); font-size: var(--type-body); color: var(--text-primary);${i > 0 ? ' border-top: 1px solid var(--border-subtle);' : ''}`}
                >
                  <span
                    class="data-mono"
                    style="min-width: 28px; text-align: right; color: var(--text-tertiary); font-size: var(--type-label);"
                  >
                    {i + 1}.
                  </span>
                  <span>{name}</span>
                </li>
              ))}
            </ol>
          </div>

          <p style="font-size: var(--type-label); color: var(--text-tertiary); line-height: 1.5; margin-top: 8px;">
            Feature selection picks the most informative signals from your home's data.
            Using mRMR (minimum Redundancy Maximum Relevance), ARIA selects features
            that are individually useful but not redundant with each other.
          </p>
        </div>
      )}
    </CollapsibleSection>
  );
}

// ─── Section B: Model Health ──────────────────────────────────────────────────

function ModelHealth({ models, loading }) {
  const reference = models?.reference;
  const incremental = models?.incremental;
  const forecaster = models?.forecaster;
  const isEmpty = !reference && !incremental && !forecaster;

  return (
    <CollapsibleSection
      title="Model Health"
      subtitle="Reference comparison, training mode, forecaster"
      summary={isEmpty ? 'awaiting data' : (reference?.interpretation || 'active')}
      defaultOpen={true}
      loading={loading}
    >
      {isEmpty ? (
        <Callout>Model health data appears after the first training cycle.</Callout>
      ) : (
        <div class="space-y-4">
          {/* Reference model comparison */}
          {reference && (
            <div class="t-frame" data-label="Reference Model">
              <div class="flex items-center gap-3 flex-wrap" style="margin-bottom: 12px;">
                <Badge
                  label={reference.interpretation || 'unknown'}
                  color={interpretationColor(reference.interpretation)}
                />
              </div>

              <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div class="t-frame" data-label="Primary Trend">
                  <span class="data-mono" style="font-size: var(--type-headline); color: var(--text-primary);">
                    {reference.primary_trend != null ? formatPct(reference.primary_trend) : '\u2014'}
                  </span>
                </div>
                <div class="t-frame" data-label="Reference Trend">
                  <span class="data-mono" style="font-size: var(--type-headline); color: var(--text-primary);">
                    {reference.reference_trend != null ? formatPct(reference.reference_trend) : '\u2014'}
                  </span>
                </div>
                <div class="t-frame" data-label="Divergence">
                  <span class="data-mono" style="font-size: var(--type-headline); color: var(--text-primary);">
                    {reference.divergence != null ? formatPct(reference.divergence) : '\u2014'}
                  </span>
                </div>
              </div>

              <p style="font-size: var(--type-label); color: var(--text-tertiary); line-height: 1.5; margin-top: 12px;">
                A reference model runs alongside ARIA's main model without any meta-learning
                adjustments. Comparing the two reveals whether accuracy changes come from your
                home changing (drift) or from ARIA's learning process.
              </p>
            </div>
          )}

          {/* Incremental training */}
          {incremental && (
            <div class="t-frame" data-label="Incremental Training">
              <div class="flex items-center gap-3 flex-wrap" style="margin-bottom: 12px;">
                <Badge
                  label={incremental.mode === 'incremental' ? 'Incremental' : 'Full Retrain'}
                  color={incremental.mode === 'incremental' ? 'var(--accent)' : 'var(--text-secondary)'}
                />
                {incremental.tree_count != null && (
                  <span class="data-mono" style="font-size: var(--type-label); color: var(--text-tertiary);">
                    {incremental.tree_count} / {incremental.max_trees || '\u2014'} trees
                  </span>
                )}
              </div>

              <p style="font-size: var(--type-label); color: var(--text-tertiary); line-height: 1.5;">
                Instead of rebuilding models from scratch, incremental training adds new trees
                to existing models — faster and preserves learned knowledge.
              </p>
            </div>
          )}

          {/* Forecaster backend */}
          {forecaster && (
            <div class="t-frame" data-label="Forecaster Backend">
              <div class="flex items-center gap-3 flex-wrap">
                <Badge
                  label={forecaster.backend || 'auto'}
                  color="var(--accent)"
                />
                {forecaster.backend === 'neuralprophet' && forecaster.config && (
                  <span class="data-mono" style="font-size: var(--type-label); color: var(--text-tertiary);">
                    AR({forecaster.config.ar_order || '?'}) &middot; {forecaster.config.epochs || '?'} epochs &middot; lr {forecaster.config.learning_rate || '?'}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </CollapsibleSection>
  );
}

// ─── Section C: Training History ──────────────────────────────────────────────

function TrainingHistory({ models, loading }) {
  const mlModels = models?.ml_models;
  const isEmpty = !mlModels;

  const lastTrained = mlModels?.last_trained;
  const scores = mlModels?.scores;
  const hasScores = scores && typeof scores === 'object' && Object.keys(scores).length > 0;

  return (
    <CollapsibleSection
      title="Training History"
      subtitle="Last retrain and model accuracy"
      summary={isEmpty ? 'no data' : (lastTrained ? formatDate(lastTrained) : 'pending')}
      defaultOpen={true}
      loading={loading}
    >
      {isEmpty ? (
        <Callout>Training history will populate after the first ML training cycle.</Callout>
      ) : (
        <div class="space-y-4">
          {lastTrained && (
            <div class="t-frame" data-label="Last Trained">
              <span class="data-mono" style="font-size: var(--type-headline); color: var(--accent);">
                {formatDate(lastTrained)}
              </span>
            </div>
          )}

          {hasScores && (
            <div class="t-frame" data-label="Model Scores">
              <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: var(--type-body);">
                  <thead>
                    <tr style="border-bottom: 2px solid var(--border-subtle);">
                      <th style="text-align: left; padding: 8px 12px; color: var(--text-secondary); font-weight: 600;">Model</th>
                      <th style="text-align: right; padding: 8px 12px; color: var(--text-secondary); font-weight: 600;">R&sup2;</th>
                      <th style="text-align: right; padding: 8px 12px; color: var(--text-secondary); font-weight: 600;">MAE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(scores).map(([name, vals]) => (
                      <tr key={name} style="border-bottom: 1px solid var(--border-subtle);">
                        <td style="padding: 8px 12px; color: var(--text-primary);">{name}</td>
                        <td style="text-align: right; padding: 8px 12px; color: var(--accent);">
                          {vals?.r2 != null ? vals.r2.toFixed(3) : '\u2014'}
                        </td>
                        <td style="text-align: right; padding: 8px 12px; color: var(--text-primary);">
                          {vals?.mae != null ? vals.mae.toFixed(3) : '\u2014'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <p style="font-size: var(--type-label); color: var(--text-tertiary); line-height: 1.5; margin-top: 8px;">
            Training history shows when ARIA last retrained its models and how accurate
            they are. R&sup2; measures how well the model explains variance (1.0 = perfect),
            MAE measures average prediction error (lower = better).
          </p>
        </div>
      )}
    </CollapsibleSection>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function MLEngine() {
  const [features, setFeatures] = useState(null);
  const [models, setModels] = useState(null);
  const [drift, setDrift] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  function load() {
    setLoading(true);
    setError(null);
    Promise.all([
      fetchJson('/api/ml/features'),
      fetchJson('/api/ml/models'),
      fetchJson('/api/ml/drift'),
    ])
      .then(([f, m, d]) => {
        setFeatures(f);
        setModels(m);
        setDrift(d);
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }

  useEffect(() => { load(); }, []);

  if (error) {
    return (
      <div>
        <PageBanner page="MLENGINE" />
        <ErrorState error={error} onRetry={load} />
      </div>
    );
  }

  if (loading) {
    return (
      <div>
        <PageBanner page="MLENGINE" />
        <LoadingState type="full" />
      </div>
    );
  }

  return (
    <div class="space-y-6">
      <PageBanner page="MLENGINE" subtitle="Feature selection, model health, and training history" />

      <FeatureSelection features={features} loading={false} />
      <ModelHealth models={models} loading={false} />
      <TrainingHistory models={models} loading={false} />
    </div>
  );
}
