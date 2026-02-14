import { useState } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import HeroCard from '../components/HeroCard.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Return inline style string based on confidence threshold. */
function confidenceColor(confidence) {
  if (confidence >= 0.7) return 'background: var(--status-healthy-glow); color: var(--status-healthy);';
  if (confidence >= 0.4) return 'background: var(--status-warning-glow); color: var(--status-warning);';
  return 'background: var(--status-error-glow); color: var(--status-error);';
}

function PredictionCard({ prediction }) {
  const [expanded, setExpanded] = useState(false);

  const confidence = prediction.confidence ?? 0;
  const pct = Math.round(confidence * 100);

  return (
    <div class="t-frame" data-label={prediction.entity_id || prediction.metric || 'prediction'} style="padding: 1rem;">
      {/* Header */}
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-bold truncate mr-2 data-mono" style="color: var(--text-primary)">
          {prediction.entity_id}
        </h3>
        <span class="inline-block px-2 py-0.5 rounded-full text-xs font-medium whitespace-nowrap" style={confidenceColor(confidence)}>
          {pct}%
        </span>
      </div>

      {/* Predicted value */}
      <div class="flex items-center gap-2 mb-3">
        <span class="text-xs" style="color: var(--text-tertiary)">Predicted:</span>
        <StatusBadge state={String(prediction.predicted_value ?? '')} />
        {prediction.current_value != null && (
          <span class="text-xs ml-2" style="color: var(--text-tertiary)">
            current: {prediction.current_value}
          </span>
        )}
      </div>

      {/* Confidence bar */}
      <div class="mb-3">
        <div class="flex items-center justify-between text-xs mb-1" style="color: var(--text-tertiary)">
          <span>Confidence</span>
          <span>{pct}%</span>
        </div>
        <div class="h-2 rounded-full" style="background: var(--bg-inset)">
          <div
            class="h-2 rounded-full transition-all"
            style={`background: var(--accent); width: ${pct}%`}
          />
        </div>
      </div>

      {/* Model info */}
      <div class="text-xs space-y-0.5" style="color: var(--text-tertiary)">
        {prediction.model && (
          <div>Model: <span style="color: var(--text-secondary)">{prediction.model}</span></div>
        )}
        {prediction.prediction_time && (
          <div>Time: <span style="color: var(--text-secondary)">{new Date(prediction.prediction_time).toLocaleString()}</span></div>
        )}
      </div>

      {/* Expandable details */}
      {prediction.features && (
        <div class="mt-3">
          <button
            onClick={() => setExpanded(!expanded)}
            class="text-sm cursor-pointer"
            style="color: var(--accent)"
          >
            {expanded ? 'Hide details' : 'Show details'}
          </button>
          {expanded && (
            <pre class="mt-2 p-3 text-xs overflow-x-auto" style="background: var(--bg-inset); color: var(--text-primary); border-radius: var(--radius); font-family: var(--font-mono)">
              {JSON.stringify(prediction.features, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export default function Predictions() {
  const { data, loading, error, refetch } = useCache('ml_predictions');

  const { metadata, predictions } = useComputed(() => {
    if (!data || !data.data) return { metadata: null, predictions: [] };
    const inner = data.data;
    return {
      metadata: inner.metadata || null,
      predictions: inner.predictions || [],
    };
  }, [data]);

  const pageSubtitle = "ML model predictions for individual entity states. Each card shows what the model expects and how confident it is.";

  if (loading && !data) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Predictions</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
        </div>
        <LoadingState type="cards" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Predictions</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
        </div>
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6 animate-page-enter">
      <div class="t-section-header" style="padding-bottom: 8px;">
        <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Predictions</h1>
        <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
      </div>

      {/* Hero — what ARIA expects */}
      <HeroCard
        value={predictions.length}
        label="predictions"
        delta={predictions.length > 0 ? `${predictions.filter(p => (p.confidence ?? 0) >= 0.7).length} high confidence` : null}
        loading={loading}
      />

      {/* Metadata summary */}
      {metadata && (
        <div class="flex flex-wrap gap-3 text-sm" style="color: var(--text-tertiary)">
          {metadata.model_version && (
            <span style="background: var(--bg-surface-raised); border-radius: var(--radius); padding: 0.25rem 0.5rem;">v{metadata.model_version}</span>
          )}
          {metadata.features_used != null && (
            <span style="background: var(--bg-surface-raised); border-radius: var(--radius); padding: 0.25rem 0.5rem;">{metadata.features_used} features</span>
          )}
          {metadata.generated_at && (
            <span style="background: var(--bg-surface-raised); border-radius: var(--radius); padding: 0.25rem 0.5rem;">
              Generated: {new Date(metadata.generated_at).toLocaleString()}
            </span>
          )}
        </div>
      )}

      {predictions.length === 0 ? (
        <div class="t-callout" style="padding: 0.75rem;">
          <span class="text-sm" style="color: var(--text-secondary)">No entity-level predictions yet. The ML engine trains models after 14+ days of data, then predicts individual entity states. Until then, aggregate predictions are available on the Intelligence page.</span>
        </div>
      ) : (
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 stagger-children">
          {predictions.map((pred, i) => (
            <PredictionCard key={pred.entity_id || i} prediction={pred} />
          ))}
        </div>
      )}
    </div>
  );
}
