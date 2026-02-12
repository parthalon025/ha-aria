import { useState } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import StatusBadge from '../components/StatusBadge.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Return badge color classes based on confidence threshold. */
function confidenceColor(confidence) {
  if (confidence >= 0.7) return 'bg-green-100 text-green-700';
  if (confidence >= 0.4) return 'bg-amber-100 text-amber-700';
  return 'bg-red-100 text-red-700';
}

function PredictionCard({ prediction }) {
  const [expanded, setExpanded] = useState(false);

  const confidence = prediction.confidence ?? 0;
  const pct = Math.round(confidence * 100);

  return (
    <div class="bg-white rounded-lg shadow-sm p-4">
      {/* Header */}
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-bold text-gray-900 truncate mr-2 font-mono">
          {prediction.entity_id}
        </h3>
        <span class={`inline-block px-2 py-0.5 rounded-full text-xs font-medium whitespace-nowrap ${confidenceColor(confidence)}`}>
          {pct}%
        </span>
      </div>

      {/* Predicted value */}
      <div class="flex items-center gap-2 mb-3">
        <span class="text-xs text-gray-500">Predicted:</span>
        <StatusBadge state={String(prediction.predicted_value ?? '')} />
        {prediction.current_value != null && (
          <span class="text-xs text-gray-400 ml-2">
            current: {prediction.current_value}
          </span>
        )}
      </div>

      {/* Confidence bar */}
      <div class="mb-3">
        <div class="flex items-center justify-between text-xs text-gray-500 mb-1">
          <span>Confidence</span>
          <span>{pct}%</span>
        </div>
        <div class="h-2 rounded-full bg-gray-200">
          <div
            class="h-2 rounded-full bg-blue-500 transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Model info */}
      <div class="text-xs text-gray-400 space-y-0.5">
        {prediction.model && (
          <div>Model: <span class="text-gray-600">{prediction.model}</span></div>
        )}
        {prediction.prediction_time && (
          <div>Time: <span class="text-gray-600">{new Date(prediction.prediction_time).toLocaleString()}</span></div>
        )}
      </div>

      {/* Expandable details */}
      {prediction.features && (
        <div class="mt-3">
          <button
            onClick={() => setExpanded(!expanded)}
            class="text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
          >
            {expanded ? 'Hide details' : 'Show details'}
          </button>
          {expanded && (
            <pre class="mt-2 bg-gray-900 text-gray-100 p-3 rounded text-xs font-mono overflow-x-auto">
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
          <h1 class="text-2xl font-bold text-gray-900">Predictions</h1>
          <p class="text-sm text-gray-500">{pageSubtitle}</p>
        </div>
        <LoadingState type="cards" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Predictions</h1>
          <p class="text-sm text-gray-500">{pageSubtitle}</p>
        </div>
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Predictions</h1>
        <p class="text-sm text-gray-500">{pageSubtitle}</p>
      </div>

      {/* Metadata summary */}
      {metadata && (
        <div class="flex flex-wrap gap-3 text-sm text-gray-500">
          {metadata.model_version && (
            <span class="bg-gray-100 rounded px-2 py-1">v{metadata.model_version}</span>
          )}
          {metadata.features_used != null && (
            <span class="bg-gray-100 rounded px-2 py-1">{metadata.features_used} features</span>
          )}
          {metadata.generated_at && (
            <span class="bg-gray-100 rounded px-2 py-1">
              Generated: {new Date(metadata.generated_at).toLocaleString()}
            </span>
          )}
        </div>
      )}

      {predictions.length === 0 ? (
        <div class="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-800">
          No entity-level predictions yet. The ML engine trains models after 14+ days of data, then predicts individual entity states. Until then, aggregate predictions are available on the Intelligence page.
        </div>
      ) : (
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {predictions.map((pred, i) => (
            <PredictionCard key={pred.entity_id || i} prediction={pred} />
          ))}
        </div>
      )}
    </div>
  );
}
