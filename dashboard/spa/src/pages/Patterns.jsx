import { useState } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Type badge color by pattern type. */
function typeBadgeColor(type) {
  switch ((type || '').toLowerCase()) {
    case 'temporal': return 'bg-blue-100 text-blue-700';
    case 'correlation': return 'bg-purple-100 text-purple-700';
    case 'sequence': return 'bg-amber-100 text-amber-700';
    case 'anomaly': return 'bg-red-100 text-red-700';
    default: return 'bg-gray-100 text-gray-700';
  }
}

function PatternCard({ pattern }) {
  const [expanded, setExpanded] = useState(false);

  const confidence = pattern.confidence ?? 0;
  const pct = Math.round(confidence * 100);
  const entities = pattern.entities || [];

  return (
    <div class="bg-white rounded-lg shadow-sm p-5">
      {/* Header */}
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-base font-bold text-gray-900">{pattern.name || 'Unnamed pattern'}</h3>
        {pattern.type && (
          <span class={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${typeBadgeColor(pattern.type)}`}>
            {pattern.type}
          </span>
        )}
      </div>

      {/* Description */}
      {pattern.description && (
        <p class="text-sm text-gray-600 mb-3">{pattern.description}</p>
      )}

      {/* Details grid */}
      <div class="grid grid-cols-2 gap-x-6 gap-y-2 text-sm mb-3">
        {pattern.frequency && (
          <div>
            <span class="text-gray-400">Frequency</span>
            <div class="font-medium text-gray-700">{pattern.frequency}</div>
          </div>
        )}
        {entities.length > 0 && (
          <div>
            <span class="text-gray-400">Entities</span>
            <div class="font-medium text-gray-700">{entities.length}</div>
          </div>
        )}
        <div>
          <span class="text-gray-400">Confidence</span>
          <div class="font-medium text-gray-700">{pct}%</div>
        </div>
        {pattern.time_window && (
          <div>
            <span class="text-gray-400">Time window</span>
            <div class="font-medium text-gray-700">{pattern.time_window}</div>
          </div>
        )}
        {pattern.support != null && (
          <div>
            <span class="text-gray-400">Support</span>
            <div class="font-medium text-gray-700">{Math.round(pattern.support * 100)}%</div>
          </div>
        )}
      </div>

      {/* Entity list */}
      {entities.length > 0 && (
        <div class="flex flex-wrap gap-1 mb-3">
          {entities.map((eid) => (
            <span key={eid} class="inline-block px-1.5 py-0.5 bg-gray-100 rounded text-xs font-mono text-gray-600">
              {eid}
            </span>
          ))}
        </div>
      )}

      {/* Expandable raw data */}
      {pattern.raw_data && (
        <div>
          <button
            onClick={() => setExpanded(!expanded)}
            class="text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
          >
            {expanded ? 'Hide raw data' : 'Show raw data'}
          </button>
          {expanded && (
            <pre class="mt-2 bg-gray-900 text-gray-100 p-3 rounded text-xs font-mono overflow-x-auto">
              {JSON.stringify(pattern.raw_data, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export default function Patterns() {
  const { data, loading, error, refetch } = useCache('patterns');

  const { metadata, patterns } = useComputed(() => {
    if (!data || !data.data) return { metadata: null, patterns: [] };
    const inner = data.data;
    return {
      metadata: inner.metadata || null,
      patterns: inner.patterns || [],
    };
  }, [data]);

  const pageSubtitle = "Recurring behaviors the system has found — things that happen at the same time, in the same sequence, or that correlate with each other.";

  if (loading && !data) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Patterns</h1>
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
          <h1 class="text-2xl font-bold text-gray-900">Patterns</h1>
          <p class="text-sm text-gray-500">{pageSubtitle}</p>
        </div>
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Patterns</h1>
        <p class="text-sm text-gray-500">{pageSubtitle}</p>
      </div>

      {/* Metadata summary */}
      {metadata && (
        <div class="flex flex-wrap gap-3 text-sm text-gray-500">
          {metadata.time_range && (
            <span class="bg-gray-100 rounded px-2 py-1">Range: {metadata.time_range}</span>
          )}
          {metadata.pattern_count != null && (
            <span class="bg-gray-100 rounded px-2 py-1">{metadata.pattern_count} patterns</span>
          )}
          {metadata.analyzed_at && (
            <span class="bg-gray-100 rounded px-2 py-1">
              Analyzed: {new Date(metadata.analyzed_at).toLocaleString()}
            </span>
          )}
        </div>
      )}

      {patterns.length === 0 ? (
        <div class="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-800">
          No patterns detected yet. The pattern recognition module analyzes HA logbook sequences to find temporal, correlation, and sequence patterns. It needs several days of logbook data with meaningful device events to identify reliable patterns.
        </div>
      ) : (
        <div class="space-y-4">
          {patterns.map((pat, i) => (
            <PatternCard key={pat.name || i} pattern={pat} />
          ))}
        </div>
      )}
    </div>
  );
}
