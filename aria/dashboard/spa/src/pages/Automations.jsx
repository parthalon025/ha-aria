import { useState } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { baseUrl } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Status badge color classes. */
function statusColor(status) {
  switch ((status || '').toLowerCase()) {
    case 'approved': return 'bg-green-100 text-green-700';
    case 'rejected': return 'bg-red-100 text-red-700';
    default: return 'bg-amber-100 text-amber-700'; // pending
  }
}

/** Render a trigger/condition/action object as readable key-value pairs. */
function ObjectDisplay({ obj }) {
  if (!obj) return null;

  if (typeof obj === 'string') {
    return <span class="text-sm text-gray-700">{obj}</span>;
  }

  return (
    <dl class="text-xs text-gray-600 space-y-0.5">
      {Object.entries(obj).map(([k, v]) => (
        <div key={k} class="flex gap-2">
          <dt class="text-gray-400 min-w-[80px]">{k}:</dt>
          <dd class="text-gray-700 break-all">
            {typeof v === 'object' ? JSON.stringify(v) : String(v)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function AutomationCard({ suggestion, onStatusChange, updating }) {
  const confidence = suggestion.confidence ?? 0;
  const pct = Math.round(confidence * 100);
  const status = (suggestion.status || 'pending').toLowerCase();

  const borderClass = status === 'approved'
    ? 'border-l-4 border-green-500'
    : status === 'rejected'
      ? 'border-l-4 border-red-500 opacity-60'
      : '';

  return (
    <div class={`bg-white rounded-lg shadow-sm p-5 ${borderClass}`}>
      {/* Header */}
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-base font-bold text-gray-900">{suggestion.name || 'Unnamed automation'}</h3>
        <span class={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${statusColor(status)}`}>
          {status}
        </span>
      </div>

      {/* Description */}
      {suggestion.description && (
        <p class="text-sm text-gray-600 mb-3">{suggestion.description}</p>
      )}

      {/* Confidence bar */}
      <div class="mb-4">
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

      {/* Trigger */}
      {suggestion.trigger && (
        <div class="mb-3">
          <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Trigger</h4>
          <ObjectDisplay obj={suggestion.trigger} />
        </div>
      )}

      {/* Conditions */}
      {suggestion.conditions && suggestion.conditions.length > 0 && (
        <div class="mb-3">
          <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Conditions</h4>
          <div class="space-y-1">
            {suggestion.conditions.map((cond, i) => (
              <ObjectDisplay key={i} obj={cond} />
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      {suggestion.actions && suggestion.actions.length > 0 && (
        <div class="mb-3">
          <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Actions</h4>
          <div class="space-y-1">
            {suggestion.actions.map((action, i) => (
              <ObjectDisplay key={i} obj={action} />
            ))}
          </div>
        </div>
      )}

      {/* YAML preview */}
      {suggestion.yaml && (
        <details class="mb-3">
          <summary class="text-sm text-blue-600 hover:text-blue-800 cursor-pointer">
            Show YAML
          </summary>
          <pre class="mt-2 bg-gray-900 text-gray-100 p-3 rounded text-xs font-mono overflow-x-auto">
            {suggestion.yaml}
          </pre>
        </details>
      )}

      {/* Approve / Reject buttons */}
      {status === 'pending' && (
        <div class="flex gap-2 mt-4">
          <button
            onClick={() => onStatusChange(suggestion.id, 'approved')}
            disabled={updating}
            class="px-4 py-1.5 text-sm font-medium text-white bg-green-600 hover:bg-green-700 disabled:opacity-50 rounded-md transition-colors"
          >
            Approve
          </button>
          <button
            onClick={() => onStatusChange(suggestion.id, 'rejected')}
            disabled={updating}
            class="px-4 py-1.5 text-sm font-medium text-white bg-red-600 hover:bg-red-700 disabled:opacity-50 rounded-md transition-colors"
          >
            Reject
          </button>
        </div>
      )}
    </div>
  );
}

export default function Automations() {
  const { data, loading, error, refetch } = useCache('automation_suggestions');
  const [updating, setUpdating] = useState(false);
  const [updateError, setUpdateError] = useState(null);
  // Local overrides for optimistic updates (id → status)
  const [localStatuses, setLocalStatuses] = useState({});

  const { metadata, suggestions } = useComputed(() => {
    if (!data || !data.data) return { metadata: null, suggestions: [] };
    const inner = data.data;
    return {
      metadata: inner.metadata || null,
      suggestions: inner.suggestions || [],
    };
  }, [data]);

  // Apply local status overrides
  const displaySuggestions = useComputed(() => {
    return suggestions.map((s) => {
      if (localStatuses[s.id]) {
        return { ...s, status: localStatuses[s.id] };
      }
      return s;
    });
  }, [suggestions, localStatuses]);

  async function handleStatusChange(id, newStatus) {
    setUpdating(true);
    setUpdateError(null);

    // Optimistic update
    setLocalStatuses((prev) => ({ ...prev, [id]: newStatus }));

    try {
      // Build updated suggestions array with new status
      const updatedSuggestions = suggestions.map((s) =>
        s.id === id ? { ...s, status: newStatus } : s
      );
      const updatedData = {
        ...(data.data || {}),
        suggestions: updatedSuggestions,
      };

      const res = await fetch(`${baseUrl}/api/cache/automation_suggestions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: updatedData }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      // Clear optimistic override and refetch authoritative state
      setLocalStatuses((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      refetch();
    } catch (err) {
      // Revert optimistic update
      setLocalStatuses((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      setUpdateError(err.message || String(err));
    } finally {
      setUpdating(false);
    }
  }

  const pageSubtitle = "Automation ideas generated from discovered patterns. Review, approve, or reject — approved automations can be exported to HA.";

  if (loading && !data) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Automations</h1>
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
          <h1 class="text-2xl font-bold text-gray-900">Automations</h1>
          <p class="text-sm text-gray-500">{pageSubtitle}</p>
        </div>
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Automations</h1>
        <p class="text-sm text-gray-500">{pageSubtitle}</p>
      </div>

      {/* Metadata summary */}
      {metadata && (
        <div class="flex flex-wrap gap-3 text-sm text-gray-500">
          {metadata.total_suggestions != null && (
            <span class="bg-gray-100 rounded px-2 py-1">{metadata.total_suggestions} suggestions</span>
          )}
          {metadata.model && (
            <span class="bg-gray-100 rounded px-2 py-1">Model: {metadata.model}</span>
          )}
          {metadata.generated_at && (
            <span class="bg-gray-100 rounded px-2 py-1">
              Generated: {new Date(metadata.generated_at).toLocaleString()}
            </span>
          )}
        </div>
      )}

      {/* Update error */}
      {updateError && (
        <ErrorState error={updateError} onRetry={() => setUpdateError(null)} />
      )}

      {displaySuggestions.length === 0 ? (
        <div class="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-800">
          No automation suggestions yet. The orchestrator generates suggestions when it finds patterns with high enough confidence and matching capabilities. This requires the pattern recognition and discovery modules to have populated data first.
        </div>
      ) : (
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {displaySuggestions.map((sug, i) => (
            <AutomationCard
              key={sug.id || i}
              suggestion={sug}
              onStatusChange={handleStatusChange}
              updating={updating}
            />
          ))}
        </div>
      )}
    </div>
  );
}
