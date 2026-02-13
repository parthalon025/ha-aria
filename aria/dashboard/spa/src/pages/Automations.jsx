import { useState } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { baseUrl } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Status badge inline style. */
function statusStyle(status) {
  switch ((status || '').toLowerCase()) {
    case 'approved': return 'background: rgba(34,197,94,0.15); color: var(--status-healthy);';
    case 'rejected': return 'background: rgba(239,68,68,0.15); color: var(--status-error);';
    default: return 'background: rgba(245,158,11,0.15); color: var(--status-warning);'; // pending
  }
}

/** Render a trigger/condition/action object as readable key-value pairs. */
function ObjectDisplay({ obj }) {
  if (!obj) return null;

  if (typeof obj === 'string') {
    return <span class="text-sm" style="color: var(--text-secondary)">{obj}</span>;
  }

  return (
    <dl class="text-xs space-y-0.5" style="color: var(--text-secondary)">
      {Object.entries(obj).map(([k, v]) => (
        <div key={k} class="flex gap-2">
          <dt class="min-w-[80px]" style="color: var(--text-tertiary)">{k}:</dt>
          <dd class="break-all" style="color: var(--text-secondary)">
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

  const borderStyle = status === 'approved'
    ? 'border-left: 4px solid var(--status-healthy);'
    : status === 'rejected'
      ? 'border-left: 4px solid var(--status-error); opacity: 0.6;'
      : '';

  return (
    <div class="t-card" style={`padding: 1.25rem; ${borderStyle}`}>
      {/* Header */}
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-base font-bold" style="color: var(--text-primary)">{suggestion.name || 'Unnamed automation'}</h3>
        <span class="inline-block px-2 py-0.5 rounded-full text-xs font-medium" style={statusStyle(status)}>
          {status}
        </span>
      </div>

      {/* Description */}
      {suggestion.description && (
        <p class="text-sm mb-3" style="color: var(--text-secondary)">{suggestion.description}</p>
      )}

      {/* Confidence bar */}
      <div class="mb-4">
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

      {/* Trigger */}
      {suggestion.trigger && (
        <div class="mb-3">
          <h4 class="text-xs font-semibold uppercase tracking-wide mb-1" style="color: var(--text-tertiary)">Trigger</h4>
          <ObjectDisplay obj={suggestion.trigger} />
        </div>
      )}

      {/* Conditions */}
      {suggestion.conditions && suggestion.conditions.length > 0 && (
        <div class="mb-3">
          <h4 class="text-xs font-semibold uppercase tracking-wide mb-1" style="color: var(--text-tertiary)">Conditions</h4>
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
          <h4 class="text-xs font-semibold uppercase tracking-wide mb-1" style="color: var(--text-tertiary)">Actions</h4>
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
          <summary class="text-sm cursor-pointer" style="color: var(--accent)">
            Show YAML
          </summary>
          <pre class="mt-2 p-3 text-xs overflow-x-auto" style="background: var(--bg-inset); color: var(--text-primary); border-radius: var(--radius); font-family: var(--font-mono)">
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
            class="t-btn px-4 py-1.5 text-sm font-medium disabled:opacity-50 transition-colors"
            style="background: var(--status-healthy); color: #fff; border: none;"
          >
            Approve
          </button>
          <button
            onClick={() => onStatusChange(suggestion.id, 'rejected')}
            disabled={updating}
            class="t-btn px-4 py-1.5 text-sm font-medium disabled:opacity-50 transition-colors"
            style="background: var(--status-error); color: #fff; border: none;"
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
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Automations</h1>
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
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Automations</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
        </div>
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6">
      <div class="animate-fade-in-up">
        <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Automations</h1>
        <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
      </div>

      {/* Metadata summary */}
      {metadata && (
        <div class="flex flex-wrap gap-3 text-sm animate-fade-in-up delay-100" style="color: var(--text-tertiary)">
          {metadata.total_suggestions != null && (
            <span style="background: var(--bg-surface-raised); border-radius: var(--radius); padding: 0.25rem 0.5rem;">{metadata.total_suggestions} suggestions</span>
          )}
          {metadata.model && (
            <span style="background: var(--bg-surface-raised); border-radius: var(--radius); padding: 0.25rem 0.5rem;">Model: {metadata.model}</span>
          )}
          {metadata.generated_at && (
            <span style="background: var(--bg-surface-raised); border-radius: var(--radius); padding: 0.25rem 0.5rem;">
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
        <div class="t-callout animate-fade-in-up delay-200" style="padding: 0.75rem;">
          <span class="text-sm" style="color: var(--text-secondary)">No automation suggestions yet. The orchestrator generates suggestions when it finds patterns with high enough confidence and matching capabilities. This requires the pattern recognition and discovery modules to have populated data first.</span>
        </div>
      ) : (
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 stagger-children">
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
