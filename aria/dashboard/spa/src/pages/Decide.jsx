import { useState } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { baseUrl } from '../api.js';
import HeroCard from '../components/HeroCard.jsx';
import PageBanner from '../components/PageBanner.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import { Section, Callout } from './intelligence/utils.jsx';

function statusStyle(status) {
  switch ((status || '').toLowerCase()) {
    case 'approved': return 'background: var(--status-healthy-glow); color: var(--status-healthy);';
    case 'rejected': return 'background: var(--status-error-glow); color: var(--status-error);';
    case 'deferred': return 'background: var(--bg-surface-raised); color: var(--text-tertiary);';
    default: return 'background: var(--status-warning-glow); color: var(--status-warning);';
  }
}

function RecommendationCard({ suggestion, onAction, updating }) {
  const confidence = suggestion.confidence ?? 0;
  const pct = Math.round(confidence * 100);
  const status = (suggestion.status || 'pending').toLowerCase();

  return (
    <div class="t-frame" data-label={suggestion.name || 'recommendation'} style="padding: 1.25rem;">
      <div class="flex items-center justify-between mb-2">
        <span class="text-base font-bold" style="color: var(--text-primary)">{suggestion.name || 'Unnamed'}</span>
        <span class="inline-block px-2 py-0.5 rounded-full text-xs font-medium" style={statusStyle(status)}>{status}</span>
      </div>
      {suggestion.description && (
        <p class="text-sm mb-3" style="color: var(--text-secondary)">{suggestion.description}</p>
      )}
      <div class="flex items-center gap-4 text-xs mb-3" style="color: var(--text-tertiary)">
        <span>Confidence: <span class="data-mono font-medium" style="color: var(--text-secondary)">{pct}%</span></span>
        {suggestion.occurrence_count != null && (
          <span>Occurrences: <span class="data-mono font-medium" style="color: var(--text-secondary)">{suggestion.occurrence_count}</span></span>
        )}
      </div>
      {suggestion.yaml && (
        <details class="mb-3">
          <summary class="text-sm cursor-pointer" style="color: var(--accent)">Show YAML</summary>
          <pre class="mt-2 p-3 text-xs overflow-x-auto" style="background: var(--bg-inset); color: var(--text-primary); border-radius: var(--radius); font-family: var(--font-mono)">{suggestion.yaml}</pre>
        </details>
      )}
      {status === 'pending' && (
        <div class="flex gap-2 mt-4">
          <button onClick={() => onAction(suggestion.id, 'approved')} disabled={updating}
            class="t-btn px-4 py-1.5 text-sm font-medium disabled:opacity-50"
            style="background: var(--status-healthy); color: var(--text-inverse); border: none; cursor: pointer;">Approve</button>
          <button onClick={() => onAction(suggestion.id, 'rejected')} disabled={updating}
            class="t-btn px-4 py-1.5 text-sm font-medium disabled:opacity-50"
            style="background: var(--status-error); color: var(--text-inverse); border: none; cursor: pointer;">Reject</button>
          <button onClick={() => onAction(suggestion.id, 'deferred')} disabled={updating}
            class="t-btn px-4 py-1.5 text-sm font-medium disabled:opacity-50"
            style="background: var(--bg-surface-raised); color: var(--text-secondary); border: 1px solid var(--border-subtle); cursor: pointer;">Defer</button>
        </div>
      )}
    </div>
  );
}

export default function Decide() {
  const { data, loading, error, refetch } = useCache('automation_suggestions');
  const [updating, setUpdating] = useState(false);
  const [updateError, setUpdateError] = useState(null);
  const [localStatuses, setLocalStatuses] = useState({});

  const suggestions = useComputed(() => {
    if (!data || !data.data) return [];
    return data.data.suggestions || [];
  }, [data]);

  const displaySuggestions = useComputed(() => {
    return suggestions.map((item) => localStatuses[item.id] ? { ...item, status: localStatuses[item.id] } : item);
  }, [suggestions, localStatuses]);

  const pending = displaySuggestions.filter((item) => (item.status || 'pending').toLowerCase() === 'pending');
  const approved = displaySuggestions.filter((item) => (item.status || '').toLowerCase() === 'approved');
  const rejected = displaySuggestions.filter((item) => (item.status || '').toLowerCase() === 'rejected');
  const deferred = displaySuggestions.filter((item) => (item.status || '').toLowerCase() === 'deferred');

  async function handleAction(id, newStatus) {
    setUpdating(true);
    setUpdateError(null);
    setLocalStatuses((prev) => ({ ...prev, [id]: newStatus }));
    try {
      const updatedSuggestions = suggestions.map((item) =>
        item.id === id ? { ...item, status: newStatus } : item
      );
      const updatedData = { ...(data.data || {}), suggestions: updatedSuggestions };
      const res = await fetch(`${baseUrl}/api/cache/automation_suggestions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: updatedData }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setLocalStatuses((prev) => { const next = { ...prev }; delete next[id]; return next; });
      refetch();
    } catch (err) {
      setLocalStatuses((prev) => { const next = { ...prev }; delete next[id]; return next; });
      setUpdateError(err.message || String(err));
    } finally {
      setUpdating(false);
    }
  }

  if (loading && !data) {
    return (
      <div class="space-y-6">
        <PageBanner page="DECIDE" subtitle="Automation suggestions ARIA has generated based on patterns in your home. Review each one and choose to approve, reject, or defer." />
        <LoadingState type="cards" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <PageBanner page="DECIDE" subtitle="Automation suggestions ARIA has generated based on patterns in your home. Review each one and choose to approve, reject, or defer." />
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="DECIDE" subtitle="Automation suggestions ARIA has generated based on patterns in your home. Review each one and choose to approve, reject, or defer." />

      <HeroCard
        value={pending.length}
        label="pending review"
        delta={`${approved.length} approved \u00b7 ${rejected.length} rejected \u00b7 ${deferred.length} deferred`}
        loading={loading}
      />

      {updateError && <ErrorState error={updateError} onRetry={() => setUpdateError(null)} />}

      {pending.length === 0 && displaySuggestions.length === 0 ? (
        <Callout>No automation suggestions yet. ARIA generates recommendations when it finds patterns with high confidence and matching capabilities.</Callout>
      ) : (
        <>
          {pending.length > 0 && (
            <Section title="Pending Review" summary={`${pending.length} pending`}>
              <div class="space-y-4">
                {pending.map((sug, idx) => (
                  <RecommendationCard key={sug.id || idx} suggestion={sug} onAction={handleAction} updating={updating} />
                ))}
              </div>
            </Section>
          )}

          {(approved.length > 0 || rejected.length > 0 || deferred.length > 0) && (
            <Section title="History" subtitle="Previously reviewed recommendations." defaultOpen={pending.length === 0}>
              <div class="flex gap-4 mb-4 text-sm">
                <span style="color: var(--status-healthy)">{approved.length} approved</span>
                <span style="color: var(--status-error)">{rejected.length} rejected</span>
                <span style="color: var(--text-tertiary)">{deferred.length} deferred</span>
                {displaySuggestions.length > 0 && (
                  <span style="color: var(--text-secondary)">
                    Acceptance rate: {Math.round((approved.length / displaySuggestions.length) * 100)}%
                  </span>
                )}
              </div>
              <div class="space-y-3">
                {[...approved, ...rejected, ...deferred].map((sug, idx) => (
                  <RecommendationCard key={sug.id || idx} suggestion={sug} onAction={handleAction} updating={updating} />
                ))}
              </div>
            </Section>
          )}
        </>
      )}
    </div>
  );
}
