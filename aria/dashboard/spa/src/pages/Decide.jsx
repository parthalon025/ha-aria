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

const STATUS_ICONS = {
  approved: '\u2713',
  rejected: '\u2717',
  deferred: '\u23F8',
  pending: '\u25CF',
};

/** Detect conflicts: overlapping triggers/entities between suggestions. */
function detectConflicts(suggestion, allSuggestions) {
  const conflicts = [];
  const entities = suggestion.entities || [];
  const trigger = suggestion.trigger || '';
  if (!entities.length && !trigger) return conflicts;

  for (const other of allSuggestions) {
    if (other.id === suggestion.id) continue;
    if ((other.status || 'pending').toLowerCase() === 'rejected') continue;

    const otherEntities = other.entities || [];
    const overlap = entities.filter((ent) => otherEntities.includes(ent));
    const triggerMatch = trigger && other.trigger && trigger === other.trigger;

    if (overlap.length > 0 || triggerMatch) {
      conflicts.push({
        name: other.name || 'Unnamed',
        reason: overlap.length > 0
          ? `Shares ${overlap.length} entit${overlap.length === 1 ? 'y' : 'ies'}: ${overlap.slice(0, 3).join(', ')}${overlap.length > 3 ? '...' : ''}`
          : 'Same trigger',
      });
    }
  }
  return conflicts;
}

/** Confidence tier badge with semantic coloring */
function ConfidenceBadge({ confidence }) {
  const pct = Math.round((confidence ?? 0) * 100);
  let color, bg, tier;
  if (pct >= 80) {
    color = 'var(--status-healthy)';
    bg = 'var(--status-healthy-glow)';
    tier = 'High';
  } else if (pct >= 50) {
    color = 'var(--status-warning)';
    bg = 'var(--status-warning-glow)';
    tier = 'Medium';
  } else {
    color = 'var(--status-error)';
    bg = 'var(--status-error-glow)';
    tier = 'Low';
  }
  return (
    <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium" style={`background: ${bg}; color: ${color}`}>
      <span class="data-mono">{pct}%</span>
      <span>{tier}</span>
    </span>
  );
}

/** Visual health bar — shows approval/rejection/deferral proportions */
function HealthBar({ approved, rejected, deferred, total }) {
  if (total === 0) return null;
  const apPct = (approved / total) * 100;
  const rjPct = (rejected / total) * 100;
  const dfPct = (deferred / total) * 100;
  const pdPct = 100 - apPct - rjPct - dfPct;

  return (
    <div class="t-frame" data-label="decision health" style="padding: 1rem;">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs font-bold uppercase" style="color: var(--text-tertiary)">Decision Health</span>
        <span class="text-xs data-mono" style="color: var(--text-secondary)">
          {total} total
        </span>
      </div>
      <div class="flex rounded overflow-hidden" style="height: 8px; background: var(--bg-inset);">
        {apPct > 0 && <div style={`width: ${apPct}%; background: var(--status-healthy); transition: width 0.3s ease;`} />}
        {rjPct > 0 && <div style={`width: ${rjPct}%; background: var(--status-error); transition: width 0.3s ease;`} />}
        {dfPct > 0 && <div style={`width: ${dfPct}%; background: var(--text-tertiary); transition: width 0.3s ease;`} />}
        {pdPct > 0 && <div style={`width: ${pdPct}%; background: var(--status-warning); opacity: 0.4; transition: width 0.3s ease;`} />}
      </div>
      <div class="flex gap-4 mt-2 text-xs" style="color: var(--text-tertiary)">
        <span><span style="color: var(--status-healthy)">{'\u25CF'}</span> Approved {Math.round(apPct)}%</span>
        <span><span style="color: var(--status-error)">{'\u25CF'}</span> Rejected {Math.round(rjPct)}%</span>
        <span><span style="color: var(--text-tertiary)">{'\u25CF'}</span> Deferred {Math.round(dfPct)}%</span>
        {pdPct > 0 && <span><span style="color: var(--status-warning); opacity: 0.6">{'\u25CF'}</span> Pending {Math.round(pdPct)}%</span>}
      </div>
    </div>
  );
}

function RecommendationCard({ suggestion, onAction, onUndo, updating, undoable, allSuggestions }) {
  const status = (suggestion.status || 'pending').toLowerCase();
  const isGapFill = suggestion.gap_fill || suggestion.fills_gap || suggestion.coverage_gap;
  const gapLabel = suggestion.gap_fill || suggestion.coverage_gap;
  const icon = STATUS_ICONS[status] || STATUS_ICONS.pending;
  const conflicts = allSuggestions ? detectConflicts(suggestion, allSuggestions) : [];

  return (
    <div class="t-frame clickable-data" data-label={suggestion.name || 'recommendation'} style="padding: 1.25rem; cursor: pointer;" onClick={() => { window.location.hash = `#/detail/suggestion/${suggestion.id || suggestion.name || 'unknown'}`; }}>
      <div class="flex items-center justify-between mb-2">
        <div class="flex items-center gap-2 min-w-0">
          <span class="text-base font-bold truncate" style="color: var(--text-primary)">{suggestion.name || 'Unnamed'}</span>
          {isGapFill && (
            <span class="inline-block px-1.5 py-0.5 rounded text-xs font-medium flex-shrink-0" style="background: var(--accent-glow); color: var(--accent);">
              Gap fill{typeof gapLabel === 'string' ? `: ${gapLabel}` : ''}
            </span>
          )}
        </div>
        <div class="flex items-center gap-2 flex-shrink-0">
          <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium" style={statusStyle(status)}>
            <span>{icon}</span> {status}
          </span>
          {undoable && (
            <button
              onClick={(evt) => { evt.stopPropagation(); onUndo(suggestion.id); }}
              class="px-2 py-0.5 text-xs rounded"
              style="background: var(--bg-surface-raised); color: var(--accent); border: 1px solid var(--border-subtle); cursor: pointer;"
              title="Undo last action"
            >
              {'\u21A9'} Undo
            </button>
          )}
        </div>
      </div>

      {conflicts.length > 0 && (
        <div class="mb-2 px-2 py-1.5 rounded text-xs" style="background: var(--status-warning-glow); border: 1px solid var(--status-warning); color: var(--status-warning);">
          <span class="font-bold">{'\u26A0'} Conflict{conflicts.length > 1 ? 's' : ''}:</span>
          {conflicts.map((conflict, ci) => (
            <span key={ci}> {conflict.name} ({conflict.reason}){ci < conflicts.length - 1 ? ',' : ''}</span>
          ))}
        </div>
      )}

      {suggestion.description && (
        <p class="text-sm mb-3" style="color: var(--text-secondary)">{suggestion.description}</p>
      )}

      <div class="flex items-center gap-4 text-xs mb-3" style="color: var(--text-tertiary)">
        <ConfidenceBadge confidence={suggestion.confidence} />
        {suggestion.occurrence_count != null && (
          <span>Occurrences: <span class="data-mono font-medium" style="color: var(--text-secondary)">{suggestion.occurrence_count}</span></span>
        )}
        {suggestion.source && (
          <span>Source: <span class="font-medium" style="color: var(--text-secondary)">{suggestion.source}</span></span>
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
          <button onClick={(evt) => { evt.stopPropagation(); onAction(suggestion.id, 'approved'); }} disabled={updating}
            class="t-btn px-4 py-1.5 text-sm font-medium disabled:opacity-50"
            style="background: var(--status-healthy); color: var(--text-inverse); border: none; cursor: pointer;">Approve</button>
          <button onClick={(evt) => { evt.stopPropagation(); onAction(suggestion.id, 'rejected'); }} disabled={updating}
            class="t-btn px-4 py-1.5 text-sm font-medium disabled:opacity-50"
            style="background: var(--status-error); color: var(--text-inverse); border: none; cursor: pointer;">Reject</button>
          <button onClick={(evt) => { evt.stopPropagation(); onAction(suggestion.id, 'deferred'); }} disabled={updating}
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
  /** Track recently changed IDs for undo — maps id to previous status */
  const [undoHistory, setUndoHistory] = useState({});

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

  async function persistStatuses(updatedSuggestions) {
    const updatedData = { ...(data.data || {}), suggestions: updatedSuggestions };
    const res = await fetch(`${baseUrl}/api/cache/automation_suggestions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: updatedData }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  }

  async function handleAction(id, newStatus) {
    setUpdating(true);
    setUpdateError(null);
    // Record previous status for undo
    const prevItem = displaySuggestions.find((item) => item.id === id);
    const prevStatus = prevItem ? (prevItem.status || 'pending') : 'pending';
    setUndoHistory((prev) => ({ ...prev, [id]: prevStatus }));
    setLocalStatuses((prev) => ({ ...prev, [id]: newStatus }));
    try {
      const updatedSuggestions = suggestions.map((item) =>
        item.id === id ? { ...item, status: newStatus } : item
      );
      await persistStatuses(updatedSuggestions);
      setLocalStatuses((prev) => { const next = { ...prev }; delete next[id]; return next; });
      refetch();
    } catch (err) {
      setLocalStatuses((prev) => { const next = { ...prev }; delete next[id]; return next; });
      setUndoHistory((prev) => { const next = { ...prev }; delete next[id]; return next; });
      setUpdateError(err.message || String(err));
    } finally {
      setUpdating(false);
    }
  }

  async function handleUndo(id) {
    const prevStatus = undoHistory[id];
    if (!prevStatus) return;
    setUpdating(true);
    setUpdateError(null);
    setLocalStatuses((prev) => ({ ...prev, [id]: prevStatus }));
    setUndoHistory((prev) => { const next = { ...prev }; delete next[id]; return next; });
    try {
      const updatedSuggestions = suggestions.map((item) =>
        item.id === id ? { ...item, status: prevStatus } : item
      );
      await persistStatuses(updatedSuggestions);
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

  const reviewed = approved.length + rejected.length + deferred.length;

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="DECIDE" subtitle="Automation suggestions ARIA has generated based on patterns in your home. Review each one and choose to approve, reject, or defer." />

      <HeroCard
        value={pending.length}
        label="pending review"
        delta={`${approved.length} approved \u00b7 ${rejected.length} rejected \u00b7 ${deferred.length} deferred`}
        loading={loading}
      />

      {reviewed > 0 && (
        <HealthBar
          approved={approved.length}
          rejected={rejected.length}
          deferred={deferred.length}
          total={displaySuggestions.length}
        />
      )}

      {updateError && <ErrorState error={updateError} onRetry={() => setUpdateError(null)} />}

      {pending.length === 0 && displaySuggestions.length === 0 ? (
        <Callout>No automation suggestions yet. ARIA generates recommendations when it finds patterns with high confidence and matching capabilities.</Callout>
      ) : (
        <>
          {pending.length > 0 && (
            <Section title="Pending Review" summary={`${pending.length} pending`}>
              <div class="space-y-4">
                {pending.map((sug, idx) => (
                  <RecommendationCard
                    key={sug.id || idx}
                    suggestion={sug}
                    onAction={handleAction}
                    onUndo={handleUndo}
                    updating={updating}
                    undoable={!!undoHistory[sug.id]}
                    allSuggestions={displaySuggestions}
                  />
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
                  <RecommendationCard
                    key={sug.id || idx}
                    suggestion={sug}
                    onAction={handleAction}
                    onUndo={handleUndo}
                    updating={updating}
                    undoable={!!undoHistory[sug.id]}
                    allSuggestions={displaySuggestions}
                  />
                ))}
              </div>
            </Section>
          )}
        </>
      )}
    </div>
  );
}
