/**
 * CurationDetail — Detail renderer for entity curation decisions.
 * Three-section layout: Summary (status + tier), Explanation (entity info), History (audit trail).
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';
import { relativeTime } from '../intelligence/utils.jsx';

function curationStatusColor(status) {
  if (status === 'approved') return 'background: var(--status-healthy-glow); color: var(--status-healthy);';
  if (status === 'rejected') return 'background: var(--status-error-glow); color: var(--status-error);';
  return 'background: var(--bg-inset); color: var(--text-secondary);';
}

export default function CurationDetail({ id, type: _type }) {
  const [curation, setCuration] = useState(null);
  const [audit, setAudit] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      fetchJson('/api/curation'),
      fetchJson(`/api/audit/curation/${encodeURIComponent(id)}`).catch(err => { console.warn('Optional fetch failed:', err.message); return null; }),
    ])
      .then(([curationResult, auditResult]) => {
        const data = curationResult?.data || curationResult || {};
        const entities = data.entities || data;

        // Find by entity_id
        let found = null;
        if (entities[id]) {
          found = entities[id];
        } else if (Array.isArray(entities)) {
          found = entities.find((c) => c.entity_id === id);
        } else {
          // Try iterating object values
          const match = Object.entries(entities).find(
            ([key, val]) => key === id || val?.entity_id === id
          );
          if (match) found = match[1];
        }
        setCuration(found);

        if (auditResult) {
          const entries = Array.isArray(auditResult) ? auditResult : (auditResult?.entries || auditResult?.trail || []);
          const sorted = [...entries].sort((a, b) => {
            const ta = new Date(a.timestamp || a.time || 0).getTime();
            const tb = new Date(b.timestamp || b.time || 0).getTime();
            return tb - ta;
          });
          setAudit(sorted);
        }
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!curation) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No curation data found for: {id}
        </p>
      </div>
    );
  }

  const status = curation.status || 'pending';
  const statsItems = [
    { label: 'Entity ID', value: curation.entity_id || id },
    { label: 'Status', value: status },
  ];
  if (curation.tier) {
    statsItems.push({ label: 'Tier', value: curation.tier });
  }
  if (curation.decided_by) {
    statsItems.push({ label: 'Decided By', value: curation.decided_by });
  }

  const domain = id.includes('.') ? id.split('.')[0] : (curation.domain || null);

  return (
    <div class="space-y-6">
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        <div style="margin-bottom: 12px;">
          <span
            style={`display: inline-block; padding: 2px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: var(--type-label); text-transform: uppercase; ${curationStatusColor(status)}`}
          >
            {status}
          </span>
        </div>
        <StatsGrid items={statsItems} />
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        <div class="space-y-1">
          {domain && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Domain</span>
              <span style="color: var(--text-secondary);">{domain}</span>
            </div>
          )}
          {curation.device_class && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Device Class</span>
              <span style="color: var(--text-secondary);">{curation.device_class}</span>
            </div>
          )}
          {curation.area && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Area</span>
              <span style="color: var(--text-secondary);">{curation.area}</span>
            </div>
          )}
          {curation.state_change_frequency !== null && curation.state_change_frequency !== undefined && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">State Change Freq</span>
              <span style="color: var(--text-secondary);">
                {typeof curation.state_change_frequency === 'number'
                  ? curation.state_change_frequency.toFixed(2)
                  : String(curation.state_change_frequency)}
              </span>
            </div>
          )}
          {curation.reason && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Reason</span>
              <span style="color: var(--text-secondary); max-width: 60%; text-align: right; word-break: break-word;">
                {curation.reason}
              </span>
            </div>
          )}
          {curation.category && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Category</span>
              <span style="color: var(--text-secondary);">{curation.category}</span>
            </div>
          )}
        </div>

        {!domain && !curation.device_class && !curation.area && !curation.reason && (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No additional details available
          </p>
        )}
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        {audit.length > 0 ? (
          <div class="space-y-2">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Audit Trail ({audit.length})
            </span>
            {audit.map((entry, idx) => (
              <div key={idx} class="flex gap-3 items-center" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary); min-width: 60px; flex-shrink: 0;">
                  {relativeTime(entry.timestamp || entry.time)}
                </span>
                <span
                  style={`padding: 1px 6px; border-radius: 3px; font-size: var(--type-label); ${curationStatusColor(entry.status || entry.action || 'update')}`}
                >
                  {entry.status || entry.action || 'update'}
                </span>
                {entry.decided_by && (
                  <span style="color: var(--text-tertiary); flex: 1; text-align: right;">
                    {entry.decided_by}
                  </span>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No audit history available
          </p>
        )}
      </div>
    </div>
  );
}
