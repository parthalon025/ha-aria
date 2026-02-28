/**
 * EntityDetail — Detail renderer for HA entities.
 * Three-section layout: Summary (state + stats), Explanation (attributes + curation), History (audit).
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';
import { relativeTime } from '../intelligence/utils.jsx';

export default function EntityDetail({ id, type: _type }) {
  const [entity, setEntity] = useState(null);
  const [curation, setCuration] = useState(null);
  const [audit, setAudit] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      fetchJson('/api/cache/entities'),
      fetchJson('/api/curation').catch(err => { console.warn('Optional fetch failed:', err.message); return null; }),
    ])
      .then(([entitiesResult, curationResult]) => {
        // Find entity in cache — could be nested in data or a flat array
        const data = entitiesResult?.data || entitiesResult || {};
        let found = null;

        if (Array.isArray(data)) {
          found = data.find((e) => e.entity_id === id);
        } else if (data.entities) {
          if (Array.isArray(data.entities)) {
            found = data.entities.find((e) => e.entity_id === id);
          } else {
            found = data.entities[id] || null;
          }
        } else if (data[id]) {
          found = data[id];
        }

        setEntity(found);

        // Extract curation for this entity
        if (curationResult) {
          const cData = curationResult?.data || curationResult || {};
          const entities = cData.entities || cData;
          if (entities[id]) {
            setCuration(entities[id]);
          } else if (Array.isArray(entities)) {
            setCuration(entities.find((c) => c.entity_id === id) || null);
          }
        }

        // Fetch audit trail if curation exists
        if (curationResult) {
          fetchJson(`/api/audit/curation/${encodeURIComponent(id)}`)
            .then((auditData) => setAudit(auditData))
            .catch(err => { console.warn('Optional fetch failed:', err.message); return null; });
        }
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!entity) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No entity found: {id}
        </p>
      </div>
    );
  }

  const domain = entity.entity_id ? entity.entity_id.split('.')[0] : (entity.domain || '\u2014');
  const statsItems = [
    { label: 'Entity ID', value: entity.entity_id || id },
    { label: 'State', value: entity.state || '\u2014' },
    { label: 'Domain', value: domain },
  ];
  if (entity.device_class) {
    statsItems.push({ label: 'Device Class', value: entity.device_class });
  }
  if (entity.area || entity.area_id) {
    statsItems.push({ label: 'Area', value: entity.area || entity.area_id });
  }
  if (curation) {
    if (curation.tier) statsItems.push({ label: 'Tier', value: curation.tier });
    if (curation.status) statsItems.push({ label: 'Status', value: curation.status });
  }

  // Entity attributes
  const attributes = entity.attributes || {};
  const attrEntries = Object.entries(attributes).filter(
    ([key]) => !key.startsWith('_') && key !== 'friendly_name'
  );

  // Audit trail entries
  const auditEntries = Array.isArray(audit) ? audit : (audit?.entries || audit?.trail || []);

  return (
    <div class="space-y-6">
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        <StatsGrid items={statsItems} />
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        {/* Entity attributes */}
        {attrEntries.length > 0 && (
          <div class="space-y-1" style="margin-bottom: 16px;">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Attributes
            </span>
            {attrEntries.map(([key, val]) => (
              <div key={key} class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary);">{key}</span>
                <span style="color: var(--text-secondary); max-width: 60%; text-align: right; word-break: break-word;">
                  {typeof val === 'object' ? JSON.stringify(val) : String(val ?? '\u2014')}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Curation details */}
        {curation ? (
          <div class="space-y-1">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Curation Classification
            </span>
            {curation.tier && (
              <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary);">Tier</span>
                <span style="color: var(--text-secondary);">{curation.tier}</span>
              </div>
            )}
            {curation.reason && (
              <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary);">Reason</span>
                <span style="color: var(--text-secondary);">{curation.reason}</span>
              </div>
            )}
            {curation.category && (
              <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary);">Category</span>
                <span style="color: var(--text-secondary);">{curation.category}</span>
              </div>
            )}
          </div>
        ) : attrEntries.length === 0 ? (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No additional details available
          </p>
        ) : null}
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        {entity.last_changed && (
          <p style="color: var(--text-secondary); font-family: var(--font-mono); font-size: var(--type-label); margin-bottom: 12px;">
            Last changed: {relativeTime(entity.last_changed)}
          </p>
        )}
        {auditEntries.length > 0 ? (
          <div class="space-y-1">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Curation Audit Trail
            </span>
            {auditEntries.map((entry, idx) => (
              <div key={idx} class="flex gap-3" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary); min-width: 60px; flex-shrink: 0;">
                  {relativeTime(entry.timestamp || entry.time)}
                </span>
                <span style="color: var(--text-secondary);">
                  {entry.action || entry.event || 'update'}
                </span>
                <span style="color: var(--text-tertiary); flex: 1; text-align: right;">
                  {entry.detail || entry.reason || ''}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            {entity.last_changed ? '' : 'No historical data available'}
          </p>
        )}
      </div>
    </div>
  );
}
