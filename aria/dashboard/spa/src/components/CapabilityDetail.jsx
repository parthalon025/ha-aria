import { useState } from 'preact/hooks';
import { putJson } from '../api.js';
import UsefulnessBar from './UsefulnessBar.jsx';

/**
 * Format an ISO timestamp to a readable date string.
 */
function formatDate(iso) {
  if (!iso) return '\u2014';
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
    });
  } catch {
    return iso;
  }
}

/**
 * CapabilityDetail — expanded view for a single capability.
 *
 * Shows:
 * - 5 usefulness component bars (predictability, stability, coverage, activity, cohesion)
 * - Metadata: description, source, layer, naming_method, first_seen, promoted_at
 * - Temporal pattern (peak_hours, weekday_bias) for behavioral capabilities
 * - Entity list (first 5, expandable)
 * - Action buttons: Promote / Archive
 *
 * @param {Object} props
 * @param {string} props.name - Capability name
 * @param {Object} props.capability - Full capability data object
 * @param {Function} props.onAction - Called after promote/archive to trigger refetch
 */
export default function CapabilityDetail({ name, capability, onAction }) {
  const [showAllEntities, setShowAllEntities] = useState(false);
  const [busy, setBusy] = useState(null); // 'promote' | 'archive' | null

  const uc = capability.usefulness_components || {};
  const entities = capability.entities || [];
  const temporal = capability.temporal_pattern || {};
  const status = capability.status || 'promoted';

  const usefulnessFields = [
    { key: 'predictability', label: 'Predictability' },
    { key: 'stability', label: 'Stability' },
    { key: 'coverage', label: 'Coverage' },
    { key: 'activity', label: 'Activity' },
    { key: 'cohesion', label: 'Cohesion' },
  ];

  const metaFields = [
    { key: 'description', label: 'Description' },
    { key: 'source', label: 'Source' },
    { key: 'layer', label: 'Layer' },
    { key: 'naming_method', label: 'Naming' },
    { key: 'first_seen', label: 'First seen', format: formatDate },
    { key: 'promoted_at', label: 'Promoted', format: formatDate },
  ];

  async function handleAction(action) {
    setBusy(action);
    try {
      await putJson(`/api/capabilities/${name}/${action}`, {});
      if (onAction) onAction();
    } catch (err) {
      console.error(`Failed to ${action} capability:`, err);
    } finally {
      setBusy(null);
    }
  }

  const visibleEntities = showAllEntities ? entities : entities.slice(0, 5);
  const hasMoreEntities = entities.length > 5;

  return (
    <div class="space-y-4" style="padding-top: 12px; border-top: 1px solid var(--border-subtle);">
      {/* Usefulness component bars */}
      {Object.keys(uc).length > 0 && (
        <div class="space-y-2">
          <span
            style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
          >
            Usefulness breakdown
          </span>
          {usefulnessFields.map(({ key, label }) =>
            uc[key] !== null && uc[key] !== undefined ? (
              <UsefulnessBar key={key} value={uc[key]} label={label} />
            ) : null
          )}
        </div>
      )}

      {/* Metadata */}
      <div class="space-y-1">
        {metaFields.map(({ key, label, format }) => {
          const val = capability[key];
          if (val === null || val === undefined) return null;
          const display = format ? format(val) : String(val);
          return (
            <div key={key} class="flex justify-between" style="font-size: var(--type-label); font-family: var(--font-mono);">
              <span style="color: var(--text-tertiary);">{label}</span>
              <span style="color: var(--text-secondary);">{display}</span>
            </div>
          );
        })}
      </div>

      {/* Temporal pattern */}
      {(temporal.peak_hours || temporal.weekday_bias) && (
        <div class="space-y-1">
          <span
            style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
          >
            Temporal pattern
          </span>
          {temporal.peak_hours && (
            <div class="flex justify-between" style="font-size: var(--type-label); font-family: var(--font-mono);">
              <span style="color: var(--text-tertiary);">Peak hours</span>
              <span style="color: var(--text-secondary);">
                {Array.isArray(temporal.peak_hours) ? temporal.peak_hours.join(', ') : String(temporal.peak_hours)}
              </span>
            </div>
          )}
          {temporal.weekday_bias && (
            <div class="flex justify-between" style="font-size: var(--type-label); font-family: var(--font-mono);">
              <span style="color: var(--text-tertiary);">Weekday bias</span>
              <span style="color: var(--text-secondary);">{String(temporal.weekday_bias)}</span>
            </div>
          )}
        </div>
      )}

      {/* Entity list */}
      {entities.length > 0 && (
        <div>
          <span
            style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
          >
            Entities ({entities.length})
          </span>
          <ul class="space-y-0.5" style="margin-top: 4px;">
            {visibleEntities.map((eid) => (
              <li
                key={eid}
                class="truncate data-mono"
                style="font-size: var(--type-label); color: var(--text-tertiary);"
              >
                {eid}
              </li>
            ))}
          </ul>
          {hasMoreEntities && (
            <button
              onClick={() => setShowAllEntities(!showAllEntities)}
              class="cursor-pointer"
              style="font-size: var(--type-label); color: var(--accent); background: none; border: none; padding: 0; margin-top: 4px; font-family: var(--font-mono);"
            >
              {showAllEntities ? 'Show less' : `Show all ${entities.length}`}
            </button>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div class="flex gap-2" style="padding-top: 8px;">
        {status !== 'promoted' && (
          <button
            onClick={() => handleAction('promote')}
            disabled={busy !== null}
            class="cursor-pointer"
            style={`font-size: var(--type-label); font-family: var(--font-mono); padding: 4px 12px; border: 1px solid var(--status-healthy); color: var(--status-healthy); background: var(--status-healthy-glow); border-radius: 4px; opacity: ${busy ? '0.5' : '1'};`}
          >
            {busy === 'promote' ? 'Promoting...' : 'Promote'}
          </button>
        )}
        {status !== 'archived' && (
          <button
            onClick={() => handleAction('archive')}
            disabled={busy !== null}
            class="cursor-pointer"
            style={`font-size: var(--type-label); font-family: var(--font-mono); padding: 4px 12px; border: 1px solid var(--text-tertiary); color: var(--text-tertiary); background: none; border-radius: 4px; opacity: ${busy ? '0.5' : '1'};`}
          >
            {busy === 'archive' ? 'Archiving...' : 'Archive'}
          </button>
        )}
      </div>
    </div>
  );
}
