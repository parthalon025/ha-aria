import { useState } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import HeroCard from '../components/HeroCard.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Humanize a snake_case name: "power_monitoring" -> "Power monitoring" */
function humanize(name) {
  const spaced = (name || '').replace(/_/g, ' ');
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** Known capability detail fields (not entity lists or metadata). */
const DETAIL_FIELDS = new Set([
  'supports_color', 'supports_brightness', 'supports_color_temp',
  'measurement_unit', 'modes', 'can_predict',
]);

function CapabilityCard({ name, capability }) {
  const [expanded, setExpanded] = useState(false);

  const entities = capability.entities || [];
  const entityCount = capability.entity_count || entities.length;

  // Collect detail fields
  const details = Object.entries(capability).filter(
    ([k]) => DETAIL_FIELDS.has(k)
  );

  const visibleEntities = expanded ? entities : entities.slice(0, 5);
  const hasMore = entities.length > 5;

  return (
    <div class="t-frame" data-label={humanize(name)} style="padding: 1rem;">
      {/* Header */}
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-base font-bold" style="color: var(--text-primary)">{humanize(name)}</h3>
        <span class="inline-block px-2 py-0.5 rounded-full text-xs font-medium" style="background: var(--accent-glow); color: var(--accent)">
          {entityCount} {entityCount === 1 ? 'entity' : 'entities'}
        </span>
      </div>

      {/* Detail fields */}
      {details.length > 0 && (
        <dl class="text-sm mb-3 space-y-1" style="color: var(--text-secondary)">
          {details.map(([key, val]) => (
            <div key={key} class="flex justify-between">
              <dt style="color: var(--text-tertiary)">{humanize(key)}</dt>
              <dd class="font-medium" style="color: var(--text-secondary)">{String(val)}</dd>
            </div>
          ))}
        </dl>
      )}

      {/* Entity list */}
      {entities.length > 0 && (
        <div>
          <ul class="text-xs space-y-0.5" style="color: var(--text-tertiary)">
            {visibleEntities.map((eid) => (
              <li key={eid} class="truncate data-mono">{eid}</li>
            ))}
          </ul>
          {hasMore && (
            <button
              onClick={() => setExpanded(!expanded)}
              class="text-sm cursor-pointer mt-1"
              style="color: var(--accent)"
            >
              {expanded ? 'Show less' : `Show all ${entities.length}`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default function Capabilities() {
  const { data, loading, error, refetch } = useCache('capabilities');

  const capabilities = useComputed(() => {
    if (!data || !data.data) return [];
    const inner = data.data;
    return Object.entries(inner).map(([name, cap]) => ({ name, ...cap }));
  }, [data]);

  // Total entity count across all capabilities
  const totalEntities = useComputed(() => {
    return capabilities.reduce((sum, cap) => sum + (cap.entity_count || (cap.entities || []).length), 0);
  }, [capabilities]);

  const pageSubtitle = "What your home can do — detected automatically from your entities. Each card represents a capability with its supporting entities.";

  if (loading && !data) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Capabilities</h1>
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
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Capabilities</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
        </div>
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6 animate-page-enter">
      <div class="t-section-header" style="padding-bottom: 8px;">
        <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Capabilities</h1>
        <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
      </div>

      {/* Hero — what ARIA can measure */}
      <HeroCard
        value={capabilities.length}
        label="capabilities"
        delta={`across ${totalEntities} entities`}
        loading={loading}
      />

      {capabilities.length === 0 ? (
        <div class="t-callout" style="padding: 0.75rem;">
          <span class="text-sm" style="color: var(--text-secondary)">No capabilities detected yet. Capabilities are identified during discovery by matching entity domains and device classes to known patterns (e.g., power monitoring, lighting, occupancy). They appear after the first successful discovery scan.</span>
        </div>
      ) : (
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 stagger-children">
          {capabilities.map((cap) => (
            <CapabilityCard key={cap.name} name={cap.name} capability={cap} />
          ))}
        </div>
      )}
    </div>
  );
}
