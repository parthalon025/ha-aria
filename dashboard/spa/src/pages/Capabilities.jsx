import { useState } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Humanize a snake_case name: "power_monitoring" → "Power monitoring" */
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
    <div class="bg-white rounded-lg shadow-sm p-4">
      {/* Header */}
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-base font-bold text-gray-900">{humanize(name)}</h3>
        <span class="inline-block px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700">
          {entityCount} {entityCount === 1 ? 'entity' : 'entities'}
        </span>
      </div>

      {/* Detail fields */}
      {details.length > 0 && (
        <dl class="text-sm text-gray-600 mb-3 space-y-1">
          {details.map(([key, val]) => (
            <div key={key} class="flex justify-between">
              <dt class="text-gray-500">{humanize(key)}</dt>
              <dd class="font-medium text-gray-700">{String(val)}</dd>
            </div>
          ))}
        </dl>
      )}

      {/* Entity list */}
      {entities.length > 0 && (
        <div>
          <ul class="text-xs text-gray-500 space-y-0.5">
            {visibleEntities.map((eid) => (
              <li key={eid} class="truncate font-mono">{eid}</li>
            ))}
          </ul>
          {hasMore && (
            <button
              onClick={() => setExpanded(!expanded)}
              class="text-sm text-blue-600 hover:text-blue-800 cursor-pointer mt-1"
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

  const pageSubtitle = "What your home can do — detected automatically from your entities. Each card represents a capability with its supporting entities.";

  if (loading && !data) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Capabilities</h1>
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
          <h1 class="text-2xl font-bold text-gray-900">Capabilities</h1>
          <p class="text-sm text-gray-500">{pageSubtitle}</p>
        </div>
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Capabilities</h1>
        <p class="text-sm text-gray-500">{pageSubtitle}</p>
      </div>

      {capabilities.length === 0 ? (
        <div class="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-800">
          No capabilities detected yet. Capabilities are identified during discovery by matching entity domains and device classes to known patterns (e.g., power monitoring, lighting, occupancy). They appear after the first successful discovery scan.
        </div>
      ) : (
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {capabilities.map((cap) => (
            <CapabilityCard key={cap.name} name={cap.name} capability={cap} />
          ))}
        </div>
      )}
    </div>
  );
}
