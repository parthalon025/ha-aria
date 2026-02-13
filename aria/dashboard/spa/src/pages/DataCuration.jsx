import { useState, useEffect } from 'preact/hooks';
import { fetchJson, putJson, postJson } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

const STATUS_COLORS = {
  auto_excluded: 'bg-gray-100 text-gray-600',
  excluded: 'bg-red-100 text-red-700',
  included: 'bg-green-100 text-green-700',
  promoted: 'bg-blue-100 text-blue-700',
};

function SummaryBar({ summary }) {
  const total = summary?.total ?? 0;
  const perStatus = summary?.per_status ?? {};
  const stats = [
    { label: 'Total', value: total, cls: 'text-gray-700' },
    { label: 'Auto-Excluded', value: perStatus.auto_excluded ?? 0, cls: 'text-gray-500' },
    { label: 'Excluded', value: perStatus.excluded ?? 0, cls: 'text-red-600' },
    { label: 'Included', value: perStatus.included ?? 0, cls: 'text-green-600' },
    { label: 'Promoted', value: perStatus.promoted ?? 0, cls: 'text-blue-600' },
  ];

  return (
    <div class="grid grid-cols-2 sm:grid-cols-5 gap-3">
      {stats.map((s) => (
        <div key={s.label} class="bg-white rounded-lg shadow-sm p-3 text-center">
          <div class={`text-xl font-bold ${s.cls}`}>{s.value}</div>
          <div class="text-xs text-gray-400">{s.label}</div>
        </div>
      ))}
    </div>
  );
}

function TierSection({ tier, label, entities, defaultOpen, onOverride, onBulk }) {
  const [open, setOpen] = useState(defaultOpen);
  const [search, setSearch] = useState('');

  const filtered = search
    ? entities.filter((e) =>
        e.entity_id.toLowerCase().includes(search.toLowerCase()) ||
        (e.reason || '').toLowerCase().includes(search.toLowerCase())
      )
    : entities;

  // Group by reason for tier 1, by group_id for tier 2
  const grouped = {};
  for (const e of filtered) {
    const groupKey = tier === 1 ? (e.reason || 'Other') : (e.group_id || e.entity_id);
    if (!grouped[groupKey]) grouped[groupKey] = [];
    grouped[groupKey].push(e);
  }

  return (
    <section class="bg-white rounded-lg shadow-sm">
      <button
        class="w-full flex items-center justify-between px-4 py-3 text-left"
        onClick={() => setOpen(!open)}
      >
        <div class="flex items-center gap-2">
          <h3 class="text-sm font-bold text-gray-700">Tier {tier}: {label}</h3>
          <span class="text-xs bg-gray-100 text-gray-500 rounded-full px-2 py-0.5">{entities.length}</span>
        </div>
        <span class="text-gray-400 text-xs">{open ? '\u25B2' : '\u25BC'}</span>
      </button>

      {open && (
        <div class="px-4 pb-3 space-y-3">
          {entities.length > 10 && (
            <input
              type="text"
              placeholder="Search entity or reason..."
              value={search}
              onInput={(e) => setSearch(e.target.value)}
              class="w-full border border-gray-200 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          )}

          {tier <= 2 ? (
            // Grouped view for tiers 1-2
            Object.entries(grouped).map(([groupKey, items]) => (
              <div key={groupKey} class="border border-gray-100 rounded p-3 space-y-2">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-medium text-gray-600 truncate">{groupKey}</span>
                  <div class="flex gap-1 flex-shrink-0">
                    {tier === 1 && (
                      <button
                        class="text-xs bg-green-50 text-green-600 px-2 py-0.5 rounded hover:bg-green-100"
                        onClick={() => onBulk(items.map((e) => e.entity_id), 'included')}
                      >
                        Include All
                      </button>
                    )}
                    {tier === 2 && (
                      <>
                        <button
                          class="text-xs bg-green-50 text-green-600 px-2 py-0.5 rounded hover:bg-green-100"
                          onClick={() => onBulk(items.map((e) => e.entity_id), 'included')}
                        >
                          Approve All
                        </button>
                        <button
                          class="text-xs bg-red-50 text-red-600 px-2 py-0.5 rounded hover:bg-red-100"
                          onClick={() => onBulk(items.map((e) => e.entity_id), 'excluded')}
                        >
                          Reject All
                        </button>
                      </>
                    )}
                  </div>
                </div>
                <div class="space-y-1">
                  {items.slice(0, 20).map((e) => (
                    <EntityRow key={e.entity_id} entity={e} onOverride={onOverride} compact />
                  ))}
                  {items.length > 20 && (
                    <p class="text-xs text-gray-400">...and {items.length - 20} more</p>
                  )}
                </div>
              </div>
            ))
          ) : (
            // Table view for tier 3
            <div class="space-y-1">
              {filtered.slice(0, 100).map((e) => (
                <EntityRow key={e.entity_id} entity={e} onOverride={onOverride} />
              ))}
              {filtered.length > 100 && (
                <p class="text-xs text-gray-400">Showing 100 of {filtered.length}. Use search to narrow.</p>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function EntityRow({ entity, onOverride, compact }) {
  const statusCls = STATUS_COLORS[entity.status] || 'bg-gray-100 text-gray-600';

  return (
    <div class={`flex items-center gap-2 ${compact ? 'py-0.5' : 'py-1.5 border-b border-gray-50 last:border-0'}`}>
      <span class="text-xs text-gray-600 truncate flex-1 font-mono">{entity.entity_id}</span>
      <span class={`text-[10px] font-medium rounded px-1.5 py-0.5 flex-shrink-0 ${statusCls}`}>
        {entity.status}
      </span>
      {entity.human_override && (
        <span class="text-[10px] text-purple-500 flex-shrink-0">manual</span>
      )}
      {!compact && (
        <div class="flex gap-1 flex-shrink-0">
          {entity.status !== 'included' && (
            <button
              class="text-[10px] text-green-600 hover:underline"
              onClick={() => onOverride(entity.entity_id, 'included')}
            >
              Include
            </button>
          )}
          {entity.status !== 'excluded' && (
            <button
              class="text-[10px] text-red-600 hover:underline"
              onClick={() => onOverride(entity.entity_id, 'excluded')}
            >
              Exclude
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default function DataCuration() {
  const [curations, setCurations] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function fetchAll() {
    setLoading(true);
    setError(null);
    try {
      const [cData, sData] = await Promise.all([
        fetchJson('/api/curation'),
        fetchJson('/api/curation/summary'),
      ]);
      setCurations(cData.curations || []);
      setSummary(sData);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleOverride(entityId, status) {
    try {
      await putJson(`/api/curation/${entityId}`, { status });
      await fetchAll();
    } catch (e) {
      console.error('Override failed:', e);
    }
  }

  async function handleBulk(entityIds, status) {
    try {
      await postJson('/api/curation/bulk', { entity_ids: entityIds, status });
      await fetchAll();
    } catch (e) {
      console.error('Bulk update failed:', e);
    }
  }

  useEffect(() => { fetchAll(); }, []);

  const tier1 = curations.filter((c) => c.tier === 1);
  const tier2 = curations.filter((c) => c.tier === 2);
  const tier3 = curations.filter((c) => c.tier === 3);

  if (loading && curations.length === 0) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Data Curation</h1>
          <p class="text-sm text-gray-500">Classify entities into include/exclude tiers for the shadow engine.</p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Data Curation</h1>
          <p class="text-sm text-gray-500">Classify entities into include/exclude tiers for the shadow engine.</p>
        </div>
        <ErrorState error={error} onRetry={fetchAll} />
      </div>
    );
  }

  if (curations.length === 0) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Data Curation</h1>
          <p class="text-sm text-gray-500">Classify entities into include/exclude tiers for the shadow engine.</p>
        </div>
        <div class="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-600">
          No entity classifications yet. The data quality module will classify entities after discovery runs.
        </div>
      </div>
    );
  }

  return (
    <div class="space-y-6">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Data Curation</h1>
        <p class="text-sm text-gray-500">Classify entities into include/exclude tiers for the shadow engine.</p>
      </div>

      <SummaryBar summary={summary} />

      <div class="space-y-4">
        <TierSection
          tier={1} label="Auto-Excluded" entities={tier1}
          defaultOpen={false} onOverride={handleOverride} onBulk={handleBulk}
        />
        <TierSection
          tier={2} label="Edge Cases" entities={tier2}
          defaultOpen={true} onOverride={handleOverride} onBulk={handleBulk}
        />
        <TierSection
          tier={3} label="Included" entities={tier3}
          defaultOpen={true} onOverride={handleOverride} onBulk={handleBulk}
        />
      </div>
    </div>
  );
}
