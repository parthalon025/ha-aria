import { useState, useEffect } from 'preact/hooks';
import { fetchJson, putJson, postJson } from '../api.js';
import { CURATION_GROUP_PREVIEW, CURATION_TABLE_MAX } from '../constants.js';
import HeroCard from '../components/HeroCard.jsx';
import PageBanner from '../components/PageBanner.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

const STATUS_COLORS = {
  auto_excluded: 'background: var(--bg-surface-raised); color: var(--text-tertiary);',
  excluded: 'background: var(--status-error-glow); color: var(--status-error);',
  included: 'background: var(--status-healthy-glow); color: var(--status-healthy);',
  promoted: 'background: var(--accent-glow); color: var(--accent);',
};

function SummaryBar({ summary }) {
  const total = summary?.total ?? 0;
  const perStatus = summary?.per_status ?? {};
  const stats = [
    { label: 'Total', value: total, style: 'color: var(--text-secondary)' },
    { label: 'Auto-Excluded', value: perStatus.auto_excluded ?? 0, style: 'color: var(--text-tertiary)' },
    { label: 'Excluded', value: perStatus.excluded ?? 0, style: 'color: var(--status-error)' },
    { label: 'Included', value: perStatus.included ?? 0, style: 'color: var(--status-healthy)' },
    { label: 'Promoted', value: perStatus.promoted ?? 0, style: 'color: var(--accent)' },
  ];

  return (
    <div class="grid grid-cols-2 sm:grid-cols-5 gap-3">
      {stats.map((s) => (
        <div key={s.label} class="t-frame" data-label={s.label} style="padding: 0.75rem; text-align: center;">
          <div class="text-xl font-bold" style={s.style}>{s.value}</div>
          <div class="text-xs" style="color: var(--text-tertiary)">{s.label}</div>
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
    <section class="t-frame" data-label={`tier ${tier}: ${label.toLowerCase()}`}>
      <button
        class="w-full flex items-center justify-between px-4 py-3 text-left"
        onClick={() => setOpen(!open)}
      >
        <div class="flex items-center gap-2">
          <h3 class="text-sm font-bold" style="color: var(--text-secondary)">Tier {tier}: {label}</h3>
          <span class="text-xs rounded-full px-2 py-0.5" style="background: var(--bg-surface-raised); color: var(--text-tertiary)">{entities.length}</span>
        </div>
        <span class="text-xs" style="color: var(--text-tertiary)">{open ? '\u25B2' : '\u25BC'}</span>
      </button>

      {open && (
        <div class="px-4 pb-3 space-y-3">
          {entities.length > 10 && (
            <input
              type="text"
              placeholder="Search entity or reason..."
              value={search}
              onInput={(e) => setSearch(e.target.value)}
              class="t-input w-full px-3 py-1.5 text-sm"
            />
          )}

          {tier <= 2 ? (
            // Grouped view for tiers 1-2
            Object.entries(grouped).map(([groupKey, items]) => (
              <div key={groupKey} class="p-3 space-y-2" style="border: 1px solid var(--border-subtle); border-radius: var(--radius)">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-medium truncate" style="color: var(--text-secondary)">{groupKey}</span>
                  <div class="flex gap-1 flex-shrink-0">
                    {tier === 1 && (
                      <button
                        class="t-btn text-xs px-2 py-0.5"
                        style="background: var(--status-healthy-glow); color: var(--status-healthy); border: none;"
                        onClick={() => onBulk(items.map((e) => e.entity_id), 'included')}
                      >
                        Include All
                      </button>
                    )}
                    {tier === 2 && (
                      <>
                        <button
                          class="t-btn text-xs px-2 py-0.5"
                          style="background: var(--status-healthy-glow); color: var(--status-healthy); border: none;"
                          onClick={() => onBulk(items.map((e) => e.entity_id), 'included')}
                        >
                          Approve All
                        </button>
                        <button
                          class="t-btn text-xs px-2 py-0.5"
                          style="background: var(--status-error-glow); color: var(--status-error); border: none;"
                          onClick={() => onBulk(items.map((e) => e.entity_id), 'excluded')}
                        >
                          Reject All
                        </button>
                      </>
                    )}
                  </div>
                </div>
                <div class="space-y-1">
                  {items.slice(0, CURATION_GROUP_PREVIEW).map((e) => (
                    <EntityRow key={e.entity_id} entity={e} onOverride={onOverride} compact />
                  ))}
                  {items.length > CURATION_GROUP_PREVIEW && (
                    <p class="text-xs" style="color: var(--text-tertiary)">...and {items.length - CURATION_GROUP_PREVIEW} more</p>
                  )}
                </div>
              </div>
            ))
          ) : (
            // Table view for tier 3
            <div class="space-y-1">
              {filtered.slice(0, CURATION_TABLE_MAX).map((e) => (
                <EntityRow key={e.entity_id} entity={e} onOverride={onOverride} />
              ))}
              {filtered.length > CURATION_TABLE_MAX && (
                <p class="text-xs" style="color: var(--text-tertiary)">Showing {CURATION_TABLE_MAX} of {filtered.length}. Use search to narrow.</p>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function EntityRow({ entity, onOverride, compact }) {
  const statusStyle = STATUS_COLORS[entity.status] || 'background: var(--bg-surface-raised); color: var(--text-secondary);';

  return (
    <div class={`flex items-center gap-2 ${compact ? 'py-0.5' : 'py-1.5'}`} style={compact ? '' : 'border-bottom: 1px solid var(--border-subtle)'}>
      <span class="text-xs truncate flex-1 data-mono" style="color: var(--text-secondary)">{entity.entity_id}</span>
      <span class="text-[10px] font-medium flex-shrink-0 px-1.5 py-0.5" style={`border-radius: var(--radius); ${statusStyle}`}>
        {entity.status}
      </span>
      {entity.human_override && (
        <span class="text-[10px] flex-shrink-0" style="color: var(--accent-dim)">manual</span>
      )}
      {!compact && (
        <div class="flex gap-1 flex-shrink-0">
          {entity.status !== 'included' && (
            <button
              class="text-[10px]"
              style="color: var(--status-healthy)"
              onClick={() => onOverride(entity.entity_id, 'included')}
            >
              Include
            </button>
          )}
          {entity.status !== 'excluded' && (
            <button
              class="text-[10px]"
              style="color: var(--status-error)"
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
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Data Curation</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">Classify entities into include/exclude tiers for the shadow engine.</p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Data Curation</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">Classify entities into include/exclude tiers for the shadow engine.</p>
        </div>
        <ErrorState error={error} onRetry={fetchAll} />
      </div>
    );
  }

  if (curations.length === 0) {
    return (
      <div class="space-y-6">
        <PageBanner page="CURATION" subtitle="Entity-level noise control and classification." />
        <div class="t-callout" style="padding: 1rem;">
          <span class="text-sm" style="color: var(--text-secondary)">No entity classifications yet. The data quality module will classify entities after discovery runs.</span>
        </div>
      </div>
    );
  }

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="CURATION" subtitle="Entity-level noise control and classification." />

      {/* Hero — what feeds the pipeline */}
      <HeroCard
        value={summary?.per_status?.included ?? 0}
        label="entities included"
        delta={`${summary?.per_status?.excluded ?? 0} excluded`}
      />

      <SummaryBar summary={summary} />

      <div class="space-y-4 stagger-children">
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
