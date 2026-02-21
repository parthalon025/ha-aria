import { useState, useEffect, useRef, useMemo } from 'preact/hooks';
import { fetchJson, putJson, postJson } from '../api.js';
import useCache from '../hooks/useCache.js';
import PageBanner from '../components/PageBanner.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Group configs by category for collapsible sections. */
function groupByCategory(configs) {
  const groups = {};
  for (const c of configs) {
    const cat = c.category || 'Other';
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(c);
  }
  return groups;
}

function ParamControl({ config, onUpdate, descMode }) {
  const [value, setValue] = useState(config.value);
  const [saving, setSaving] = useState(false);
  const timerRef = useRef(null);
  const isDefault = config.value === config.default_value;

  useEffect(() => { setValue(config.value); }, [config.value]);

  function debouncedSave(newVal) {
    setValue(newVal);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      setSaving(true);
      try {
        await putJson(`/api/config/${config.key}`, { value: String(newVal) });
        onUpdate();
      } catch (e) {
        console.error('Config save failed:', e);
      } finally {
        setSaving(false);
      }
    }, 500);
  }

  async function handleReset() {
    setSaving(true);
    try {
      await postJson(`/api/config/reset/${config.key}`, {});
      setValue(config.default_value);
      onUpdate();
    } catch (e) {
      console.error('Config reset failed:', e);
    } finally {
      setSaving(false);
    }
  }

  const vtype = config.value_type;

  return (
    <div class="flex flex-col sm:flex-row sm:items-center gap-2 py-3" style="border-bottom: 1px solid var(--border-subtle)">
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <a href={`#/detail/config/${config.key}`} class="clickable-data text-sm font-medium" style="color: var(--text-secondary); text-decoration: none;">{config.label || config.key}</a>
          {saving && <span class="text-xs" style="color: var(--accent)">Saving...</span>}
        </div>
        {(() => {
          const desc = descMode === 'simple'
            ? (config.description_layman || config.description)
            : (config.description_technical || config.description);
          return desc ? <p class="text-xs mt-0.5" style="color: var(--text-tertiary)">{desc}</p> : null;
        })()}
      </div>
      <div class="flex items-center gap-2 flex-shrink-0">
        {vtype === 'number' && (
          <div class="flex items-center gap-2">
            <input
              type="range"
              min={config.min_value ?? 0}
              max={config.max_value ?? 100}
              step={config.step ?? 1}
              value={value}
              onInput={(e) => debouncedSave(e.target.value)}
              class="w-28"
              style="accent-color: var(--accent)"
            />
            <span class="text-sm w-16 text-right data-mono" style="color: var(--text-secondary)">{value}</span>
          </div>
        )}
        {vtype === 'boolean' && (
          <button
            class="relative w-10 h-5 rounded-full transition-colors"
            style={value === 'true' || value === '1' ? 'background: var(--accent)' : 'background: var(--bg-inset)'}
            onClick={() => debouncedSave(value === 'true' || value === '1' ? 'false' : 'true')}
          >
            <span class={`absolute top-0.5 w-4 h-4 rounded-full shadow transition-transform ${
              value === 'true' || value === '1' ? 'translate-x-5' : 'translate-x-0.5'
            }`} style="background: var(--bg-surface)" />
          </button>
        )}
        {vtype === 'string' && (
          <input
            type="text"
            value={value}
            onInput={(e) => debouncedSave(e.target.value)}
            class="t-input px-2 py-1 text-sm w-48"
          />
        )}
        {vtype === 'select' && (
          <select
            value={value}
            onChange={(e) => debouncedSave(e.target.value)}
            class="t-input px-2 py-1 text-sm"
          >
            {(config.options || '').split(',').map((o) => (
              <option key={o.trim()} value={o.trim()}>{o.trim()}</option>
            ))}
          </select>
        )}
        <button
          onClick={handleReset}
          disabled={isDefault}
          class="text-xs px-2 py-1"
          style={`border-radius: var(--radius); ${
            isDefault
              ? 'color: var(--text-tertiary); cursor: default;'
              : 'color: var(--accent); cursor: pointer;'
          }`}
        >
          Reset
        </button>
      </div>
    </div>
  );
}

function CategorySection({ category, configs, onUpdate, descMode }) {
  const [open, setOpen] = useState(true);

  return (
    <section class="t-frame" data-label={category.toLowerCase()}>
      <button
        class="w-full flex items-center justify-between px-4 py-3 text-left"
        onClick={() => setOpen(!open)}
      >
        <h3 class="text-sm font-bold" style="color: var(--text-secondary)">{category}</h3>
        <span class="text-xs" style="color: var(--text-tertiary)">{open ? '\u25B2' : '\u25BC'} {configs.length} param{configs.length !== 1 ? 's' : ''}</span>
      </button>
      {open && (
        <div class="px-4 pb-3">
          {configs.map((c) => (
            <ParamControl key={c.key} config={c} onUpdate={onUpdate} descMode={descMode} />
          ))}
        </div>
      )}
    </section>
  );
}

// Extended labels for Settings — supplements the shared DOMAIN_LABELS from utils
import { DOMAIN_LABELS as BASE_DOMAIN_LABELS } from './intelligence/utils.jsx';
const DOMAIN_LABELS = {
  ...BASE_DOMAIN_LABELS,
  automation: 'Automations',
  script: 'Scripts',
  input_boolean: 'Input Booleans',
  input_number: 'Input Numbers',
  input_select: 'Input Selects',
  camera: 'Cameras',
};

/** Toggle pill for filter items */
function FilterToggle({ label, active, count, onToggle }) {
  return (
    <button
      class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors"
      style={active
        ? 'background: var(--accent-glow); color: var(--accent); border: 1px solid var(--accent); cursor: pointer;'
        : 'background: var(--bg-surface-raised); color: var(--text-tertiary); border: 1px solid var(--border-subtle); cursor: pointer;'
      }
      onClick={onToggle}
    >
      <span>{label}</span>
      {count != null && <span class="data-mono" style="opacity: 0.7">{count}</span>}
    </button>
  );
}

/** Data Filtering section — area/domain/entity toggles for controlling what ARIA monitors */
function DataFilteringSection() {
  const entities = useCache('entities');
  const areas = useCache('areas');
  const devices = useCache('devices');

  const [filterOpen, setFilterOpen] = useState(true);
  const [areaSearch, setAreaSearch] = useState('');
  const [domainSearch, setDomainSearch] = useState('');
  const [entitySearch, setEntitySearch] = useState('');
  const [activeTab, setActiveTab] = useState('areas');

  // Excluded items stored in config — local state for optimistic UI
  const [excludedAreas, setExcludedAreas] = useState(new Set());
  const [excludedDomains, setExcludedDomains] = useState(new Set());
  const [excludedEntities, setExcludedEntities] = useState(new Set());
  const [filterSaving, setFilterSaving] = useState(false);
  const [filterLoaded, setFilterLoaded] = useState(false);
  const saveTimerRef = useRef(null);

  // Load existing filter config
  useEffect(() => {
    fetchJson('/api/config').then((cfgData) => {
      const cfgs = cfgData.configs || [];
      for (const c of cfgs) {
        if (c.key === 'filter.exclude_areas' && c.value) {
          setExcludedAreas(new Set(c.value.split(',').map((s) => s.trim()).filter(Boolean)));
        }
        if (c.key === 'filter.exclude_domains' && c.value) {
          setExcludedDomains(new Set(c.value.split(',').map((s) => s.trim()).filter(Boolean)));
        }
        if (c.key === 'filter.exclude_entities' && c.value) {
          setExcludedEntities(new Set(c.value.split(',').map((s) => s.trim()).filter(Boolean)));
        }
      }
      setFilterLoaded(true);
    }).catch(() => {
      setFilterLoaded(true);
    });
  }, []);

  // Debounced save of filter config
  function saveFilters(key, values) {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(async () => {
      setFilterSaving(true);
      try {
        await putJson(`/api/config/${key}`, { value: [...values].join(',') });
      } catch (err) {
        console.error('Filter save failed:', err);
      } finally {
        setFilterSaving(false);
      }
    }, 600);
  }

  // Extract unique areas
  const areaList = useMemo(() => {
    const areasDict = areas.data?.data || {};
    return Object.entries(areasDict)
      .map(([id, area]) => ({ id, name: area.name || id }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [areas.data]);

  // Extract unique domains with entity counts
  const domainList = useMemo(() => {
    const entityArray = Object.values(entities.data?.data || {});
    const counts = {};
    for (const ent of entityArray) {
      const domain = (ent.entity_id || '').split('.')[0];
      if (domain) counts[domain] = (counts[domain] || 0) + 1;
    }
    return Object.entries(counts)
      .map(([domain, count]) => ({ id: domain, name: DOMAIN_LABELS[domain] || domain, count }))
      .sort((a, b) => b.count - a.count);
  }, [entities.data]);

  // Entity list for granular filtering
  const entityList = useMemo(() => {
    const entityDict = entities.data?.data || {};
    const devicesDict = devices.data?.data || {};
    const areasDict = areas.data?.data || {};
    return Object.entries(entityDict)
      .map(([id, ent]) => {
        const device = ent.device_id ? devicesDict[ent.device_id] : null;
        const areaId = ent.area_id || (device ? device.area_id : null);
        const areaName = areaId && areasDict[areaId] ? areasDict[areaId].name : '';
        return {
          id: ent.entity_id || id,
          name: ent.friendly_name || ent.entity_id || id,
          domain: (ent.entity_id || id).split('.')[0],
          area: areaName,
        };
      })
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [entities.data, devices.data, areas.data]);

  function toggleArea(areaId) {
    setExcludedAreas((prev) => {
      const next = new Set(prev);
      if (next.has(areaId)) next.delete(areaId);
      else next.add(areaId);
      saveFilters('filter.exclude_areas', next);
      return next;
    });
  }

  function toggleDomain(domain) {
    setExcludedDomains((prev) => {
      const next = new Set(prev);
      if (next.has(domain)) next.delete(domain);
      else next.add(domain);
      saveFilters('filter.exclude_domains', next);
      return next;
    });
  }

  function toggleEntity(entityId) {
    setExcludedEntities((prev) => {
      const next = new Set(prev);
      if (next.has(entityId)) next.delete(entityId);
      else next.add(entityId);
      saveFilters('filter.exclude_entities', next);
      return next;
    });
  }

  const cacheLoading = entities.loading || areas.loading || devices.loading;
  const totalExcluded = excludedAreas.size + excludedDomains.size + excludedEntities.size;

  const filteredAreas = areaSearch
    ? areaList.filter((a) => a.name.toLowerCase().includes(areaSearch.toLowerCase()))
    : areaList;

  const filteredDomains = domainSearch
    ? domainList.filter((d) => d.name.toLowerCase().includes(domainSearch.toLowerCase()) || d.id.toLowerCase().includes(domainSearch.toLowerCase()))
    : domainList;

  const filteredEntities = entitySearch
    ? entityList.filter((e) => e.name.toLowerCase().includes(entitySearch.toLowerCase()) || e.id.toLowerCase().includes(entitySearch.toLowerCase()))
    : entityList;

  // Cap visible entities at 100 for performance
  const visibleEntities = filteredEntities.slice(0, 100);
  const entityOverflow = filteredEntities.length - visibleEntities.length;

  const tabs = [
    { key: 'areas', label: 'Areas', count: areaList.length, excluded: excludedAreas.size },
    { key: 'domains', label: 'Domains', count: domainList.length, excluded: excludedDomains.size },
    { key: 'entities', label: 'Entities', count: entityList.length, excluded: excludedEntities.size },
  ];

  return (
    <section class="t-frame" data-label="data filtering">
      <button
        class="w-full flex items-center justify-between px-4 py-3 text-left"
        onClick={() => setFilterOpen(!filterOpen)}
      >
        <div>
          <h3 class="text-sm font-bold" style="color: var(--text-secondary)">Data Filtering</h3>
          <p class="text-xs mt-0.5" style="color: var(--text-tertiary)">Control which areas, domains, and entities ARIA monitors and analyzes.</p>
        </div>
        <div class="flex items-center gap-2">
          {filterSaving && <span class="text-xs" style="color: var(--accent)">Saving...</span>}
          {totalExcluded > 0 && (
            <span class="inline-block px-2 py-0.5 rounded-full text-xs font-medium" style="background: var(--status-warning-glow); color: var(--status-warning)">
              {totalExcluded} excluded
            </span>
          )}
          <span class="text-xs" style="color: var(--text-tertiary)">{filterOpen ? '\u25B2' : '\u25BC'}</span>
        </div>
      </button>

      {filterOpen && (
        <div class="px-4 pb-4">
          {cacheLoading && !filterLoaded ? (
            <div class="text-xs py-4 text-center" style="color: var(--text-tertiary)">Loading entity data...</div>
          ) : (
            <>
              {/* Tab bar */}
              <div class="flex gap-1 mb-4 p-0.5 rounded" style="background: var(--bg-inset)">
                {tabs.map((tab) => (
                  <button
                    key={tab.key}
                    class="flex-1 px-3 py-1.5 rounded text-xs font-medium transition-colors"
                    style={activeTab === tab.key
                      ? 'background: var(--bg-surface); color: var(--text-primary); box-shadow: var(--card-shadow);'
                      : 'color: var(--text-tertiary); cursor: pointer;'
                    }
                    onClick={() => setActiveTab(tab.key)}
                  >
                    {tab.label}
                    <span class="data-mono ml-1" style="opacity: 0.6">{tab.count}</span>
                    {tab.excluded > 0 && (
                      <span class="ml-1" style="color: var(--status-warning)">(-{tab.excluded})</span>
                    )}
                  </button>
                ))}
              </div>

              {/* Areas tab */}
              {activeTab === 'areas' && (
                <div>
                  <input
                    type="text"
                    placeholder="Search areas..."
                    value={areaSearch}
                    onInput={(e) => setAreaSearch(e.target.value)}
                    class="t-input w-full px-3 py-1.5 text-sm mb-3"
                  />
                  {filteredAreas.length === 0 ? (
                    <p class="text-xs py-2" style="color: var(--text-tertiary)">No areas found.</p>
                  ) : (
                    <div class="flex flex-wrap gap-2">
                      {filteredAreas.map((area) => (
                        <FilterToggle
                          key={area.id}
                          label={area.name}
                          active={!excludedAreas.has(area.id)}
                          onToggle={() => toggleArea(area.id)}
                        />
                      ))}
                    </div>
                  )}
                  {excludedAreas.size > 0 && (
                    <p class="text-xs mt-3" style="color: var(--status-warning)">
                      {excludedAreas.size} area{excludedAreas.size !== 1 ? 's' : ''} excluded from monitoring. Click to re-enable.
                    </p>
                  )}
                </div>
              )}

              {/* Domains tab */}
              {activeTab === 'domains' && (
                <div>
                  <input
                    type="text"
                    placeholder="Search domains..."
                    value={domainSearch}
                    onInput={(e) => setDomainSearch(e.target.value)}
                    class="t-input w-full px-3 py-1.5 text-sm mb-3"
                  />
                  {filteredDomains.length === 0 ? (
                    <p class="text-xs py-2" style="color: var(--text-tertiary)">No domains found.</p>
                  ) : (
                    <div class="flex flex-wrap gap-2">
                      {filteredDomains.map((domain) => (
                        <FilterToggle
                          key={domain.id}
                          label={domain.name}
                          count={domain.count}
                          active={!excludedDomains.has(domain.id)}
                          onToggle={() => toggleDomain(domain.id)}
                        />
                      ))}
                    </div>
                  )}
                  {excludedDomains.size > 0 && (
                    <p class="text-xs mt-3" style="color: var(--status-warning)">
                      {excludedDomains.size} domain{excludedDomains.size !== 1 ? 's' : ''} excluded from monitoring. Entities in excluded domains will not be tracked.
                    </p>
                  )}
                </div>
              )}

              {/* Entities tab */}
              {activeTab === 'entities' && (
                <div>
                  <input
                    type="text"
                    placeholder="Search entities by name or ID..."
                    value={entitySearch}
                    onInput={(e) => setEntitySearch(e.target.value)}
                    class="t-input w-full px-3 py-1.5 text-sm mb-3"
                  />
                  {visibleEntities.length === 0 ? (
                    <p class="text-xs py-2" style="color: var(--text-tertiary)">
                      {entitySearch ? 'No entities match your search.' : 'No entities discovered yet.'}
                    </p>
                  ) : (
                    <div class="space-y-0">
                      {visibleEntities.map((ent) => (
                        <div
                          key={ent.id}
                          class="flex items-center gap-3 py-2"
                          style="border-bottom: 1px solid var(--border-subtle)"
                        >
                          <button
                            class="relative w-8 h-4 rounded-full transition-colors flex-shrink-0"
                            style={!excludedEntities.has(ent.id) ? 'background: var(--accent)' : 'background: var(--bg-inset)'}
                            onClick={() => toggleEntity(ent.id)}
                          >
                            <span class={`absolute top-0.5 w-3 h-3 rounded-full shadow transition-transform ${
                              !excludedEntities.has(ent.id) ? 'translate-x-4' : 'translate-x-0.5'
                            }`} style="background: var(--bg-surface)" />
                          </button>
                          <div class="flex-1 min-w-0">
                            <div class="text-xs font-medium truncate" style={excludedEntities.has(ent.id) ? 'color: var(--text-tertiary); text-decoration: line-through;' : 'color: var(--text-secondary)'}>{ent.name}</div>
                            <div class="text-xs truncate" style="color: var(--text-tertiary)">{ent.id}{ent.area ? ` \u00b7 ${ent.area}` : ''}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {entityOverflow > 0 && (
                    <p class="text-xs mt-2" style="color: var(--text-tertiary)">
                      Showing {visibleEntities.length} of {filteredEntities.length} entities. Use search to narrow results.
                    </p>
                  )}
                  {excludedEntities.size > 0 && (
                    <p class="text-xs mt-3" style="color: var(--status-warning)">
                      {excludedEntities.size} entit{excludedEntities.size !== 1 ? 'ies' : 'y'} individually excluded from monitoring.
                    </p>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </section>
  );
}

export default function Settings() {
  const [configs, setConfigs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [descMode, setDescMode] = useState('simple');

  async function fetchConfigs() {
    try {
      const data = await fetchJson('/api/config');
      setConfigs(data.configs || []);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchConfigs(); }, []);

  if (loading && configs.length === 0) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Settings</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">Configure shadow engine, activity monitor, and data quality parameters.</p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Settings</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">Configure shadow engine, activity monitor, and data quality parameters.</p>
        </div>
        <ErrorState error={error} onRetry={fetchConfigs} />
      </div>
    );
  }

  const groups = groupByCategory(configs);
  const modified = configs.filter((c) => c.value !== c.default_value).length;

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="SETTINGS" subtitle={`${configs.length} parameters across ${Object.keys(groups).length} categories.${modified > 0 ? ` ${modified} modified from defaults.` : ''}`} />

      <div class="flex items-center justify-end mb-2">
        <button
          class="text-xs px-3 py-1.5"
          style={`border-radius: var(--radius); border: 1px solid var(--border-subtle); color: var(--text-secondary); background: var(--bg-surface); cursor: pointer;`}
          onClick={() => setDescMode(mode => mode === 'simple' ? 'technical' : 'simple')}
        >
          {descMode === 'simple' ? 'Show Technical' : 'Show Simple'}
        </button>
      </div>

      {/* Data Filtering — area/domain/entity toggles */}
      <DataFilteringSection />

      <div class="space-y-4">
        {Object.entries(groups).map(([cat, items]) => (
          <CategorySection key={cat} category={cat} configs={items} onUpdate={fetchConfigs} descMode={descMode} />
        ))}
      </div>
    </div>
  );
}
