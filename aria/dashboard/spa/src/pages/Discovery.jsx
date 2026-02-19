import { useState, useMemo, useEffect } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { fetchJson, putJson } from '../api.js';
import HeroCard from '../components/HeroCard.jsx';
import PageBanner from '../components/PageBanner.jsx';
import StatsGrid from '../components/StatsGrid.jsx';
import DataTable from '../components/DataTable.jsx';
import DomainChart from '../components/DomainChart.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import InlineSettings from '../components/InlineSettings.jsx';

// Stable empty-object references to avoid useMemo recomputation during loading
const EMPTY_OBJ = {};

/** Resolve effective area: entity's own area_id, or its parent device's area_id */
function getEffectiveArea(entity, devicesDict) {
  if (entity.area_id) return entity.area_id;
  if (entity.device_id) {
    const device = devicesDict[entity.device_id];
    if (device && device.area_id) return device.area_id;
  }
  return null;
}

export default function Discovery() {
  const entities = useCache('entities');
  const devices = useCache('devices');
  const areas = useCache('areas');
  const capabilities = useCache('capabilities');

  // Filter state for entity table
  const [domainFilter, setDomainFilter] = useState('');
  const [stateFilter, setStateFilter] = useState('');
  const [areaFilter, setAreaFilter] = useState('');
  const [hideUnavailable, setHideUnavailable] = useState(false);
  const [savingIgnore, setSavingIgnore] = useState(false);

  // Sync hideUnavailable toggle from backend config on mount
  useEffect(() => {
    fetchJson('/api/config/curation.unavailable_grace_hours')
      .then((cfg) => {
        const val = Number(cfg?.value ?? 0);
        if (val > 0) setHideUnavailable(true);
      })
      .catch(() => {});
  }, []);

  const cacheLoading = entities.loading || devices.loading || areas.loading || capabilities.loading;
  const cacheError = entities.error || devices.error || areas.error || capabilities.error;

  // Unpack dicts (stable fallback prevents useMemo churn during loading)
  const entitiesDict = entities.data?.data || EMPTY_OBJ;
  const devicesDict = devices.data?.data || EMPTY_OBJ;
  const areasDict = areas.data?.data || EMPTY_OBJ;
  const capsDict = capabilities.data?.data || EMPTY_OBJ;

  // Entity array
  const entityArray = useComputed(
    () => Object.entries(entitiesDict).map(([id, e]) => ({ entity_id: id, ...e })),
    [entitiesDict]
  );

  // Unavailable counts (total + long-unavailable for context)
  const unavailableCounts = useComputed(() => {
    const unavailable = entityArray.filter((e) => e.state === 'unavailable');
    const total = unavailable.length;
    // Count entities unavailable > 24h (using last_changed)
    const now = Date.now();
    const longUnavailable = unavailable.filter((e) => {
      if (!e.last_changed) return false;
      const changedMs = new Date(e.last_changed).getTime();
      return (now - changedMs) > 24 * 60 * 60 * 1000;
    }).length;
    return { total, longUnavailable };
  }, [entityArray]);
  const unavailableCount = unavailableCounts.total;

  // Stats
  const stats = useComputed(() => {
    if (!entities.data) return null;
    return [
      { label: 'Entities', value: entityArray.length.toLocaleString() },
      { label: 'Devices', value: Object.keys(devicesDict).length.toLocaleString() },
      { label: 'Areas', value: Object.keys(areasDict).length.toLocaleString() },
      { label: 'Capabilities', value: Object.keys(capsDict).length.toLocaleString() },
      {
        label: 'Unavailable',
        value: unavailableCount,
        subtitle: unavailableCounts.longUnavailable > 0 ? `${unavailableCounts.longUnavailable} > 24h` : undefined,
        warning: unavailableCount > 100,
      },
    ];
  }, [entities.data, entityArray, devicesDict, areasDict, capsDict, unavailableCount, unavailableCounts]);

  // Domain breakdown for chart
  const domainBreakdown = useComputed(() => {
    const counts = {};
    for (const e of entityArray) {
      const d = e.domain || e.entity_id?.split('.')[0] || 'unknown';
      counts[d] = (counts[d] || 0) + 1;
    }
    return Object.entries(counts)
      .map(([domain, count]) => ({ domain, count }))
      .sort((a, b) => b.count - a.count);
  }, [entityArray]);

  // Area entity counts (resolves through device when entity has no direct area)
  const areaCounts = useComputed(() => {
    const counts = {};
    for (const e of entityArray) {
      const area = getEffectiveArea(e, devicesDict);
      if (area) {
        counts[area] = (counts[area] || 0) + 1;
      }
    }
    return Object.entries(areasDict).map(([id, a]) => ({
      area_id: id,
      name: a.name || id,
      count: counts[id] || 0,
    })).sort((a, b) => b.count - a.count);
  }, [entityArray, areasDict, devicesDict]);

  // Unique domains/states/areas for filter dropdowns
  const uniqueDomains = useComputed(() => {
    const set = new Set(entityArray.map((e) => e.domain || e.entity_id?.split('.')[0]));
    return [...set].sort();
  }, [entityArray]);

  const uniqueStates = useComputed(() => {
    const set = new Set(entityArray.map((e) => e.state).filter(Boolean));
    return [...set].sort();
  }, [entityArray]);

  const uniqueAreas = useComputed(() => {
    return Object.entries(areasDict)
      .map(([id, a]) => ({ id, name: a.name || id }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [areasDict]);

  // Filtered entity data for table
  const filteredEntities = useMemo(() => {
    let arr = entityArray;
    if (hideUnavailable) {
      arr = arr.filter((e) => e.state !== 'unavailable');
    }
    if (domainFilter) {
      arr = arr.filter((e) => (e.domain || e.entity_id?.split('.')[0]) === domainFilter);
    }
    if (stateFilter) {
      arr = arr.filter((e) => e.state === stateFilter);
    }
    if (areaFilter) {
      arr = arr.filter((e) => getEffectiveArea(e, devicesDict) === areaFilter);
    }
    return arr;
  }, [entityArray, hideUnavailable, domainFilter, stateFilter, areaFilter, devicesDict]);

  // Entity table columns
  const entityColumns = [
    {
      key: 'friendly_name',
      label: 'Entity',
      sortable: true,
      render: (_, row) => (
        <div>
          <div class="font-semibold" style="color: var(--text-primary)">{row.friendly_name || row.entity_id}</div>
          <div class="text-xs data-mono" style="color: var(--text-tertiary)">{row.entity_id}</div>
        </div>
      ),
    },
    {
      key: 'state',
      label: 'State',
      sortable: true,
      render: (val, row) => (
        <span class="inline-flex items-center gap-1">
          <StatusBadge state={val} />
          {row.unit_of_measurement && (
            <span class="text-xs" style="color: var(--text-tertiary)">{row.unit_of_measurement}</span>
          )}
        </span>
      ),
    },
    {
      key: 'domain',
      label: 'Domain',
      sortable: true,
      render: (val, row) => {
        const domain = val || row.entity_id?.split('.')[0] || '\u2014';
        return (
          <span class="inline-block px-2 py-0.5 text-xs font-medium" style="background: var(--bg-inset); color: var(--text-primary); border-radius: var(--radius);">
            {domain}
          </span>
        );
      },
    },
    {
      key: 'device_class',
      label: 'Class',
      sortable: true,
      render: (val) => val || '\u2014',
    },
    {
      key: 'area_id',
      label: 'Area',
      sortable: true,
      render: (val, row) => {
        const effectiveArea = getEffectiveArea(row, devicesDict);
        if (!effectiveArea) return '\u2014';
        const area = areasDict[effectiveArea];
        return area ? area.name : effectiveArea;
      },
    },
    {
      key: 'device_id',
      label: 'Device',
      sortable: true,
      render: (val) => {
        if (!val) return '\u2014';
        const device = devicesDict[val];
        return device ? device.name : val;
      },
    },
  ];

  // Device table
  const deviceArray = useComputed(
    () => Object.entries(devicesDict).map(([id, d]) => ({
      device_id: id,
      ...d,
      area_name: d.area_id && areasDict[d.area_id] ? areasDict[d.area_id].name : (d.area_id || '\u2014'),
    })),
    [devicesDict, areasDict]
  );

  const deviceColumns = [
    {
      key: 'name',
      label: 'Name',
      sortable: true,
      render: (val, row) => (
        <div>
          <div class="font-semibold" style="color: var(--text-primary)">{val || row.device_id}</div>
          <div class="text-xs data-mono" style="color: var(--text-tertiary)">{row.device_id}</div>
        </div>
      ),
    },
    { key: 'manufacturer', label: 'Manufacturer', sortable: true, render: (val) => val || '\u2014' },
    { key: 'model', label: 'Model', sortable: true, render: (val) => val || '\u2014' },
    { key: 'area_name', label: 'Area', sortable: true, render: (val) => val || '\u2014' },
  ];

  // Filter dropdown JSX for entity table
  const entityFilters = (
    <div class="flex flex-wrap gap-3">
      <select
        value={domainFilter}
        onChange={(e) => setDomainFilter(e.target.value)}
        class="t-input px-2 py-1.5 text-sm"
      >
        <option value="">All domains</option>
        {uniqueDomains.map((d) => (
          <option key={d} value={d}>{d}</option>
        ))}
      </select>
      <select
        value={stateFilter}
        onChange={(e) => setStateFilter(e.target.value)}
        class="t-input px-2 py-1.5 text-sm"
      >
        <option value="">All states</option>
        <option value="on">on</option>
        <option value="off">off</option>
        <option value="unavailable">unavailable</option>
        {uniqueStates
          .filter((s) => s !== 'on' && s !== 'off' && s !== 'unavailable')
          .map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
      </select>
      <select
        value={areaFilter}
        onChange={(e) => setAreaFilter(e.target.value)}
        class="t-input px-2 py-1.5 text-sm"
      >
        <option value="">All areas</option>
        {uniqueAreas.map((a) => (
          <option key={a.id} value={a.id}>{a.name}</option>
        ))}
      </select>
      <button
        onClick={async () => {
          const next = !hideUnavailable;
          setHideUnavailable(next);
          setSavingIgnore(true);
          try {
            await putJson('/api/config/curation.unavailable_grace_hours', {
              value: next ? '24' : '0',
              changed_by: 'dashboard',
            });
          } catch (err) {
            console.error('Failed to update unavailable_grace_hours:', err);
          } finally {
            setSavingIgnore(false);
          }
        }}
        disabled={savingIgnore}
        class="px-2 py-1.5 text-sm font-medium disabled:opacity-50"
        style={hideUnavailable
          ? "color: var(--bg-primary); background: var(--accent); border-radius: var(--radius);"
          : "color: var(--text-tertiary); background: var(--bg-inset); border-radius: var(--radius);"}
        title={hideUnavailable
          ? 'Hiding unavailable entities (also excluded from backend analysis after 24h)'
          : 'Click to hide unavailable entities from view and backend analysis'}
      >
        {hideUnavailable ? `Ignoring ${unavailableCount} unavailable` : `Ignore unavailable (${unavailableCount})`}
      </button>
      {(domainFilter || stateFilter || areaFilter || hideUnavailable) && (
        <button
          onClick={() => { setDomainFilter(''); setStateFilter(''); setAreaFilter(''); setHideUnavailable(false); }}
          class="px-2 py-1.5 text-sm"
          style="color: var(--accent);"
        >
          Clear filters
        </button>
      )}
    </div>
  );

  const pageSubtitle = "Everything the hub found by scanning your Home Assistant instance. Rescans automatically every 24 hours.";

  // Loading
  if (cacheLoading && !entities.data) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Discovery</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  // Error
  if (cacheError) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Discovery</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
        </div>
        <ErrorState error={cacheError} onRetry={() => { entities.refetch(); devices.refetch(); areas.refetch(); capabilities.refetch(); }} />
      </div>
    );
  }

  const areaCount = Object.keys(areasDict).length;

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="DISCOVERY" subtitle="Every entity, device, and area in your home." />

      {/* Hero — what ARIA sees */}
      <HeroCard
        value={entityArray.length.toLocaleString()}
        label="entities discovered"
        delta={`across ${areaCount} area${areaCount !== 1 ? 's' : ''}`}
        loading={cacheLoading}
      />

      {/* Stats */}
      {stats ? <StatsGrid items={stats} /> : <LoadingState type="stats" />}

      {/* Domain Breakdown */}
      <section>
        <div class="t-section-header mb-4" style="padding-bottom: 6px;">
          <h2 class="text-lg font-semibold" style="color: var(--text-primary)">Domain Breakdown</h2>
          <p class="text-sm" style="color: var(--text-tertiary)">How your entities are distributed across HA domains. Larger bars = more entities in that domain.</p>
        </div>
        <div class="t-frame" data-label="domains" style="padding: 1rem;">
          <DomainChart data={domainBreakdown} total={entityArray.length} />
        </div>
      </section>

      {/* Area Grid */}
      {areaCounts.length > 0 && (
        <section>
          <div class="t-section-header mb-4" style="padding-bottom: 6px;">
            <h2 class="text-lg font-semibold" style="color: var(--text-primary)">Areas</h2>
            <p class="text-sm" style="color: var(--text-tertiary)">Physical locations defined in Home Assistant, with entity counts per area.</p>
          </div>
          <div class="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {areaCounts.map((a) => (
              <div key={a.area_id} class="t-frame" data-label={a.name} style="padding: 1rem;">
                <div class="font-semibold text-sm truncate" style="color: var(--text-primary)" title={a.name}>{a.name}</div>
                <div class="text-sm mt-1" style="color: var(--text-tertiary)">{a.count} entities</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Entity Table */}
      <section>
        <div class="t-section-header mb-4" style="padding-bottom: 6px;">
          <h2 class="text-lg font-semibold" style="color: var(--text-primary)">Entities</h2>
          <p class="text-sm" style="color: var(--text-tertiary)">Every entity registered in HA — sensors, switches, lights, and more. Filter by domain, state, or area.</p>
        </div>
        <DataTable
          columns={entityColumns}
          data={filteredEntities}
          searchFields={['entity_id', 'friendly_name', 'domain', 'device_class']}
          pageSize={50}
          searchPlaceholder="Search entities..."
          filterContent={entityFilters}
        />
      </section>

      {/* Device Table */}
      <section>
        <div class="t-section-header mb-4" style="padding-bottom: 6px;">
          <h2 class="text-lg font-semibold" style="color: var(--text-primary)">Devices</h2>
          <p class="text-sm" style="color: var(--text-tertiary)">Physical devices registered in HA. Each device groups multiple entities (e.g., a thermostat has temperature, humidity, and mode entities).</p>
        </div>
        <DataTable
          columns={deviceColumns}
          data={deviceArray}
          searchFields={['name', 'device_id', 'manufacturer', 'model']}
          pageSize={50}
          searchPlaceholder="Search devices..."
        />
      </section>

      {/* Data Quality Settings */}
      <InlineSettings
        categories={['Data Quality']}
        title="Data Quality"
        subtitle="Control which entities are excluded from analysis. Excluded entities are hidden from ML pipelines, predictions, and pattern detection."
      />
    </div>
  );
}
