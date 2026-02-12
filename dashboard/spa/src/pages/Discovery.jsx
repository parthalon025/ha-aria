import { useState, useMemo } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import StatsGrid from '../components/StatsGrid.jsx';
import DataTable from '../components/DataTable.jsx';
import DomainChart from '../components/DomainChart.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

// Stable empty-object references to avoid useMemo recomputation during loading
const EMPTY_OBJ = {};

export default function Discovery() {
  const entities = useCache('entities');
  const devices = useCache('devices');
  const areas = useCache('areas');
  const capabilities = useCache('capabilities');

  // Filter state for entity table
  const [domainFilter, setDomainFilter] = useState('');
  const [stateFilter, setStateFilter] = useState('');
  const [areaFilter, setAreaFilter] = useState('');

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

  // Unavailable count
  const unavailableCount = useComputed(
    () => entityArray.filter((e) => e.state === 'unavailable').length,
    [entityArray]
  );

  // Stats
  const stats = useComputed(() => {
    if (!entities.data) return null;
    return [
      { label: 'Entities', value: entityArray.length.toLocaleString() },
      { label: 'Devices', value: Object.keys(devicesDict).length.toLocaleString() },
      { label: 'Areas', value: Object.keys(areasDict).length.toLocaleString() },
      { label: 'Capabilities', value: Object.keys(capsDict).length.toLocaleString() },
      { label: 'Unavailable', value: unavailableCount, warning: unavailableCount > 100 },
    ];
  }, [entities.data, entityArray, devicesDict, areasDict, capsDict, unavailableCount]);

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

  // Area entity counts
  const areaCounts = useComputed(() => {
    const counts = {};
    for (const e of entityArray) {
      if (e.area_id) {
        counts[e.area_id] = (counts[e.area_id] || 0) + 1;
      }
    }
    return Object.entries(areasDict).map(([id, a]) => ({
      area_id: id,
      name: a.name || id,
      count: counts[id] || 0,
    })).sort((a, b) => b.count - a.count);
  }, [entityArray, areasDict]);

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
    if (domainFilter) {
      arr = arr.filter((e) => (e.domain || e.entity_id?.split('.')[0]) === domainFilter);
    }
    if (stateFilter) {
      arr = arr.filter((e) => e.state === stateFilter);
    }
    if (areaFilter) {
      arr = arr.filter((e) => e.area_id === areaFilter);
    }
    return arr;
  }, [entityArray, domainFilter, stateFilter, areaFilter]);

  // Entity table columns
  const entityColumns = [
    {
      key: 'friendly_name',
      label: 'Entity',
      sortable: true,
      render: (_, row) => (
        <div>
          <div class="font-semibold text-gray-900">{row.friendly_name || row.entity_id}</div>
          <div class="text-xs font-mono text-gray-400">{row.entity_id}</div>
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
            <span class="text-xs text-gray-400">{row.unit_of_measurement}</span>
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
          <span class="inline-block px-2 py-0.5 rounded text-xs font-medium bg-gray-800 text-white">
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
      render: (val) => {
        if (!val) return '\u2014';
        const area = areasDict[val];
        return area ? area.name : val;
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
          <div class="font-semibold text-gray-900">{val || row.device_id}</div>
          <div class="text-xs font-mono text-gray-400">{row.device_id}</div>
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
        class="bg-white border border-gray-300 rounded-md px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="">All domains</option>
        {uniqueDomains.map((d) => (
          <option key={d} value={d}>{d}</option>
        ))}
      </select>
      <select
        value={stateFilter}
        onChange={(e) => setStateFilter(e.target.value)}
        class="bg-white border border-gray-300 rounded-md px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
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
        class="bg-white border border-gray-300 rounded-md px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="">All areas</option>
        {uniqueAreas.map((a) => (
          <option key={a.id} value={a.id}>{a.name}</option>
        ))}
      </select>
      {(domainFilter || stateFilter || areaFilter) && (
        <button
          onClick={() => { setDomainFilter(''); setStateFilter(''); setAreaFilter(''); }}
          class="px-2 py-1.5 text-sm text-blue-600 hover:text-blue-800"
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
          <h1 class="text-2xl font-bold text-gray-900">Discovery</h1>
          <p class="text-sm text-gray-500">{pageSubtitle}</p>
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
          <h1 class="text-2xl font-bold text-gray-900">Discovery</h1>
          <p class="text-sm text-gray-500">{pageSubtitle}</p>
        </div>
        <ErrorState error={cacheError} onRetry={() => { entities.refetch(); devices.refetch(); areas.refetch(); capabilities.refetch(); }} />
      </div>
    );
  }

  return (
    <div class="space-y-6">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Discovery</h1>
        <p class="text-sm text-gray-500">{pageSubtitle}</p>
      </div>

      {/* Stats */}
      {stats ? <StatsGrid items={stats} /> : <LoadingState type="stats" />}

      {/* Domain Breakdown */}
      <section>
        <div class="mb-4">
          <h2 class="text-lg font-semibold text-gray-900">Domain Breakdown</h2>
          <p class="text-sm text-gray-500">How your entities are distributed across HA domains. Larger bars = more entities in that domain.</p>
        </div>
        <div class="bg-white rounded-lg shadow-sm p-4">
          <DomainChart data={domainBreakdown} total={entityArray.length} />
        </div>
      </section>

      {/* Area Grid */}
      {areaCounts.length > 0 && (
        <section>
          <div class="mb-4">
            <h2 class="text-lg font-semibold text-gray-900">Areas</h2>
            <p class="text-sm text-gray-500">Physical locations defined in Home Assistant, with entity counts per area.</p>
          </div>
          <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {areaCounts.map((a) => (
              <div key={a.area_id} class="bg-white rounded-lg shadow-sm p-4">
                <div class="font-semibold text-gray-900 text-sm truncate" title={a.name}>{a.name}</div>
                <div class="text-sm text-gray-400 mt-1">{a.count} entities</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Entity Table */}
      <section>
        <div class="mb-4">
          <h2 class="text-lg font-semibold text-gray-900">Entities</h2>
          <p class="text-sm text-gray-500">Every entity registered in HA — sensors, switches, lights, and more. Filter by domain, state, or area.</p>
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
        <div class="mb-4">
          <h2 class="text-lg font-semibold text-gray-900">Devices</h2>
          <p class="text-sm text-gray-500">Physical devices registered in HA. Each device groups multiple entities (e.g., a thermostat has temperature, humidity, and mode entities).</p>
        </div>
        <DataTable
          columns={deviceColumns}
          data={deviceArray}
          searchFields={['name', 'device_id', 'manufacturer', 'model']}
          pageSize={50}
          searchPlaceholder="Search devices..."
        />
      </section>
    </div>
  );
}
