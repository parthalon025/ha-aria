import { useState, useEffect } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { fetchJson } from '../api.js';
import StatsGrid from '../components/StatsGrid.jsx';
import DataTable from '../components/DataTable.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

function ExecutiveSummary({ entityCount, capCount, moduleCount, maturity }) {
  const phase = maturity ? maturity.phase : null;
  const days = maturity ? maturity.days_of_data : 0;

  return (
    <section class="bg-white rounded-lg shadow-sm p-5 space-y-4 border-l-4 border-blue-500">
      <div>
        <h2 class="text-base font-bold text-gray-900">Why does this exist?</h2>
        <p class="text-sm text-gray-700 leading-relaxed mt-1">
          Home Assistant tells you what's happening right now. It doesn't tell you what
          <span class="italic"> should</span> be happening, what changed from yesterday, or what
          you could automate but haven't. This system closes that gap — it watches your home
          over weeks and months, learns what normal looks like, and surfaces the things you'd
          only notice if you were staring at dashboards all day.
        </p>
      </div>

      <div>
        <h2 class="text-base font-bold text-gray-900">What is this?</h2>
        <p class="text-sm text-gray-700 leading-relaxed mt-1">
          The HA Intelligence Hub watches your Home Assistant instance and learns how your home
          behaves over time. It scans every entity, detects what your home can do, and builds
          an increasingly accurate model of what "normal" looks like — so it can predict what
          should happen next and eventually suggest automations you haven't thought of.
        </p>
      </div>

      <div>
        <h2 class="text-base font-bold text-gray-900">How does it work?</h2>
        <p class="text-sm text-gray-700 leading-relaxed mt-1">
          Six modules run as a pipeline, each building on the last:
        </p>
        <ol class="text-sm text-gray-700 mt-2 space-y-1 list-decimal ml-5">
          <li><span class="font-medium">Discovery</span> scans HA for entities, devices, and areas every 24 hours.</li>
          <li><span class="font-medium">Capabilities</span> identifies what your home can do (lighting, power monitoring, climate, locks, etc.).</li>
          <li><span class="font-medium">Intelligence</span> collects daily and intraday snapshots via cron, building baselines and predictions.</li>
          <li><span class="font-medium">Patterns</span> analyzes logbook sequences to find recurring behaviors.</li>
          <li><span class="font-medium">ML Engine</span> trains models after 14 days to make entity-level predictions.</li>
          <li><span class="font-medium">Orchestrator</span> combines patterns + capabilities to suggest new automations.</li>
        </ol>
      </div>

      <div>
        <h2 class="text-base font-bold text-gray-900">Where are we now?</h2>
        <p class="text-sm text-gray-700 leading-relaxed mt-1">
          {entityCount > 0 ? (
            <>
              Discovery has found <span class="font-medium">{entityCount.toLocaleString()} entities</span>
              {capCount > 0 && <> across <span class="font-medium">{capCount} capabilities</span></>}
              {moduleCount > 0 && <>, with <span class="font-medium">{moduleCount} modules</span> running</>}.
            </>
          ) : (
            'Waiting for first discovery scan to complete.'
          )}
          {phase && (
            <>
              {' '}The intelligence engine is in the <span class="font-medium">{phase}</span> phase
              {days > 0 && <> with <span class="font-medium">{days} day{days !== 1 ? 's' : ''}</span> of data</>}.
              {phase === 'collecting' && ' It needs 7 days of snapshots before baselines become reliable, and 14 days before ML models activate.'}
              {phase === 'baselines' && ' Statistical baselines are active. ML models will activate after 14 days.'}
              {phase === 'ml-training' && ' ML models are training. Predictions now blend statistics with machine learning.'}
              {phase === 'ml-active' && ' Full intelligence is active — baselines, ML predictions, and meta-learning are all running.'}
            </>
          )}
        </p>
      </div>

      <p class="text-xs text-gray-400">
        Use the sidebar to explore each module's output. Every page explains what it shows and what to expect as data accumulates.
      </p>
    </section>
  );
}

function ShadowSummary({ shadowAccuracy }) {
  if (!shadowAccuracy) return null;

  const total = shadowAccuracy.predictions_total ?? 0;
  const acc = shadowAccuracy.overall_accuracy ?? 0;
  const stage = shadowAccuracy.stage || 'backtest';

  return (
    <section class="bg-white rounded-lg shadow-sm p-4 border-l-4 border-blue-500">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <span class="text-xs font-medium bg-blue-100 text-blue-700 rounded-full px-2.5 py-0.5 capitalize">{stage}</span>
          <span class="text-sm text-gray-700">
            {total > 0
              ? `${total} prediction${total !== 1 ? 's' : ''}, ${Math.round(acc)}% accuracy`
              : 'No predictions yet'}
          </span>
        </div>
        <a href="#/shadow" class="text-sm text-blue-600 hover:text-blue-800 font-medium">View details &rarr;</a>
      </div>
    </section>
  );
}

export default function Home() {
  const entities = useCache('entities');
  const devices = useCache('devices');
  const areas = useCache('areas');
  const capabilities = useCache('capabilities');
  const intelligence = useCache('intelligence');

  // Direct API fetches for health and events
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [events, setEvents] = useState(null);
  const [eventsError, setEventsError] = useState(null);
  const [shadowAccuracy, setShadowAccuracy] = useState(null);

  useEffect(() => {
    fetchJson('/health')
      .then((d) => { setHealth(d); setHealthError(null); })
      .catch((err) => setHealthError(err.message || String(err)));
  }, []);

  useEffect(() => {
    fetchJson('/api/events?limit=20')
      .then((d) => { setEvents(d); setEventsError(null); })
      .catch((err) => setEventsError(err.message || String(err)));
  }, []);

  useEffect(() => {
    fetchJson('/api/shadow/accuracy')
      .then((d) => setShadowAccuracy(d))
      .catch(() => {}); // silently hide on error — shadow is non-essential on Home
  }, []);

  const cacheLoading = entities.loading || devices.loading || areas.loading || capabilities.loading;
  const cacheError = entities.error || devices.error || areas.error || capabilities.error;

  // Stats
  const stats = useComputed(() => {
    if (!entities.data || !devices.data || !areas.data || !capabilities.data) return null;

    const entityCount = Object.keys(entities.data.data || {}).length;
    const deviceCount = Object.keys(devices.data.data || {}).length;
    const areaCount = Object.keys(areas.data.data || {}).length;
    const capCount = Object.keys(capabilities.data.data || {}).length;
    const moduleCount = health ? Object.keys(health.modules || {}).length : 0;

    return {
      items: [
        { label: 'Entities', value: entityCount.toLocaleString() },
        { label: 'Devices', value: deviceCount.toLocaleString() },
        { label: 'Areas', value: areaCount.toLocaleString() },
        { label: 'Capabilities', value: capCount.toLocaleString() },
        { label: 'Modules', value: moduleCount },
      ],
      entityCount,
      capCount,
      moduleCount,
    };
  }, [entities.data, devices.data, areas.data, capabilities.data, health]);

  // Intelligence maturity for executive summary
  const maturity = useComputed(() => {
    if (!intelligence.data || !intelligence.data.data) return null;
    return intelligence.data.data.data_maturity || null;
  }, [intelligence.data]);

  // Module list from health
  const modules = useComputed(() => {
    if (!health || !health.modules) return [];
    return Object.entries(health.modules).map(([name, info]) => ({
      name,
      registered: info.registered,
    }));
  }, [health]);

  // Events table data
  const eventRows = useComputed(() => {
    if (!events || !events.events) return [];
    return events.events.map((e) => ({
      id: e.id,
      time: e.timestamp ? new Date(e.timestamp).toLocaleString() : '\u2014',
      type: e.event_type || '\u2014',
      category: e.category || '\u2014',
      details: e.data ? JSON.stringify(e.data).slice(0, 120) : '\u2014',
    }));
  }, [events]);

  const eventColumns = [
    { key: 'time', label: 'Time', sortable: true },
    { key: 'type', label: 'Type', sortable: true },
    { key: 'category', label: 'Category', sortable: true },
    { key: 'details', label: 'Details', className: 'max-w-xs truncate' },
  ];

  if (cacheLoading && !entities.data) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p class="text-sm text-gray-500">System overview — entities discovered, modules running, and recent hub activity.</p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (cacheError) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p class="text-sm text-gray-500">System overview — entities discovered, modules running, and recent hub activity.</p>
        </div>
        <ErrorState error={cacheError} onRetry={() => { entities.refetch(); devices.refetch(); areas.refetch(); capabilities.refetch(); }} />
      </div>
    );
  }

  return (
    <div class="space-y-6">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p class="text-sm text-gray-500">System overview — entities discovered, modules running, and recent hub activity.</p>
      </div>

      {/* Executive Summary */}
      <ExecutiveSummary
        entityCount={stats ? stats.entityCount : 0}
        capCount={stats ? stats.capCount : 0}
        moduleCount={stats ? stats.moduleCount : 0}
        maturity={maturity}
      />

      {/* Stats */}
      {stats ? <StatsGrid items={stats.items} /> : <LoadingState type="stats" />}

      {/* Shadow Mode Summary */}
      <ShadowSummary shadowAccuracy={shadowAccuracy} />

      {/* Module Health */}
      <section>
        <div class="mb-4">
          <h2 class="text-lg font-semibold text-gray-900">Module Health</h2>
          <p class="text-sm text-gray-500">Each module handles a piece of the intelligence pipeline. Green means registered and running.</p>
        </div>
        {healthError ? (
          <ErrorState error={healthError} onRetry={() => fetchJson('/health').then((d) => { setHealth(d); setHealthError(null); }).catch((e) => setHealthError(e.message))} />
        ) : modules.length === 0 ? (
          <div class="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-800">Modules are initializing. Discovery scans your HA instance first, then other modules load sequentially.</div>
        ) : (
          <div class="bg-white rounded-lg shadow-sm p-4">
            <div class="flex flex-wrap gap-4">
              {modules.map((m) => (
                <div key={m.name} class="flex items-center gap-2">
                  <span class={`inline-block w-2.5 h-2.5 rounded-full ${m.registered ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span class="text-sm text-gray-700">{m.name}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Cache Categories */}
      {health && health.cache && health.cache.categories && (
        <section>
          <div class="mb-4">
            <h2 class="text-lg font-semibold text-gray-900">Cache Categories</h2>
            <p class="text-sm text-gray-500">Data stored by each module. Categories update automatically when modules refresh or new data arrives via WebSocket.</p>
          </div>
          <div class="bg-white rounded-lg shadow-sm p-4">
            <div class="flex flex-wrap gap-3">
              {health.cache.categories.map((cat) => {
                // Try to find last_updated from the loaded cache data
                const cacheMap = { entities, devices, areas, capabilities };
                const catData = cacheMap[cat];
                const lastUpdated = catData && catData.data ? catData.data.last_updated : null;

                return (
                  <div key={cat} class="flex items-center gap-2 bg-gray-50 rounded px-3 py-1.5">
                    <span class="text-sm font-medium text-gray-700">{cat}</span>
                    {lastUpdated && (
                      <span class="text-xs text-gray-400">
                        {new Date(lastUpdated).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </section>
      )}

      {/* Recent Events */}
      <section>
        <div class="mb-4">
          <h2 class="text-lg font-semibold text-gray-900">Recent Events</h2>
          <p class="text-sm text-gray-500">Internal hub activity — module registrations, cache updates, and scheduled tasks. Not HA device events.</p>
        </div>
        {eventsError ? (
          <ErrorState error={eventsError} onRetry={() => fetchJson('/api/events?limit=20').then((d) => { setEvents(d); setEventsError(null); }).catch((e) => setEventsError(e.message))} />
        ) : !events ? (
          <LoadingState type="table" />
        ) : (
          <DataTable
            columns={eventColumns}
            data={eventRows}
            searchFields={['type', 'category', 'details']}
            pageSize={20}
            searchPlaceholder="Search events..."
          />
        )}
      </section>
    </div>
  );
}
