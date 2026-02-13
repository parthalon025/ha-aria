import { useState, useEffect } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { fetchJson } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STATUS = {
  HEALTHY: { label: 'Healthy', bg: 'bg-green-100', text: 'text-green-700', dot: 'bg-green-500' },
  WAITING: { label: 'Waiting', bg: 'bg-gray-100', text: 'text-gray-500', dot: 'bg-gray-400' },
  REVIEW:  { label: 'Review', bg: 'bg-amber-100', text: 'text-amber-700', dot: 'bg-amber-500', pulse: true },
  BLOCKED: { label: 'Blocked', bg: 'bg-red-100', text: 'text-red-700', dot: 'bg-red-500' },
};

const NODE_META = {
  discovery:           { icon: '\uD83D\uDD0D', label: 'Discovery',        lane: 0 },
  activity_monitor:    { icon: '\uD83D\uDCE1', label: 'Activity Monitor', lane: 0 },
  data_quality:        { icon: '\uD83C\uDFAF', label: 'Data Curation',    lane: 0 },
  intelligence:        { icon: '\uD83E\uDDE0', label: 'Intelligence',     lane: 1 },
  ml_engine:           { icon: '\uD83D\uDCCA', label: 'ML Engine',        lane: 1 },
  pattern_recognition: { icon: '\uD83D\uDD17', label: 'Patterns',         lane: 1 },
  shadow_engine:       { icon: '\uD83D\uDC7B', label: 'Shadow Engine',    lane: 2 },
  orchestrator:        { icon: '\u26A1',        label: 'Orchestrator',     lane: 2 },
  pipeline_gates:      { icon: '\uD83D\uDEA6', label: 'Pipeline Gates',   lane: 2 },
};

const LANES = [
  { title: 'Data Collection', subtitle: 'What feeds the system', color: 'border-emerald-400' },
  { title: 'Learning',        subtitle: 'How the system learns', color: 'border-violet-400' },
  { title: 'Actions',         subtitle: 'What the system produces', color: 'border-amber-400' },
];

const PHASES = ['collecting', 'baselines', 'ml-training', 'ml-active'];
const PHASE_LABELS = ['Collecting', 'Baselines', 'ML Training', 'ML Active'];
const PHASE_MILESTONES = [
  'Gathering daily snapshots',
  'Statistical baselines active',
  'Training ML models',
  'Full intelligence active',
];

// ---------------------------------------------------------------------------
// Status computation
// ---------------------------------------------------------------------------

function computeNodeStatus(nodeId, data) {
  const { health, entities, activity, curation, intelligence, shadow, pipeline } = data;
  const modules = health ? health.modules || {} : {};
  const isRegistered = (name) => modules[name] && modules[name].registered;

  const maturity = intelligence ? (intelligence.data_maturity || null) : null;
  const days = maturity ? maturity.days_of_data || 0 : 0;
  const phase = maturity ? maturity.phase || null : null;
  const mlActive = phase === 'ml-active';

  const entityCount = entities && entities.data ? Object.keys(entities.data || {}).length : 0;
  const wsObj = activity && activity.data ? (activity.data.websocket || null) : null;
  const snapshotCount = maturity ? (maturity.intraday_count || 0) : 0;
  const predTotal = shadow ? (shadow.predictions_total || 0) : 0;
  const stage = pipeline ? (pipeline.current_stage || 'backtest') : 'backtest';

  const curationTotal = curation ? (curation.total || 0) : 0;

  switch (nodeId) {
    case 'discovery':
      if (!isRegistered('discovery')) return STATUS.BLOCKED;
      return entityCount > 0 ? STATUS.HEALTHY : STATUS.WAITING;

    case 'activity_monitor':
      if (!isRegistered('activity_monitor')) return STATUS.BLOCKED;
      if (wsObj === null) return STATUS.WAITING;
      return wsObj && wsObj.connected ? STATUS.HEALTHY : STATUS.BLOCKED;

    case 'data_quality':
      return curationTotal > 0 ? STATUS.HEALTHY : STATUS.WAITING;

    case 'intelligence':
      if (!isRegistered('intelligence')) return STATUS.BLOCKED;
      if (!maturity) return STATUS.WAITING;
      return snapshotCount > 0 ? STATUS.HEALTHY : STATUS.WAITING;

    case 'ml_engine':
      if (!isRegistered('ml_engine')) return STATUS.BLOCKED;
      if (mlActive) return STATUS.HEALTHY;
      if (days >= 14) return STATUS.REVIEW;
      return STATUS.WAITING;

    case 'pattern_recognition':
      if (!isRegistered('pattern_recognition')) return STATUS.BLOCKED;
      return days >= 1 ? STATUS.HEALTHY : STATUS.WAITING;

    case 'shadow_engine':
      if (!isRegistered('shadow_engine')) return STATUS.BLOCKED;
      return predTotal > 0 ? STATUS.HEALTHY : STATUS.WAITING;

    case 'orchestrator':
      if (!isRegistered('orchestrator')) return STATUS.BLOCKED;
      return mlActive ? STATUS.HEALTHY : STATUS.WAITING;

    case 'pipeline_gates':
      return stage === 'autonomous' ? STATUS.HEALTHY : STATUS.WAITING;

    default:
      return STATUS.WAITING;
  }
}

function computeNodeStats(data) {
  const { entities, activity, curation, intelligence, shadow, pipeline } = data;

  const entityCount = entities && entities.data ? Object.keys(entities.data || {}).length : 0;
  const maturity = intelligence ? (intelligence.data_maturity || null) : null;
  const days = maturity ? maturity.days_of_data || 0 : 0;
  const phase = maturity ? maturity.phase || null : null;
  const snapshotCount = maturity ? (maturity.intraday_count || 0) : 0;
  const mlActive = phase === 'ml-active';

  const actRate = activity && activity.data ? (activity.data.activity_rate || null) : null;
  const rate = actRate ? actRate.current : null;

  const perStatus = curation ? (curation.per_status || {}) : {};
  const included = perStatus.included || 0;
  const excluded = (perStatus.excluded || 0) + (perStatus.auto_excluded || 0);

  const predTotal = shadow ? (shadow.predictions_total || 0) : 0;
  const acc = shadow ? Math.round(shadow.overall_accuracy || 0) : 0;

  const stage = pipeline ? (pipeline.current_stage || 'backtest') : 'backtest';

  return {
    discovery: entityCount > 0 ? `${entityCount} entities discovered` : 'Scanning...',
    activity_monitor: rate !== null ? `${rate} events/min` : 'Connecting to HA...',
    data_quality: curation ? `${included} included, ${excluded} filtered` : 'Loading...',
    intelligence: snapshotCount > 0 ? `Day ${days}/7 \u2014 ${snapshotCount} snapshots` : 'Waiting for first snapshot',
    ml_engine: mlActive ? 'Models trained and active' : (days >= 14 ? 'Ready to train' : `${Math.max(0, 14 - days)} days until ML training`),
    pattern_recognition: days >= 1 ? 'Analyzing event sequences' : 'Needs activity data',
    shadow_engine: predTotal > 0 ? `${predTotal} predictions, ${acc}% accuracy` : 'No predictions yet',
    orchestrator: mlActive ? 'Generating automation suggestions' : 'Waiting for ML + patterns',
    pipeline_gates: `Stage: ${stage}`,
  };
}

// ---------------------------------------------------------------------------
// Small components
// ---------------------------------------------------------------------------

function StatusChip({ status }) {
  return (
    <span class={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium ${status.bg} ${status.text}`}>
      <span class={`w-1.5 h-1.5 rounded-full ${status.dot}${status.pulse ? ' animate-pulse-amber' : ''}`} />
      {status.label}
    </span>
  );
}

function LaneHeader({ lane }) {
  return (
    <div class={`border-b-2 ${lane.color} pb-2 mb-4`}>
      <h3 class="text-sm font-semibold text-gray-900">{lane.title}</h3>
      <p class="text-xs text-gray-500">{lane.subtitle}</p>
    </div>
  );
}

function DownArrow() {
  return (
    <div class="flex justify-center md:hidden py-1">
      <svg width="20" height="24" viewBox="0 0 20 24" fill="none">
        <path d="M10 2 L10 18 M4 14 L10 20 L16 14" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    </div>
  );
}

function LaneArrow() {
  return (
    <div class="hidden md:flex items-center justify-center px-2">
      <svg width="40" height="24" viewBox="0 0 40 24" fill="none">
        <line x1="0" y1="12" x2="30" y2="12" stroke="#94A3B8" stroke-width="2" stroke-dasharray="6 4" class="animate-dash-flow" />
        <path d="M28 6 L36 12 L28 18" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />
      </svg>
    </div>
  );
}

function PipelineNode({ nodeId, status, stat, link }) {
  const meta = NODE_META[nodeId];
  if (!meta) return null;

  const inner = (
    <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-3 hover:shadow-md transition-shadow">
      <div class="flex items-center gap-2 mb-1">
        <span class="text-lg">{meta.icon}</span>
        <span class="text-sm font-medium text-gray-900">{meta.label}</span>
      </div>
      <StatusChip status={status} />
      {stat && <p class="text-xs text-gray-500 mt-1.5">{stat}</p>}
    </div>
  );

  if (link) {
    return <a href={link} class="block no-underline">{inner}</a>;
  }
  return inner;
}

function YouNode({ title, guidance, linkHref, linkLabel }) {
  return (
    <div class="border-2 border-dashed border-blue-300 rounded-lg p-3 bg-blue-50/50">
      <div class="flex items-center gap-2 mb-1">
        <span class="text-xs font-bold bg-blue-600 text-white rounded px-1.5 py-0.5">YOU</span>
        <span class="text-sm font-medium text-gray-900">{title}</span>
      </div>
      <p class="text-xs text-gray-600">{guidance}</p>
      {linkHref && (
        <a href={linkHref} class="text-xs text-blue-600 hover:text-blue-800 font-medium mt-1 inline-block">
          {linkLabel || 'View'} &rarr;
        </a>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// JourneyProgress
// ---------------------------------------------------------------------------

function JourneyProgress({ maturity, shadowStage }) {
  const phase = maturity ? maturity.phase || 'collecting' : 'collecting';
  const activeIdx = PHASES.indexOf(phase);

  return (
    <section class="bg-white rounded-lg shadow-sm p-4 animate-fade-in-up">
      <h2 class="text-sm font-semibold text-gray-900 mb-3">System Maturity</h2>
      <div class="flex items-center gap-1 mb-2">
        {PHASES.map((p, i) => {
          let bg = 'bg-gray-200';
          if (i < activeIdx) bg = 'bg-green-500';
          else if (i === activeIdx) bg = 'bg-blue-500';
          return <div key={p} class={`h-2 flex-1 rounded-full ${bg}`} />;
        })}
      </div>
      <div class="flex justify-between text-xs">
        {PHASES.map((p, i) => {
          let color = 'text-gray-400';
          if (i < activeIdx) color = 'text-green-600';
          else if (i === activeIdx) color = 'text-blue-600 font-medium';
          return <span key={p} class={color}>{PHASE_LABELS[i]}</span>;
        })}
      </div>
      <p class="text-xs text-gray-500 mt-2">
        {activeIdx >= 0 && activeIdx < PHASE_MILESTONES.length ? PHASE_MILESTONES[activeIdx] : ''}
        {shadowStage ? ` \u2014 Pipeline: ${shadowStage}` : ''}
      </p>
    </section>
  );
}

// ---------------------------------------------------------------------------
// RightNowStrip
// ---------------------------------------------------------------------------

function RightNowStrip({ activity, intraday }) {
  // activity = useCache result; activity.data = {category, data: {occupancy, websocket, ...}}
  const inner = activity && activity.data ? (activity.data.data || null) : null;
  const ws = inner ? (inner.websocket || null) : null;
  const actRate = inner ? (inner.activity_rate || null) : null;
  const evRate = actRate ? actRate.current : null;
  const occ = inner ? (inner.occupancy || null) : null;

  // intraday_trend is a list of hourly snapshots — use the latest
  const latest = Array.isArray(intraday) && intraday.length > 0 ? intraday[intraday.length - 1] : null;
  const lightsOn = latest ? (latest.lights_on ?? null) : null;
  const powerW = latest ? (latest.power_watts ?? null) : null;

  return (
    <section class="bg-white rounded-lg shadow-sm p-3 animate-fade-in-up delay-100">
      <div class="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
        <div class="flex items-center gap-1.5">
          <span class="text-gray-500">Occupancy</span>
          <span class="font-medium text-gray-900">{occ && occ.anyone_home ? 'Home' : occ ? 'Away' : '\u2014'}</span>
        </div>
        <div class="flex items-center gap-1.5">
          <span class="text-gray-500">Events</span>
          <span class="font-medium text-gray-900">{evRate != null ? `${evRate}/min` : '\u2014'}</span>
        </div>
        <div class="flex items-center gap-1.5">
          <span class="text-gray-500">Lights</span>
          <span class="font-medium text-gray-900">{lightsOn != null ? `${lightsOn} on` : '\u2014'}</span>
        </div>
        <div class="flex items-center gap-1.5">
          <span class="text-gray-500">Power</span>
          <span class="font-medium text-gray-900">{powerW != null ? `${Math.round(powerW)} W` : '\u2014'}</span>
        </div>
        <div class="flex items-center gap-1.5">
          <span class={`w-2 h-2 rounded-full ${ws && ws.connected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span class="text-gray-500">WebSocket</span>
          <span class="font-medium text-gray-900">{ws && ws.connected ? 'Connected' : ws === null ? '\u2014' : 'Disconnected'}</span>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// PipelineFlow
// ---------------------------------------------------------------------------

const LANE_NODES = [
  ['discovery', 'activity_monitor', 'data_quality'],
  ['intelligence', 'ml_engine', 'pattern_recognition'],
  ['shadow_engine', 'orchestrator', 'pipeline_gates'],
];

const NODE_LINKS = {
  discovery: '#/discovery',
  activity_monitor: '#/intelligence',
  data_quality: '#/data-curation',
  intelligence: '#/intelligence',
  ml_engine: '#/intelligence',
  pattern_recognition: '#/patterns',
  shadow_engine: '#/shadow',
  orchestrator: '#/automations',
  pipeline_gates: '#/shadow',
};

function PipelineFlow({ statusData, curation, maturity, shadow, pipeline }) {
  const statuses = {};
  const statStrings = computeNodeStats(statusData);
  Object.keys(NODE_META).forEach((nid) => {
    statuses[nid] = computeNodeStatus(nid, statusData);
  });

  const included = curation && curation.per_status ? (curation.per_status.included || 0) : 0;
  const mlActive = maturity && maturity.phase === 'ml-active';
  const days = maturity ? (maturity.days_of_data || 0) : 0;
  const stage = pipeline ? (pipeline.current_stage || 'backtest') : 'backtest';

  const youNodes = [
    {
      title: 'Curate Your Entities',
      guidance: `${included} entities feeding data. Exclude noisy sensors to improve signal quality.`,
      linkHref: '#/data-curation',
      linkLabel: 'Data Curation',
    },
    {
      title: 'Adjust Parameters',
      guidance: mlActive
        ? 'Fine-tune training parameters'
        : `${Math.max(0, 14 - days)} more days of data needed. No action needed.`,
      linkHref: '#/settings',
      linkLabel: 'Settings',
    },
    {
      title: 'Review & Advance',
      guidance: stage === 'autonomous'
        ? 'System is fully autonomous'
        : `Pipeline at ${stage} stage. Accuracy gates control progression.`,
      linkHref: '#/shadow',
      linkLabel: 'Shadow Mode',
    },
  ];

  return (
    <section class="animate-fade-in-up delay-200">
      <div class="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr_auto_1fr] gap-4 md:gap-2">
        {LANE_NODES.map((nodes, laneIdx) => {
          const lane = LANES[laneIdx];
          const you = youNodes[laneIdx];
          return (
            <>
              {/* Lane arrow before 2nd and 3rd lanes (desktop only) */}
              {laneIdx > 0 && <LaneArrow />}
              <div key={laneIdx}>
                <LaneHeader lane={lane} />
                <div class="space-y-2">
                  {nodes.map((nid, nIdx) => (
                    <>
                      {nIdx > 0 && <DownArrow />}
                      <PipelineNode
                        key={nid}
                        nodeId={nid}
                        status={statuses[nid]}
                        stat={statStrings[nid]}
                        link={NODE_LINKS[nid]}
                      />
                    </>
                  ))}
                  <DownArrow />
                  <YouNode
                    title={you.title}
                    guidance={you.guidance}
                    linkHref={you.linkHref}
                    linkLabel={you.linkLabel}
                  />
                </div>
              </div>
            </>
          );
        })}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Home (default export)
// ---------------------------------------------------------------------------

export default function Home() {
  const intelligence = useCache('intelligence');
  const activity = useCache('activity_summary');
  const entities = useCache('entities');

  const [health, setHealth] = useState(null);
  const [shadow, setShadow] = useState(null);
  const [pipeline, setPipeline] = useState(null);
  const [curation, setCuration] = useState(null);
  const [fetchError, setFetchError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetchJson('/health').catch(() => null),
      fetchJson('/api/shadow/accuracy').catch(() => null),
      fetchJson('/api/pipeline').catch(() => null),
      fetchJson('/api/curation/summary').catch(() => null),
    ]).then(([h, s, p, c]) => {
      setHealth(h);
      setShadow(s);
      setPipeline(p);
      setCuration(c);
    }).catch((err) => setFetchError(err.message || String(err)));
  }, []);

  const loading = intelligence.loading || activity.loading || entities.loading;
  const cacheError = intelligence.error || activity.error || entities.error;

  const maturity = useComputed(() => {
    if (!intelligence.data || !intelligence.data.data) return null;
    return intelligence.data.data.data_maturity || null;
  }, [intelligence.data]);

  const intraday = useComputed(() => {
    if (!intelligence.data || !intelligence.data.data) return null;
    return intelligence.data.data.intraday_trend || null;
  }, [intelligence.data]);

  const shadowStage = useComputed(() => {
    return pipeline ? (pipeline.current_stage || null) : null;
  }, [pipeline]);

  const statusData = useComputed(() => ({
    health,
    entities: entities.data,
    activity: activity.data,
    curation,
    intelligence: intelligence.data ? intelligence.data.data : null,
    shadow,
    pipeline,
  }), [health, entities.data, activity.data, curation, intelligence.data, shadow, pipeline]);

  if (loading && !intelligence.data) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Pipeline Dashboard</h1>
          <p class="text-sm text-gray-500">Live system overview — data flow, module health, and your next steps.</p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (cacheError) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Pipeline Dashboard</h1>
          <p class="text-sm text-gray-500">Live system overview — data flow, module health, and your next steps.</p>
        </div>
        <ErrorState
          error={cacheError}
          onRetry={() => { intelligence.refetch(); activity.refetch(); entities.refetch(); }}
        />
      </div>
    );
  }

  if (fetchError) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Pipeline Dashboard</h1>
          <p class="text-sm text-gray-500">Live system overview — data flow, module health, and your next steps.</p>
        </div>
        <ErrorState error={fetchError} />
      </div>
    );
  }

  return (
    <div class="space-y-6">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Pipeline Dashboard</h1>
        <p class="text-sm text-gray-500">Live system overview — data flow, module health, and your next steps.</p>
      </div>

      <JourneyProgress maturity={maturity} shadowStage={shadowStage} />

      <RightNowStrip activity={activity} intraday={intraday} />

      <PipelineFlow
        statusData={statusData}
        curation={curation}
        maturity={maturity}
        shadow={shadow}
        pipeline={pipeline}
      />
    </div>
  );
}
