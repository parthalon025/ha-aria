import { useState, useEffect, useRef } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { getCategory } from '../store.js';
import { fetchJson } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import AriaLogo from '../components/AriaLogo.jsx';
import HeroCard from '../components/HeroCard.jsx';
import PageBanner from '../components/PageBanner.jsx';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PHASES = ['collecting', 'baselines', 'ml-training', 'ml-active'];
const PHASE_LABELS = ['Collecting', 'Baselines', 'ML Training', 'ML Active'];
const PHASE_MILESTONES = [
  'Gathering daily snapshots',
  'Statistical baselines active',
  'Training ML models',
  'Full intelligence active',
];

// ---------------------------------------------------------------------------
// Bus Architecture Diagram
// ---------------------------------------------------------------------------

const PLANE_DATA = {
  data: {
    label: 'DATA PLANE',
    nodes: [
      { id: 'discovery', label: 'Discovery', metricKey: 'entity_count' },
      { id: 'activity_monitor', label: 'Activity', metricKey: 'event_rate' },
      { id: 'data_quality', label: 'Curation', metricKey: 'included_count' },
      { id: 'activity_labeler', label: 'Labeler', metricKey: 'current_activity' },
    ],
  },
  learning: {
    label: 'LEARNING PLANE',
    nodes: [
      { id: 'intelligence', label: 'Intel.', metricKey: 'day_count' },
      { id: 'ml_engine', label: 'ML Engine', metricKey: 'mean_r2' },
      { id: 'pattern_recognition', label: 'Patterns', metricKey: 'sequence_count' },
      { id: 'drift_monitor', label: 'Drift', metricKey: 'drift_count' },
    ],
  },
  action: {
    label: 'ACTION PLANE',
    nodes: [
      { id: 'shadow_engine', label: 'Shadow', metricKey: 'accuracy' },
      { id: 'orchestrator', label: 'Orchestr.', metricKey: 'pending_count' },
      { id: 'pipeline_gates', label: 'Gates', metricKey: 'pipeline_stage' },
      { id: 'feedback_health', label: 'Feedback', metricKey: 'feedback_fresh' },
    ],
  },
};

function getNodeStatus(moduleStatuses, nodeId) {
  const status = moduleStatuses?.[nodeId];
  if (status === 'running') return 'healthy';
  if (status === 'failed') return 'blocked';
  if (status === 'starting') return 'waiting';
  return 'waiting';
}

function getNodeMetric(cacheData, node) {
  const caps = cacheData?.capabilities?.data || {};
  const pipeline = cacheData?.pipeline?.data || {};
  const shadow = cacheData?.shadow_accuracy?.data || {};
  const activity = cacheData?.activity_labels?.data || {};
  const feedback = cacheData?.feedback_health || {};

  switch (node.id) {
    case 'discovery': {
      const count = Object.values(caps).filter((entry) => entry && typeof entry === 'object' && entry.entities).reduce((sum, entry) => sum + (entry.entities?.length || 0), 0);
      return count ? `${count} entities` : '\u2014';
    }
    case 'activity_monitor': return pipeline?.events_per_minute ? `${pipeline.events_per_minute.toFixed(1)} ev/m` : '\u2014';
    case 'data_quality': return pipeline?.included_entities ? `${pipeline.included_entities} incl.` : '\u2014';
    case 'activity_labeler': {
      const curr = activity?.current_activity;
      return curr?.predicted || '\u2014';
    }
    case 'intelligence': return pipeline?.intelligence_day ? `Day ${pipeline.intelligence_day}` : '\u2014';
    case 'ml_engine': {
      const mlCaps = Object.values(caps).filter((entry) => entry?.ml_accuracy);
      if (mlCaps.length === 0) return '\u2014';
      const avgR2 = mlCaps.reduce((s, entry) => s + (entry.ml_accuracy.mean_r2 || 0), 0) / mlCaps.length;
      return `R\u00B2: ${avgR2.toFixed(2)}`;
    }
    case 'pattern_recognition': return '\u2014';
    case 'drift_monitor': {
      const drifted = Object.values(caps).filter((entry) => entry?.drift_flagged).length;
      return `${drifted} flagged`;
    }
    case 'shadow_engine': return shadow?.overall_accuracy ? `${(shadow.overall_accuracy * 100).toFixed(0)}%` : '\u2014';
    case 'orchestrator': return '\u2014';
    case 'pipeline_gates': return pipeline?.stage || 'shadow';
    case 'feedback_health': {
      const fresh = (feedback?.capabilities_with_ml_feedback || 0) + (feedback?.capabilities_with_shadow_feedback || 0);
      return `${fresh} fresh`;
    }
    default: return '\u2014';
  }
}

function ModuleNode({ x, y, status, label, metric, glowing, animateMetric }) {
  const colors = {
    healthy: 'var(--status-healthy)',
    waiting: 'var(--status-waiting)',
    blocked: 'var(--status-error)',
    review: 'var(--status-warning)',
  };
  const color = colors[status] || colors.waiting;

  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect width="180" height="55" rx="4" fill="var(--bg-surface)" stroke={glowing ? 'var(--accent)' : 'var(--border-primary)'} stroke-width={glowing ? '2' : '1'} />
      {glowing && (
        <rect width="180" height="55" rx="4" fill="none" stroke="var(--accent)" stroke-width="1" opacity="0.3">
          <animate attributeName="opacity" values="0.3;0.1;0.3" dur="2s" repeatCount="1" />
        </rect>
      )}
      <circle cx="16" cy="16" r="5" fill={color} filter="url(#led-glow)">
        {(status === 'healthy' || glowing) && <animate attributeName="opacity" values="1;0.6;1" dur={glowing ? '1s' : '3s'} repeatCount="indefinite" />}
      </circle>
      <text x="28" y="20" fill="var(--text-primary)" font-size="11" font-weight="600" font-family="var(--font-mono)">{label}</text>
      <text x="16" y="42" fill="var(--text-tertiary)" font-size="10" font-family="var(--font-mono)" class={animateMetric ? 't2-typewriter' : ''}>
        {metric}
      </text>
    </g>
  );
}

function PlaneLabel({ x, y, label }) {
  return <text x={x} y={y} text-anchor="middle" fill="var(--text-tertiary)" font-size="10" font-weight="700" font-family="var(--font-mono)" letter-spacing="2">{label}</text>;
}

function BusConnector({ x, y1, y2, label, stale, flashing }) {
  const lineClass = stale ? '' : 'bus-connector-line';
  const stroke = stale ? 'var(--status-error)' : 'var(--border-primary)';
  const dasharray = stale ? '2 4' : undefined;
  const flashClass = flashing ? ' animate-data-refresh' : '';

  return (
    <g class={flashClass}>
      <line
        x1={x} y1={y1} x2={x} y2={y2}
        class={lineClass}
        stroke={stroke}
        stroke-width="1"
        stroke-dasharray={dasharray}
      />
      {stale && (
        <g transform={`translate(${x - 4}, ${(y1 + y2) / 2 - 6})`}>
          <circle cx="4" cy="4" r="6" fill="var(--status-error)" opacity="0.2" />
          <text x="4" y="7" text-anchor="middle" fill="var(--status-error)" font-size="9" font-weight="bold" font-family="var(--font-mono)">!</text>
        </g>
      )}
      {label && <text x={x + 4} y={(y1 + y2) / 2} fill={stale ? 'var(--status-error)' : 'var(--text-tertiary)'} font-size="8" font-family="var(--font-mono)">{label}</text>}
    </g>
  );
}

// Map cache categories to connector indices for flash animation
const CACHE_CONNECTOR_MAP = {
  capabilities: { section: 'data', index: 0 },
  entities: { section: 'data', index: 0 },
  activity_summary: { section: 'data', index: 1 },
  entity_curation: { section: 'data', index: 2 },
  activity_labels: { section: 'data', index: 3 },
  ml_accuracy: { section: 'learning', index: 1 },
  shadow_accuracy: { section: 'learning', index: 1 },
};

const STALE_THRESHOLD_MS = 48 * 60 * 60 * 1000; // 48 hours

function isStaleTimestamp(ts) {
  if (!ts) return true;
  const d = typeof ts === 'string' ? new Date(ts).getTime() : ts;
  if (isNaN(d)) return true;
  return Date.now() - d > STALE_THRESHOLD_MS;
}

function computeBusLoad(feedbackHealth) {
  if (!feedbackHealth) return 1.5;
  const metrics = [
    feedbackHealth.capabilities_with_ml_feedback,
    feedbackHealth.capabilities_with_shadow_feedback,
    feedbackHealth.total_feedback_entries,
    feedbackHealth.feedback_sources_active,
    feedbackHealth.capabilities_with_drift_feedback,
  ];
  const active = metrics.filter((v) => v && v > 0).length;
  if (active >= 5) return 3;
  if (active >= 3) return 2.5;
  if (active >= 1) return 2;
  return 1.5;
}

function BusArchitecture({ moduleStatuses, cacheData, feedbackHealth }) {
  const nodeX = [50, 260, 470, 680];
  const connectorLabelsData = ['entities', 'events', 'rules', 'labels'];
  const connectorLabelsLearning = ['', 'accuracy', '', ''];

  // --- Animation 1: Cache update flash ---
  const [flashingConnectors, setFlashingConnectors] = useState({});
  const flashTimerRef = useRef({});

  useEffect(() => {
    const categories = Object.keys(CACHE_CONNECTOR_MAP);
    const lastFetchedRef = {};

    // Snapshot current lastFetched values
    categories.forEach((cat) => {
      const sig = getCategory(cat);
      lastFetchedRef[cat] = sig.value.lastFetched || 0;
    });

    const interval = setInterval(() => {
      categories.forEach((cat) => {
        const sig = getCategory(cat);
        const current = sig.value.lastFetched || 0;
        if (current > lastFetchedRef[cat] && lastFetchedRef[cat] > 0) {
          const mapping = CACHE_CONNECTOR_MAP[cat];
          const connKey = `${mapping.section}-${mapping.index}`;
          setFlashingConnectors((prev) => ({ ...prev, [connKey]: true }));

          // Clear after 300ms
          clearTimeout(flashTimerRef.current[connKey]);
          flashTimerRef.current[connKey] = setTimeout(() => {
            setFlashingConnectors((prev) => ({ ...prev, [connKey]: false }));
          }, 300);
        }
        lastFetchedRef[cat] = current;
      });
    }, 200);

    return () => {
      clearInterval(interval);
      Object.values(flashTimerRef.current).forEach(clearTimeout);
    };
  }, []);

  // --- Animation 3: Activity Labeler pulse ---
  const prevActivityRef = useRef(null);
  const [labelerGlowing, setLabelerGlowing] = useState(false);
  const [labelerAnimateMetric, setLabelerAnimateMetric] = useState(false);
  const glowTimerRef = useRef(null);

  const currentActivity = cacheData?.activity_labels?.data?.current_activity?.predicted || null;

  useEffect(() => {
    if (prevActivityRef.current !== null && currentActivity !== prevActivityRef.current && currentActivity !== null) {
      setLabelerGlowing(true);
      setLabelerAnimateMetric(true);
      clearTimeout(glowTimerRef.current);
      glowTimerRef.current = setTimeout(() => {
        setLabelerGlowing(false);
        setLabelerAnimateMetric(false);
      }, 2000);
    }
    prevActivityRef.current = currentActivity;
    return () => clearTimeout(glowTimerRef.current);
  }, [currentActivity]);

  // --- Animation 4: Stale indicators ---
  const feedbackTimestamps = feedbackHealth || {};
  const staleLearning = [
    false,
    isStaleTimestamp(feedbackTimestamps.last_ml_feedback_at),
    false,
    false,
  ];
  const staleFeedback = [
    isStaleTimestamp(feedbackTimestamps.last_shadow_feedback_at),
    false,
    false,
    isStaleTimestamp(feedbackTimestamps.last_feedback_at),
  ];

  // --- Animation 5: Bus load indicator ---
  const busLoad = computeBusLoad(feedbackHealth);

  function renderPlane(planeKey, yOffset) {
    const plane = PLANE_DATA[planeKey];
    return (
      <g transform={`translate(0, ${yOffset})`}>
        <PlaneLabel x={450} y={15} label={plane.label} />
        {plane.nodes.map((node, idx) => {
          const isLabeler = node.id === 'activity_labeler';
          return (
            <ModuleNode
              key={node.id}
              x={nodeX[idx]}
              y={25}
              status={getNodeStatus(moduleStatuses, node.id)}
              label={node.label}
              metric={getNodeMetric(cacheData, node)}
              glowing={isLabeler && labelerGlowing}
              animateMetric={isLabeler && labelerAnimateMetric}
            />
          );
        })}
      </g>
    );
  }

  // Full feedback loop path: ML Engine → down to feedback bus → left → up to capabilities bus → right → down to Discovery
  // ML Engine node is at nodeX[1]=260 + 90 center = 350, learning plane y=150+25+55=230
  // Feedback bus at y=272 (260+12 center)
  // Capabilities bus at y=122 (110+12 center)
  // Discovery node is at nodeX[0]=50 + 90 center = 140, data plane y=25+55=80
  const tracerPath = 'M350,230 L350,272 L140,272 L140,122 L870,122 L870,175';

  return (
    <section class="t-terminal-bg rounded-lg p-4 overflow-x-auto">
      <svg viewBox="0 0 900 530" class="w-full" style="min-width: 700px; max-width: 100%;">
        {/* Defs for animations and filters */}
        <defs>
          <pattern id="bus-flow" width="20" height="4" patternUnits="userSpaceOnUse">
            <rect width="10" height="4" fill="var(--accent)" opacity="0.5">
              <animate attributeName="x" from="-20" to="20" dur="1.5s" repeatCount="indefinite" />
            </rect>
          </pattern>
          <pattern id="feedback-flow" width="20" height="4" patternUnits="userSpaceOnUse">
            <rect width="10" height="4" fill="var(--status-healthy)" opacity="0.5">
              <animate attributeName="x" from="20" to="-20" dur="2s" repeatCount="indefinite" />
            </rect>
          </pattern>
          <filter id="led-glow">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="tracer-glow">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* DATA PLANE */}
        {renderPlane('data', 0)}

        {/* Connectors: data plane -> capabilities bus */}
        {nodeX.map((nx, idx) => (
          <BusConnector
            key={`dc${idx}`}
            x={nx + 90}
            y1={80}
            y2={110}
            label={connectorLabelsData[idx]}
            flashing={flashingConnectors[`data-${idx}`]}
          />
        ))}

        {/* CAPABILITIES BUS */}
        <g transform="translate(0, 110)">
          <rect x="30" y="0" width="840" height="24" rx="4" fill="url(#bus-flow)" />
          <rect x="30" y="0" width="840" height="24" rx="4" fill="none" stroke="var(--accent)" stroke-width={busLoad} opacity="0.4" style="transition: stroke-width 0.5s ease;" />
          <text x="450" y="16" text-anchor="middle" fill="var(--accent)" font-size="9" font-family="var(--font-mono)">
            CAPABILITIES BUS  [entities] [activity] [curation] [labels] [usefulness]
          </text>
          <circle r="3" fill="var(--accent)">
            <animateMotion dur="3s" repeatCount="indefinite" path="M30,12 L870,12" />
          </circle>
        </g>

        {/* Connectors: capabilities bus -> learning plane */}
        {nodeX.map((nx, idx) => (
          <BusConnector key={`cl${idx}`} x={nx + 90} y1={134} y2={160} />
        ))}

        {/* LEARNING PLANE */}
        {renderPlane('learning', 150)}

        {/* Connectors: learning plane -> feedback bus */}
        {nodeX.map((nx, idx) => (
          <BusConnector
            key={`lf${idx}`}
            x={nx + 90}
            y1={230}
            y2={260}
            label={connectorLabelsLearning[idx]}
            stale={staleLearning[idx]}
            flashing={flashingConnectors[`learning-${idx}`]}
          />
        ))}

        {/* FEEDBACK BUS */}
        <g transform="translate(0, 260)">
          <rect x="30" y="0" width="840" height="24" rx="4" fill="url(#feedback-flow)" />
          <rect x="30" y="0" width="840" height="24" rx="4" fill="none" stroke="var(--status-healthy)" stroke-width={busLoad} opacity="0.4" style="transition: stroke-width 0.5s ease;" />
          <text x="450" y="16" text-anchor="middle" fill="var(--status-healthy)" font-size="9" font-family="var(--font-mono)">
            FEEDBACK BUS  [accuracy] [hit_rate] [suggestions] [drift] [corrections]
          </text>
          <circle r="3" fill="var(--status-healthy)">
            <animateMotion dur="4s" repeatCount="indefinite" path="M870,12 L30,12" />
          </circle>
        </g>

        {/* Connectors: feedback bus -> action plane */}
        {nodeX.map((nx, idx) => (
          <BusConnector
            key={`fa${idx}`}
            x={nx + 90}
            y1={284}
            y2={310}
            stale={staleFeedback[idx]}
          />
        ))}

        {/* ACTION PLANE */}
        {renderPlane('action', 300)}

        {/* YOU node at bottom */}
        <g transform="translate(310, 390)">
          <rect x="0" y="0" width="280" height="35" rx="4" fill="var(--bg-inset)" stroke="var(--accent)" stroke-width="2" />
          <text x="140" y="22" text-anchor="middle" fill="var(--accent)" font-size="12" font-weight="bold" font-family="var(--font-mono)">
            {'YOU:  Label \u00B7 Curate \u00B7 Review \u00B7 Advance'}
          </text>
        </g>

        {/* Feedback loop tracer — full loop path */}
        {/* Lead tracer */}
        <circle r="4" fill="var(--status-healthy)" opacity="0.9" filter="url(#tracer-glow)">
          <animateMotion dur="10s" repeatCount="indefinite" path={tracerPath} />
          <animate attributeName="r" values="4;6;4" dur="10s" repeatCount="indefinite" />
        </circle>
        {/* Trailing glow (larger, dimmer, 0.5s behind) */}
        <circle r="7" fill="var(--status-healthy)" opacity="0.25" filter="url(#tracer-glow)">
          <animateMotion dur="10s" repeatCount="indefinite" path={tracerPath} begin="0.5s" />
          <animate attributeName="r" values="7;10;7" dur="10s" repeatCount="indefinite" begin="0.5s" />
        </circle>
      </svg>
    </section>
  );
}

// ---------------------------------------------------------------------------
// JourneyProgress
// ---------------------------------------------------------------------------

function JourneyProgress({ maturity, shadowStage }) {
  const phase = maturity ? maturity.phase || 'collecting' : 'collecting';
  const activeIdx = PHASES.indexOf(phase);

  return (
    <section class="t-frame" data-label="system maturity">
      <div class="flex items-center gap-1 mb-2">
        {PHASES.map((p, i) => {
          let bg = 'var(--bg-inset)';
          if (i < activeIdx) bg = 'var(--status-healthy)';
          else if (i === activeIdx) bg = 'var(--accent)';
          return <div key={p} class="h-2 flex-1" style={`border-radius: var(--radius); background: ${bg};`} />;
        })}
      </div>
      <div class="flex justify-between text-xs">
        {PHASES.map((p, i) => {
          let color = 'var(--text-tertiary)';
          let weight = 'normal';
          if (i < activeIdx) color = 'var(--status-healthy)';
          else if (i === activeIdx) { color = 'var(--accent)'; weight = '500'; }
          return <span key={p} style={`color: ${color}; font-weight: ${weight};`}>{PHASE_LABELS[i]}</span>;
        })}
      </div>
      <p class="text-xs mt-2" style="color: var(--text-tertiary)">
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
    <section class="t-frame" data-label="live metrics">
      <div class="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
        <div class="flex items-center gap-1.5">
          <span style="color: var(--text-tertiary)">Occupancy</span>
          <span class="font-medium" style="color: var(--text-primary)">{occ && occ.anyone_home ? 'Home' : occ ? 'Away' : '\u2014'}</span>
        </div>
        <div class="flex items-center gap-1.5">
          <span style="color: var(--text-tertiary)">Events</span>
          <span class="data-mono font-medium" style="color: var(--text-primary)">{evRate != null ? `${evRate}/min` : '\u2014'}</span>
        </div>
        <div class="flex items-center gap-1.5">
          <span style="color: var(--text-tertiary)">Lights</span>
          <span class="data-mono font-medium" style="color: var(--text-primary)">{lightsOn != null ? `${lightsOn} on` : '\u2014'}</span>
        </div>
        <div class="flex items-center gap-1.5">
          <span style="color: var(--text-tertiary)">Power</span>
          <span class="data-mono font-medium" style="color: var(--text-primary)">{powerW != null ? `${Math.round(powerW)} W` : '\u2014'}</span>
        </div>
        <div class="flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full" style={`background: ${ws && ws.connected ? 'var(--status-healthy)' : 'var(--status-error)'};`} />
          <span style="color: var(--text-tertiary)">WebSocket</span>
          <span class="font-medium" style="color: var(--text-primary)">{ws && ws.connected ? 'Connected' : ws === null ? '\u2014' : 'Disconnected'}</span>
        </div>
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
  const [feedbackHealth, setFeedbackHealth] = useState(null);
  const [fetchError, setFetchError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetchJson('/health').catch(() => null),
      fetchJson('/api/shadow/accuracy').catch(() => null),
      fetchJson('/api/pipeline').catch(() => null),
      fetchJson('/api/curation/summary').catch(() => null),
      fetchJson('/api/capabilities/feedback/health').catch(() => null),
      fetchJson('/api/activity/current').catch(() => null),
    ]).then(([hlth, s, p, c, fb, act]) => {
      setHealth(hlth);
      setShadow(s);
      setPipeline(p);
      setCuration(c);
      setFeedbackHealth(fb);
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

  const cacheData = useComputed(() => ({
    capabilities: entities.data,
    pipeline: { data: pipeline },
    shadow_accuracy: { data: shadow },
    activity_labels: activity.data,
    feedback_health: feedbackHealth,
  }), [entities.data, pipeline, shadow, activity.data, feedbackHealth]);

  if (loading && !intelligence.data) {
    return (
      <div class="space-y-6">
        <div class="t-frame" data-label="aria">
          <AriaLogo className="w-24 mb-1" color="var(--text-primary)" />
          <p class="text-sm" style="color: var(--text-tertiary); font-family: var(--font-mono);">
            Live system overview — data flow, module health, and your next steps.
          </p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (cacheError) {
    return (
      <div class="space-y-6">
        <div class="t-frame" data-label="aria">
          <AriaLogo className="w-24 mb-1" color="var(--text-primary)" />
          <p class="text-sm" style="color: var(--text-tertiary); font-family: var(--font-mono);">
            Live system overview — data flow, module health, and your next steps.
          </p>
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
        <PageBanner page="HOME" subtitle="Live system overview — data flow, module health, and your next steps." />
        <ErrorState error={fetchError} />
      </div>
    );
  }

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="HOME" subtitle="Live system overview — data flow, module health, and your next steps." />

      <HeroCard
        value={pipeline ? pipeline.current_stage : 'starting'}
        label="pipeline stage"
        delta={maturity ? `Day ${maturity.days_of_data} \u00b7 ${maturity.phase}` : null}
      />

      <JourneyProgress maturity={maturity} shadowStage={shadowStage} />

      <RightNowStrip activity={activity} intraday={intraday} />

      <BusArchitecture
        moduleStatuses={health?.modules || {}}
        cacheData={cacheData}
        feedbackHealth={feedbackHealth}
      />
    </div>
  );
}
