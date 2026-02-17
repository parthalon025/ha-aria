import { useState, useEffect, useRef } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { fetchJson } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import AriaLogo from '../components/AriaLogo.jsx';
import HeroCard from '../components/HeroCard.jsx';
import PageBanner from '../components/PageBanner.jsx';
import PresenceCard from '../components/PresenceCard.jsx';

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

const FLOW_INTAKE = [
  { id: 'engine', label: 'Engine', metricKey: 'pipeline_day' },
  { id: 'discovery', label: 'Discovery', metricKey: 'entity_count' },
  { id: 'activity_monitor', label: 'Activity', metricKey: 'event_rate' },
  { id: 'presence', label: 'Presence', metricKey: 'presence_status' },
];

const FLOW_PROCESSING = [
  { id: 'intelligence', label: 'Intelligence', metricKey: 'day_count' },
  { id: 'ml_engine', label: 'ML Engine', metricKey: 'mean_r2' },
  { id: 'shadow_engine', label: 'Shadow', metricKey: 'accuracy' },
  { id: 'pattern_recognition', label: 'Patterns', metricKey: 'sequence_count' },
];

const FLOW_ENRICHMENT = [
  { id: 'orchestrator', label: 'Orchestrator', metricKey: 'pending_count' },
  { id: 'organic_discovery', label: 'Organic Disc.', metricKey: 'organic_count' },
  { id: 'data_quality', label: 'Curation', metricKey: 'included_count' },
  { id: 'activity_labeler', label: 'Labeler', metricKey: 'current_activity' },
];

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
  switch (node.id) {
    case 'engine': return pipeline?.intelligence_day ? `Day ${pipeline.intelligence_day}` : '\u2014';
    case 'discovery': {
      const count = Object.values(caps).filter((entry) => entry && typeof entry === 'object' && entry.entities).reduce((sum, entry) => sum + (entry.entities?.length || 0), 0);
      return count ? `${count} entities` : '\u2014';
    }
    case 'activity_monitor': return pipeline?.events_per_minute ? `${pipeline.events_per_minute.toFixed(1)} ev/m` : '\u2014';
    case 'presence': return '\u2014';
    case 'intelligence': return pipeline?.intelligence_day ? `Day ${pipeline.intelligence_day}` : '\u2014';
    case 'ml_engine': {
      const mlCaps = Object.values(caps).filter((entry) => entry?.ml_accuracy);
      if (mlCaps.length === 0) return '\u2014';
      const avgR2 = mlCaps.reduce((s, entry) => s + (entry.ml_accuracy.mean_r2 || 0), 0) / mlCaps.length;
      return `R\u00B2: ${avgR2.toFixed(2)}`;
    }
    case 'shadow_engine': return shadow?.overall_accuracy ? `${(shadow.overall_accuracy * 100).toFixed(0)}%` : '\u2014';
    case 'pattern_recognition': return '\u2014';
    case 'orchestrator': return '\u2014';
    case 'organic_discovery': {
      const organic = Object.values(caps).filter((entry) => entry?.source === 'organic').length;
      return organic ? `${organic} organic` : '\u2014';
    }
    case 'data_quality': return pipeline?.included_entities ? `${pipeline.included_entities} incl.` : '\u2014';
    case 'activity_labeler': {
      const curr = activity?.current_activity;
      return curr?.predicted || '\u2014';
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


function BusArchitecture({ moduleStatuses, cacheData }) {
  const nodeX = [30, 235, 440, 645];
  const nodeW = 180;

  // --- Activity Labeler pulse animation (kept from original) ---
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

  function renderRow(nodes, yOffset) {
    return nodes.map((node, idx) => {
      const isLabeler = node.id === 'activity_labeler';
      return (
        <ModuleNode
          key={node.id}
          x={nodeX[idx]}
          y={yOffset}
          status={node.id === 'engine' ? 'healthy' : getNodeStatus(moduleStatuses, node.id)}
          label={node.label}
          metric={getNodeMetric(cacheData, node)}
          glowing={isLabeler && labelerGlowing}
          animateMetric={isLabeler && labelerAnimateMetric}
        />
      );
    });
  }

  // Arrow helper — downward arrow with optional label
  function Arrow({ x, y, label, length, labelSide }) {
    const len = length || 18;
    const side = labelSide || 'right';
    const lx = side === 'left' ? x - 6 : x + 6;
    const anchor = side === 'left' ? 'end' : 'start';
    return (
      <g>
        <line x1={x} y1={y} x2={x} y2={y + len} stroke="var(--border-primary)" stroke-width="1" />
        <polygon points={`${x - 3},${y + len - 4} ${x + 3},${y + len - 4} ${x},${y + len}`} fill="var(--border-primary)" />
        {label && <text x={lx} y={y + len / 2 + 3} text-anchor={anchor} fill="var(--text-tertiary)" font-size="7" font-family="var(--font-mono)">{label}</text>}
      </g>
    );
  }

  // Labeled connection line between two points (no arrowhead, dashed)
  function DataLine({ x1, y1, x2, y2, label, color }) {
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    return (
      <g>
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color || 'var(--border-primary)'} stroke-width="0.7" stroke-dasharray="3 2" opacity="0.6" />
        {label && <text x={mx} y={my - 3} text-anchor="middle" fill={color || 'var(--text-tertiary)'} font-size="6.5" font-family="var(--font-mono)">{label}</text>}
      </g>
    );
  }

  // Banner bar helper
  function Banner({ y, label, sublabel, color }) {
    return (
      <g>
        <rect x="20" y={y} width="820" height="28" rx="4" fill="var(--bg-inset)" stroke={color || 'var(--accent)'} stroke-width="1.5" />
        <text x="430" y={y + 13} text-anchor="middle" fill={color || 'var(--accent)'} font-size="11" font-weight="700" font-family="var(--font-mono)">{label}</text>
        {sublabel && <text x="430" y={y + 24} text-anchor="middle" fill="var(--text-tertiary)" font-size="8" font-family="var(--font-mono)">{sublabel}</text>}
      </g>
    );
  }

  // Section label helper
  function SectionLabel({ y, label }) {
    return <text x="430" y={y} text-anchor="middle" fill="var(--text-tertiary)" font-size="9" font-weight="700" font-family="var(--font-mono)" letter-spacing="2">{label}</text>;
  }

  // Center x for each node column
  const cx = nodeX.map((x) => x + nodeW / 2);
  // Bottom y of a node at yOffset
  const nodeBot = (yOff) => yOff + 55;

  // Vertical tracer path down center
  const tracerPath = 'M430,28 L430,70 L430,290 L430,380 L430,500 L430,580 L430,640';

  return (
    <section class="t-terminal-bg rounded-lg p-4 overflow-x-auto">
      <svg viewBox="0 0 860 670" class="w-full" style="min-width: 700px; max-width: 100%;">
        <defs>
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

        {/* === Layer 1: HOME ASSISTANT === */}
        <Banner y={0} label="HOME ASSISTANT" sublabel="REST /api/states \u00B7 WebSocket state_changed \u00B7 MQTT frigate/events" />

        {/* HA → Intake: exact protocol per module */}
        <Arrow x={cx[0]} y={28} label="REST /api/states" length={30} />
        <Arrow x={cx[1]} y={28} label="REST + WS registries" length={30} />
        <Arrow x={cx[2]} y={28} label="WS state_changed" length={30} />
        <Arrow x={cx[3]} y={28} label="MQTT + WS sensors" length={30} />

        {/* === Layer 2: INTAKE === */}
        <SectionLabel y={70} label="INTAKE" />
        {renderRow(FLOW_INTAKE, 76)}

        {/* Engine Pipeline detail box */}
        <g transform="translate(30, 140)">
          <rect width="800" height="55" rx="4" fill="none" stroke="var(--border-primary)" stroke-width="1" stroke-dasharray="4 2" />
          <text x="12" y="14" fill="var(--text-tertiary)" font-size="8" font-weight="600" font-family="var(--font-mono)">ENGINE PIPELINE  ~/ha-logs/intelligence/</text>
          <text x="12" y="28" fill="var(--text-tertiary)" font-size="7.5" font-family="var(--font-mono)">
            {'intraday/HH00.json \u2192 daily/YYYY-MM-DD.json \u2192 baselines.json \u2192 ml_models/*.pkl'}
          </text>
          <text x="12" y="42" fill="var(--text-tertiary)" font-size="7.5" font-family="var(--font-mono)">
            {'\u2192 predictions.json \u2192 correlations.json \u2192 entity_correlations.json \u2192 sequence_anomalies.json'}
          </text>
        </g>

        {/* Intake → Hub Cache: exact cache categories written */}
        <Arrow x={cx[0]} y={198} label="intelligence (JSON)" length={28} />
        <Arrow x={cx[1]} y={198} label="entities, devices, areas, capabilities" length={28} />
        <Arrow x={cx[2]} y={198} label="activity_log, activity_summary" length={28} />
        <Arrow x={cx[3]} y={198} label="presence (Bayesian)" length={28} />

        {/* === Layer 3: HUB CACHE === */}
        <Banner y={230} label="HUB CACHE" sublabel="SQLite hub.db \u00B7 15 cache categories \u00B7 WebSocket push on update" color="var(--status-healthy)" />

        {/* Cache → Processing: what each module reads */}
        <Arrow x={cx[0]} y={258} label="engine JSON + activity" length={28} />
        <Arrow x={cx[1]} y={258} label="capabilities + activity_log" length={28} />
        <Arrow x={cx[2]} y={258} label="state_changed events" length={28} />
        <Arrow x={cx[3]} y={258} label="logbook + snapshots" length={28} />

        {/* === Layer 4: PROCESSING === */}
        <SectionLabel y={296} label="PROCESSING" />
        {renderRow(FLOW_PROCESSING, 302)}

        {/* Processing writes back to cache — show specific outputs */}
        <g transform="translate(0, 362)">
          {/* Intelligence writes */}
          <text x={cx[0]} y={0} text-anchor="middle" fill="var(--text-tertiary)" font-size="6.5" font-family="var(--font-mono)">{'\u2193 intelligence cache'}</text>
          {/* ML Engine writes */}
          <text x={cx[1]} y={0} text-anchor="middle" fill="var(--text-tertiary)" font-size="6.5" font-family="var(--font-mono)">{'\u2193 ml_predictions, capabilities'}</text>
          {/* Shadow writes */}
          <text x={cx[2]} y={0} text-anchor="middle" fill="var(--text-tertiary)" font-size="6.5" font-family="var(--font-mono)">{'\u2193 predictions tbl, pipeline_state'}</text>
          {/* Patterns writes */}
          <text x={cx[3]} y={0} text-anchor="middle" fill="var(--text-tertiary)" font-size="6.5" font-family="var(--font-mono)">{'\u2193 patterns cache'}</text>
        </g>

        {/* Feedback loops: ML→capabilities, Shadow→capabilities */}
        <DataLine x1={cx[1]} y1={370} x2={cx[1] - 60} y2={380} label="ml_accuracy \u2192 capabilities" color="var(--status-warning)" />
        <DataLine x1={cx[2]} y1={370} x2={cx[2] - 60} y2={380} label="hit_rate \u2192 capabilities" color="var(--status-warning)" />

        {/* === Layer 5: ENRICHMENT === */}
        <SectionLabel y={396} label="ENRICHMENT" />
        {renderRow(FLOW_ENRICHMENT, 402)}

        {/* Enrichment writes */}
        <g transform="translate(0, 462)">
          <text x={cx[0]} y={0} text-anchor="middle" fill="var(--text-tertiary)" font-size="6.5" font-family="var(--font-mono)">{'automation_suggestions'}</text>
          <text x={cx[1]} y={0} text-anchor="middle" fill="var(--text-tertiary)" font-size="6.5" font-family="var(--font-mono)">{'capabilities (organic)'}</text>
          <text x={cx[2]} y={0} text-anchor="middle" fill="var(--text-tertiary)" font-size="6.5" font-family="var(--font-mono)">{'entity_curation tbl'}</text>
          <text x={cx[3]} y={0} text-anchor="middle" fill="var(--text-tertiary)" font-size="6.5" font-family="var(--font-mono)">{'activity_labels'}</text>
        </g>

        {/* Cross-module reads shown as dashed lines */}
        {/* Orchestrator reads patterns */}
        <DataLine x1={cx[3]} y1={nodeBot(302)} x2={cx[0]} y2={402} label="reads patterns" />
        {/* Organic discovery reads entities + activity_log */}
        <DataLine x1={cx[1] - 40} y1={nodeBot(302) + 20} x2={cx[1]} y2={402} label="reads entities + activity" />

        {/* Arrows to YOU */}
        <Arrow x={430} y={475} length={22} />

        {/* === Layer 6: YOU === */}
        <g transform="translate(200, 502)">
          <rect x="0" y="0" width="460" height="35" rx="4" fill="var(--bg-inset)" stroke="var(--accent)" stroke-width="2" />
          <text x="230" y="14" text-anchor="middle" fill="var(--accent)" font-size="11" font-weight="bold" font-family="var(--font-mono)">
            {'YOU:  Label \u00B7 Curate \u00B7 Review \u00B7 Advance'}
          </text>
          <text x="230" y="28" text-anchor="middle" fill="var(--text-tertiary)" font-size="7.5" font-family="var(--font-mono)">
            {'corrections \u2192 activity_labels  |  curation rules \u2192 entity_curation  |  approve \u2192 automation_suggestions'}
          </text>
        </g>

        {/* Arrow to Dashboard */}
        <Arrow x={430} y={537} length={22} />

        {/* === Layer 7: DASHBOARD === */}
        <Banner y={563} label="DASHBOARD" sublabel="13 pages \u00B7 WebSocket push \u00B7 reads all cache categories via /api/cache/*" />

        {/* Flowing dot tracer along center vertical */}
        <circle r="3" fill="var(--accent)" opacity="0.7" filter="url(#tracer-glow)">
          <animateMotion dur="8s" repeatCount="indefinite" path={tracerPath} />
        </circle>
        <circle r="5" fill="var(--accent)" opacity="0.2" filter="url(#tracer-glow)">
          <animateMotion dur="8s" repeatCount="indefinite" path={tracerPath} begin="0.4s" />
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
  const [fetchError, setFetchError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetchJson('/health').catch(() => null),
      fetchJson('/api/shadow/accuracy').catch(() => null),
      fetchJson('/api/pipeline').catch(() => null),
      fetchJson('/api/curation/summary').catch(() => null),
      fetchJson('/api/activity/current').catch(() => null),
    ]).then(([hlth, s, p, c, act]) => {
      setHealth(hlth);
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

  const cacheData = useComputed(() => ({
    capabilities: entities.data,
    pipeline: { data: pipeline },
    shadow_accuracy: { data: shadow },
    activity_labels: activity.data,
  }), [entities.data, pipeline, shadow, activity.data]);

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

      <PresenceCard />

      <BusArchitecture
        moduleStatuses={health?.modules || {}}
        cacheData={cacheData}
      />
    </div>
  );
}
