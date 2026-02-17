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

// Progressive disclosure: detail shown on hover (Shneiderman 1996)
const NODE_DETAIL = {
  engine: {
    protocol: 'REST /api/states (systemd timers)',
    reads: 'All entity states, logbook, calendar, weather',
    writes: '~/ha-logs/intelligence/*.json (predictions, baselines, correlations, anomalies)',
  },
  discovery: {
    protocol: 'REST registries + WS registry_updated',
    reads: 'Entity/device/area registries from HA',
    writes: 'entities, devices, areas, capabilities (seed)',
  },
  activity_monitor: {
    protocol: 'WS state_changed (real-time)',
    reads: 'All state_changed events, REST /api/states (startup seed)',
    writes: 'activity_log (15-min windows), activity_summary (live)',
  },
  presence: {
    protocol: 'MQTT frigate/events + WS state_changed',
    reads: 'Camera person/face detections, motion/light/dimmer sensors',
    writes: 'presence (Bayesian per-room probability)',
  },
  intelligence: {
    protocol: 'Reads engine JSON files from disk',
    reads: 'Engine outputs + activity_log + activity_summary',
    writes: 'intelligence (consolidated: maturity, predictions, trends, drift)',
  },
  ml_engine: {
    protocol: 'Trains from daily snapshot files',
    reads: 'capabilities, activity_log, daily/*.json snapshots',
    writes: 'ml_predictions, ml_training_metadata, capabilities (ml_accuracy feedback)',
  },
  shadow_engine: {
    protocol: 'Subscribes to hub state_changed events',
    reads: 'Entity state changes, activity context',
    writes: 'predictions table, pipeline_state, capabilities (hit_rate feedback)',
  },
  pattern_recognition: {
    protocol: 'Reads logbook JSON files',
    reads: '~/ha-logs/logbook/*.json, intraday snapshots',
    writes: 'patterns (temporal sequences per area)',
  },
  orchestrator: {
    protocol: 'Can call HA /api/automation/trigger',
    reads: 'patterns cache',
    writes: 'automation_suggestions, pending_automations',
  },
  organic_discovery: {
    protocol: 'Weekly timer + Ollama queue (LLM naming)',
    reads: 'entities, devices, areas, activity_log, discovery_history',
    writes: 'capabilities (organic), discovery_settings, discovery_history',
  },
  data_quality: {
    protocol: 'Classifies entities into tiers 1/2/3',
    reads: 'entities, activity_log (event rates)',
    writes: 'entity_curation table (included/excluded/tier)',
  },
  activity_labeler: {
    protocol: 'Ollama queue for LLM fallback',
    reads: 'Sensor context: power, lights, motion, occupancy, time',
    writes: 'activity_labels (predictions + corrections + classifier)',
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

function ModuleNode({ x, y, status, label, metric, glowing, animateMetric, nodeId, onHover }) {
  const colors = {
    healthy: 'var(--status-healthy)',
    waiting: 'var(--status-waiting)',
    blocked: 'var(--status-error)',
    review: 'var(--status-warning)',
  };
  const color = colors[status] || colors.waiting;

  return (
    <g
      transform={`translate(${x}, ${y})`}
      onMouseEnter={() => onHover && onHover(nodeId)}
      onMouseLeave={() => onHover && onHover(null)}
      style="cursor: default;"
    >
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
  // Layout constants — generous spacing (Gestalt proximity principle)
  const nodeX = [30, 235, 440, 645];
  const nodeW = 180;
  const cx = nodeX.map((x) => x + nodeW / 2);

  // --- Hover state for progressive disclosure (Shneiderman 1996) ---
  const [hoveredNode, setHoveredNode] = useState(null);

  // --- Activity Labeler pulse animation ---
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
          nodeId={node.id}
          onHover={setHoveredNode}
        />
      );
    });
  }

  // --- Primitive helpers ---

  // Downward arrow with color-coded stroke (preattentive: Treisman 1985)
  // type: 'external' (accent), 'cache' (green), 'internal' (default gray)
  function Arrow({ x, y, label, length, type }) {
    const len = length || 20;
    const colors = { external: 'var(--accent)', cache: 'var(--status-healthy)', feedback: 'var(--status-warning)' };
    const stroke = colors[type] || 'var(--border-primary)';
    return (
      <g>
        <line x1={x} y1={y} x2={x} y2={y + len} stroke={stroke} stroke-width="1" />
        <polygon points={`${x - 3},${y + len - 4} ${x + 3},${y + len - 4} ${x},${y + len}`} fill={stroke} />
        {label && <text x={x + 6} y={y + len / 2 + 3} fill="var(--text-tertiary)" font-size="8" font-family="var(--font-mono)">{label}</text>}
      </g>
    );
  }

  // Swim lane background (Gestalt enclosure — strongest grouping cue)
  function SwimLane({ y, height, label, color }) {
    return (
      <g>
        <rect x="15" y={y} width="830" height={height} rx="6" fill={color || 'var(--bg-surface)'} opacity="0.15" stroke="var(--border-primary)" stroke-width="0.5" />
        <text x="28" y={y + 14} fill="var(--text-tertiary)" font-size="9" font-weight="700" font-family="var(--font-mono)" letter-spacing="2" opacity="0.7">{label}</text>
      </g>
    );
  }

  // Banner divider bar
  function Banner({ y, label, sublabel, color }) {
    return (
      <g>
        <rect x="20" y={y} width="820" height={sublabel ? 28 : 22} rx="4" fill="var(--bg-inset)" stroke={color || 'var(--accent)'} stroke-width="1.5" />
        <text x="430" y={y + (sublabel ? 13 : 15)} text-anchor="middle" fill={color || 'var(--accent)'} font-size="11" font-weight="700" font-family="var(--font-mono)">{label}</text>
        {sublabel && <text x="430" y={y + 24} text-anchor="middle" fill="var(--text-tertiary)" font-size="8" font-family="var(--font-mono)">{sublabel}</text>}
      </g>
    );
  }

  // Feedback arc — curved dashed line returning upward (amber, visually distinct)
  function FeedbackArc({ x, y1, y2, label }) {
    const curveX = x + 95;
    const path = `M${x},${y1} Q${curveX},${(y1 + y2) / 2} ${x},${y2}`;
    return (
      <g>
        <path d={path} fill="none" stroke="var(--status-warning)" stroke-width="1" stroke-dasharray="4 3" opacity="0.7" />
        <polygon points={`${x - 3},${y2 + 3} ${x + 3},${y2 + 3} ${x},${y2}`} fill="var(--status-warning)" opacity="0.7" />
        {label && <text x={curveX - 8} y={(y1 + y2) / 2 + 3} fill="var(--status-warning)" font-size="7.5" font-family="var(--font-mono)" text-anchor="end">{label}</text>}
      </g>
    );
  }

  // --- Hover detail panel (progressive disclosure) ---
  function DetailPanel() {
    if (!hoveredNode || !NODE_DETAIL[hoveredNode]) return null;
    const d = NODE_DETAIL[hoveredNode];
    const panelY = 542;
    return (
      <g>
        <rect x="20" y={panelY} width="820" height="56" rx="4" fill="var(--bg-surface)" stroke="var(--accent)" stroke-width="1" opacity="0.95" />
        <text x="32" y={panelY + 14} fill="var(--accent)" font-size="10" font-weight="700" font-family="var(--font-mono)">{hoveredNode.replace(/_/g, ' ').toUpperCase()}</text>
        <text x="32" y={panelY + 28} fill="var(--text-tertiary)" font-size="8" font-family="var(--font-mono)">
          {`\u25B6 ${d.protocol}    \u2502  reads: ${d.reads}`}
        </text>
        <text x="32" y={panelY + 42} fill="var(--text-tertiary)" font-size="8" font-family="var(--font-mono)">
          {`\u25BC writes: ${d.writes}`}
        </text>
      </g>
    );
  }

  // Tracer path — follows center vertical through diagram
  const tracerPath = 'M430,28 L430,65 L430,258 L430,345 L430,445 L430,510 L430,540';

  // --- Y coordinates (consistent 70px between layers, uniform spacing) ---
  const yHA = 0;
  const yIntake = 52;           // swim lane top
  const yIntakeNodes = 70;      // nodes inside lane
  const yIntakeBot = 140;       // swim lane bottom
  const yCache = 152;           // hub cache banner
  const yProc = 192;            // swim lane top
  const yProcNodes = 210;       // nodes inside lane
  const yProcBot = 282;         // swim lane bottom
  const yEnrich = 294;          // swim lane top
  const yEnrichNodes = 312;     // nodes inside lane
  const yEnrichBot = 382;       // swim lane bottom
  const yYou = 398;             // YOU banner
  const yDash = 452;            // dashboard banner
  const svgHeight = hoveredNode ? 605 : 490;

  return (
    <section class="t-terminal-bg rounded-lg p-4 overflow-x-auto">
      <svg viewBox={`0 0 860 ${svgHeight}`} class="w-full" style="min-width: 700px; max-width: 100%; transition: height 0.2s ease;">
        <defs>
          <filter id="led-glow">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="tracer-glow">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* ═══ HOME ASSISTANT ═══ */}
        <Banner y={yHA} label="HOME ASSISTANT" sublabel="REST \u00B7 WebSocket \u00B7 MQTT" />

        {/* HA → Intake arrows — color-coded by protocol (preattentive processing) */}
        <Arrow x={cx[0]} y={28} label="REST /api/states" length={24} type="external" />
        <Arrow x={cx[1]} y={28} label="REST + WS registries" length={24} type="external" />
        <Arrow x={cx[2]} y={28} label="WS state_changed" length={24} type="external" />
        <Arrow x={cx[3]} y={28} label="MQTT + WS" length={24} type="external" />

        {/* ═══ INTAKE swim lane ═══ (Gestalt enclosure) */}
        <SwimLane y={yIntake} height={yIntakeBot - yIntake} label="INTAKE" color="var(--accent)" />
        {renderRow(FLOW_INTAKE, yIntakeNodes)}

        {/* Intake → Cache arrows — show what each writes */}
        <Arrow x={cx[0]} y={yIntakeBot} label="JSON files" length={12} type="cache" />
        <Arrow x={cx[1]} y={yIntakeBot} label="entities, caps" length={12} type="cache" />
        <Arrow x={cx[2]} y={yIntakeBot} label="activity_log" length={12} type="cache" />
        <Arrow x={cx[3]} y={yIntakeBot} label="presence" length={12} type="cache" />

        {/* ═══ HUB CACHE ═══ */}
        <Banner y={yCache} label="HUB CACHE" sublabel="SQLite hub.db \u00B7 15 categories \u00B7 WebSocket push" color="var(--status-healthy)" />

        {/* Cache → Processing arrows — show what each reads */}
        <Arrow x={cx[0]} y={yCache + 28} label="engine JSON" length={12} />
        <Arrow x={cx[1]} y={yCache + 28} label="caps + activity" length={12} />
        <Arrow x={cx[2]} y={yCache + 28} label="state events" length={12} />
        <Arrow x={cx[3]} y={yCache + 28} label="logbook" length={12} />

        {/* ═══ PROCESSING swim lane ═══ */}
        <SwimLane y={yProc} height={yProcBot - yProc} label="PROCESSING" color="var(--status-healthy)" />
        {renderRow(FLOW_PROCESSING, yProcNodes)}

        {/* Feedback arcs — ML and Shadow write back to capabilities (amber, distinct) */}
        <FeedbackArc x={cx[1] + 90} y1={yProcNodes + 30} y2={yCache + 14} label="ml_accuracy" />
        <FeedbackArc x={cx[2] + 90} y1={yProcNodes + 30} y2={yCache + 14} label="hit_rate" />

        {/* Processing → Enrichment arrows */}
        <Arrow x={cx[0]} y={yProcBot} label="intelligence" length={12} type="cache" />
        <Arrow x={cx[1]} y={yProcBot} label="ml_predictions" length={12} type="cache" />
        <Arrow x={cx[2]} y={yProcBot} label="pipeline_state" length={12} type="cache" />
        <Arrow x={cx[3]} y={yProcBot} label="patterns" length={12} type="cache" />

        {/* ═══ ENRICHMENT swim lane ═══ */}
        <SwimLane y={yEnrich} height={yEnrichBot - yEnrich} label="ENRICHMENT" color="var(--status-warning)" />
        {renderRow(FLOW_ENRICHMENT, yEnrichNodes)}

        {/* Enrichment output labels (short, 1-2 words — Tufte data-ink ratio) */}
        <text x={cx[0]} y={yEnrichBot - 4} text-anchor="middle" fill="var(--text-tertiary)" font-size="7.5" font-family="var(--font-mono)" opacity="0.8">{'\u2193 suggestions'}</text>
        <text x={cx[1]} y={yEnrichBot - 4} text-anchor="middle" fill="var(--text-tertiary)" font-size="7.5" font-family="var(--font-mono)" opacity="0.8">{'\u2193 organic caps'}</text>
        <text x={cx[2]} y={yEnrichBot - 4} text-anchor="middle" fill="var(--text-tertiary)" font-size="7.5" font-family="var(--font-mono)" opacity="0.8">{'\u2193 curation tbl'}</text>
        <text x={cx[3]} y={yEnrichBot - 4} text-anchor="middle" fill="var(--text-tertiary)" font-size="7.5" font-family="var(--font-mono)" opacity="0.8">{'\u2193 labels'}</text>

        {/* ═══ YOU ═══ */}
        <Arrow x={430} y={yEnrichBot} length={16} />
        <g transform={`translate(220, ${yYou})`}>
          <rect x="0" y="0" width="420" height="28" rx="4" fill="var(--bg-inset)" stroke="var(--accent)" stroke-width="2" />
          <text x="210" y="18" text-anchor="middle" fill="var(--accent)" font-size="11" font-weight="bold" font-family="var(--font-mono)">
            {'YOU:  Label \u00B7 Curate \u00B7 Review \u00B7 Advance'}
          </text>
        </g>

        {/* ═══ DASHBOARD ═══ */}
        <Arrow x={430} y={yYou + 28} length={16} />
        <Banner y={yDash} label="DASHBOARD" sublabel="13 pages \u00B7 WebSocket push" />

        {/* Flowing tracer dot — single animation, restrained (Tufte: minimize decoration) */}
        <circle r="3" fill="var(--accent)" opacity="0.6" filter="url(#tracer-glow)">
          <animateMotion dur="8s" repeatCount="indefinite" path={tracerPath} />
        </circle>

        {/* Hover detail panel — appears at bottom when any module is hovered */}
        <DetailPanel />
      </svg>

      {/* Color legend — below SVG, not inside it (reduces SVG clutter) */}
      <div class="flex gap-4 mt-2 text-xs" style="color: var(--text-tertiary); font-family: var(--font-mono);">
        <span><span style="color: var(--accent);">{'\u25CF'}</span> external API</span>
        <span><span style="color: var(--status-healthy);">{'\u25CF'}</span> cache write</span>
        <span><span style="color: var(--status-warning);">{'\u25CF'}</span> feedback loop</span>
        <span style="opacity: 0.6;">hover any module for detail</span>
      </div>
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
