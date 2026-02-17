import { useState, useRef, useEffect, useMemo } from 'preact/hooks';
import { computeLayout, computeTraceback } from '../lib/sankeyLayout.js';
import { ALL_NODES, LINKS, NODE_DETAIL, getNodeMetric, ACTION_CONDITIONS } from '../lib/pipelineGraph.js';
import PipelineStepper from './PipelineStepper.jsx';

// --- Color mapping ---
const FLOW_COLORS = {
  data: 'var(--accent)',
  cache: 'var(--status-healthy)',
  feedback: 'var(--status-warning)',
};

const STATUS_COLORS = {
  healthy: 'var(--status-healthy)',
  warning: 'var(--status-warning)',
  blocked: 'var(--status-error)',
  waiting: 'var(--status-waiting)',
};

function getModuleStatus(moduleStatuses, nodeId) {
  const s = moduleStatuses?.[nodeId];
  if (s === 'running') return 'healthy';
  if (s === 'failed') return 'blocked';
  if (s === 'starting') return 'waiting';
  return 'waiting';
}

function getGroupHealth(moduleStatuses, childIds) {
  const statuses = childIds.map((id) => getModuleStatus(moduleStatuses, id));
  if (statuses.includes('blocked')) return 'blocked';
  if (statuses.includes('warning')) return 'warning';
  if (statuses.some((s) => s === 'healthy')) return 'healthy';
  return 'waiting';
}

// --- SVG Primitives ---

function SvgSparkline({ data, x, y, w, h, color }) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) =>
    `${x + (i / (data.length - 1)) * w},${y + h - ((v - min) / range) * h}`
  ).join(' ');
  return <polyline points={points} fill="none" stroke={color} stroke-width="1.5" opacity="0.7" />;
}

function getNodeSparklineData(cacheData, nodeId) {
  const sparklineMap = {
    intelligence: () => cacheData?.intelligence?.data?.intraday_trend,
    shadow_engine: () => cacheData?.shadow_accuracy?.history,
    ml_engine: () => cacheData?.ml_pipeline?.training?.r2_history,
    activity_monitor: () => cacheData?.pipeline?.event_rate_history,
  };
  const getter = sparklineMap[nodeId];
  if (!getter) return null;
  const data = getter();
  return Array.isArray(data) && data.length >= 2 ? data : null;
}

function getNodeFreshness(cacheData, nodeId) {
  const categoryMap = {
    discovery: 'capabilities',
    activity_monitor: 'activity_summary',
    presence: 'presence',
    intelligence: 'intelligence',
    ml_engine: 'ml_pipeline',
    shadow_engine: 'shadow_accuracy',
    pattern_recognition: 'patterns',
    data_quality: 'curation',
    orchestrator: 'automations',
    organic_discovery: 'capabilities',
    activity_labeler: 'activity_labels',
  };
  const cat = categoryMap[nodeId];
  if (!cat) return null;
  const ts = cacheData?.[cat]?.timestamp || cacheData?.[cat]?.updated_at;
  if (!ts) return null;
  const age = Date.now() - new Date(ts).getTime();
  const minutes = Math.floor(age / 60000);
  if (minutes < 1) return { text: '<1m', status: 'healthy' };
  if (minutes < 15) return { text: `${minutes}m`, status: 'healthy' };
  if (minutes < 60) return { text: `${minutes}m`, status: 'warning' };
  const hours = Math.floor(minutes / 60);
  return { text: `${hours}h`, status: 'blocked' };
}

function SankeyNode({ node, status, metric, onClick, highlighted, dimmed, onMouseEnter, onMouseLeave, sparklineData, freshness, pulsing }) {
  const color = STATUS_COLORS[status] || STATUS_COLORS.waiting;
  const opacity = dimmed ? 0.12 : 1;

  return (
    <g
      transform={`translate(${node.x}, ${node.y})`}
      onClick={() => onClick && onClick(node)}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      style={`cursor: ${onClick ? 'pointer' : 'default'}; opacity: ${opacity}; transition: opacity 0.3s;`}
    >
      <rect
        width={node.w}
        height={node.h}
        rx="4"
        fill="var(--bg-surface)"
        stroke={highlighted ? 'var(--accent)' : pulsing ? 'var(--accent)' : 'var(--border-primary)'}
        stroke-width={highlighted ? '2' : pulsing ? '2' : '1'}
      >
        {pulsing && (
          <animate attributeName="opacity" values="1;0.5;1" dur="0.6s" repeatCount="1" />
        )}
      </rect>
      {/* LED */}
      <circle cx="14" cy="14" r="4" fill={color}>
        {status === 'healthy' && (
          <animate attributeName="opacity" values="1;0.6;1" dur="3s" repeatCount="indefinite" />
        )}
      </circle>
      {/* Label */}
      <text x="26" y="18" fill="var(--text-primary)" font-size="11" font-weight="600" font-family="var(--font-mono)">
        {node.label}
      </text>
      {/* Freshness timestamp (expanded nodes only) */}
      {freshness && (
        <text x={node.w - 28} y="14" fill={STATUS_COLORS[freshness.status]} font-size="8" font-family="var(--font-mono)" text-anchor="end">
          {freshness.text}
        </text>
      )}
      {/* Sparkline or Metric */}
      {sparklineData ? (
        <SvgSparkline data={sparklineData} x={60} y={22} w={70} h={14} color={color} />
      ) : null}
      <text x="14" y={node.h - 10} fill="var(--text-tertiary)" font-size="10" font-family="var(--font-mono)">
        {metric}
      </text>
    </g>
  );
}

function SankeyFlow({ link, dimmed, highlighted }) {
  const color = FLOW_COLORS[link.type] || 'var(--border-primary)';
  const opacity = dimmed ? 0.05 : (0.3 + Math.min(0.4, link.value * 0.04));
  const strokeW = highlighted ? 2 : 0;
  const dashArray = link.type === 'feedback' ? '4 3' : 'none';

  return (
    <g style="transition: opacity 0.3s;">
      <path
        d={link.path}
        fill={color}
        opacity={opacity}
        stroke={highlighted ? color : 'none'}
        stroke-width={strokeW}
        stroke-dasharray={dashArray}
      />
    </g>
  );
}

function BusBar({ busBar }) {
  return (
    <g>
      <rect
        x={busBar.x + 10}
        y={busBar.y}
        width={busBar.width - 20}
        height={busBar.height}
        rx="4"
        fill="var(--bg-terminal)"
        stroke="var(--status-healthy)"
        stroke-width="1.5"
      />
      <text
        x={busBar.x + busBar.width / 2}
        y={busBar.y + 18}
        text-anchor="middle"
        fill="var(--status-healthy)"
        font-size="10"
        font-weight="700"
        font-family="var(--font-mono)"
      >
        {'HUB CACHE \u00B7 hub.db \u00B7 15 categories'}
      </text>
    </g>
  );
}

function ActionStrip({ cacheData }) {
  const action = useMemo(() => {
    for (const cond of ACTION_CONDITIONS) {
      if (cond.test(cacheData)) {
        const text = typeof cond.text === 'function' ? cond.text(cacheData) : cond.text;
        return { text, href: cond.href };
      }
    }
    return null;
  }, [cacheData]);

  if (!action) return null;

  return (
    <div class="t-frame mt-4" data-label="next action">
      {action.href ? (
        <a href={action.href} class="text-sm font-medium" style="color: var(--accent); font-family: var(--font-mono); text-decoration: none;">
          {action.text}
        </a>
      ) : (
        <span class="text-sm" style="color: var(--text-tertiary); font-family: var(--font-mono);">
          {action.text}
        </span>
      )}
    </div>
  );
}

// --- Detail Panel (hover) ---

function DetailPanel({ nodeId, svgWidth, y }) {
  if (!nodeId || !NODE_DETAIL[nodeId]) return null;
  const d = NODE_DETAIL[nodeId];
  return (
    <g>
      <rect x="20" y={y} width={svgWidth - 40} height="56" rx="4" fill="var(--bg-surface)" stroke="var(--accent)" stroke-width="1" opacity="0.95" />
      <text x="32" y={y + 14} fill="var(--accent)" font-size="10" font-weight="700" font-family="var(--font-mono)">
        {nodeId.replace(/_/g, ' ').toUpperCase()}
      </text>
      <text x="32" y={y + 28} fill="var(--text-tertiary)" font-size="8" font-family="var(--font-mono)">
        {`\u25B6 ${d.protocol}  \u2502  reads: ${d.reads}`}
      </text>
      <text x="32" y={y + 42} fill="var(--text-tertiary)" font-size="8" font-family="var(--font-mono)">
        {`\u25BC writes: ${d.writes}`}
      </text>
    </g>
  );
}

// --- Main Component ---

export default function PipelineSankey({ moduleStatuses, cacheData }) {
  const containerRef = useRef(null);
  const [width, setWidth] = useState(860);
  const [expandedColumn, setExpandedColumn] = useState(-1);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [traceTarget, setTraceTarget] = useState(null);
  const [transitioning, setTransitioning] = useState(false);
  const prevTimestampsRef = useRef({});
  const [pulsingNodes, setPulsingNodes] = useState(new Set());

  // Responsive width
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect?.width;
      if (w && w > 0) setWidth(w);
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Compute layout
  const layout = useMemo(
    () => computeLayout({ nodes: ALL_NODES, links: LINKS, width, expandedColumn }),
    [width, expandedColumn]
  );

  // Compute trace-back highlight set
  const traceSet = useMemo(
    () => (traceTarget ? computeTraceback(traceTarget, LINKS, NODE_DETAIL) : null),
    [traceTarget]
  );

  function handleNodeClick(node) {
    if (node.isGroup) {
      setTransitioning(true);
      setTimeout(() => {
        setExpandedColumn((prev) => (prev === node.column ? -1 : node.column));
        setTimeout(() => setTransitioning(false), 150);
      }, 150);
      setTraceTarget(null);
    } else if (node.column === 4) {
      // Output node — toggle trace-back
      setTraceTarget((prev) => (prev === node.id ? null : node.id));
    }
  }

  function isNodeDimmed(node) {
    if (!traceSet) return false;
    if (node.isGroup) return !node.childIds?.some((id) => traceSet.has(id));
    return !traceSet.has(node.id);
  }

  function isLinkDimmed(link) {
    if (!traceSet) return false;
    return !traceSet.has(link.source) || !traceSet.has(link.target);
  }

  // Detect cache timestamp changes and pulse affected nodes
  useEffect(() => {
    const categoryToNodes = {
      capabilities: ['discovery', 'organic_discovery'],
      activity_summary: ['activity_monitor'],
      presence: ['presence'],
      intelligence: ['intelligence'],
      ml_pipeline: ['ml_engine'],
      shadow_accuracy: ['shadow_engine'],
      patterns: ['pattern_recognition'],
      curation: ['data_quality'],
      automations: ['orchestrator'],
      activity_labels: ['activity_labeler'],
    };

    const newPulsing = new Set();
    const currentTimestamps = {};

    for (const [cat, nodeIds] of Object.entries(categoryToNodes)) {
      const ts = cacheData?.[cat]?.timestamp || cacheData?.[cat]?.updated_at || null;
      currentTimestamps[cat] = ts;
      if (ts && prevTimestampsRef.current[cat] && ts !== prevTimestampsRef.current[cat]) {
        for (const id of nodeIds) newPulsing.add(id);
      }
    }

    prevTimestampsRef.current = currentTimestamps;

    if (newPulsing.size > 0) {
      setPulsingNodes(newPulsing);
      const timer = setTimeout(() => setPulsingNodes(new Set()), 600);
      return () => clearTimeout(timer);
    }
  }, [cacheData]);

  const svgHeight = layout.svgHeight + (hoveredNode ? 66 : 0);

  // Mobile responsive switch
  if (width < 640) {
    return <PipelineStepper moduleStatuses={moduleStatuses} cacheData={cacheData} />;
  }

  return (
    <section ref={containerRef} class="t-terminal-bg rounded-lg p-4 overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${svgHeight}`}
        class="w-full"
        style="min-width: 600px; max-width: 100%; transition: height 0.2s ease;"
      >
        <defs>
          <filter id="led-glow-sankey">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Flows (behind nodes) */}
        <g style={`opacity: ${transitioning ? 0 : 1}; transition: opacity 150ms;`}>
        {layout.links.map((link) => (
          <SankeyFlow
            key={`${link.source}-${link.target}`}
            link={link}
            dimmed={isLinkDimmed(link)}
            highlighted={traceSet && !isLinkDimmed(link)}
          />
        ))}
        </g>

        {/* Trace labels on highlighted flows */}
        {traceSet && layout.links
          .filter((link) => !isLinkDimmed(link))
          .map((link) => {
            const midX = (link.x0 + link.x1) / 2;
            const midY = (link.y0 + link.y1) / 2;
            const label = getNodeMetric(cacheData, link.source);
            if (!label || label === '\u2014') return null;
            return (
              <text
                key={`trace-${link.source}-${link.target}`}
                x={midX}
                y={midY - 6}
                text-anchor="middle"
                fill="var(--accent)"
                font-size="7"
                font-family="var(--font-mono)"
                opacity="0.8"
              >
                {label}
              </text>
            );
          })}

        {/* Bus Bar */}
        <BusBar busBar={layout.busBar} />

        {/* Nodes */}
        {layout.nodes.map((node) => {
          const status = node.isGroup
            ? getGroupHealth(moduleStatuses, node.childIds || [])
            : (node.column === 0 ? 'healthy' : getModuleStatus(moduleStatuses, node.id));
          const metric = node.isGroup
            ? `${node.children?.length || 0} modules`
            : getNodeMetric(cacheData, node.id);
          const isExpanded = !node.isGroup && node.column === expandedColumn;
          const sparklineData = isExpanded ? getNodeSparklineData(cacheData, node.id) : null;
          const freshness = isExpanded ? getNodeFreshness(cacheData, node.id) : null;

          return (
            <SankeyNode
              key={node.id}
              node={node}
              status={status}
              metric={metric}
              onClick={handleNodeClick}
              highlighted={traceSet && traceSet.has(node.id)}
              dimmed={isNodeDimmed(node)}
              onMouseEnter={() => !node.isGroup && setHoveredNode(node.id)}
              onMouseLeave={() => setHoveredNode(null)}
              sparklineData={sparklineData}
              freshness={freshness}
              pulsing={pulsingNodes.has(node.id) || (node.isGroup && node.childIds?.some(id => pulsingNodes.has(id)))}
            />
          );
        })}

        {/* Detail panel on hover */}
        {hoveredNode && (
          <DetailPanel nodeId={hoveredNode} svgWidth={width} y={layout.svgHeight - 10} />
        )}
      </svg>

      {/* Color legend */}
      <div class="flex gap-4 mt-2 text-xs" style="color: var(--text-tertiary); font-family: var(--font-mono);">
        <span><span style="color: var(--accent);">{'\u25CF'}</span> data flow</span>
        <span><span style="color: var(--status-healthy);">{'\u25CF'}</span> cache read/write</span>
        <span><span style="color: var(--status-warning);">{'\u25CF'}</span> feedback loop</span>
        <span style="opacity: 0.6;">click column to expand · click output to trace</span>
      </div>

      {/* Action strip */}
      <ActionStrip cacheData={cacheData} />
    </section>
  );
}
