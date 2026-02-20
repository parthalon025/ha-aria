# Pipeline Sankey Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the BusArchitecture swim-lane diagram on the ARIA Home page with a bus-bar Sankey flow visualization that shows the full data pipeline from HA sources to automation suggestions, with progressive disclosure and trace-back debugging.

**Architecture:** Custom SVG Sankey layout engine (no library) renders 45 nodes in 5 collapsible column groups connected by cubic Bezier ribbon flows through a central horizontal cache bus bar. Three interaction layers: glance (collapsed columns), expand (individual nodes with sparklines), and trace-back (critical path highlighting). Mobile falls back to a vertical pipeline stepper.

**Tech Stack:** Preact + JSX, SVG, existing CSS custom properties, existing `useCache` hook + `fetchJson`, existing `TimeChart` compact mode for sparklines.

**Design doc:** `docs/plans/2026-02-17-pipeline-sankey-design.md`

---

## Task 1: Sankey Layout Engine

**Files:**
- Create: `aria/dashboard/spa/src/lib/sankeyLayout.js`
- Test: Manual — visual inspection after Task 3

This is a pure data → coordinates function. No JSX, no DOM. Takes node/link definitions + container dimensions, returns positioned nodes and Bezier path data.

**Step 1: Create the layout module with node positioning**

```js
// aria/dashboard/spa/src/lib/sankeyLayout.js

/**
 * Custom Sankey layout engine for ARIA pipeline diagram.
 * No dependencies. Takes graph definition, returns SVG coordinates.
 *
 * Layout: 5 columns of nodes above/below a horizontal bus bar.
 * Flows route through the bus bar (down into it, up out of it).
 */

// --- Constants ---
const BUS_BAR_HEIGHT = 28;
const NODE_WIDTH = 140;
const NODE_HEIGHT_COLLAPSED = 60;
const NODE_HEIGHT_EXPANDED = 40;
const NODE_PAD_Y = 8;
const COLUMN_PAD_X = 24;

/**
 * Compute layout for the Sankey diagram.
 *
 * @param {Object} params
 * @param {Array<{id, column, label, metricKey}>} params.nodes - All 45 nodes
 * @param {Array<{source, target, value, type}>} params.links - Flows between nodes
 * @param {number} params.width - Available SVG width
 * @param {number} params.expandedColumn - Index of expanded column (0-4), or -1
 * @returns {{ nodes: Array, links: Array, busBar: Object, svgHeight: number }}
 */
export function computeLayout({ nodes, links, width, expandedColumn }) {
  const columns = [[], [], [], [], []];
  nodes.forEach((n) => columns[n.column].push(n));

  // --- X positions: distribute 5 columns evenly ---
  const usableWidth = width - NODE_WIDTH;
  const colSpacing = usableWidth / 4;
  const colX = columns.map((_, i) => Math.round(i * colSpacing));

  // --- Y positions: nodes above bus bar, stacked per column ---
  const busBarY = computeBusBarY(columns, expandedColumn);

  const positioned = [];
  columns.forEach((col, ci) => {
    const isExpanded = ci === expandedColumn;
    const nodeH = isExpanded ? NODE_HEIGHT_EXPANDED : NODE_HEIGHT_COLLAPSED;
    // Collapsed: single group node. Expanded: all individual nodes.
    const visibleNodes = isExpanded ? col : [groupNode(col, ci)];
    let y = 40; // below PageBanner area
    visibleNodes.forEach((node) => {
      positioned.push({
        ...node,
        x: colX[ci],
        y,
        w: isExpanded ? NODE_WIDTH : NODE_WIDTH + 40,
        h: nodeH,
        cx: colX[ci] + (isExpanded ? NODE_WIDTH : NODE_WIDTH + 40) / 2,
        cy: y + nodeH / 2,
        bottomY: y + nodeH,
        isGroup: !isExpanded,
      });
      y += nodeH + NODE_PAD_Y;
    });
  });

  // --- Bus bar ---
  const busBar = { x: 0, y: busBarY, width, height: BUS_BAR_HEIGHT };

  // --- Link paths: route through bus bar ---
  const positionedLinks = links.map((link) => {
    const src = positioned.find((n) => n.id === link.source || (n.isGroup && n.childIds?.includes(link.source)));
    const tgt = positioned.find((n) => n.id === link.target || (n.isGroup && n.childIds?.includes(link.target)));
    if (!src || !tgt) return null;
    return {
      ...link,
      path: computeLinkPath(src, tgt, busBar, link.value),
      sourceNode: src,
      targetNode: tgt,
    };
  }).filter(Boolean);

  const svgHeight = busBarY + BUS_BAR_HEIGHT + 80; // room for action strip

  return { nodes: positioned, links: positionedLinks, busBar, svgHeight };
}

function computeBusBarY(columns, expandedColumn) {
  let maxNodesInCol = 1;
  columns.forEach((col, ci) => {
    if (ci === expandedColumn) {
      maxNodesInCol = Math.max(maxNodesInCol, col.length);
    }
  });
  return 40 + maxNodesInCol * (NODE_HEIGHT_EXPANDED + NODE_PAD_Y) + 20;
}

function groupNode(col, columnIndex) {
  const COLUMN_LABELS = ['SOURCES', 'INTAKE', 'PROCESSING', 'ENRICHMENT', 'OUTPUTS'];
  return {
    id: `group_${columnIndex}`,
    column: columnIndex,
    label: COLUMN_LABELS[columnIndex],
    isGroup: true,
    childIds: col.map((n) => n.id),
    children: col,
  };
}

/**
 * Compute SVG path for a Sankey link that routes through the bus bar.
 * Returns an SVG path string for a filled ribbon (two Bezier curves).
 */
function computeLinkPath(src, tgt, busBar, value) {
  const halfW = Math.max(2, Math.min(20, value * 0.5));
  const x0 = src.x + src.w;
  const y0 = src.cy;
  const x1 = tgt.x;
  const y1 = tgt.cy;

  // If source and target are in the same half (both above bar), direct Bezier.
  // Otherwise route through bus bar.
  const midX = (x0 + x1) / 2;

  // Simple direct curve (both above bus bar)
  return {
    d: [
      `M ${x0},${y0 - halfW}`,
      `C ${midX},${y0 - halfW} ${midX},${y1 - halfW} ${x1},${y1 - halfW}`,
      `L ${x1},${y1 + halfW}`,
      `C ${midX},${y1 + halfW} ${midX},${y0 + halfW} ${x0},${y0 + halfW}`,
      'Z',
    ].join(' '),
    halfW,
  };
}

/**
 * Compute the critical path trace from an output node back to sources.
 *
 * @param {string} outputNodeId - The output node to trace from
 * @param {Array} links - All links
 * @param {Object} nodeDetail - NODE_DETAIL map (reads/writes per module)
 * @returns {Set<string>} Set of node IDs on the critical path
 */
export function computeTraceback(outputNodeId, links, nodeDetail) {
  const ancestors = new Set([outputNodeId]);
  const queue = [outputNodeId];
  while (queue.length > 0) {
    const current = queue.shift();
    links.forEach((link) => {
      if (link.target === current && !ancestors.has(link.source)) {
        ancestors.add(link.source);
        queue.push(link.source);
      }
    });
  }
  return ancestors;
}
```

**Step 2: Verify the module is importable**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && node -e "import('./src/lib/sankeyLayout.js').then(m => console.log('OK:', Object.keys(m)))"`
Expected: `OK: [ 'computeLayout', 'computeTraceback' ]`

**Step 3: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/lib/sankeyLayout.js
git commit -m "feat(dashboard): add custom Sankey layout engine"
```

---

## Task 2: Pipeline Data Model

**Files:**
- Create: `aria/dashboard/spa/src/lib/pipelineGraph.js`

This defines the 45 nodes, their columns, and all links between them — the data that feeds the layout engine. Also contains the NODE_DETAIL map (migrated from Home.jsx) and the action strip conditions.

**Step 1: Create the pipeline graph definition**

```js
// aria/dashboard/spa/src/lib/pipelineGraph.js

/**
 * ARIA pipeline graph definition — 45 nodes, 5 columns, all links.
 * This is the single source of truth for the Sankey topology.
 * When modules are added/removed, update HERE (not the JSX).
 */

// --- Column 0: External Data Sources ---
export const SOURCES = [
  { id: 'rest_states', column: 0, label: 'REST /api/states', metricKey: 'entity_count' },
  { id: 'rest_registries', column: 0, label: 'REST Registries', metricKey: null },
  { id: 'ws_state_changed', column: 0, label: 'WS state_changed', metricKey: 'event_rate' },
  { id: 'ws_registry', column: 0, label: 'WS registry_updated', metricKey: null },
  { id: 'mqtt_frigate', column: 0, label: 'MQTT frigate/events', metricKey: 'mqtt_status' },
  { id: 'logbook_json', column: 0, label: 'Logbook JSON', metricKey: null },
  { id: 'snapshot_json', column: 0, label: 'Snapshot JSON', metricKey: 'snapshot_count' },
  { id: 'ollama_queue', column: 0, label: 'Ollama Queue', metricKey: null },
];

// --- Column 1: Intake Modules ---
export const INTAKE = [
  { id: 'discovery', column: 1, label: 'Discovery', metricKey: 'entity_count' },
  { id: 'activity_monitor', column: 1, label: 'Activity Monitor', metricKey: 'event_rate' },
  { id: 'presence', column: 1, label: 'Presence', metricKey: 'presence_status' },
  { id: 'engine', column: 1, label: 'Engine (Batch)', metricKey: 'pipeline_day' },
];

// --- Column 2: Processing Modules ---
export const PROCESSING = [
  { id: 'intelligence', column: 2, label: 'Intelligence', metricKey: 'day_count' },
  { id: 'ml_engine', column: 2, label: 'ML Engine', metricKey: 'mean_r2' },
  { id: 'shadow_engine', column: 2, label: 'Shadow Engine', metricKey: 'accuracy' },
  { id: 'pattern_recognition', column: 2, label: 'Patterns', metricKey: 'sequence_count' },
  { id: 'data_quality', column: 2, label: 'Data Quality', metricKey: 'included_count' },
];

// --- Column 3: Enrichment Modules ---
export const ENRICHMENT = [
  { id: 'orchestrator', column: 3, label: 'Orchestrator', metricKey: 'suggestion_count' },
  { id: 'organic_discovery', column: 3, label: 'Organic Discovery', metricKey: 'organic_count' },
  { id: 'activity_labeler', column: 3, label: 'Activity Labeler', metricKey: 'current_activity' },
];

// --- Column 4: API Outputs ---
export const OUTPUTS = [
  { id: 'out_auto_suggestions', column: 4, label: 'Suggestions', metricKey: 'suggestion_count', page: '/automations' },
  { id: 'out_pending_auto', column: 4, label: 'Pending Automations', metricKey: null, page: '/automations' },
  { id: 'out_created_auto', column: 4, label: 'Created Automations', metricKey: null, page: '/automations' },
  { id: 'out_ml_predictions', column: 4, label: 'ML Predictions', metricKey: null, page: '/predictions' },
  { id: 'out_ml_drift', column: 4, label: 'ML Drift / Anomalies', metricKey: null, page: '/ml-engine' },
  { id: 'out_shadow_preds', column: 4, label: 'Shadow Predictions', metricKey: null, page: '/shadow' },
  { id: 'out_shadow_accuracy', column: 4, label: 'Shadow Accuracy', metricKey: null, page: '/shadow' },
  { id: 'out_pipeline_stage', column: 4, label: 'Pipeline Stage', metricKey: null, page: '/' },
  { id: 'out_activity_labels', column: 4, label: 'Activity Labels', metricKey: null, page: '/intelligence' },
  { id: 'out_patterns', column: 4, label: 'Patterns', metricKey: null, page: '/patterns' },
  { id: 'out_intelligence', column: 4, label: 'Intelligence', metricKey: null, page: '/intelligence' },
  { id: 'out_presence', column: 4, label: 'Presence Map', metricKey: null, page: '/presence' },
  { id: 'out_curation', column: 4, label: 'Entity Curation', metricKey: null, page: '/data-curation' },
  { id: 'out_capabilities', column: 4, label: 'Capabilities', metricKey: null, page: '/capabilities' },
  { id: 'out_validation', column: 4, label: 'Validation', metricKey: null, page: '/validation' },
];

export const ALL_NODES = [...SOURCES, ...INTAKE, ...PROCESSING, ...ENRICHMENT, ...OUTPUTS];

// --- Links: source → target with relative value (drives flow width) ---
// type: 'data' (cyan), 'cache' (green), 'feedback' (amber)
export const LINKS = [
  // Sources → Intake
  { source: 'rest_states', target: 'engine', value: 10, type: 'data' },
  { source: 'rest_states', target: 'activity_monitor', value: 3, type: 'data' },
  { source: 'rest_registries', target: 'discovery', value: 8, type: 'data' },
  { source: 'ws_registry', target: 'discovery', value: 2, type: 'data' },
  { source: 'ws_state_changed', target: 'activity_monitor', value: 8, type: 'data' },
  { source: 'ws_state_changed', target: 'presence', value: 3, type: 'data' },
  { source: 'mqtt_frigate', target: 'presence', value: 4, type: 'data' },
  { source: 'logbook_json', target: 'engine', value: 5, type: 'data' },
  { source: 'logbook_json', target: 'pattern_recognition', value: 5, type: 'data' },
  { source: 'snapshot_json', target: 'engine', value: 4, type: 'data' },
  { source: 'snapshot_json', target: 'ml_engine', value: 4, type: 'data' },
  { source: 'ollama_queue', target: 'organic_discovery', value: 2, type: 'data' },
  { source: 'ollama_queue', target: 'activity_labeler', value: 2, type: 'data' },

  // Intake → Processing (through cache)
  { source: 'discovery', target: 'data_quality', value: 6, type: 'cache' },
  { source: 'discovery', target: 'organic_discovery', value: 4, type: 'cache' },
  { source: 'discovery', target: 'ml_engine', value: 3, type: 'cache' },
  { source: 'activity_monitor', target: 'intelligence', value: 5, type: 'cache' },
  { source: 'activity_monitor', target: 'shadow_engine', value: 5, type: 'cache' },
  { source: 'activity_monitor', target: 'ml_engine', value: 3, type: 'cache' },
  { source: 'activity_monitor', target: 'activity_labeler', value: 3, type: 'cache' },
  { source: 'presence', target: 'activity_labeler', value: 3, type: 'cache' },
  { source: 'engine', target: 'intelligence', value: 6, type: 'cache' },

  // Processing → Enrichment
  { source: 'intelligence', target: 'activity_labeler', value: 3, type: 'cache' },
  { source: 'pattern_recognition', target: 'orchestrator', value: 5, type: 'cache' },
  { source: 'shadow_engine', target: 'orchestrator', value: 2, type: 'cache' },

  // Enrichment → Outputs
  { source: 'orchestrator', target: 'out_auto_suggestions', value: 4, type: 'cache' },
  { source: 'orchestrator', target: 'out_pending_auto', value: 2, type: 'cache' },
  { source: 'orchestrator', target: 'out_created_auto', value: 2, type: 'cache' },
  { source: 'ml_engine', target: 'out_ml_predictions', value: 3, type: 'cache' },
  { source: 'ml_engine', target: 'out_ml_drift', value: 2, type: 'cache' },
  { source: 'shadow_engine', target: 'out_shadow_preds', value: 3, type: 'cache' },
  { source: 'shadow_engine', target: 'out_shadow_accuracy', value: 2, type: 'cache' },
  { source: 'shadow_engine', target: 'out_pipeline_stage', value: 2, type: 'cache' },
  { source: 'activity_labeler', target: 'out_activity_labels', value: 3, type: 'cache' },
  { source: 'pattern_recognition', target: 'out_patterns', value: 3, type: 'cache' },
  { source: 'intelligence', target: 'out_intelligence', value: 4, type: 'cache' },
  { source: 'presence', target: 'out_presence', value: 3, type: 'cache' },
  { source: 'data_quality', target: 'out_curation', value: 3, type: 'cache' },
  { source: 'discovery', target: 'out_capabilities', value: 3, type: 'cache' },
  { source: 'organic_discovery', target: 'out_capabilities', value: 2, type: 'cache' },

  // Feedback loops (reverse direction, amber)
  { source: 'ml_engine', target: 'discovery', value: 2, type: 'feedback' },
  { source: 'shadow_engine', target: 'discovery', value: 2, type: 'feedback' },
];

// --- NODE_DETAIL: progressive disclosure data (migrated from Home.jsx) ---
export const NODE_DETAIL = {
  rest_states: {
    protocol: 'HTTP GET /api/states (systemd timers)',
    reads: 'All HA entity states',
    writes: 'Consumed by Engine + Activity Monitor',
  },
  rest_registries: {
    protocol: 'HTTP GET /api/config/*_registry',
    reads: 'Entity, device, area registries',
    writes: 'Consumed by Discovery module',
  },
  ws_state_changed: {
    protocol: 'WebSocket subscribe_events',
    reads: 'Real-time state change events',
    writes: 'Consumed by Activity Monitor + Presence',
  },
  ws_registry: {
    protocol: 'WebSocket registry_updated',
    reads: 'Registry change notifications',
    writes: 'Consumed by Discovery module',
  },
  mqtt_frigate: {
    protocol: 'MQTT frigate/events (<mqtt-broker-ip>:1883)',
    reads: 'Person/face detection from cameras',
    writes: 'Consumed by Presence module',
  },
  logbook_json: {
    protocol: 'Disk files (ha-log-sync timer, every 15m)',
    reads: '~/ha-logs/logbook/*.json',
    writes: 'Consumed by Engine + Pattern Recognition',
  },
  snapshot_json: {
    protocol: 'Disk files (engine timers)',
    reads: '~/ha-logs/intelligence/daily/*.json',
    writes: 'Consumed by Engine + ML Engine',
  },
  ollama_queue: {
    protocol: 'HTTP POST to Ollama Queue (port 7683)',
    reads: 'LLM inference requests',
    writes: 'Consumed by Organic Discovery + Activity Labeler',
  },
  discovery: {
    protocol: 'REST registries + WS registry_updated',
    reads: 'Entity/device/area registries from HA',
    writes: 'entities, devices, areas, capabilities (seed), discovery_metadata',
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
  engine: {
    protocol: 'REST /api/states (systemd timers)',
    reads: 'All entity states, logbook, calendar, weather',
    writes: '~/ha-logs/intelligence/*.json (predictions, baselines, correlations, anomalies)',
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
  data_quality: {
    protocol: 'Classifies entities into tiers 1/2/3',
    reads: 'entities, activity_log (event rates)',
    writes: 'entity_curation table (included/excluded/tier)',
  },
  orchestrator: {
    protocol: 'Can call HA /api/automation/trigger',
    reads: 'patterns cache',
    writes: 'automation_suggestions, pending_automations, created_automations',
  },
  organic_discovery: {
    protocol: 'Weekly timer + Ollama queue (LLM naming)',
    reads: 'entities, devices, areas, activity_log, discovery_history',
    writes: 'capabilities (organic), discovery_settings, discovery_history',
  },
  activity_labeler: {
    protocol: 'Ollama queue for LLM fallback',
    reads: 'Sensor context: power, lights, motion, occupancy, time',
    writes: 'activity_labels (predictions + corrections + classifier)',
  },
};

// --- Metric extraction (migrated + extended from Home.jsx getNodeMetric) ---
export function getNodeMetric(cacheData, nodeId) {
  const caps = cacheData?.capabilities?.data || {};
  const pipeline = cacheData?.pipeline || {};
  const shadow = cacheData?.shadow_accuracy || {};
  const activity = cacheData?.activity_labels?.data || {};
  const intelligence = cacheData?.intelligence?.data || {};
  const curation = cacheData?.curation || {};
  const mlPipeline = cacheData?.ml_pipeline || {};
  const presence = cacheData?.presence?.data || {};

  switch (nodeId) {
    // Sources
    case 'rest_states': {
      const count = Object.keys(caps).filter((k) => caps[k]?.entities).reduce((s, k) => s + (caps[k].entities?.length || 0), 0);
      return count ? `${count} entities` : '\u2014';
    }
    case 'ws_state_changed': return pipeline?.events_per_minute ? `${pipeline.events_per_minute.toFixed(1)} ev/m` : '\u2014';
    case 'mqtt_frigate': return presence?.mqtt_connected ? 'Connected' : 'Disconnected';

    // Intake
    case 'engine': return pipeline?.intelligence_day ? `Day ${pipeline.intelligence_day}` : '\u2014';
    case 'discovery': {
      const count = Object.keys(caps).filter((k) => caps[k]?.entities).reduce((s, k) => s + (caps[k].entities?.length || 0), 0);
      return count ? `${count} entities` : '\u2014';
    }
    case 'activity_monitor': return pipeline?.events_per_minute ? `${pipeline.events_per_minute.toFixed(1)} ev/m` : '\u2014';
    case 'presence': {
      const rooms = presence?.occupied_rooms;
      return rooms?.length ? `${rooms.length} rooms` : '\u2014';
    }

    // Processing
    case 'intelligence': return pipeline?.intelligence_day ? `Day ${pipeline.intelligence_day}` : '\u2014';
    case 'ml_engine': {
      const mlCaps = Object.values(caps).filter((e) => e?.ml_accuracy);
      if (mlCaps.length === 0) return mlPipeline?.training?.models?.length ? `${mlPipeline.training.models.length} models` : '\u2014';
      const avgR2 = mlCaps.reduce((s, e) => s + (e.ml_accuracy.mean_r2 || 0), 0) / mlCaps.length;
      return `R\u00B2: ${avgR2.toFixed(2)}`;
    }
    case 'shadow_engine': return shadow?.overall_accuracy ? `${(shadow.overall_accuracy * 100).toFixed(0)}%` : '\u2014';
    case 'pattern_recognition': return '\u2014';
    case 'data_quality': return curation?.included ? `${curation.included} incl.` : pipeline?.included_entities ? `${pipeline.included_entities} incl.` : '\u2014';

    // Enrichment
    case 'orchestrator': return '\u2014';
    case 'organic_discovery': {
      const organic = Object.values(caps).filter((e) => e?.source === 'organic').length;
      return organic ? `${organic} organic` : '\u2014';
    }
    case 'activity_labeler': return activity?.current_activity?.predicted || '\u2014';

    // Outputs (show page name)
    default: return '\u2014';
  }
}

// --- Action strip conditions (priority order) ---
export const ACTION_CONDITIONS = [
  {
    id: 'advance_pipeline',
    test: (d) => d.pipeline?.can_advance,
    text: 'Pipeline gate met \u2014 advance to next stage \u2192',
    href: '#/shadow',
  },
  {
    id: 'edge_entities',
    test: (d) => (d.curation?.edge_case || 0) > 0,
    text: (d) => `Review ${d.curation.edge_case} edge-case entities \u2192`,
    href: '#/data-curation',
  },
  {
    id: 'shadow_disagreements',
    test: (d) => (d.shadow_accuracy?.disagreement_count || 0) > 5,
    text: (d) => `Review ${d.shadow_accuracy.disagreement_count} high-confidence disagreements \u2192`,
    href: '#/shadow',
  },
  {
    id: 'organic_candidates',
    test: (d) => (d.candidates?.length || 0) > 0,
    text: (d) => `Review ${d.candidates.length} discovered capabilities \u2192`,
    href: '#/discovery',
  },
  {
    id: 'automation_suggestions',
    test: (d) => (d.intelligence?.automation_suggestions?.length || 0) > 0,
    text: (d) => `Review ${d.intelligence.automation_suggestions.length} automation suggestions \u2192`,
    href: '#/automations',
  },
  {
    id: 'all_healthy',
    test: () => true,
    text: 'System healthy \u2014 no action needed',
    href: null,
  },
];
```

**Step 2: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/lib/pipelineGraph.js
git commit -m "feat(dashboard): add pipeline graph data model for Sankey"
```

---

## Task 3: PipelineSankey Component (Collapsed View)

**Files:**
- Create: `aria/dashboard/spa/src/components/PipelineSankey.jsx`

The main SVG component rendering the collapsed 5-column view with bus bar and flow ribbons.

**Step 1: Create the component with collapsed column rendering**

```jsx
// aria/dashboard/spa/src/components/PipelineSankey.jsx

import { useState, useRef, useEffect, useMemo } from 'preact/hooks';
import { computeLayout, computeTraceback } from '../lib/sankeyLayout.js';
import { ALL_NODES, LINKS, NODE_DETAIL, getNodeMetric, ACTION_CONDITIONS } from '../lib/pipelineGraph.js';

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

function SankeyNode({ node, status, metric, onClick, highlighted, dimmed }) {
  const color = STATUS_COLORS[status] || STATUS_COLORS.waiting;
  const opacity = dimmed ? 0.12 : 1;

  return (
    <g
      transform={`translate(${node.x}, ${node.y})`}
      onClick={() => onClick && onClick(node)}
      style={`cursor: ${onClick ? 'pointer' : 'default'}; opacity: ${opacity}; transition: opacity 0.3s;`}
    >
      <rect
        width={node.w}
        height={node.h}
        rx="4"
        fill="var(--bg-surface)"
        stroke={highlighted ? 'var(--accent)' : 'var(--border-primary)'}
        stroke-width={highlighted ? '2' : '1'}
      />
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
      {/* Metric */}
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
    <g style={`transition: opacity 0.3s;`}>
      <path
        d={link.path.d}
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
        x={busBar.width / 2}
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
      setExpandedColumn((prev) => (prev === node.column ? -1 : node.column));
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

  const svgHeight = layout.svgHeight + (hoveredNode ? 66 : 0);

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
        {layout.links.map((link) => (
          <SankeyFlow
            key={`${link.source}-${link.target}`}
            link={link}
            dimmed={isLinkDimmed(link)}
            highlighted={traceSet && !isLinkDimmed(link)}
          />
        ))}

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
        <span style="opacity: 0.6;">click column to expand \u00B7 click output to trace</span>
      </div>

      {/* Action strip */}
      <ActionStrip cacheData={cacheData} />
    </section>
  );
}
```

**Step 2: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/components/PipelineSankey.jsx
git commit -m "feat(dashboard): add PipelineSankey component with collapsed view"
```

---

## Task 4: Wire PipelineSankey Into Home Page

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Home.jsx`

Replace `BusArchitecture` with `PipelineSankey`. Keep all other Home page sections (PageBanner, HeroCard, JourneyProgress, RightNowStrip, PresenceCard).

**Step 1: Update imports and data fetching**

In `Home.jsx`, replace the `BusArchitecture` import and add the new component. Add the `GET /api/ml/pipeline` fetch to the existing `Promise.all`.

Changes to make:
- Remove: `BusArchitecture` function and all its helpers (`ModuleNode`, `Arrow`, `SwimLane`, `Banner`, `FeedbackArc`, `DetailPanel`, `FLOW_INTAKE`, `FLOW_PROCESSING`, `FLOW_ENRICHMENT`, `NODE_DETAIL`, `getNodeStatus`, `getNodeMetric`) — lines 16-429
- Add: `import PipelineSankey from '../components/PipelineSankey.jsx';`
- Add `fetchJson('/api/ml/pipeline').catch(() => null)` to the existing `Promise.all` on line 529
- Add `const [mlPipeline, setMlPipeline] = useState(null);` to state
- Update `cacheData` computation to include `ml_pipeline` and `curation`
- Replace `<BusArchitecture ... />` with `<PipelineSankey ... />` in the return JSX

**Step 2: Update the cacheData shape**

The current `cacheData` computed value (line 560-565) becomes:
```js
const cacheData = useComputed(() => ({
  capabilities: entities.data,
  pipeline: pipeline,
  shadow_accuracy: shadow,
  activity_labels: activity.data,
  intelligence: intelligence.data,
  ml_pipeline: mlPipeline,
  curation: curation,
  presence: /* from useCache('presence') or null */,
}), [entities.data, pipeline, shadow, activity.data, intelligence.data, mlPipeline, curation]);
```

**Step 3: Replace the JSX**

In the return block (~line 623), replace:
```jsx
<BusArchitecture
  moduleStatuses={health?.modules || {}}
  cacheData={cacheData}
/>
```
with:
```jsx
<PipelineSankey
  moduleStatuses={health?.modules || {}}
  cacheData={cacheData}
/>
```

**Step 4: Build and smoke test**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`
Expected: Build succeeds with no errors.

Then verify the dashboard loads: `curl -s http://127.0.0.1:8001/ui/ | head -5`
Expected: HTML with `<script src="./dist/bundle.js">`

**Step 5: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/pages/Home.jsx
git commit -m "feat(dashboard): replace BusArchitecture with PipelineSankey on Home page"
```

---

## Task 5: Mobile Pipeline Stepper

**Files:**
- Create: `aria/dashboard/spa/src/components/PipelineStepper.jsx`
- Modify: `aria/dashboard/spa/src/components/PipelineSankey.jsx`

On phone (<640px), render a vertical stepper instead of the SVG Sankey. The `PipelineSankey` component detects screen width and switches.

**Step 1: Create the stepper component**

```jsx
// aria/dashboard/spa/src/components/PipelineStepper.jsx

import { useState, useMemo } from 'preact/hooks';
import { SOURCES, INTAKE, PROCESSING, ENRICHMENT, OUTPUTS, getNodeMetric, NODE_DETAIL, ACTION_CONDITIONS } from '../lib/pipelineGraph.js';

const STAGES = [
  { label: 'SOURCES', nodes: SOURCES },
  { label: 'INTAKE', nodes: INTAKE },
  { label: 'PROCESSING', nodes: PROCESSING },
  { label: 'ENRICHMENT', nodes: ENRICHMENT },
  { label: 'OUTPUTS', nodes: OUTPUTS },
];

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

export default function PipelineStepper({ moduleStatuses, cacheData }) {
  const [expandedStage, setExpandedStage] = useState(-1);

  const action = useMemo(() => {
    for (const cond of ACTION_CONDITIONS) {
      if (cond.test(cacheData)) {
        const text = typeof cond.text === 'function' ? cond.text(cacheData) : cond.text;
        return { text, href: cond.href };
      }
    }
    return null;
  }, [cacheData]);

  return (
    <section class="t-frame" data-label="pipeline">
      <div class="space-y-1">
        {STAGES.map((stage, idx) => {
          const isExpanded = expandedStage === idx;
          return (
            <div key={stage.label}>
              {/* Stage row */}
              <button
                class="w-full flex items-center gap-2 py-2 px-1 text-left"
                style="background: none; border: none; cursor: pointer;"
                onClick={() => setExpandedStage(isExpanded ? -1 : idx)}
              >
                <span
                  class="w-2 h-2 rounded-full flex-shrink-0"
                  style={`background: var(--status-healthy);`}
                />
                <span class="flex-1 text-xs font-semibold" style="color: var(--text-primary); font-family: var(--font-mono);">
                  {stage.label}
                </span>
                <span class="text-xs" style="color: var(--text-tertiary); font-family: var(--font-mono);">
                  {stage.nodes.length} modules
                </span>
              </button>

              {/* Expanded nodes */}
              {isExpanded && (
                <div class="pl-4 pb-2 space-y-1">
                  {stage.nodes.map((node) => {
                    const status = node.column === 0 ? 'healthy' : getModuleStatus(moduleStatuses, node.id);
                    const metric = getNodeMetric(cacheData, node.id);
                    return (
                      <div key={node.id} class="flex items-center gap-2 py-1">
                        <span
                          class="w-1.5 h-1.5 rounded-full flex-shrink-0"
                          style={`background: ${STATUS_COLORS[status]};`}
                        />
                        <span class="flex-1 text-xs" style="color: var(--text-secondary); font-family: var(--font-mono);">
                          {node.label}
                        </span>
                        <span class="text-xs" style="color: var(--text-tertiary); font-family: var(--font-mono);">
                          {metric}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Connector line between stages */}
              {idx < STAGES.length - 1 && idx !== 1 && (
                <div class="ml-1 h-3 border-l" style="border-color: var(--border-primary);" />
              )}

              {/* Bus bar between Intake and Processing */}
              {idx === 1 && (
                <div
                  class="my-1 mx-1 px-2 py-1 rounded text-center text-xs font-bold"
                  style="background: var(--bg-terminal); border: 1px solid var(--status-healthy); color: var(--status-healthy); font-family: var(--font-mono);"
                >
                  HUB CACHE
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Action strip */}
      {action && (
        <div class="mt-3 pt-3" style="border-top: 1px solid var(--border-primary);">
          {action.href ? (
            <a href={action.href} class="text-xs font-medium" style="color: var(--accent); font-family: var(--font-mono); text-decoration: none;">
              {action.text}
            </a>
          ) : (
            <span class="text-xs" style="color: var(--text-tertiary); font-family: var(--font-mono);">
              {action.text}
            </span>
          )}
        </div>
      )}
    </section>
  );
}
```

**Step 2: Add responsive switch in PipelineSankey**

At the top of `PipelineSankey.jsx`, add width detection and conditional render:

```jsx
import PipelineStepper from './PipelineStepper.jsx';

// Inside PipelineSankey, after the width state:
const isMobile = width < 640;

// In the return, wrap:
if (isMobile) {
  return <PipelineStepper moduleStatuses={moduleStatuses} cacheData={cacheData} />;
}
// ... existing SVG return
```

**Step 3: Build and verify**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`
Expected: Build succeeds.

**Step 4: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/components/PipelineStepper.jsx aria/dashboard/spa/src/components/PipelineSankey.jsx
git commit -m "feat(dashboard): add mobile pipeline stepper with responsive switch"
```

---

## Task 6: Expand/Collapse with Sparklines

**Files:**
- Modify: `aria/dashboard/spa/src/components/PipelineSankey.jsx`
- Modify: `aria/dashboard/spa/src/lib/sankeyLayout.js`

Add the expand/collapse interaction with fade transition, and integrate `TimeChart` compact sparklines on expanded nodes.

**Step 1: Add fade transition state**

In `PipelineSankey.jsx`, add a `transitioning` state that triggers the two-phase fade:

```jsx
const [transitioning, setTransitioning] = useState(false);

function handleColumnClick(columnIndex) {
  setTransitioning(true);
  // Phase 1: fade out (150ms)
  setTimeout(() => {
    setExpandedColumn((prev) => (prev === columnIndex ? -1 : columnIndex));
    // Phase 2: fade in happens via CSS transition on opacity
    setTimeout(() => setTransitioning(false), 150);
  }, 150);
  setTraceTarget(null);
}
```

Apply `opacity: transitioning ? 0 : 1` with `transition: opacity 150ms` to the flows group.

**Step 2: Add sparkline rendering to expanded nodes**

Import `TimeChart` and render a compact sparkline next to each expanded node's metric. The sparkline data comes from the cache entry's historical values if available — otherwise skip. Use a placeholder `[timestamps, values]` array from the last 7 data points in the cache entry.

For the initial implementation, render sparklines only for nodes that have `sparkData` in the `cacheData` (intelligence intraday_trend, shadow accuracy history). This avoids the new backend endpoint — we use what's already fetched.

**Step 3: Add freshness timestamps**

Each expanded node shows relative time since last cache update. Compute from `cacheData[category]?.timestamp` or `cacheData[category]?.updated_at`.

Color: green if <15min, amber if <1h, red if >1h.

**Step 4: Build and verify**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`
Expected: Build succeeds.

**Step 5: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/components/PipelineSankey.jsx aria/dashboard/spa/src/lib/sankeyLayout.js
git commit -m "feat(dashboard): add expand/collapse with sparklines and freshness"
```

---

## Task 7: Trace-Back Highlighting

**Files:**
- Modify: `aria/dashboard/spa/src/components/PipelineSankey.jsx`
- Modify: `aria/dashboard/spa/src/lib/sankeyLayout.js`

Wire up the trace-back feature: clicking an Output node highlights its critical path and shows data values along each hop.

**Step 1: Enhance computeTraceback to return path with labels**

Update `sankeyLayout.js` to return an ordered list of nodes on the critical path (not just a Set), plus the link values between them for display as hop labels.

**Step 2: Render trace labels on highlighted flows**

When `traceTarget` is set, render small `<text>` labels at the midpoint of each highlighted flow showing the data value (e.g., "3,065 states", "12 sequences").

Use the `getNodeMetric` values for each node on the trace as the hop labels.

**Step 3: Build and verify**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`

**Step 4: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/components/PipelineSankey.jsx aria/dashboard/spa/src/lib/sankeyLayout.js
git commit -m "feat(dashboard): add trace-back highlighting with data labels"
```

---

## Task 8: Data Update Pulse Animation

**Files:**
- Modify: `aria/dashboard/spa/src/components/PipelineSankey.jsx`

When WebSocket pushes a cache update, the affected node briefly pulses.

**Step 1: Track last-updated timestamps per cache category**

Use `useRef` to store previous timestamps. On each render, compare current vs previous. If changed, add the node ID to a `pulsingNodes` set, clear after 500ms.

**Step 2: Apply pulse CSS class**

Pulsing nodes get `t2-tick-flash` class on their `<rect>`. The existing CSS animation handles the visual.

**Step 3: Build and verify**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`

**Step 4: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/components/PipelineSankey.jsx
git commit -m "feat(dashboard): add data update pulse on WebSocket cache changes"
```

---

## Task 9: Clean Up Old BusArchitecture Code

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Home.jsx`

Remove all remnants of the old `BusArchitecture` if not already done in Task 4. Verify no dead code remains.

**Step 1: Remove old constants and components**

Delete: `FLOW_INTAKE`, `FLOW_PROCESSING`, `FLOW_ENRICHMENT`, `NODE_DETAIL` (now in `pipelineGraph.js`), `ModuleNode`, `BusArchitecture`, `Arrow`, `SwimLane`, `Banner`, `FeedbackArc`, old `DetailPanel`, `getNodeStatus`, `getNodeMetric`.

Keep: `PHASES`, `PHASE_LABELS`, `PHASE_MILESTONES`, `JourneyProgress`, `RightNowStrip`, `Home` (default export).

**Step 2: Verify build**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`

**Step 3: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/pages/Home.jsx
git commit -m "refactor(dashboard): remove old BusArchitecture code from Home page"
```

---

## Task 10: Final Build, Smoke Test, and Deploy

**Files:**
- None new — verification only

**Step 1: Full build**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`
Expected: Clean build, no warnings.

**Step 2: Restart service**

Run: `systemctl --user restart aria-hub`
Wait 5 seconds, then: `systemctl --user status aria-hub`
Expected: Active (running).

**Step 3: Smoke test all Home page sections**

Run: `curl -s http://127.0.0.1:8001/health | python3 -m json.tool`
Expected: `"status": "ok"` with module statuses.

Run: `curl -s http://127.0.0.1:8001/api/pipeline | python3 -m json.tool`
Expected: Pipeline data with `current_stage`.

Run: `curl -s http://127.0.0.1:8001/api/ml/pipeline | python3 -m json.tool`
Expected: ML pipeline data (or empty object if no training yet).

**Step 4: Visual verification**

Open `http://127.0.0.1:8001/ui/` in browser. Verify:
- [ ] 5 column groups visible with bus bar
- [ ] Flow ribbons connect columns through bus bar
- [ ] Color legend below SVG
- [ ] Action strip shows appropriate action
- [ ] Click a column → expands to show individual nodes
- [ ] Click an output node → trace-back highlights critical path
- [ ] Hover expanded node → detail panel appears
- [ ] Resize browser to <640px → switches to pipeline stepper

**Step 5: Final commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add -A
git commit -m "feat(dashboard): pipeline Sankey visualization complete

Replaces BusArchitecture swim-lane diagram with bus-bar Sankey.
- 45 nodes across 5 collapsible column groups
- Central cache bus bar (architecturally honest hub-and-spoke)
- 3-layer progressive disclosure (glance/expand/trace-back)
- Context-aware action strip
- Mobile pipeline stepper fallback
- Data update pulse animation
- Science-backed: Tufte, Cleveland & McGill, Treisman, Gestalt, Miller, Shneiderman"
```
