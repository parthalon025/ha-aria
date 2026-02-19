/**
 * ARIA pipeline graph definition — lean audit (2026-02-19).
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
  { id: 'patterns', column: 2, label: 'Patterns', metricKey: null },
  { id: 'shadow_engine', column: 2, label: 'Shadow Engine', metricKey: 'accuracy' },
  { id: 'trajectory_classifier', column: 2, label: 'Trajectory Classifier', metricKey: 'sequence_count', tierGated: 3 },
];

// --- Column 3: Enrichment Modules ---
export const ENRICHMENT = [
  { id: 'orchestrator', column: 3, label: 'Orchestrator', metricKey: 'suggestion_count' },
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
  { source: 'logbook_json', target: 'patterns', value: 5, type: 'data' },
  { source: 'snapshot_json', target: 'engine', value: 4, type: 'data' },
  { source: 'snapshot_json', target: 'ml_engine', value: 4, type: 'data' },

  // Intake → Processing (through cache)
  { source: 'discovery', target: 'ml_engine', value: 3, type: 'cache' },
  { source: 'activity_monitor', target: 'intelligence', value: 5, type: 'cache' },
  { source: 'activity_monitor', target: 'shadow_engine', value: 5, type: 'cache' },
  { source: 'activity_monitor', target: 'ml_engine', value: 3, type: 'cache' },
  { source: 'engine', target: 'intelligence', value: 6, type: 'cache' },

  // Processing → Processing (internal)
  { source: 'shadow_engine', target: 'trajectory_classifier', value: 3, type: 'cache' },

  // Processing → Enrichment
  { source: 'patterns', target: 'orchestrator', value: 5, type: 'cache' },
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
  { source: 'patterns', target: 'out_patterns', value: 3, type: 'cache' },
  { source: 'intelligence', target: 'out_intelligence', value: 4, type: 'cache' },
  { source: 'presence', target: 'out_presence', value: 3, type: 'cache' },
  { source: 'discovery', target: 'out_curation', value: 3, type: 'cache' },
  { source: 'discovery', target: 'out_capabilities', value: 3, type: 'cache' },

  // Feedback loops (reverse direction, amber)
  { source: 'ml_engine', target: 'discovery', value: 2, type: 'feedback' },
  { source: 'shadow_engine', target: 'discovery', value: 2, type: 'feedback' },
];

// --- NODE_DETAIL: progressive disclosure data ---
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
    protocol: 'MQTT frigate/events',
    reads: 'Person/face detection from cameras',
    writes: 'Consumed by Presence module',
  },
  logbook_json: {
    protocol: 'Disk files (ha-log-sync timer, every 15m)',
    reads: 'Logbook JSON files',
    writes: 'Consumed by Engine + Trajectory Classifier',
  },
  snapshot_json: {
    protocol: 'Disk files (engine timers)',
    reads: 'Daily snapshot JSON files',
    writes: 'Consumed by Engine + ML Engine',
  },
  discovery: {
    protocol: 'REST registries + WS registry_updated',
    reads: 'Entity/device/area registries from HA',
    writes: 'entities, devices, areas, capabilities (seed), discovery_metadata, entity_curation',
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
    writes: 'Engine JSON files (predictions, baselines, correlations, anomalies)',
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
  trajectory_classifier: {
    protocol: 'Subscribes to shadow_resolved events',
    reads: 'Shadow resolution outcomes, feature snapshots',
    writes: 'trajectory classification, anomaly explanations',
    tierGated: 3,
  },
  patterns: {
    protocol: 'Reads logbook JSON files from disk',
    reads: 'Logbook data, intraday snapshots',
    writes: 'patterns cache (detected event patterns, association rules)',
  },
  orchestrator: {
    protocol: 'Can call HA /api/automation/trigger',
    reads: 'patterns cache',
    writes: 'automation_suggestions, pending_automations, created_automations',
  },
};

// --- Metric extraction ---
export function getNodeMetric(cacheData, nodeId) {
  const caps = cacheData?.capabilities?.data || {};
  const pipeline = cacheData?.pipeline || {};
  const shadow = cacheData?.shadow_accuracy || {};
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
    case 'patterns': return '\u2014';
    case 'shadow_engine': return shadow?.overall_accuracy ? `${(shadow.overall_accuracy * 100).toFixed(0)}%` : '\u2014';
    case 'trajectory_classifier': return '\u2014';

    // Enrichment
    case 'orchestrator': return '\u2014';

    // Outputs (show page name)
    default: return '\u2014';
  }
}

// --- Tier-gating helper ---
// Returns the minimum tier required for a node, or 0 if not tier-gated.
export function getNodeTierGate(nodeId) {
  const node = ALL_NODES.find((n) => n.id === nodeId);
  return node?.tierGated || NODE_DETAIL[nodeId]?.tierGated || 0;
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
