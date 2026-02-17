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
    protocol: 'MQTT frigate/events (192.168.1.35:1883)',
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

// --- Metric extraction ---
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
