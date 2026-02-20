# Pipeline Sankey Accuracy Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix every topology, metric, and navigation inaccuracy in the pipeline Sankey and make all nodes clickable.

**Architecture:** All topology lives in `pipelineGraph.js` (single source of truth). Both `PipelineSankey.jsx` (desktop) and `PipelineStepper.jsx` (mobile) consume it. Fix the data layer first, then fix the rendering layer.

**Tech Stack:** Preact, esbuild, SVG

**Design:** `docs/plans/2026-02-20-pipeline-sankey-accuracy-design.md`

**PRD:** `tasks/prd.json` (8 tasks)

## Quality Gates

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

---

### Task 1: Remove 6 incorrect links (PRD T1)

**Files:**
- Modify: `aria/dashboard/spa/src/lib/pipelineGraph.js:62-108` (LINKS array)

**Step 1: Remove these 6 entries from the LINKS array:**

```javascript
// DELETE these lines:
{ source: 'logbook_json', target: 'patterns', value: 5, type: 'data' },
{ source: 'snapshot_json', target: 'ml_engine', value: 4, type: 'data' },
{ source: 'shadow_engine', target: 'orchestrator', value: 2, type: 'cache' },
{ source: 'shadow_engine', target: 'out_pipeline_stage', value: 2, type: 'cache' },
{ source: 'ml_engine', target: 'discovery', value: 2, type: 'feedback' },
{ source: 'shadow_engine', target: 'discovery', value: 2, type: 'feedback' },
```

Also remove `{ source: 'rest_states', target: 'activity_monitor', value: 3, type: 'data' }` — activity monitor uses WebSocket for its primary feed, REST only for one-shot startup seed. The `ws_state_changed` → `activity_monitor` link (line 68) already represents this correctly.

**Step 2: Verify removed**

```bash
grep -c "shadow_engine.*orchestrator\|logbook_json.*patterns\|snapshot_json.*ml_engine" aria/dashboard/spa/src/lib/pipelineGraph.js
# Expected: 0
```

**Step 3: Commit**

```bash
git add aria/dashboard/spa/src/lib/pipelineGraph.js
git commit -m "fix(sankey): remove 7 incorrect pipeline links"
```

---

### Task 2: Add 6 missing links (PRD T2)

**Files:**
- Modify: `aria/dashboard/spa/src/lib/pipelineGraph.js` (LINKS array)

**Step 1: Add these links to the LINKS array in the appropriate sections:**

```javascript
// Intake → Processing (through cache) — add after existing entries in this section
{ source: 'engine', target: 'ml_engine', value: 5, type: 'cache' },
{ source: 'engine', target: 'patterns', value: 4, type: 'cache' },

// Intake → Intake (cache dependency) — new section
{ source: 'discovery', target: 'presence', value: 2, type: 'cache' },
{ source: 'discovery', target: 'activity_monitor', value: 2, type: 'cache' },

// Intake → Processing
{ source: 'discovery', target: 'shadow_engine', value: 2, type: 'cache' },

// Processing → Enrichment — add to existing section
{ source: 'intelligence', target: 'orchestrator', value: 3, type: 'cache' },
```

Also add feedback links targeting capabilities cache (conceptually back to discovery which manages it):

```javascript
// Feedback loops (reverse direction, amber) — capabilities feedback
{ source: 'ml_engine', target: 'discovery', value: 2, type: 'feedback' },
{ source: 'shadow_engine', target: 'discovery', value: 2, type: 'feedback' },
```

Wait — these feedback links were removed in Task 1 because the target was wrong. The reality is feedback goes to the **capabilities cache**, which is managed by discovery. The Sankey doesn't have a separate capabilities node. Since discovery owns the capabilities cache, keeping `discovery` as the target is the least-wrong representation. **Re-add the feedback links** with an inline comment explaining the semantics:

```javascript
// Feedback loops — ml/shadow write accuracy to capabilities cache (owned by discovery)
{ source: 'ml_engine', target: 'discovery', value: 2, type: 'feedback' },
{ source: 'shadow_engine', target: 'discovery', value: 2, type: 'feedback' },
```

**Step 2: Verify**

```bash
grep "source: 'engine', target: 'ml_engine'" aria/dashboard/spa/src/lib/pipelineGraph.js
grep "source: 'discovery', target: 'presence'" aria/dashboard/spa/src/lib/pipelineGraph.js
# Expected: both match
```

**Step 3: Commit**

```bash
git add aria/dashboard/spa/src/lib/pipelineGraph.js
git commit -m "fix(sankey): add 6 missing pipeline links + feedback annotations"
```

---

### Task 3: Rework output column — 14 nodes → 7 OODA-aligned (PRD T3)

**Files:**
- Modify: `aria/dashboard/spa/src/lib/pipelineGraph.js:41-56` (OUTPUTS array)
- Modify: `aria/dashboard/spa/src/lib/pipelineGraph.js:90-104` (LINKS to outputs)

**Step 1: Replace the entire OUTPUTS array:**

```javascript
// --- Column 4: OODA Page Outputs ---
export const OUTPUTS = [
  { id: 'out_observe', column: 4, label: 'Observe', page: '#/observe' },
  { id: 'out_understand', column: 4, label: 'Understand', page: '#/understand' },
  { id: 'out_decide', column: 4, label: 'Decide', page: '#/decide' },
  { id: 'out_capabilities', column: 4, label: 'Capabilities', page: '#/capabilities' },
  { id: 'out_ml_models', column: 4, label: 'ML Models', page: '#/ml-engine' },
  { id: 'out_curation', column: 4, label: 'Data Curation', page: '#/data-curation' },
  { id: 'out_validation', column: 4, label: 'Validation', page: '#/validation' },
];
```

**Step 2: Replace all output links.** Delete every link whose target starts with `out_` and replace with:

```javascript
// Processing/Enrichment → Outputs (OODA pages)
{ source: 'presence', target: 'out_observe', value: 3, type: 'cache' },
{ source: 'activity_monitor', target: 'out_observe', value: 3, type: 'cache' },
{ source: 'intelligence', target: 'out_observe', value: 2, type: 'cache' },
{ source: 'intelligence', target: 'out_understand', value: 4, type: 'cache' },
{ source: 'ml_engine', target: 'out_understand', value: 3, type: 'cache' },
{ source: 'shadow_engine', target: 'out_understand', value: 3, type: 'cache' },
{ source: 'patterns', target: 'out_understand', value: 2, type: 'cache' },
{ source: 'orchestrator', target: 'out_decide', value: 4, type: 'cache' },
{ source: 'discovery', target: 'out_capabilities', value: 3, type: 'cache' },
{ source: 'ml_engine', target: 'out_ml_models', value: 3, type: 'cache' },
{ source: 'discovery', target: 'out_curation', value: 3, type: 'cache' },
```

Note: `out_validation` has no direct module link — validation is a hub-level function, not a module output. It can remain unlinked or get a thin link from intelligence if desired.

**Step 3: Verify**

```bash
grep "out_observe\|out_understand\|out_decide" aria/dashboard/spa/src/lib/pipelineGraph.js | head -5
grep "out_auto_suggestions\|out_shadow_preds" aria/dashboard/spa/src/lib/pipelineGraph.js
# Expected: first shows matches, second shows nothing
```

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/lib/pipelineGraph.js
git commit -m "fix(sankey): replace 14 stale outputs with 7 OODA-aligned nodes"
```

---

### Task 4: Fix NODE_DETAIL and ACTION_CONDITIONS (PRD T4)

**Files:**
- Modify: `aria/dashboard/spa/src/lib/pipelineGraph.js:110-290`

**Step 1: Fix NODE_DETAIL entries:**

Change `logbook_json`:
```javascript
logbook_json: {
  protocol: 'Disk files (ha-log-sync timer, every 15m)',
  reads: 'Logbook JSON files',
  writes: 'Consumed by Engine + Patterns',  // was "Engine + Trajectory Classifier"
},
```

Change `orchestrator`:
```javascript
orchestrator: {
  protocol: 'Can call HA /api/automation/trigger',
  reads: 'patterns, automation_suggestions, pending_automations caches',  // was just "patterns cache"
  writes: 'automation_suggestions, pending_automations, created_automations',
},
```

Add NODE_DETAIL entries for the new output nodes:
```javascript
out_observe: {
  protocol: 'Dashboard page',
  reads: 'presence, activity_summary, intelligence caches',
  writes: 'User interaction (view)',
},
out_understand: {
  protocol: 'Dashboard page',
  reads: 'intelligence, ml_predictions, shadow_accuracy, patterns caches',
  writes: 'User interaction (view)',
},
out_decide: {
  protocol: 'Dashboard page',
  reads: 'automation_suggestions, pending_automations caches',
  writes: 'User interaction (approve/reject automations)',
},
out_capabilities: {
  protocol: 'Dashboard page',
  reads: 'capabilities registry, candidates',
  writes: 'User interaction (promote/demote)',
},
out_ml_models: {
  protocol: 'Dashboard page',
  reads: 'ml_models, ml_drift, ml_features, ml_hardware caches',
  writes: 'User interaction (view)',
},
out_curation: {
  protocol: 'Dashboard page',
  reads: 'curation, curation_summary caches',
  writes: 'User interaction (tier assignment)',
},
out_validation: {
  protocol: 'Dashboard page',
  reads: 'validation_latest cache',
  writes: 'User interaction (view)',
},
```

Remove NODE_DETAIL entries for deleted output nodes (`out_auto_suggestions`, `out_pending_auto`, `out_created_auto`, `out_ml_predictions`, `out_ml_drift`, `out_shadow_preds`, `out_shadow_accuracy`, `out_pipeline_stage`, `out_patterns`, `out_intelligence`, `out_presence`, `out_curation` [old], `out_capabilities` [old], `out_validation` [old]). Note: old output nodes didn't have NODE_DETAIL entries — they used the default `getNodeMetric` fallback. So nothing to remove here.

**Step 2: Fix ACTION_CONDITIONS:**

```javascript
export const ACTION_CONDITIONS = [
  {
    id: 'advance_pipeline',
    test: (d) => d.pipeline?.can_advance,
    text: 'Pipeline gate met \u2014 advance to next stage \u2192',
    href: '#/understand',  // was #/shadow
  },
  {
    id: 'edge_entities',
    test: (d) => (d.curation?.edge_case || 0) > 0,
    text: (d) => `Review ${d.curation.edge_case} edge-case entities \u2192`,
    href: '#/data-curation',  // unchanged
  },
  {
    id: 'shadow_disagreements',
    test: (d) => (d.shadow_accuracy?.disagreement_count || 0) > 5,
    text: (d) => `Review ${d.shadow_accuracy.disagreement_count} high-confidence disagreements \u2192`,
    href: '#/understand',  // was #/shadow
  },
  {
    id: 'automation_suggestions',
    test: (d) => (d.intelligence?.automation_suggestions?.length || 0) > 0,
    text: (d) => `Review ${d.intelligence.automation_suggestions.length} automation suggestions \u2192`,
    href: '#/decide',  // was #/automations
  },
  {
    id: 'all_healthy',
    test: () => true,
    text: 'System healthy \u2014 no action needed',
    href: null,
  },
];
```

**Step 3: Verify**

```bash
grep "href:" aria/dashboard/spa/src/lib/pipelineGraph.js
# Expected: #/understand, #/data-curation, #/understand, #/decide, null — no #/shadow or #/automations
```

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/lib/pipelineGraph.js
git commit -m "fix(sankey): correct NODE_DETAIL text and ACTION_CONDITIONS routes"
```

---

### Task 5: Fix freshness and remove dead sparklines (PRD T5)

**Files:**
- Modify: `aria/dashboard/spa/src/components/PipelineSankey.jsx:52-90`

**Step 1: Fix `getNodeSparklineData` — remove all dead paths:**

```javascript
function getNodeSparklineData(cacheData, nodeId) {
  // All sparkline paths referenced non-existent backend data arrays.
  // Kept as a stub for future use when backend adds time-series endpoints.
  return null;
}
```

**Step 2: Fix `getNodeFreshness` categoryMap:**

```javascript
function getNodeFreshness(cacheData, nodeId) {
  const categoryMap = {
    discovery: 'capabilities',
    activity_monitor: 'activity_summary',
    presence: 'presence',
    intelligence: 'intelligence',
    ml_engine: 'ml_pipeline',
    shadow_engine: 'shadow_accuracy',
    patterns: 'patterns',           // was pattern_recognition (wrong key)
    orchestrator: 'automation_suggestions',  // was automations (wrong category)
  };
  // ... rest unchanged
```

**Step 3: Fix `categoryToNodes` in the pulsing useEffect (around line 313):**

```javascript
const categoryToNodes = {
  capabilities: ['discovery'],              // removed organic_discovery
  activity_summary: ['activity_monitor'],
  presence: ['presence'],
  intelligence: ['intelligence'],
  ml_pipeline: ['ml_engine'],
  shadow_accuracy: ['shadow_engine'],
  patterns: ['patterns'],                   // was pattern_recognition
  automation_suggestions: ['orchestrator'],  // was automations→orchestrator; was curation→data_quality
};
```

Remove dead entries: `data_quality`, `organic_discovery`, `activity_labeler`, `activity_labels`.

**Step 4: Verify**

```bash
grep "pattern_recognition\|activity_labeler\|organic_discovery\|data_quality" aria/dashboard/spa/src/components/PipelineSankey.jsx
# Expected: no matches
grep "patterns.*patterns\|orchestrator.*automation_suggestions" aria/dashboard/spa/src/components/PipelineSankey.jsx
# Expected: 2 matches
```

**Step 5: Commit**

```bash
git add aria/dashboard/spa/src/components/PipelineSankey.jsx
git commit -m "fix(sankey): correct freshness mappings, remove dead sparklines"
```

---

### Task 6: Make output nodes navigate to pages (PRD T6)

**Files:**
- Modify: `aria/dashboard/spa/src/components/PipelineSankey.jsx:283-298` (handleNodeClick)

**Step 1: Update `handleNodeClick` to navigate output nodes to their pages:**

```javascript
function handleNodeClick(node) {
  if (node.isGroup) {
    setTransitioning(true);
    setTimeout(() => {
      setExpandedColumn((prev) => (prev === node.column ? -1 : node.column));
      setTimeout(() => setTransitioning(false), 150);
    }, 150);
    setTraceTarget(null);
  } else if (node.column === 4 && node.page) {
    // Output node — navigate to OODA page
    window.location.hash = node.page;
  } else if (node.column === 0) {
    // Source node — no navigation, detail panel on hover is sufficient
    return;
  } else {
    // Module node — navigate to module detail
    window.location.hash = `#/detail/module/${node.id}`;
  }
}
```

**Step 2: Verify**

```bash
grep "node.page" aria/dashboard/spa/src/components/PipelineSankey.jsx
# Expected: at least 1 match
```

**Step 3: Commit**

```bash
git add aria/dashboard/spa/src/components/PipelineSankey.jsx
git commit -m "fix(sankey): output nodes navigate to OODA pages on click"
```

---

### Task 7: Update PipelineStepper — clickability + status (PRD T7)

**Files:**
- Modify: `aria/dashboard/spa/src/components/PipelineStepper.jsx`

**Step 1: Make expanded nodes clickable.** Replace the static `<div>` node rows with clickable elements:

```jsx
{stage.nodes.map((node) => {
  const status = node.column === 0 ? 'healthy' : getModuleStatus(moduleStatuses, node.id);
  const metric = getNodeMetric(cacheData, node.id);
  const href = node.column === 4 && node.page
    ? node.page
    : node.column > 0 && node.column < 4
      ? `#/detail/module/${node.id}`
      : null;

  const Row = href ? 'a' : 'div';
  return (
    <Row
      key={node.id}
      {...(href ? { href, style: 'text-decoration: none; color: inherit; display: flex;' } : {})}
      class={`flex items-center gap-2 py-1 ${href ? 'clickable-data' : ''}`}
    >
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
    </Row>
  );
})}
```

**Step 2: Fix the stage header LED** to reflect group health instead of hardcoded green:

Import `getNodeTierGate` from pipelineGraph.js if not already imported. Update the stage LED:

```jsx
<span
  class="w-2 h-2 rounded-full flex-shrink-0"
  style={`background: ${(() => {
    const statuses = stage.nodes
      .filter(n => n.column > 0)
      .map(n => getModuleStatus(moduleStatuses, n.id));
    if (statuses.includes('blocked')) return STATUS_COLORS.blocked;
    if (statuses.includes('warning')) return STATUS_COLORS.warning;
    if (statuses.some(s => s === 'healthy')) return STATUS_COLORS.healthy;
    return STATUS_COLORS.waiting;
  })()};`}
/>
```

**Step 3: Verify**

```bash
grep "#/detail/module" aria/dashboard/spa/src/components/PipelineStepper.jsx
grep "node.page" aria/dashboard/spa/src/components/PipelineStepper.jsx
# Expected: both match
```

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/components/PipelineStepper.jsx
git commit -m "fix(stepper): add click navigation and real status colors"
```

---

### Task 8: Build and verify (PRD T8)

**Step 1: Build SPA**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

Expected: Clean build, no errors, bundle.js generated.

**Step 2: Verify bundle exists**

```bash
test -f aria/dashboard/spa/dist/bundle.js && echo "OK"
```

**Step 3: Commit progress**

```bash
# Update progress.txt and prd.json
git add tasks/prd.json progress.txt
git commit -m "docs: pipeline Sankey accuracy — all tasks pass"
```

---

## Batch Strategy

| Batch | Tasks | Description |
|-------|-------|-------------|
| 1 | T1-T4 | `pipelineGraph.js` — all topology, output, detail, and route fixes |
| 2 | T5-T7 | `PipelineSankey.jsx` + `PipelineStepper.jsx` — metrics, click, mobile |
| 3 | T8 | Build verification |
