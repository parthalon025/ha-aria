# Interactive Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add headless screenshot audit, clickable detail sub-pages with SUPERHOT visuals, and per-module data source configuration to the ARIA dashboard.

**Architecture:** Three-phase build on existing Preact SPA + FastAPI backend. Phase 1 adds Puppeteer audit tooling. Phase 2 adds a `#/detail/:type/:id` route with 13 type-specific renderers and click handlers across all pages. Phase 3 adds backend config endpoints and frontend toggle components for module data sources.

**Tech Stack:** Preact (JSX), esbuild, Puppeteer (headless Chrome), FastAPI (Python), SQLite config store, CSS custom properties (SUPERHOT design language)

## Quality Gates

Run between every batch of tasks:

```bash
# SPA build check
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build

# Python tests (hub suite — fastest feedback)
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/ -x --timeout=120 -q

# Python compile check on new files
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m py_compile aria/hub/api.py
```

## PRD Cross-Reference

Tasks map to `tasks/prd.json` IDs: T1-T15.

---

### Task 1: Install Puppeteer and create screenshot audit scaffold (PRD T1)

**Files:**
- Modify: `aria/dashboard/spa/package.json`
- Create: `aria/dashboard/spa/scripts/screenshot-audit.js`
- Create: `docs/audit/` directory

**Step 1: Add Puppeteer devDependency**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm install --save-dev puppeteer
```

**Step 2: Create docs/audit directory**

```bash
mkdir -p /home/justin/Documents/projects/ha-aria/docs/audit/screenshots
```

**Step 3: Create screenshot-audit.js scaffold**

Create `aria/dashboard/spa/scripts/screenshot-audit.js`:

```javascript
import puppeteer from 'puppeteer';
import { writeFileSync, mkdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SCREENSHOTS_DIR = join(__dirname, '..', '..', '..', '..', 'docs', 'audit', 'screenshots');
const REPORT_PATH = join(__dirname, '..', '..', '..', '..', 'docs', 'audit', 'audit-report.md');
const BASE_URL = process.env.ARIA_URL || 'http://127.0.0.1:8001';

const ROUTES = [
  { name: 'home', path: '/' },
  { name: 'observe', path: '/observe' },
  { name: 'understand', path: '/understand' },
  { name: 'decide', path: '/decide' },
  { name: 'discovery', path: '/discovery' },
  { name: 'capabilities', path: '/capabilities' },
  { name: 'ml-engine', path: '/ml-engine' },
  { name: 'data-curation', path: '/data-curation' },
  { name: 'validation', path: '/validation' },
  { name: 'settings', path: '/settings' },
  { name: 'guide', path: '/guide' },
];

// API endpoints to check per page
const PAGE_API_MAP = {
  home: ['/health', '/api/ml/anomalies', '/api/shadow/accuracy', '/api/pipeline',
         '/api/cache/intelligence', '/api/cache/activity_summary', '/api/cache/automation_suggestions', '/api/cache/entities'],
  observe: ['/api/cache/intelligence', '/api/cache/activity_summary', '/api/cache/presence'],
  understand: ['/api/ml/anomalies', '/api/shadow/accuracy', '/api/ml/drift', '/api/ml/shap', '/api/patterns'],
  decide: ['/api/cache/automation_suggestions', '/api/automations/feedback'],
  discovery: ['/api/discovery/status', '/api/settings/discovery'],
  capabilities: ['/api/capabilities/registry', '/api/capabilities/candidates'],
  'ml-engine': ['/api/ml/models', '/api/ml/drift', '/api/ml/features', '/api/ml/hardware', '/api/ml/online'],
  'data-curation': ['/api/curation', '/api/curation/summary'],
  validation: ['/api/validation/latest'],
  settings: ['/api/config'],
  guide: [],
};

async function fetchApi(path) {
  try {
    const res = await fetch(`${BASE_URL}${path}`);
    if (!res.ok) return { error: `HTTP ${res.status}`, path };
    return { data: await res.json(), path };
  } catch (err) {
    return { error: err.message, path };
  }
}

async function run() {
  mkdirSync(SCREENSHOTS_DIR, { recursive: true });

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const report = ['# ARIA Dashboard Audit Report\n', `**Generated:** ${new Date().toISOString()}\n`];
  let totalMismatches = 0;

  for (const route of ROUTES) {
    console.log(`Capturing ${route.name}...`);
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });

    // Navigate and wait for content
    await page.goto(`${BASE_URL}/ui/#${route.path}`, { waitUntil: 'networkidle2', timeout: 15000 });
    // Wait for loading states to clear
    await page.waitForFunction(
      () => !document.querySelector('[class*="LoadingState"], [class*="loading"]'),
      { timeout: 10000 }
    ).catch(() => {});
    // Extra settle time for charts/animations
    await new Promise((r) => setTimeout(r, 1500));

    // Screenshot
    await page.screenshot({
      path: join(SCREENSHOTS_DIR, `${route.name}.png`),
      fullPage: true,
    });

    // Extract rendered text
    const pageText = await page.evaluate(() => document.body.innerText);

    // Check API endpoints
    const endpoints = PAGE_API_MAP[route.name] || [];
    const apiResults = await Promise.all(endpoints.map(fetchApi));

    report.push(`## ${route.name} (${route.path})\n`);
    report.push(`![${route.name}](screenshots/${route.name}.png)\n`);

    const mismatches = [];
    for (const result of apiResults) {
      if (result.error) {
        mismatches.push(`- **${result.path}**: API error — ${result.error}`);
      }
    }

    if (mismatches.length > 0) {
      totalMismatches += mismatches.length;
      report.push(`### Mismatches (${mismatches.length})\n`);
      report.push(mismatches.join('\n') + '\n');
    } else {
      report.push(`PASS — all ${endpoints.length} endpoints responding\n`);
    }

    report.push(`**Rendered text length:** ${pageText.length} chars\n`);
    await page.close();
  }

  await browser.close();

  report.unshift(''); // spacer
  report.push(`\n---\n**Summary:** ${totalMismatches} mismatches across ${ROUTES.length} pages\n`);
  if (totalMismatches === 0) {
    report.push('**Result:** PASS — all pages match backend data\n');
  }

  writeFileSync(REPORT_PATH, report.join('\n'));
  console.log(`\nAudit complete. ${totalMismatches} mismatches. Report: ${REPORT_PATH}`);
}

run().catch(console.error);
```

**Step 4: Verify**

```bash
grep -q 'puppeteer' /home/justin/Documents/projects/ha-aria/aria/dashboard/spa/package.json
test -f /home/justin/Documents/projects/ha-aria/aria/dashboard/spa/scripts/screenshot-audit.js
test -d /home/justin/Documents/projects/ha-aria/docs/audit
```

**Step 5: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/package.json aria/dashboard/spa/package-lock.json aria/dashboard/spa/scripts/screenshot-audit.js docs/audit/
git commit -m "feat(audit): add Puppeteer screenshot audit script scaffold"
```

---

### Task 2: Run screenshot audit (PRD T2-T3)

**Step 1: Ensure hub is running**

```bash
systemctl --user is-active aria-hub || systemctl --user start aria-hub
```

**Step 2: Run audit**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && node scripts/screenshot-audit.js
```

**Step 3: Verify screenshots exist**

```bash
ls -la /home/justin/Documents/projects/ha-aria/docs/audit/screenshots/*.png | wc -l
# Expected: 11
cat /home/justin/Documents/projects/ha-aria/docs/audit/audit-report.md
```

**Step 4: Review and fix mismatches (PRD T4)**

If mismatches are found, fix the relevant frontend components or API endpoints. Re-run audit until clean.

**Step 5: Commit screenshots and report**

```bash
cd /home/justin/Documents/projects/ha-aria
git add docs/audit/
git commit -m "docs(audit): initial screenshot audit of all 11 dashboard pages"
```

---

### Task 3: Create shared detail infrastructure (PRD T5)

**Files:**
- Create: `aria/dashboard/spa/src/components/Breadcrumb.jsx`
- Create: `aria/dashboard/spa/src/pages/DetailPage.jsx`
- Modify: `aria/dashboard/spa/src/index.css`
- Modify: `aria/dashboard/spa/src/app.jsx`

**Step 1: Create Breadcrumb.jsx**

```jsx
/**
 * Terminal-style breadcrumb navigation.
 * Renders: HOME / OBSERVE / ROOM: LIVING ROOM
 * Each segment is clickable and navigates via hash routing.
 */
export default function Breadcrumb({ segments }) {
  // segments: [{ label: 'HOME', href: '#/' }, { label: 'OBSERVE', href: '#/observe' }, ...]
  // Last segment is current (accent color, not clickable)
  return (
    <nav class="flex items-center gap-1 text-xs mb-2" style="font-family: var(--font-mono);">
      {segments.map((seg, i) => {
        const isLast = i === segments.length - 1;
        return (
          <span key={i} class="flex items-center gap-1">
            {i > 0 && <span style="color: var(--text-tertiary)">/</span>}
            {isLast ? (
              <span style="color: var(--accent)">{seg.label}</span>
            ) : (
              <a
                href={seg.href}
                class="clickable-data"
                style="color: var(--text-tertiary); text-decoration: none;"
              >
                {seg.label}
              </a>
            )}
          </span>
        );
      })}
    </nav>
  );
}
```

**Step 2: Add .clickable-data CSS to index.css**

Add to `aria/dashboard/spa/src/index.css` (near existing utility classes):

```css
/* Clickable data affordance — terminal-style hover */
.clickable-data {
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease;
  border-left: 2px solid transparent;
  padding-left: 4px;
}
.clickable-data:hover {
  background: var(--bg-surface-raised);
  border-left-color: var(--accent);
}
```

**Step 3: Create DetailPage.jsx**

```jsx
import { useState, useEffect } from 'preact/hooks';
import Breadcrumb from '../components/Breadcrumb.jsx';
import PageBanner from '../components/PageBanner.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

// Lazy-load detail renderers
const DETAIL_RENDERERS = {
  anomaly: () => import('./details/AnomalyDetail.jsx'),
  room: () => import('./details/RoomDetail.jsx'),
  entity: () => import('./details/EntityDetail.jsx'),
  prediction: () => import('./details/PredictionDetail.jsx'),
  suggestion: () => import('./details/SuggestionDetail.jsx'),
  capability: () => import('./details/CapabilityDetail.jsx'),
  model: () => import('./details/ModelDetail.jsx'),
  drift: () => import('./details/DriftDetail.jsx'),
  module: () => import('./details/ModuleDetail.jsx'),
  config: () => import('./details/ConfigDetail.jsx'),
  curation: () => import('./details/CurationDetail.jsx'),
  correlation: () => import('./details/CorrelationDetail.jsx'),
  baseline: () => import('./details/BaselineDetail.jsx'),
};

// Map detail types to their parent OODA page
const PARENT_PAGES = {
  anomaly: { label: 'UNDERSTAND', href: '#/understand' },
  room: { label: 'OBSERVE', href: '#/observe' },
  entity: { label: 'OBSERVE', href: '#/observe' },
  prediction: { label: 'UNDERSTAND', href: '#/understand' },
  suggestion: { label: 'DECIDE', href: '#/decide' },
  capability: { label: 'CAPABILITIES', href: '#/capabilities' },
  model: { label: 'ML ENGINE', href: '#/ml-engine' },
  drift: { label: 'ML ENGINE', href: '#/ml-engine' },
  module: { label: 'HOME', href: '#/' },
  config: { label: 'SETTINGS', href: '#/settings' },
  curation: { label: 'DATA CURATION', href: '#/data-curation' },
  correlation: { label: 'UNDERSTAND', href: '#/understand' },
  baseline: { label: 'UNDERSTAND', href: '#/understand' },
};

export default function DetailPage({ type, id, rest }) {
  const [Renderer, setRenderer] = useState(null);
  const [error, setError] = useState(null);

  // Support composite IDs: /detail/correlation/entity1/entity2
  const fullId = rest ? `${id}/${rest}` : id;

  useEffect(() => {
    const loader = DETAIL_RENDERERS[type];
    if (!loader) {
      setError(`Unknown detail type: ${type}`);
      return;
    }
    loader()
      .then((mod) => setRenderer(() => mod.default))
      .catch((err) => setError(err.message));
  }, [type]);

  const parent = PARENT_PAGES[type] || { label: 'HOME', href: '#/' };
  const breadcrumbs = [
    { label: 'HOME', href: '#/' },
    parent,
    { label: `${(type || '').toUpperCase()}: ${decodeURIComponent(fullId || '')}` },
  ];

  const bannerName = (type || '').toUpperCase();

  if (error) {
    return (
      <div class="space-y-6 animate-page-enter">
        <PageBanner page={`DETAIL + ${bannerName}`} subtitle="Detail view" />
        <ErrorState error={error} />
      </div>
    );
  }

  if (!Renderer) {
    return (
      <div class="space-y-6 animate-page-enter">
        <PageBanner page={`DETAIL + ${bannerName}`} subtitle="Loading detail..." />
        <LoadingState type="cards" />
      </div>
    );
  }

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page={`DETAIL + ${bannerName}`} subtitle={`Detailed view of ${type} data`} />
      <Breadcrumb segments={breadcrumbs} />
      <a href={parent.href} class="inline-block text-xs mb-2" style="color: var(--accent); font-family: var(--font-mono); text-decoration: none; min-height: 48px; line-height: 48px;">
        &larr; BACK
      </a>
      <Renderer id={fullId} type={type} />
    </div>
  );
}
```

**Step 4: Add route to app.jsx**

Add import and route in `app.jsx`:

```jsx
import DetailPage from './pages/DetailPage.jsx';
```

Add inside `<Router>` before the redirects:

```jsx
<DetailPage path="/detail/:type/:id/:rest*" />
```

**Step 5: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 6: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/components/Breadcrumb.jsx aria/dashboard/spa/src/pages/DetailPage.jsx aria/dashboard/spa/src/index.css aria/dashboard/spa/src/app.jsx
git commit -m "feat(dashboard): add detail page routing, breadcrumb nav, clickable-data CSS"
```

---

### Task 4: Build core detail renderers — Anomaly, Room, Entity, Prediction (PRD T6)

**Files:**
- Create: `aria/dashboard/spa/src/pages/details/AnomalyDetail.jsx`
- Create: `aria/dashboard/spa/src/pages/details/RoomDetail.jsx`
- Create: `aria/dashboard/spa/src/pages/details/EntityDetail.jsx`
- Create: `aria/dashboard/spa/src/pages/details/PredictionDetail.jsx`

**Step 1: Create details/ directory**

```bash
mkdir -p /home/justin/Documents/projects/ha-aria/aria/dashboard/spa/src/pages/details
```

**Step 2: Create AnomalyDetail.jsx**

Each renderer follows the Summary → Explanation → History pattern. Uses `fetchJson` for API data, `t-frame` containers, `HeroCard` for primary metric, `StatsGrid` for key-value pairs, SUPERHOT treatments for severity.

The renderers should:
- Call `fetchJson('/api/ml/anomalies')` and filter to the matching anomaly by entity ID
- Call `fetchJson('/api/anomalies/explain')` for SHAP + path trace
- Show `sh-threat-pulse` for critical, `sh-glitch` on severity text
- Use `TimeChart` for history if available
- Use `StatsGrid` for entity, area, detected_at, score
- Show SHAP attribution as horizontal bars

**Step 3: Create RoomDetail.jsx**

- Fetch presence data from cache (`/api/cache/presence`)
- Filter to the specific room
- Show Bayesian probability as HeroCard
- Show each signal type as a row with value, detail, timestamp
- Show identified persons in the room
- Show recent detections filtered to this room

**Step 4: Create EntityDetail.jsx**

- Fetch entity data from `/api/cache/entities` and curation from `/api/curation`
- Show entity state, domain, area, device info
- Show curation tier and status
- Show audit trail from `/api/audit/curation/{entity_id}`

**Step 5: Create PredictionDetail.jsx**

- Fetch from `/api/shadow/predictions` and filter by capability/entity
- Show prediction vs actual, confidence score
- Show Thompson Sampling stats from `/api/shadow/accuracy`

**Step 6: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 7: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/pages/details/
git commit -m "feat(dashboard): add core detail renderers — Anomaly, Room, Entity, Prediction"
```

---

### Task 5: Build remaining 9 detail renderers (PRD T7)

**Files:**
- Create: `aria/dashboard/spa/src/pages/details/SuggestionDetail.jsx`
- Create: `aria/dashboard/spa/src/pages/details/CapabilityDetail.jsx`
- Create: `aria/dashboard/spa/src/pages/details/ModelDetail.jsx`
- Create: `aria/dashboard/spa/src/pages/details/DriftDetail.jsx`
- Create: `aria/dashboard/spa/src/pages/details/ModuleDetail.jsx`
- Create: `aria/dashboard/spa/src/pages/details/ConfigDetail.jsx`
- Create: `aria/dashboard/spa/src/pages/details/CurationDetail.jsx`
- Create: `aria/dashboard/spa/src/pages/details/CorrelationDetail.jsx`
- Create: `aria/dashboard/spa/src/pages/details/BaselineDetail.jsx`

**Step 1: Create each renderer**

Each follows the same Summary → Explanation → History pattern:

- **SuggestionDetail**: `/api/cache/automation_suggestions` + feedback history. Show approve/reject/defer action buttons.
- **CapabilityDetail**: Refactor from existing `components/CapabilityDetail.jsx`. Use `/api/capabilities/registry/{id}`. Show dependency graph, health, can_predict toggle.
- **ModelDetail**: `/api/ml/models`. Show model type, accuracy metrics, last trained time, feature importance.
- **DriftDetail**: `/api/ml/drift`. Show Page-Hinkley score, ADWIN windows, rolling MAE with TimeChart.
- **ModuleDetail**: `/api/modules/{id}`. Show module status, uptime, last error, cache categories.
- **ConfigDetail**: `/api/config/{key}` + `/api/config-history`. Show current value, change history timeline, default, description.
- **CurationDetail**: `/api/curation` (filtered) + `/api/audit/curation/{entity_id}`. Show classification, override history.
- **CorrelationDetail**: Cache intelligence data. Show correlation coefficient, entity pair details, trend chart.
- **BaselineDetail**: Cache intelligence data. Show current vs baseline, deviation %, history chart.

**Step 2: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 3: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/pages/details/
git commit -m "feat(dashboard): add remaining 9 detail renderers"
```

---

### Task 6: Add click handlers to all pages (PRD T8-T10)

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Home.jsx`
- Modify: `aria/dashboard/spa/src/pages/Observe.jsx`
- Modify: `aria/dashboard/spa/src/pages/Understand.jsx`
- Modify: `aria/dashboard/spa/src/pages/Decide.jsx`
- Modify: `aria/dashboard/spa/src/pages/Discovery.jsx`
- Modify: `aria/dashboard/spa/src/pages/Capabilities.jsx`
- Modify: `aria/dashboard/spa/src/pages/MLEngine.jsx`
- Modify: `aria/dashboard/spa/src/pages/DataCuration.jsx`
- Modify: `aria/dashboard/spa/src/pages/Settings.jsx`
- Modify: `aria/dashboard/spa/src/components/HeroCard.jsx`
- Modify: `aria/dashboard/spa/src/components/PipelineSankey.jsx`
- Modify: `aria/dashboard/spa/src/components/PresenceCard.jsx`

**Step 1: Add href prop to HeroCard**

Add optional `href` prop. When provided, wrap the card in an anchor:

```jsx
// In HeroCard.jsx, add href to props
export default function HeroCard({ value, label, delta, warning, sparkData, sparkColor, href }) {
  const content = (/* existing card content */);
  if (href) {
    return <a href={href} class="clickable-data block" style="text-decoration: none; color: inherit;">{content}</a>;
  }
  return content;
}
```

**Step 2: Add click handlers to PipelineSankey**

Each Sankey node should navigate to `#/detail/module/{nodeId}` on click.

**Step 3: Add click handlers to PresenceCard**

Each room section should navigate to `#/detail/room/{roomName}` on click.

**Step 4: Wire OODA pages**

For each page, wrap data elements in clickable anchors or add onClick handlers:

- **Home.jsx**: HeroCards get `href` props (`#/detail/anomaly/all`, `#/detail/suggestion/all`, `#/detail/prediction/accuracy`)
- **Observe.jsx**: Presence rooms, live metric labels, activity items
- **Understand.jsx**: Anomaly cards → `#/detail/anomaly/{entity}`, prediction rows → `#/detail/prediction/{entity}`, drift → `#/detail/drift/all`, correlation cells → `#/detail/correlation/{e1}/{e2}`, baseline rows → `#/detail/baseline/{metric}`
- **Decide.jsx**: Suggestion cards → `#/detail/suggestion/{id}`

**Step 5: Wire system pages**

- **Discovery.jsx**: Entity rows → `#/detail/entity/{entity_id}`
- **Capabilities.jsx**: Capability cards → `#/detail/capability/{name}`
- **MLEngine.jsx**: Model cards → `#/detail/model/{name}`, drift indicators → `#/detail/drift/all`
- **DataCuration.jsx**: Entity rows → `#/detail/curation/{entity_id}`
- **Settings.jsx**: Config rows → `#/detail/config/{key}`

**Step 6: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 7: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/
git commit -m "feat(dashboard): add click handlers to all pages and shared components"
```

---

### Task 7: Build SPA and verify detail routing end-to-end (PRD T11)

**Step 1: Full SPA build**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 2: Restart hub**

```bash
systemctl --user restart aria-hub
```

**Step 3: Run screenshot audit with detail pages**

Update screenshot-audit.js to also capture a few detail page routes (e.g., `#/detail/anomaly/test`, `#/detail/room/living_room`) and re-run.

**Step 4: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/dist/ docs/audit/
git commit -m "feat(dashboard): build SPA with detail routing, update audit screenshots"
```

---

### Task 8: Create backend module source config endpoints (PRD T12)

**Files:**
- Create: `aria/hub/routes_module_config.py`
- Modify: `aria/hub/config_defaults.py`
- Modify: `aria/hub/api.py`
- Create: `tests/hub/test_module_config.py`

**Step 1: Write failing test**

Create `tests/hub/test_module_config.py`:

```python
"""Tests for module source configuration endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

MODULE_SOURCES = {
    "presence": {
        "signals": ["camera_person", "camera_face", "motion", "light_interaction",
                     "dimmer_press", "door", "media_active", "device_tracker"],
    },
    "activity": {
        "domains": ["light", "switch", "binary_sensor", "media_player", "climate", "cover"],
    },
}


@pytest.mark.asyncio
async def test_get_module_sources_returns_defaults():
    """GET /api/config/modules/presence/sources returns default enabled signals."""
    # Test implementation will use test client
    pass


@pytest.mark.asyncio
async def test_put_module_sources_updates_config():
    """PUT /api/config/modules/presence/sources disables a signal."""
    pass


@pytest.mark.asyncio
async def test_put_module_sources_prevents_empty():
    """PUT cannot disable all sources — at least one must remain."""
    pass
```

**Step 2: Run test to verify it fails**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_module_config.py -x --timeout=120 -q
```

**Step 3: Add default source configs to config_defaults.py**

Add new entries to the `CONFIG_DEFAULTS` list for each module's source configuration.

**Step 4: Create routes_module_config.py**

```python
"""Module source configuration routes."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from aria.hub.core import IntelligenceHub

logger = logging.getLogger(__name__)

# Valid modules and their source keys
MODULE_SOURCE_KEYS = {
    "presence": "presence.enabled_signals",
    "activity": "activity.enabled_domains",
    "anomaly": "anomaly.enabled_entities",
    "shadow": "shadow.enabled_capabilities",
    "discovery": "discovery.domain_filter",
}


class SourceUpdate(BaseModel):
    sources: list[str]


def _register_module_config_routes(router: APIRouter, hub: IntelligenceHub) -> None:
    @router.get("/api/config/modules/{module}/sources")
    async def get_module_sources(module: str):
        if module not in MODULE_SOURCE_KEYS:
            raise HTTPException(404, f"Unknown module: {module}")
        config_key = MODULE_SOURCE_KEYS[module]
        value = await hub.get_config(config_key)
        # Parse comma-separated string back to list
        sources = [s.strip() for s in (value or "").split(",") if s.strip()]
        return {"module": module, "sources": sources, "config_key": config_key}

    @router.put("/api/config/modules/{module}/sources")
    async def put_module_sources(module: str, body: SourceUpdate):
        if module not in MODULE_SOURCE_KEYS:
            raise HTTPException(404, f"Unknown module: {module}")
        if not body.sources:
            raise HTTPException(400, "At least one source must remain enabled")
        config_key = MODULE_SOURCE_KEYS[module]
        value = ",".join(body.sources)
        await hub.set_config(config_key, value, changed_by="user")
        return {"module": module, "sources": body.sources, "config_key": config_key}
```

**Step 5: Register in api.py**

In `create_api()`, add after existing route registrations:

```python
from aria.hub.routes_module_config import _register_module_config_routes
_register_module_config_routes(router, hub)
```

**Step 6: Run tests**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_module_config.py -x --timeout=120 -q
```

**Step 7: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/hub/routes_module_config.py aria/hub/config_defaults.py aria/hub/api.py tests/hub/test_module_config.py
git commit -m "feat(api): add module source config endpoints — GET/PUT /api/config/modules/{module}/sources"
```

---

### Task 9: Wire modules to read source config (PRD T13)

**Files:**
- Modify: `aria/modules/presence.py`
- Modify: tests as needed

**Step 1: Write failing test**

Add test in `tests/hub/test_presence.py` or `tests/hub/test_module_config.py`:

```python
@pytest.mark.asyncio
async def test_presence_skips_disabled_signal():
    """Presence module should skip signals not in enabled_signals config."""
    pass
```

**Step 2: Modify presence.py**

In the signal processing method, read enabled signals from config and skip disabled ones:

```python
# At signal processing time:
enabled = await self.hub.get_config("presence.enabled_signals")
if enabled:
    enabled_list = [s.strip() for s in enabled.split(",")]
    if signal_type not in enabled_list:
        return  # Skip disabled signal
```

**Step 3: Run tests**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_presence.py -x --timeout=120 -q
```

**Step 4: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/modules/presence.py tests/
git commit -m "feat(presence): read enabled_signals from config, skip disabled signal types"
```

---

### Task 10: Build frontend data source config components (PRD T14)

**Files:**
- Create: `aria/dashboard/spa/src/components/TerminalToggle.jsx`
- Create: `aria/dashboard/spa/src/components/DataSourceConfig.jsx`
- Modify: `aria/dashboard/spa/src/pages/Observe.jsx`
- Modify: `aria/dashboard/spa/src/pages/Understand.jsx`

**Step 1: Create TerminalToggle.jsx**

```jsx
/**
 * Terminal-style toggle: [ON ] / [OFF]
 * Monospace, no rounded iOS switches — fits the ASCII terminal aesthetic.
 */
export default function TerminalToggle({ enabled, onToggle, disabled }) {
  return (
    <button
      onClick={() => !disabled && onToggle(!enabled)}
      class="px-2 py-0.5 text-xs"
      style={`
        font-family: var(--font-mono);
        cursor: ${disabled ? 'not-allowed' : 'pointer'};
        color: ${enabled ? 'var(--status-healthy)' : 'var(--text-tertiary)'};
        background: var(--bg-inset);
        border: 1px solid ${enabled ? 'var(--status-healthy)' : 'var(--border-subtle)'};
        border-radius: var(--radius);
        opacity: ${disabled ? '0.5' : '1'};
        transition: all 0.15s ease;
      `}
      aria-pressed={enabled}
      aria-label={enabled ? 'Enabled' : 'Disabled'}
    >
      {enabled ? '[ON ]' : '[OFF]'}
    </button>
  );
}
```

**Step 2: Create DataSourceConfig.jsx**

```jsx
import { useState, useEffect } from 'preact/hooks';
import { fetchJson, putJson } from '../api.js';
import TerminalToggle from './TerminalToggle.jsx';
import { Section } from '../pages/intelligence/utils.jsx';

/**
 * Data source toggle list for a module.
 * Fetches current config from GET /api/config/modules/{module}/sources,
 * updates via PUT on toggle.
 */
export default function DataSourceConfig({ module, title, subtitle, descriptions }) {
  const [sources, setSources] = useState([]);
  const [allSources, setAllSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // descriptions: { signal_name: "Human-readable description" }

  async function fetchSources() {
    try {
      const data = await fetchJson(`/api/config/modules/${module}/sources`);
      setSources(data.sources || []);
    } catch (err) {
      console.warn(`DataSourceConfig: failed to load ${module} sources`, err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchSources(); }, [module]);

  // Derive all possible sources from descriptions keys
  useEffect(() => {
    if (descriptions) setAllSources(Object.keys(descriptions));
  }, [descriptions]);

  async function handleToggle(source, enabled) {
    const newSources = enabled
      ? [...sources, source]
      : sources.filter((s) => s !== source);

    if (newSources.length === 0) return; // Safety: prevent disabling all

    setSaving(true);
    try {
      await putJson(`/api/config/modules/${module}/sources`, { sources: newSources });
      setSources(newSources);
    } catch (err) {
      console.error('Source config save failed:', err);
    } finally {
      setSaving(false);
    }
  }

  if (loading) return null;

  return (
    <Section title={title} subtitle={subtitle} defaultOpen={false}>
      <div class="space-y-1">
        {allSources.map((source) => {
          const enabled = sources.includes(source);
          const isLast = sources.length === 1 && enabled;
          return (
            <div key={source} class="flex items-center justify-between py-2" style="border-bottom: 1px solid var(--border-subtle)">
              <div class="flex-1 min-w-0">
                <span class="text-sm font-medium" style="color: var(--text-secondary)">{source}</span>
                {descriptions[source] && (
                  <p class="text-xs mt-0.5" style="color: var(--text-tertiary)">{descriptions[source]}</p>
                )}
              </div>
              <TerminalToggle enabled={enabled} onToggle={(val) => handleToggle(source, val)} disabled={isLast || saving} />
            </div>
          );
        })}
      </div>
      {saving && <p class="text-xs mt-2" style="color: var(--accent)">Saving...</p>}
    </Section>
  );
}
```

**Step 3: Add to Observe.jsx**

Import and add below existing InlineSettings:

```jsx
import DataSourceConfig from '../components/DataSourceConfig.jsx';

// Inside return, after InlineSettings:
<DataSourceConfig
  module="presence"
  title="Presence Sources"
  subtitle="Toggle which signals feed room occupancy detection."
  descriptions={{
    camera_person: "Frigate camera person detection",
    camera_face: "Frigate face recognition",
    motion: "Motion sensor binary_sensors",
    light_interaction: "Light state changes",
    dimmer_press: "Hue dimmer button presses",
    door: "Door open/close sensors",
    media_active: "Media player active states",
    device_tracker: "Person home/away tracking",
  }}
/>

<DataSourceConfig
  module="activity"
  title="Activity Sources"
  subtitle="Toggle which entity domains are tracked for activity monitoring."
  descriptions={{
    light: "Light entities",
    switch: "Switch entities",
    binary_sensor: "Binary sensor entities",
    media_player: "Media player entities",
    climate: "Climate/HVAC entities",
    cover: "Cover/blind entities",
  }}
/>
```

**Step 4: Add to Understand.jsx**

Similar pattern for anomaly and shadow sources.

**Step 5: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 6: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/components/TerminalToggle.jsx aria/dashboard/spa/src/components/DataSourceConfig.jsx aria/dashboard/spa/src/pages/Observe.jsx aria/dashboard/spa/src/pages/Understand.jsx
git commit -m "feat(dashboard): add terminal toggle and data source config components"
```

---

### Task 11: Final build, full test suite, and re-run screenshot audit (PRD T15)

**Step 1: Rebuild SPA**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 2: Run full Python test suite**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/ -x --timeout=120 -q
```

**Step 3: Restart hub and re-run screenshot audit**

```bash
systemctl --user restart aria-hub
sleep 3
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && node scripts/screenshot-audit.js
```

**Step 4: Review final audit report**

```bash
cat /home/justin/Documents/projects/ha-aria/docs/audit/audit-report.md
```

**Step 5: Final commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add .
git commit -m "feat(dashboard): interactive dashboard — detail pages, data source config, final audit"
```

---

## Batch Execution Plan

| Batch | Tasks | Quality Gate |
|-------|-------|-------------|
| 1 | T1 (Puppeteer scaffold) | `npm run build` |
| 2 | T2 (Run audit + fix) | Screenshots exist, report clean |
| 3 | T3 (Detail infrastructure) | `npm run build` |
| 4 | T4-T5 (All 13 detail renderers) | `npm run build` |
| 5 | T6 (Click handlers on all pages) | `npm run build` |
| 6 | T7 (SPA build + verify routing) | Hub restart + manual check |
| 7 | T8 (Backend config endpoints) | `pytest tests/hub/ -k module_config` |
| 8 | T9 (Wire modules to config) | `pytest tests/hub/test_presence.py` |
| 9 | T10 (Frontend config components) | `npm run build` |
| 10 | T11 (Final verification) | Full pytest + audit |
