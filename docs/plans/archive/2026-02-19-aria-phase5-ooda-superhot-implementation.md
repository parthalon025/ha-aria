# ARIA Phase 5: OODA Dashboard + SUPERHOT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the ARIA dashboard from 14 pipeline-oriented pages to 5 OODA decision destinations, layered with superhot-ui visual effects for freshness, urgency, and dismissal.

**Architecture:** Assemble new OODA pages from existing components (no backend changes). Install superhot-ui as a local sibling dependency. Wrap data cards with ShFrozen for freshness, anomalies with ShThreatPulse, and dismiss actions with ShShatter. Old routes redirect to new destinations.

**Tech Stack:** Preact 10.25, preact-router, @preact/signals, Tailwind CSS v4, esbuild, superhot-ui (local)

**Design Doc:** `docs/plans/2026-02-19-aria-phase5-ooda-superhot-design.md`

**Quality Gates:**
- `cd aria/dashboard/spa && npm run build` (must exit 0)
- `cd aria/dashboard/spa && ls dist/bundle.js dist/bundle.css` (both must exist)

---

### Task 0: Initialize progress.txt + install superhot-ui

**Files:**
- Create: `progress.txt`
- Modify: `aria/dashboard/spa/package.json`

**Step 1: Create progress.txt**

```
echo "# Phase 5 OODA + SUPERHOT Progress" > progress.txt
echo "" >> progress.txt
echo "## Task 0: Bootstrap" >> progress.txt
echo "- [ ] Install superhot-ui" >> progress.txt
echo "- [ ] Verify build" >> progress.txt
```

**Step 2: Build superhot-ui**

superhot-ui's `dist/` is gitignored — must build before ARIA can consume it.

Run:
```bash
cd ~/Documents/projects/superhot-ui && npm run build
```
Expected: `dist/superhot.css`, `dist/superhot.js`, `dist/superhot.preact.js` all exist.

**Step 3: Install superhot-ui as local dependency**

Run:
```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm install file:../../../../projects/superhot-ui
```

Verify `package.json` now has `"superhot-ui": "file:../../../../projects/superhot-ui"` in dependencies.

**Step 4: Import superhot-ui CSS**

Add to `src/index.css` after the existing imports on line 2:

```css
@import "tailwindcss";
@import "uplot/dist/uPlot.min.css";
@import "../../projects/superhot-ui/css/superhot.css";
```

> **Why relative path?** The Tailwind CLI `@import` resolves relative paths reliably. Using the npm exports (`superhot-ui/css`) requires Tailwind to resolve node_modules, which may not work with `@tailwindcss/cli`. A direct relative path through the symlink is guaranteed to work.

**Step 5: Verify build**

Run:
```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```
Expected: Build succeeds. `dist/bundle.js` and `dist/bundle.css` both exist. `bundle.css` contains superhot-ui CSS (search for `data-sh-state`).

**Step 6: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/dashboard/spa/package.json aria/dashboard/spa/package-lock.json aria/dashboard/spa/src/index.css progress.txt
git commit -m "feat(dashboard): install superhot-ui + CSS integration"
```

---

### Task 1: Add freshness helper + new icons to Sidebar

**Files:**
- Modify: `aria/dashboard/spa/src/pages/intelligence/utils.jsx`
- Modify: `aria/dashboard/spa/src/components/Sidebar.jsx`

**Step 1: Add freshness helper to utils.jsx**

Add this export at the end of `src/pages/intelligence/utils.jsx`:

```jsx
/**
 * Extract the last_updated timestamp from a useCache() result.
 * Cache envelope shape: { category, data: {...}, last_updated }
 * Returns ISO string or null.
 */
export function cacheTimestamp(cacheData) {
  if (!cacheData || !cacheData.data) return null;
  return cacheData.last_updated || cacheData.data.last_updated || null;
}
```

**Step 2: Add new nav icons to Sidebar.jsx**

Add these icon components after the existing icons (before `getHashPath()`):

```jsx
function CompassIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
    </svg>
  );
}

function ActivityIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}

function ChevronDownIcon() {
  return (
    <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}
```

**Step 3: Restructure NAV_ITEMS**

Replace the `NAV_ITEMS` constant (lines 5-25) with:

```jsx
const NAV_ITEMS = [
  { path: '/', label: 'Home', icon: GridIcon },
  { path: '/observe', label: 'Observe', icon: EyeIcon },
  { path: '/understand', label: 'Understand', icon: BrainIcon },
  { path: '/decide', label: 'Decide', icon: ActivityIcon },
  // System section (expandable)
  { section: 'System', expandable: true },
  { path: '/discovery', label: 'Discovery', icon: SearchIcon, system: true },
  { path: '/capabilities', label: 'Capabilities', icon: ZapIcon, system: true },
  { path: '/ml-engine', label: 'ML Engine', icon: CpuIcon, system: true },
  { path: '/data-curation', label: 'Data Curation', icon: FilterIcon, system: true },
  { path: '/validation', label: 'Validation', icon: CheckIcon, system: true },
  { path: '/settings', label: 'Settings', icon: SlidersIcon, system: true },
];
```

**Step 4: Update PHONE_TABS**

Replace `PHONE_TABS` (lines 28-34) with:

```jsx
const PHONE_TABS = [
  { path: '/', label: 'Home', icon: GridIcon },
  { path: '/observe', label: 'Observe', icon: EyeIcon },
  { path: '/understand', label: 'Understand', icon: BrainIcon },
  { path: '/decide', label: 'Decide', icon: ActivityIcon },
  { key: 'more', label: 'More', icon: MoreIcon },
];
```

**Step 5: Update MORE_ITEMS**

Replace the `MORE_ITEMS` computation with:

```jsx
const MORE_ITEMS = NAV_ITEMS.filter(
  (item) => item.system
);
```

**Step 6: Add System section expand/collapse to DesktopNav**

In the `DesktopNav` component, add state for the System expandable section. Find the NAV_ITEMS.map in DesktopNav (around line 501) and update the rendering to handle `expandable` sections:

Inside DesktopNav, add state:
```jsx
const [systemOpen, setSystemOpen] = useState(false);
```

In the NAV_ITEMS.map, when rendering a section with `expandable: true`, render a clickable header. When `!systemOpen`, skip items with `system: true`:

```jsx
{NAV_ITEMS.map((item, i) => {
  if (item.section) {
    if (item.expandable) {
      return (
        <button
          key={item.section}
          onClick={() => setSystemOpen(!systemOpen)}
          class="flex items-center justify-between w-full"
          style={`padding: 0 12px; padding-top: 16px; padding-bottom: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-tertiary); border-top: 1px solid var(--border-subtle); margin-top: 8px; background: none; border-left: none; border-right: none; border-bottom: none; cursor: pointer;`}
        >
          <span>{item.section}</span>
          <span style={`transform: rotate(${systemOpen ? '180deg' : '0deg'}); transition: transform 0.2s ease;`}>
            <ChevronDownIcon />
          </span>
        </button>
      );
    }
    return (
      <div
        key={item.section}
        style={`padding: 0 12px; padding-top: 16px; padding-bottom: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-tertiary);${i > 0 ? ' border-top: 1px solid var(--border-subtle); margin-top: 8px;' : ''}`}
      >
        {item.section}
      </div>
    );
  }
  // Skip system items when collapsed
  if (item.system && !systemOpen) return null;
  const active = currentPath === item.path;
  return (
    <a
      key={item.path}
      href={`#${item.path}`}
      class="flex items-center gap-3 text-sm font-medium"
      style={active
        ? 'background: var(--bg-surface-raised); color: var(--text-primary); border-left: 2px solid var(--accent); padding: 8px 12px; border-radius: var(--radius); transition: background 0.15s ease, color 0.15s ease;'
        : 'color: var(--text-tertiary); padding: 8px 12px; border-left: 2px solid transparent; border-radius: var(--radius); transition: background 0.15s ease, color 0.15s ease;'
      }
      onMouseEnter={(ev) => { if (!active) ev.currentTarget.style.background = 'var(--bg-surface-raised)'; ev.currentTarget.style.color = 'var(--text-primary)'; }}
      onMouseLeave={(ev) => { if (!active) { ev.currentTarget.style.background = 'transparent'; ev.currentTarget.style.color = 'var(--text-tertiary)'; } }}
      aria-label={item.label}
    >
      <item.icon />
      {item.label}
    </a>
  );
})}
```

Apply the same expand/collapse logic to TabletNav's expanded view. In collapsed (icon-only) mode, show only non-system items. The PhoneNav More sheet already uses `MORE_ITEMS` which now only contains system items.

**Step 7: Verify build**

Run:
```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```
Expected: Build succeeds.

**Step 8: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/dashboard/spa/src/pages/intelligence/utils.jsx aria/dashboard/spa/src/components/Sidebar.jsx
git commit -m "feat(dashboard): OODA navigation structure + freshness helper"
```

---

### Task 2: Create PipelineStatusBar component

**Files:**
- Create: `aria/dashboard/spa/src/components/PipelineStatusBar.jsx`

**Step 1: Create the component**

```jsx
import { useState, useEffect } from 'preact/hooks';
import { wsConnected } from '../store.js';
import { fetchJson } from '../api.js';

/**
 * Compact one-line pipeline status bar for the Home page.
 * Shows: Pipeline stage · Shadow stage · WebSocket status.
 * Applies ShGlitch/ShMantra attributes when modules fail or WS is down.
 */
export default function PipelineStatusBar() {
  const [health, setHealth] = useState(null);
  const [pipeline, setPipeline] = useState(null);
  const connected = wsConnected.value;

  useEffect(() => {
    fetchJson('/health').then(setHealth).catch(() => {});
    fetchJson('/api/pipeline').then(setPipeline).catch(() => {});
  }, []);

  const pipelineStage = pipeline?.current_stage || 'starting';
  const hasFailed = health?.modules && Object.values(health.modules).some((status) => status === 'failed');

  // ShGlitch on failure, ShMantra on WS down
  const attrs = {};
  if (hasFailed) {
    attrs['data-sh-effect'] = 'glitch';
  }
  if (!connected) {
    attrs['data-sh-mantra'] = 'OFFLINE';
  }

  return (
    <div
      class="t-frame"
      style="padding: 8px 16px;"
      {...attrs}
    >
      <div class="flex items-center gap-3 text-xs" style="color: var(--text-tertiary); font-family: var(--font-mono);">
        <span>
          Pipeline: <span style={`color: ${hasFailed ? 'var(--status-error)' : 'var(--text-secondary)'}`}>{pipelineStage}</span>
        </span>
        <span style="color: var(--border-subtle);">&middot;</span>
        <span>
          Shadow: <span style="color: var(--text-secondary)">{pipeline?.current_stage || 'backtest'}</span>
        </span>
        <span style="color: var(--border-subtle);">&middot;</span>
        <span class="flex items-center gap-1">
          <span
            class="inline-block w-1.5 h-1.5 rounded-full"
            style={`background: ${connected ? 'var(--status-healthy)' : 'var(--status-error)'};`}
          />
          WebSocket: <span style={`color: ${connected ? 'var(--status-healthy)' : 'var(--status-error)'}`}>{connected ? 'connected' : 'disconnected'}</span>
        </span>
      </div>
    </div>
  );
}
```

**Step 2: Verify build**

Run:
```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```
Expected: Build succeeds (component not yet imported, but syntax checked via bundle).

Note: The component won't be tree-shaken since it's not imported yet, but we verify it compiles. It will be imported in Task 5.

**Step 3: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/dashboard/spa/src/components/PipelineStatusBar.jsx
git commit -m "feat(dashboard): add PipelineStatusBar component"
```

---

### Task 3: Create OodaSummaryCard component

**Files:**
- Create: `aria/dashboard/spa/src/components/OodaSummaryCard.jsx`

**Step 1: Create the component**

```jsx
/**
 * Clickable summary card for each OODA destination on the Home page.
 * Shows a brief metric and links to the full page.
 */
export default function OodaSummaryCard({ title, subtitle, metric, metricLabel, href, accentColor }) {
  const color = accentColor || 'var(--accent)';
  return (
    <a
      href={href}
      class="t-frame t-card-hover block"
      style="text-decoration: none; padding: 16px 20px; cursor: pointer;"
    >
      <div class="flex items-center justify-between mb-2">
        <span
          class="text-xs font-semibold uppercase"
          style={`letter-spacing: 0.05em; color: ${color};`}
        >
          {title}
        </span>
        <span class="text-xs" style="color: var(--text-tertiary);">&rarr;</span>
      </div>
      {metric != null && (
        <div class="flex items-baseline gap-2 mb-1">
          <span class="data-mono text-lg font-bold" style={`color: ${color};`}>
            {metric}
          </span>
          {metricLabel && (
            <span class="text-xs" style="color: var(--text-tertiary);">{metricLabel}</span>
          )}
        </div>
      )}
      {subtitle && (
        <p class="text-xs" style="color: var(--text-secondary);">{subtitle}</p>
      )}
    </a>
  );
}
```

**Step 2: Verify build**

Run:
```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```
Expected: Build succeeds.

**Step 3: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/dashboard/spa/src/components/OodaSummaryCard.jsx
git commit -m "feat(dashboard): add OodaSummaryCard component"
```

---

### Task 4: Create Observe page

**Files:**
- Create: `aria/dashboard/spa/src/pages/Observe.jsx`

**Step 1: Create the page**

This page assembles existing components: PresenceCard, HomeRightNow, ActivitySection, plus a live metrics strip.

```jsx
import { useState, useEffect } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { fetchJson } from '../api.js';
import { wsConnected } from '../store.js';
import { cacheTimestamp } from './intelligence/utils.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import PageBanner from '../components/PageBanner.jsx';
import PresenceCard from '../components/PresenceCard.jsx';
import { HomeRightNow } from './intelligence/HomeRightNow.jsx';
import { ActivitySection } from './intelligence/ActivitySection.jsx';

export default function Observe() {
  const intelligence = useCache('intelligence');
  const activity = useCache('activity_summary');

  const intel = useComputed(() => {
    if (!intelligence.data || !intelligence.data.data) return null;
    return intelligence.data.data;
  }, [intelligence.data]);

  const actInner = useComputed(() => {
    if (!activity.data || !activity.data.data) return null;
    return activity.data.data;
  }, [activity.data]);

  const loading = intelligence.loading || activity.loading;
  const error = intelligence.error || activity.error;

  if (loading && !intelligence.data) {
    return (
      <div class="space-y-6">
        <PageBanner page="OBSERVE" subtitle="What's happening in your home right now." />
        <LoadingState type="cards" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <PageBanner page="OBSERVE" subtitle="What's happening in your home right now." />
        <ErrorState error={error} onRetry={() => { intelligence.refetch(); activity.refetch(); }} />
      </div>
    );
  }

  // Live metrics from activity_summary cache
  const ws = actInner ? (actInner.websocket || null) : null;
  const actRate = actInner ? (actInner.activity_rate || null) : null;
  const evRate = actRate ? actRate.current : null;
  const occ = actInner ? (actInner.occupancy || null) : null;

  // Intraday metrics from intelligence cache
  const intraday = intel ? intel.intraday_trend : null;
  const latest = Array.isArray(intraday) && intraday.length > 0 ? intraday[intraday.length - 1] : null;
  const lightsOn = latest ? (latest.lights_on ?? null) : null;
  const powerW = latest ? (latest.power_watts ?? null) : null;

  const connected = wsConnected.value;
  const ts = cacheTimestamp(activity.data);

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="OBSERVE" subtitle="What's happening in your home right now." />

      {/* Live metrics strip */}
      <div class="t-frame" data-label="live metrics" data-sh-state={ts ? undefined : 'stale'}>
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
            <span class="w-2 h-2 rounded-full" style={`background: ${connected ? 'var(--status-healthy)' : 'var(--status-error)'};`} />
            <span style="color: var(--text-tertiary)">WebSocket</span>
            <span class="font-medium" style="color: var(--text-primary)">{connected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </div>

      {/* Presence */}
      <PresenceCard />

      {/* Home Right Now — metric cards */}
      {intel && (
        <HomeRightNow intraday={intel.intraday_trend} baselines={intel.baselines} />
      )}

      {/* Activity stream */}
      {intel && (
        <ActivitySection activity={intel.activity} />
      )}
    </div>
  );
}
```

**Step 2: Verify build**

Run:
```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```
Expected: Build succeeds.

**Step 3: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/dashboard/spa/src/pages/Observe.jsx
git commit -m "feat(dashboard): add Observe page"
```

---

### Task 5: Create Understand page

**Files:**
- Create: `aria/dashboard/spa/src/pages/Understand.jsx`

**Step 1: Create the page**

Assembles anomalies (primary), patterns, predictions, drift, correlations, shadow accuracy.

```jsx
import { useState, useEffect } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { fetchJson } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import PageBanner from '../components/PageBanner.jsx';
import { Section, Callout } from './intelligence/utils.jsx';
import { AnomalyAlerts } from './intelligence/AnomalyAlerts.jsx';
import { PredictionsVsActuals } from './intelligence/PredictionsVsActuals.jsx';
import { DriftStatus } from './intelligence/DriftStatus.jsx';
import { ShapAttributions } from './intelligence/ShapAttributions.jsx';
import { Baselines } from './intelligence/Baselines.jsx';
import { TrendsOverTime } from './intelligence/TrendsOverTime.jsx';
import { Correlations } from './intelligence/Correlations.jsx';

function PatternsList({ patterns }) {
  if (!patterns || patterns.length === 0) {
    return (
      <Section title="Patterns" subtitle="Recurring event sequences from your logbook.">
        <Callout>No patterns detected yet. Needs several days of logbook data.</Callout>
      </Section>
    );
  }

  return (
    <Section title="Patterns" subtitle="Recurring event sequences from your logbook." summary={`${patterns.length} pattern${patterns.length === 1 ? '' : 's'}`}>
      <div class="space-y-2">
        {patterns.map((pat, i) => (
          <div key={pat.name || i} class="t-frame p-3" data-label={pat.name || 'pattern'}>
            <div class="flex items-center justify-between">
              <span class="text-sm font-bold" style="color: var(--text-primary)">{pat.name || 'Unnamed'}</span>
              {pat.type && (
                <span class="text-xs px-2 py-0.5 rounded-full" style="background: var(--bg-surface-raised); color: var(--text-secondary)">{pat.type}</span>
              )}
            </div>
            {pat.description && <p class="text-xs mt-1" style="color: var(--text-secondary)">{pat.description}</p>}
            <div class="flex gap-4 mt-1 text-xs" style="color: var(--text-tertiary)">
              {pat.confidence != null && <span>Confidence: {Math.round(pat.confidence * 100)}%</span>}
              {pat.frequency && <span>{pat.frequency}</span>}
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}

function ShadowBrief({ accuracy }) {
  if (!accuracy) return null;

  const dailyTrend = accuracy.daily_trend || [];
  const last7 = dailyTrend.slice(-7);
  const avg7d = last7.length > 0
    ? last7.reduce((sum, day) => sum + (day.accuracy ?? 0), 0) / last7.length
    : null;

  const total = accuracy.predictions_total ?? 0;
  const stage = accuracy.stage || 'backtest';

  return (
    <Section title="Shadow Accuracy" subtitle="Predict-compare-score loop measuring forecast quality." summary={avg7d != null ? `${Math.round(avg7d * 100)}% (7d avg)` : stage}>
      <div class="t-frame p-3" data-label="shadow">
        <div class="flex items-center gap-4 text-sm">
          <span class="text-xs font-medium rounded-full px-2.5 py-0.5 capitalize" style="background: var(--accent-glow); color: var(--accent)">{stage}</span>
          {avg7d != null ? (
            <span style="color: var(--text-secondary)">
              <span class="font-bold" style={`color: ${Math.round(avg7d * 100) >= 70 ? 'var(--status-healthy)' : Math.round(avg7d * 100) >= 40 ? 'var(--status-warning)' : 'var(--status-error)'}`}>
                {Math.round(avg7d * 100)}%
              </span> trailing 7-day accuracy ({total} predictions)
            </span>
          ) : (
            <span style="color: var(--text-tertiary)">No predictions yet</span>
          )}
        </div>
      </div>
    </Section>
  );
}

export default function Understand() {
  const intelligence = useCache('intelligence');
  const patternsCache = useCache('patterns');

  const [anomalies, setAnomalies] = useState(null);
  const [drift, setDrift] = useState(null);
  const [shap, setShap] = useState(null);
  const [shadowAccuracy, setShadowAccuracy] = useState(null);

  useEffect(() => {
    fetchJson('/api/ml/anomalies').then(setAnomalies).catch(() => {});
    fetchJson('/api/ml/drift').then(setDrift).catch(() => {});
    fetchJson('/api/ml/shap').then(setShap).catch(() => {});
    fetchJson('/api/shadow/accuracy').then(setShadowAccuracy).catch(() => {});
  }, []);

  const intel = useComputed(() => {
    if (!intelligence.data || !intelligence.data.data) return null;
    return intelligence.data.data;
  }, [intelligence.data]);

  const patterns = useComputed(() => {
    if (!patternsCache.data || !patternsCache.data.data) return [];
    return patternsCache.data.data.patterns || [];
  }, [patternsCache.data]);

  const loading = intelligence.loading;
  const error = intelligence.error;

  if (loading && !intelligence.data) {
    return (
      <div class="space-y-6">
        <PageBanner page="UNDERSTAND" subtitle="What's unusual, what's repeating, and why." />
        <LoadingState type="cards" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <PageBanner page="UNDERSTAND" subtitle="What's unusual, what's repeating, and why." />
        <ErrorState error={error} onRetry={intelligence.refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-8 animate-page-enter">
      <PageBanner page="UNDERSTAND" subtitle="What's unusual, what's repeating, and why." />

      {/* Anomalies — primary output, always at top */}
      <AnomalyAlerts anomalies={anomalies} />

      {/* Patterns */}
      <PatternsList patterns={patterns} />

      {/* Predictions vs Actuals */}
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {intel && <PredictionsVsActuals predictions={intel.predictions} intradayTrend={intel.intraday_trend} />}
        <DriftStatus drift={drift} />
      </div>

      {/* SHAP */}
      <ShapAttributions shap={shap} />

      {/* Context: Baselines + Trends + Correlations */}
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {intel && <Baselines baselines={intel.baselines} />}
        {intel && <TrendsOverTime trendData={intel.trend_data} intradayTrend={intel.intraday_trend} />}
      </div>
      {intel && <Correlations correlations={intel.entity_correlations?.top_co_occurrences} />}

      {/* Shadow accuracy */}
      <ShadowBrief accuracy={shadowAccuracy} />
    </div>
  );
}
```

**Step 2: Verify build**

Run:
```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```
Expected: Build succeeds.

**Step 3: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/dashboard/spa/src/pages/Understand.jsx
git commit -m "feat(dashboard): add Understand page"
```

---

### Task 6: Create Decide page

**Files:**
- Create: `aria/dashboard/spa/src/pages/Decide.jsx`

**Step 1: Create the page**

Reuses the AutomationCard from Automations.jsx. We import the cache and rebuild the layout with a decision-focused framing.

```jsx
import { useState } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { baseUrl } from '../api.js';
import HeroCard from '../components/HeroCard.jsx';
import PageBanner from '../components/PageBanner.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import { Section, Callout } from './intelligence/utils.jsx';

/** Status badge inline style. */
function statusStyle(status) {
  switch ((status || '').toLowerCase()) {
    case 'approved': return 'background: var(--status-healthy-glow); color: var(--status-healthy);';
    case 'rejected': return 'background: var(--status-error-glow); color: var(--status-error);';
    case 'deferred': return 'background: var(--bg-surface-raised); color: var(--text-tertiary);';
    default: return 'background: var(--status-warning-glow); color: var(--status-warning);';
  }
}

function RecommendationCard({ suggestion, onAction, updating }) {
  const confidence = suggestion.confidence ?? 0;
  const pct = Math.round(confidence * 100);
  const status = (suggestion.status || 'pending').toLowerCase();

  return (
    <div class="t-frame" data-label={suggestion.name || 'recommendation'} style="padding: 1.25rem;">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-base font-bold" style="color: var(--text-primary)">{suggestion.name || 'Unnamed'}</h3>
        <span class="inline-block px-2 py-0.5 rounded-full text-xs font-medium" style={statusStyle(status)}>{status}</span>
      </div>
      {suggestion.description && (
        <p class="text-sm mb-3" style="color: var(--text-secondary)">{suggestion.description}</p>
      )}
      <div class="flex items-center gap-4 text-xs mb-3" style="color: var(--text-tertiary)">
        <span>Confidence: <span class="data-mono font-medium" style="color: var(--text-secondary)">{pct}%</span></span>
        {suggestion.occurrence_count != null && (
          <span>Occurrences: <span class="data-mono font-medium" style="color: var(--text-secondary)">{suggestion.occurrence_count}</span></span>
        )}
      </div>
      {suggestion.yaml && (
        <details class="mb-3">
          <summary class="text-sm cursor-pointer" style="color: var(--accent)">Show YAML</summary>
          <pre class="mt-2 p-3 text-xs overflow-x-auto" style="background: var(--bg-inset); color: var(--text-primary); border-radius: var(--radius); font-family: var(--font-mono)">{suggestion.yaml}</pre>
        </details>
      )}
      {status === 'pending' && (
        <div class="flex gap-2 mt-4">
          <button
            onClick={() => onAction(suggestion.id, 'approved')}
            disabled={updating}
            class="t-btn px-4 py-1.5 text-sm font-medium disabled:opacity-50"
            style="background: var(--status-healthy); color: var(--text-inverse); border: none; cursor: pointer;"
          >
            Approve
          </button>
          <button
            onClick={() => onAction(suggestion.id, 'rejected')}
            disabled={updating}
            class="t-btn px-4 py-1.5 text-sm font-medium disabled:opacity-50"
            style="background: var(--status-error); color: var(--text-inverse); border: none; cursor: pointer;"
          >
            Reject
          </button>
          <button
            onClick={() => onAction(suggestion.id, 'deferred')}
            disabled={updating}
            class="t-btn px-4 py-1.5 text-sm font-medium disabled:opacity-50"
            style="background: var(--bg-surface-raised); color: var(--text-secondary); border: 1px solid var(--border-subtle); cursor: pointer;"
          >
            Defer
          </button>
        </div>
      )}
    </div>
  );
}

export default function Decide() {
  const { data, loading, error, refetch } = useCache('automation_suggestions');
  const [updating, setUpdating] = useState(false);
  const [updateError, setUpdateError] = useState(null);
  const [localStatuses, setLocalStatuses] = useState({});

  const { suggestions } = useComputed(() => {
    if (!data || !data.data) return { suggestions: [] };
    return { suggestions: data.data.suggestions || [] };
  }, [data]);

  const displaySuggestions = useComputed(() => {
    return suggestions.map((item) => localStatuses[item.id] ? { ...item, status: localStatuses[item.id] } : item);
  }, [suggestions, localStatuses]);

  const pending = displaySuggestions.filter((item) => (item.status || 'pending').toLowerCase() === 'pending');
  const approved = displaySuggestions.filter((item) => (item.status || '').toLowerCase() === 'approved');
  const rejected = displaySuggestions.filter((item) => (item.status || '').toLowerCase() === 'rejected');
  const deferred = displaySuggestions.filter((item) => (item.status || '').toLowerCase() === 'deferred');

  async function handleAction(id, newStatus) {
    setUpdating(true);
    setUpdateError(null);
    setLocalStatuses((prev) => ({ ...prev, [id]: newStatus }));

    try {
      const updatedSuggestions = suggestions.map((item) =>
        item.id === id ? { ...item, status: newStatus } : item
      );
      const updatedData = { ...(data.data || {}), suggestions: updatedSuggestions };
      const res = await fetch(`${baseUrl}/api/cache/automation_suggestions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: updatedData }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setLocalStatuses((prev) => { const next = { ...prev }; delete next[id]; return next; });
      refetch();
    } catch (err) {
      setLocalStatuses((prev) => { const next = { ...prev }; delete next[id]; return next; });
      setUpdateError(err.message || String(err));
    } finally {
      setUpdating(false);
    }
  }

  if (loading && !data) {
    return (
      <div class="space-y-6">
        <PageBanner page="DECIDE" subtitle="Recommendations ARIA has generated. Approve, reject, or defer." />
        <LoadingState type="cards" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <PageBanner page="DECIDE" subtitle="Recommendations ARIA has generated. Approve, reject, or defer." />
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="DECIDE" subtitle="Recommendations ARIA has generated. Approve, reject, or defer." />

      <HeroCard
        value={pending.length}
        label="pending review"
        delta={`${approved.length} approved · ${rejected.length} rejected · ${deferred.length} deferred`}
        loading={loading}
      />

      {updateError && <ErrorState error={updateError} onRetry={() => setUpdateError(null)} />}

      {pending.length === 0 && displaySuggestions.length === 0 ? (
        <Callout>No automation suggestions yet. ARIA generates recommendations when it finds patterns with high confidence and matching capabilities.</Callout>
      ) : (
        <>
          {/* Pending — primary action area */}
          {pending.length > 0 && (
            <Section title="Pending Review" summary={`${pending.length} pending`}>
              <div class="space-y-4">
                {pending.map((sug, i) => (
                  <RecommendationCard key={sug.id || i} suggestion={sug} onAction={handleAction} updating={updating} />
                ))}
              </div>
            </Section>
          )}

          {/* History */}
          {(approved.length > 0 || rejected.length > 0 || deferred.length > 0) && (
            <Section title="History" subtitle="Previously reviewed recommendations." defaultOpen={pending.length === 0}>
              <div class="flex gap-4 mb-4 text-sm">
                <span style="color: var(--status-healthy)">{approved.length} approved</span>
                <span style="color: var(--status-error)">{rejected.length} rejected</span>
                <span style="color: var(--text-tertiary)">{deferred.length} deferred</span>
                {displaySuggestions.length > 0 && (
                  <span style="color: var(--text-secondary)">
                    Acceptance rate: {Math.round((approved.length / displaySuggestions.length) * 100)}%
                  </span>
                )}
              </div>
              <div class="space-y-3">
                {[...approved, ...rejected, ...deferred].map((sug, i) => (
                  <RecommendationCard key={sug.id || i} suggestion={sug} onAction={handleAction} updating={updating} />
                ))}
              </div>
            </Section>
          )}
        </>
      )}
    </div>
  );
}
```

**Step 2: Verify build**

Run:
```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```
Expected: Build succeeds.

**Step 3: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/dashboard/spa/src/pages/Decide.jsx
git commit -m "feat(dashboard): add Decide page with approve/reject/defer"
```

---

### Task 7: Rebuild Home page as OODA summary

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Home.jsx`

**Step 1: Rewrite Home.jsx**

Replace the entire file with the OODA summary layout. Three hero cards (Anomalies, Recommendations, Accuracy), pipeline status bar, OODA summary cards, compact Sankey.

```jsx
import { useState, useEffect } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { fetchJson } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import AriaLogo from '../components/AriaLogo.jsx';
import HeroCard from '../components/HeroCard.jsx';
import PageBanner from '../components/PageBanner.jsx';
import PipelineSankey from '../components/PipelineSankey.jsx';
import PipelineStatusBar from '../components/PipelineStatusBar.jsx';
import OodaSummaryCard from '../components/OodaSummaryCard.jsx';
import { relativeTime } from './intelligence/utils.jsx';

export default function Home() {
  const intelligence = useCache('intelligence');
  const activity = useCache('activity_summary');
  const entities = useCache('entities');
  const automations = useCache('automation_suggestions');

  const [health, setHealth] = useState(null);
  const [anomalies, setAnomalies] = useState(null);
  const [shadowAccuracy, setShadowAccuracy] = useState(null);
  const [pipeline, setPipeline] = useState(null);

  useEffect(() => {
    fetchJson('/health').then(setHealth).catch(() => {});
    fetchJson('/api/ml/anomalies').then(setAnomalies).catch(() => {});
    fetchJson('/api/shadow/accuracy').then(setShadowAccuracy).catch(() => {});
    fetchJson('/api/pipeline').then(setPipeline).catch(() => {});
  }, []);

  const loading = intelligence.loading;
  const error = intelligence.error;

  // ── Anomaly hero ──
  const anomalyItems = anomalies?.anomalies || [];
  const criticalCount = anomalyItems.filter((item) => item.severity === 'critical' || (item.score != null && item.score < -0.5)).length;
  const anomalyValue = anomalyItems.length > 0
    ? `${anomalyItems.length} detected${criticalCount > 0 ? ` (${criticalCount} critical)` : ''}`
    : 'Clear';
  const lastAnomaly = anomalyItems.length > 0 ? null : anomalies?.last_anomaly_at;
  const anomalyDelta = anomalyItems.length > 0 ? null : (lastAnomaly ? `last anomaly ${relativeTime(lastAnomaly)}` : null);

  // ── Recommendations hero ──
  const suggestions = useComputed(() => {
    if (!automations.data || !automations.data.data) return [];
    return automations.data.data.suggestions || [];
  }, [automations.data]);
  const pending = suggestions.filter((item) => (item.status || 'pending').toLowerCase() === 'pending');
  const approved = suggestions.filter((item) => (item.status || '').toLowerCase() === 'approved');
  const recValue = pending.length > 0
    ? `${pending.length} pending review`
    : 'None pending';
  const recDelta = pending.length > 0 ? null : (approved.length > 0 ? `${approved.length} approved this week` : null);

  // ── Accuracy hero (trailing 7-day average) ──
  const dailyTrend = shadowAccuracy?.daily_trend || [];
  const last7 = dailyTrend.slice(-7);
  const avg7d = last7.length > 0
    ? last7.reduce((sum, day) => sum + (day.accuracy ?? 0), 0) / last7.length
    : null;
  const accValue = avg7d != null ? `${Math.round(avg7d * 100)}%` : '\u2014';
  const accDelta = avg7d != null ? '7-day avg' : null;

  // ── Observe summary metrics ──
  const actInner = useComputed(() => {
    if (!activity.data || !activity.data.data) return null;
    return activity.data.data;
  }, [activity.data]);
  const occ = actInner ? (actInner.occupancy || null) : null;

  // ── Sankey cache data ──
  const cacheData = useComputed(() => ({
    capabilities: entities.data,
    pipeline: { data: pipeline },
    shadow_accuracy: { data: shadowAccuracy },
    activity_labels: activity.data,
  }), [entities.data, pipeline, shadowAccuracy, activity.data]);

  if (loading && !intelligence.data) {
    return (
      <div class="space-y-6">
        <div class="t-frame" data-label="aria">
          <AriaLogo className="w-24 mb-1" color="var(--text-primary)" />
          <p class="text-sm" style="color: var(--text-tertiary); font-family: var(--font-mono);">
            OODA decision dashboard — anomalies, recommendations, and system health.
          </p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <PageBanner page="HOME" subtitle="OODA decision dashboard — anomalies, recommendations, and system health." />
        <ErrorState error={error} onRetry={intelligence.refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="HOME" subtitle="OODA decision dashboard — anomalies, recommendations, and system health." />

      {/* Three hero cards */}
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <HeroCard
          value={anomalyValue}
          label="anomalies"
          delta={anomalyDelta}
          warning={anomalyItems.length > 0}
        />
        <HeroCard
          value={recValue}
          label="recommendations"
          delta={recDelta}
        />
        <HeroCard
          value={accValue}
          label="accuracy"
          delta={accDelta}
        />
      </div>

      {/* Pipeline status bar */}
      <PipelineStatusBar />

      {/* OODA summary cards */}
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <OodaSummaryCard
          title="Observe"
          subtitle="What's happening in your home right now."
          metric={occ && occ.anyone_home ? 'Home' : occ ? 'Away' : null}
          metricLabel="occupancy"
          href="#/observe"
        />
        <OodaSummaryCard
          title="Understand"
          subtitle="What's unusual, what's repeating, and why."
          metric={anomalyItems.length || 0}
          metricLabel={anomalyItems.length === 1 ? 'anomaly' : 'anomalies'}
          href="#/understand"
          accentColor={anomalyItems.length > 0 ? 'var(--status-warning)' : undefined}
        />
        <OodaSummaryCard
          title="Decide"
          subtitle="Automation recommendations to review."
          metric={pending.length}
          metricLabel="pending"
          href="#/decide"
        />
      </div>

      {/* Compact Pipeline Sankey */}
      <PipelineSankey
        moduleStatuses={health?.modules || {}}
        cacheData={cacheData}
      />
    </div>
  );
}
```

**Step 2: Verify build**

Run:
```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```
Expected: Build succeeds.

**Step 3: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/dashboard/spa/src/pages/Home.jsx
git commit -m "feat(dashboard): rebuild Home as OODA summary with hero cards"
```

---

### Task 8: Update app.jsx routes + redirects

**Files:**
- Modify: `aria/dashboard/spa/src/app.jsx`

**Step 1: Add imports for new pages**

Add these imports after the existing page imports:

```jsx
import Observe from './pages/Observe.jsx';
import Understand from './pages/Understand.jsx';
import Decide from './pages/Decide.jsx';
```

**Step 2: Add redirect component**

Add before `export default function App()`:

```jsx
/** Redirect old hash routes to new OODA destinations. */
function Redirect({ to }) {
  const { replace } = createHashHistory();
  useEffect(() => {
    replace(to);
  }, [to]);
  return null;
}
```

Wait — `createHashHistory` is called once at module scope. We need to use the existing `hashHistory`. But it's a module-level const not exported. Instead, just use `window.location.replace`:

```jsx
import { useEffect } from 'preact/hooks';

function Redirect({ to }) {
  useEffect(() => {
    window.location.hash = '#' + to;
  }, [to]);
  return null;
}
```

Actually, `useEffect` is already imported via Preact hooks. The simplest approach: use `hashHistory.replace` since it's already a module-level const. Add the Redirect component right after `const hashHistory = createHashHistory();`:

```jsx
function Redirect({ to }) {
  useEffect(() => {
    hashHistory.replace(to);
  }, [to]);
  return null;
}
```

**Step 3: Update Router with new routes and redirects**

Replace the Router block (lines 106-121) with:

```jsx
<Router history={hashHistory}>
  <Home path="/" />
  <Observe path="/observe" />
  <Understand path="/understand" />
  <Decide path="/decide" />
  <Discovery path="/discovery" />
  <Capabilities path="/capabilities" />
  <MLEngine path="/ml-engine" />
  <DataCuration path="/data-curation" />
  <Validation path="/validation" />
  <Settings path="/settings" />
  <Guide path="/guide" />
  {/* Redirects from old routes */}
  <Redirect path="/intelligence" to="/understand" />
  <Redirect path="/predictions" to="/understand" />
  <Redirect path="/patterns" to="/understand" />
  <Redirect path="/shadow" to="/understand" />
  <Redirect path="/automations" to="/decide" />
  <Redirect path="/presence" to="/observe" />
</Router>
```

Note: Old page imports (`Intelligence`, `Predictions`, `Patterns`, `Automations`, `Shadow`, `Presence`) are no longer used in routes. Remove their imports to avoid bundling dead code:

Remove these import lines:
```
import Predictions from './pages/Predictions.jsx';
import Patterns from './pages/Patterns.jsx';
import Automations from './pages/Automations.jsx';
import Intelligence from './pages/Intelligence.jsx';
import Shadow from './pages/Shadow.jsx';
import Presence from './pages/Presence.jsx';
```

**Step 4: Verify build**

Run:
```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```
Expected: Build succeeds. Bundle size should decrease slightly (removed dead page imports).

**Step 5: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/dashboard/spa/src/app.jsx
git commit -m "feat(dashboard): OODA routes + old route redirects"
```

---

### Task 9: Apply SUPERHOT effects to components

**Files:**
- Modify: `aria/dashboard/spa/src/components/HeroCard.jsx`
- Modify: `aria/dashboard/spa/src/components/ErrorState.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/AnomalyAlerts.jsx`

**Step 1: Wrap HeroCard with ShFrozen**

Add freshness state to HeroCard. The component receives a new optional `timestamp` prop. When provided, it applies the `data-sh-state` attribute based on data age.

Modify `src/components/HeroCard.jsx`:

```jsx
import { useEffect, useRef } from 'preact/hooks';
import TimeChart from './TimeChart.jsx';

// Freshness thresholds (seconds): 5min cooling, 30min frozen, 60min stale
const FRESHNESS_THRESHOLDS = { cooling: 300, frozen: 1800, stale: 3600 };

function computeFreshness(timestamp) {
  if (!timestamp) return null;
  const age = (Date.now() - new Date(timestamp).getTime()) / 1000;
  if (age > FRESHNESS_THRESHOLDS.stale) return 'stale';
  if (age > FRESHNESS_THRESHOLDS.frozen) return 'frozen';
  if (age > FRESHNESS_THRESHOLDS.cooling) return 'cooling';
  return 'fresh';
}

export default function HeroCard({ value, label, unit, delta, warning, loading, sparkData, sparkColor, timestamp }) {
  const cursorClass = loading ? 'cursor-working' : 'cursor-active';
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current) return;
    function update() {
      const state = computeFreshness(timestamp);
      if (state) {
        ref.current.setAttribute('data-sh-state', state);
      } else {
        ref.current.removeAttribute('data-sh-state');
      }
    }
    update();
    const interval = setInterval(update, 30000);
    return () => clearInterval(interval);
  }, [timestamp]);

  return (
    <div
      ref={ref}
      class={`t-frame ${cursorClass}`}
      data-label={label}
      style={warning ? 'border-left: 3px solid var(--status-warning);' : ''}
    >
      <div class="flex items-baseline gap-2" style="justify-content: space-between;">
        <div class="flex items-baseline gap-2">
          <span
            class="data-mono"
            style={`font-size: var(--type-hero); font-weight: 600; color: ${warning ? 'var(--status-warning)' : 'var(--accent)'}; line-height: 1;`}
          >
            {value ?? '\u2014'}
          </span>
          {unit && (
            <span
              class="data-mono"
              style="font-size: var(--type-headline); color: var(--text-tertiary);"
            >
              {unit}
            </span>
          )}
        </div>
        {sparkData && sparkData.length > 1 && sparkData[0].length > 1 && (
          <div style="width: 80px; height: 32px; flex-shrink: 0;">
            <TimeChart
              data={sparkData}
              series={[{ label: label || 'trend', color: sparkColor || 'var(--accent)', width: 1.5 }]}
              compact
            />
          </div>
        )}
      </div>
      {delta && (
        <div
          style="font-size: var(--type-label); color: var(--text-secondary); margin-top: 8px; font-family: var(--font-mono);"
        >
          {delta}
        </div>
      )}
    </div>
  );
}
```

**Step 2: Add ShGlitch to ErrorState**

Modify `src/components/ErrorState.jsx` to apply the glitch data attribute:

```jsx
export default function ErrorState({ error, onRetry }) {
  const message = error instanceof Error ? error.message : String(error || 'Unknown error');

  return (
    <div
      data-sh-effect="glitch"
      style="background: var(--bg-surface); border: 1px solid var(--status-error); border-left-width: 3px; border-radius: var(--radius); padding: 16px;"
    >
      <div class="flex items-start gap-3">
        <svg class="w-5 h-5 mt-0.5 shrink-0" style="color: var(--status-error);" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        <div class="flex-1">
          <p class="text-sm" style="color: var(--status-error);">{message}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            class="text-sm font-medium transition-colors"
            style="color: var(--status-error); background: var(--bg-surface-raised); border-radius: var(--radius); padding: 4px 12px;"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}
```

**Step 3: Add ShThreatPulse to anomaly items**

In `src/pages/intelligence/AnomalyAlerts.jsx`, add threat-pulse to critical anomaly items.

Add a check for critical severity in the anomaly rendering. Modify the anomaly item div (around line 48) to conditionally add `data-sh-effect="threat-pulse"`:

```jsx
<div key={i} class="t-frame p-3" data-label="anomaly"
  style="border-left: 3px solid var(--status-warning)"
  {...(a.severity === 'critical' || (a.score != null && a.score < -0.5) ? { 'data-sh-effect': 'threat-pulse' } : {})}>
```

**Step 4: Verify build**

Run:
```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```
Expected: Build succeeds. `bundle.css` contains superhot CSS selectors.

**Step 5: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/dashboard/spa/src/components/HeroCard.jsx aria/dashboard/spa/src/components/ErrorState.jsx aria/dashboard/spa/src/pages/intelligence/AnomalyAlerts.jsx
git commit -m "feat(dashboard): apply SUPERHOT effects (freshness, glitch, threat-pulse)"
```

---

### Task 10: Final build, quality gate, and verification

**Files:**
- Modify: `progress.txt`

**Step 1: Full clean build**

```bash
cd ~/Documents/projects/ha-aria/aria/dashboard/spa && rm -rf dist && npm run build
```

Expected: Build succeeds. Both `dist/bundle.js` and `dist/bundle.css` exist.

**Step 2: Verify bundle contains superhot CSS**

```bash
grep -c 'data-sh-state' ~/Documents/projects/ha-aria/aria/dashboard/spa/dist/bundle.css
```

Expected: Non-zero count (superhot CSS selectors present).

**Step 3: Verify new routes in bundle**

```bash
grep -c '/observe\|/understand\|/decide' ~/Documents/projects/ha-aria/aria/dashboard/spa/dist/bundle.js
```

Expected: Non-zero count (new routes present).

**Step 4: Verify old route redirects in bundle**

```bash
grep -c '/intelligence\|/predictions\|/automations' ~/Documents/projects/ha-aria/aria/dashboard/spa/dist/bundle.js
```

Expected: Non-zero count (redirect routes present).

**Step 5: Check success criteria**

1. Navigation: 5 primary destinations (Home, Observe, Understand, Decide, System expandable) — verify in Sidebar.jsx NAV_ITEMS
2. Hero cards: Anomalies + Recommendations + Accuracy on Home — verify in Home.jsx
3. SUPERHOT effects: ShFrozen (HeroCard), ShThreatPulse (AnomalyAlerts), ShGlitch (ErrorState, PipelineStatusBar), ShMantra (PipelineStatusBar) — verify in source
4. Freshness: HeroCard accepts timestamp prop, computes data-sh-state every 30s — verify in HeroCard.jsx
5. Old routes redirect: `/intelligence` → `/understand`, etc. — verify in app.jsx
6. Build passes: `npm run build` exits 0 — verified in step 1
7. Dark mode: superhot tokens use CSS custom properties that switch with `[data-theme="dark"]` — inherited from superhot-ui

**Step 6: Update progress.txt**

Append:
```
## Completed
- Task 0: Bootstrap + superhot-ui install ✓
- Task 1: Sidebar OODA nav + freshness helper ✓
- Task 2: PipelineStatusBar component ✓
- Task 3: OodaSummaryCard component ✓
- Task 4: Observe page ✓
- Task 5: Understand page ✓
- Task 6: Decide page ✓
- Task 7: Home page OODA rebuild ✓
- Task 8: Routes + redirects ✓
- Task 9: SUPERHOT effects ✓
- Task 10: Final quality gate ✓

All 7 success criteria met.
```

**Step 7: Final commit**

```bash
cd ~/Documents/projects/ha-aria
git add progress.txt
git commit -m "feat(dashboard): Phase 5 OODA + SUPERHOT complete"
```

---

## Notes for Implementer

### Cache Envelope Pattern

Every `useCache(name)` returns `{ data, loading, error, refetch }`. The `data` field is the **outer cache envelope**: `{ category, data: {...actual_data...}, last_updated }`. To get the actual data: `data.data`. To get freshness timestamp: `data.last_updated`.

### esbuild JSX Gotcha

**Never use `h` or `Fragment` as callback parameter names.** esbuild injects these as JSX factory functions via `preact-shim.js`. Using them as `.map(h => ...)` shadows the JSX factory and causes silent render failures. Use descriptive names like `item`, `entry`, `pat`.

### superhot-ui CSS Priority

ARIA's `--sh-*` tokens (if any are defined in `index.css`) override superhot-ui defaults. The import order ensures ARIA's tokens take precedence: Tailwind → uPlot → superhot-ui → ARIA theme tokens.

### Files NOT Changed

These pages are unchanged — just moved to the System expandable section in nav:
- `Discovery.jsx`, `Capabilities.jsx`, `MLEngine.jsx`, `DataCuration.jsx`, `Validation.jsx`, `Settings.jsx`, `Guide.jsx`

These pages are no longer in routes but kept on disk (components reused by new pages):
- `Intelligence.jsx` (source of component imports), `Patterns.jsx`, `Automations.jsx`, `Predictions.jsx`, `Shadow.jsx`, `Presence.jsx`
