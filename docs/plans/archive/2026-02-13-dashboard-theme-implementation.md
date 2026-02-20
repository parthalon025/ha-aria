# Dashboard Theme Redesign — Implementation Plan

## In Plain English

This is the task-by-task work order for adding dark mode and a cohesive visual style to every screen in ARIA's dashboard. Think of it like a painting contractor's job list -- which walls to prep first, what order to apply coats, and how five painters can work in different rooms without stepping on each other.

## Why This Exists

The dashboard had a split personality: one page looked like a command center, the rest looked like a generic business app. Adding dark mode was not just a color swap -- every component had colors baked in as fixed values instead of using a central palette. This plan coordinates a systematic replacement across 30+ files, organized so infrastructure goes first, shared components second, individual pages third, and a final quality audit catches anything missed. The five-agent structure ensures no two people edit the same file.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the ARIA dashboard from a generic light-mode SaaS look into a unified technical aesthetic matching the pixel-art logo, with dark/light theme toggle, rich ambient animations, and responsive tablet support.

**Architecture:** CSS custom properties define all colors as tokens in `:root` (light) and `[data-theme="dark"]` (dark). Components reference `var(--token)` via inline `style` attributes (Tailwind pre-built CSS can't handle new utility classes). Theme state lives in the Preact signals store with `localStorage` persistence. A 5-agent team with clean file ownership prevents merge conflicts.

**Tech Stack:** Preact + @preact/signals, CSS custom properties, esbuild bundler, no runtime dependencies added.

**Design doc:** `docs/plans/2026-02-13-dashboard-theme-redesign.md`

**Build command (run after every task that touches JSX/CSS):**
```bash
cd aria/dashboard/spa && npx esbuild src/index.jsx --bundle --outfile=dist/bundle.js --jsx-factory=h --jsx-fragment=Fragment --inject:src/preact-shim.js --loader:.jsx=jsx --minify
```

**Lint (run before commits):**
```bash
cd /home/justin/Documents/projects/ha-aria && python3 -m ruff check aria/ --fix && python3 -m ruff format aria/
```

---

## Phase 1: Infrastructure (Agent 1)

These tasks must complete before any other agent starts. They establish the theme token system, state management, and core UI chrome.

### Task 1.1: Add CSS Custom Property Tokens

**Files:**
- Modify: `aria/dashboard/spa/src/index.css` (lines 1–2, add tokens before animation section)

**Step 1: Add token definitions to top of index.css**

Replace the single `@import "tailwindcss";` line with the full token system. Keep the existing `@import` but add all CSS custom properties after it.

```css
@import "tailwindcss";

/* ── Theme tokens ── */
:root {
  /* Surfaces */
  --bg-base: #f8fafc;
  --bg-surface: #ffffff;
  --bg-surface-raised: #f1f5f9;
  --bg-inset: #e2e8f0;

  /* Borders */
  --border-primary: #cbd5e1;
  --border-subtle: #e2e8f0;
  --border-accent: #22d3ee;

  /* Text */
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-tertiary: #94a3b8;
  --text-inverse: #f8fafc;

  /* Accent (cyan from logo) */
  --accent: #22d3ee;
  --accent-dim: #0e7490;
  --accent-glow: rgba(34, 211, 238, 0.15);
  --accent-text: #0f172a;

  /* Status */
  --status-healthy: #22c55e;
  --status-warning: #f59e0b;
  --status-error: #ef4444;
  --status-waiting: #94a3b8;

  /* Ambient */
  --grid-line: rgba(15, 23, 42, 0.04);
  --scan-line: rgba(34, 211, 238, 0.06);
  --glow-spread: rgba(34, 211, 238, 0.08);

  /* Typography */
  --font-mono: ui-monospace, 'Cascadia Code', 'Fira Code', 'Menlo', monospace;

  /* Shape */
  --radius: 2px;
  --radius-lg: 4px;
}

[data-theme="dark"] {
  --bg-base: #0b1120;
  --bg-surface: #111827;
  --bg-surface-raised: #1a2332;
  --bg-inset: #0f172a;

  --border-primary: #2a3545;
  --border-subtle: #1e293b;
  --border-accent: #22d3ee;

  --text-primary: #e5e7eb;
  --text-secondary: #9ca3af;
  --text-tertiary: #6b7280;
  --text-inverse: #0f172a;

  --accent: #22d3ee;
  --accent-dim: #06b6d4;
  --accent-glow: rgba(34, 211, 238, 0.12);
  --accent-text: #0f172a;

  --status-healthy: #4ade80;
  --status-warning: #fbbf24;
  --status-error: #f87171;
  --status-waiting: #6b7280;

  --grid-line: rgba(34, 211, 238, 0.03);
  --scan-line: rgba(34, 211, 238, 0.04);
  --glow-spread: rgba(34, 211, 238, 0.06);
}

/* ── Global theme-aware base styles ── */
body {
  background: var(--bg-base);
  color: var(--text-primary);
  transition: background 0.3s ease, color 0.3s ease;
}

/* Monospace data class — apply to any element showing numeric/data values */
.data-mono {
  font-family: var(--font-mono);
}

/* Theme-aware card base */
.t-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  transition: background 0.3s ease, border-color 0.3s ease;
}
.t-card:hover {
  border-color: var(--border-primary);
}
.t-card-hover:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px var(--accent-glow);
  border-color: var(--accent);
}

/* Theme-aware section header with scan line */
.t-section-header {
  position: relative;
  overflow: hidden;
}
.t-section-header::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  animation: scan-sweep 4s ease-in-out infinite;
}

/* Status indicator styles */
.t-status {
  font-family: var(--font-mono);
  font-size: 0.6875rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 8px;
  border-radius: var(--radius);
}
.t-status-healthy { color: var(--status-healthy); border-left: 2px solid var(--status-healthy); }
.t-status-warning { color: var(--status-warning); border-left: 2px solid var(--status-warning); }
.t-status-error   { color: var(--status-error);   border-left: 2px solid var(--status-error); }
.t-status-waiting { color: var(--status-waiting); border-left: 2px solid var(--status-waiting); }

/* Button base */
.t-btn {
  border-radius: var(--radius);
  transition: transform 0.1s ease, box-shadow 0.2s ease;
}
.t-btn:active {
  transform: scale(0.97);
}
.t-btn-primary {
  background: var(--accent);
  color: var(--accent-text);
}
.t-btn-primary:hover {
  box-shadow: 0 2px 8px var(--accent-glow);
}
.t-btn-secondary {
  background: var(--bg-surface-raised);
  color: var(--text-secondary);
  border: 1px solid var(--border-subtle);
}
.t-btn-secondary:hover {
  border-color: var(--border-primary);
  color: var(--text-primary);
}

/* Input base */
.t-input {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius);
  color: var(--text-primary);
  transition: border-color 0.2s ease;
}
.t-input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 2px var(--accent-glow);
}

/* Callout / info box */
.t-callout {
  background: var(--bg-surface-raised);
  border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius);
  color: var(--text-secondary);
}
```

**Step 2: Build and verify**

Run: `cd aria/dashboard/spa && npx esbuild src/index.jsx --bundle --outfile=dist/bundle.js --jsx-factory=h --jsx-fragment=Fragment --inject:src/preact-shim.js --loader:.jsx=jsx --minify`
Expected: Build succeeds (exit 0).

**Step 3: Commit**

```bash
git add aria/dashboard/spa/src/index.css
git commit -m "feat(dashboard): add CSS custom property theme token system"
```

---

### Task 1.2: Add Theme State to Store

**Files:**
- Modify: `aria/dashboard/spa/src/store.js` (add theme signals and functions after wsMessage signal)

**Step 1: Add theme state management**

After the existing `const wsMessage = signal('');` (line 94), add:

```javascript
// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

/** Current theme: 'light' or 'dark'. */
const theme = signal(getInitialTheme());

/**
 * Determine initial theme from localStorage or system preference.
 * @returns {'light'|'dark'}
 */
function getInitialTheme() {
  const stored = localStorage.getItem('aria-theme');
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/**
 * Apply theme to document and persist.
 * @param {'light'|'dark'} t
 */
function setTheme(t) {
  theme.value = t;
  document.documentElement.setAttribute('data-theme', t);
  localStorage.setItem('aria-theme', t);
}

/** Toggle between light and dark. */
function toggleTheme() {
  setTheme(theme.value === 'dark' ? 'light' : 'dark');
}

// Apply theme on load
if (typeof document !== 'undefined') {
  document.documentElement.setAttribute('data-theme', theme.value);
}
```

**Step 2: Export theme functions**

Update the exports block at the bottom of store.js to include the new theme exports:

```javascript
export {
  cacheStore,
  getCategory,
  fetchCategory,
  wsConnected,
  wsMessage,
  connectWebSocket,
  disconnectWebSocket,
  theme,
  setTheme,
  toggleTheme,
};
```

**Step 3: Build and verify**

Run build command. Expected: Build succeeds.

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/store.js
git commit -m "feat(dashboard): add theme state with localStorage persistence"
```

---

### Task 1.3: Theme-Aware App Shell

**Files:**
- Modify: `aria/dashboard/spa/src/app.jsx` (update root div and body classes)
- Modify: `aria/dashboard/spa/dist/index.html` (remove hardcoded colors from body)

**Step 1: Update index.html body**

Change the body tag from:
```html
<body class="bg-gray-50 text-gray-900 min-h-screen">
```
To:
```html
<body class="min-h-screen">
```

The CSS custom properties on `body` in index.css now handle colors via theme tokens.

**Step 2: Update App root div**

In `app.jsx`, change the root div (line 96) from:
```jsx
<div class="min-h-screen bg-gray-50">
```
To:
```jsx
<div class="min-h-screen" style="background: var(--bg-base); color: var(--text-primary); transition: background 0.3s ease, color 0.3s ease;">
```

**Step 3: Build and verify**

Run build command. Expected: Build succeeds, app renders with theme tokens.

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/app.jsx aria/dashboard/spa/dist/index.html
git commit -m "feat(dashboard): theme-aware app shell"
```

---

### Task 1.4: Sidebar — Theme Toggle + Theme-Aware Styling

**Files:**
- Modify: `aria/dashboard/spa/src/components/Sidebar.jsx`

This is the largest single-file change. The sidebar needs:
1. Theme toggle button in the footer
2. All hardcoded Tailwind colors replaced with CSS var inline styles
3. Active nav item border-shimmer effect
4. Tablet icon-rail responsive behavior

**Step 1: Import theme state**

Add to the imports at top:
```javascript
import { theme, toggleTheme } from '../store.js';
```

**Step 2: Add theme toggle icons**

Add after the existing icon functions (before `getHashPath`):

```javascript
function SunIcon() {
  return (
    <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}
```

**Step 3: Replace the Sidebar component**

Replace the entire `export default function Sidebar()` with a theme-aware version. Key changes:
- All `bg-gray-900` → `var(--bg-surface)`
- All `text-gray-400` → `var(--text-tertiary)`
- All `bg-gray-800` → `var(--bg-surface-raised)`
- All `border-gray-800` → `var(--border-subtle)`
- Active state uses accent border instead of background
- Footer gets theme toggle button
- Tablet breakpoint gets icon-rail mode

The full replacement for the Sidebar function body uses inline styles with CSS vars throughout. The nav items use a left-accent border for active state. The footer section includes the theme toggle between version info and connection status.

Key patterns for each nav item:
```jsx
// Active state:
style="background: var(--bg-surface-raised); color: var(--text-primary); border-left: 2px solid var(--accent);"

// Inactive hover state:
onMouseEnter: style.background = 'var(--bg-surface-raised)'
onMouseLeave: style.background = 'transparent'
```

Theme toggle button in footer:
```jsx
<button
  onClick={toggleTheme}
  style="background: var(--bg-surface-raised); color: var(--text-secondary); border: 1px solid var(--border-subtle); border-radius: var(--radius); padding: 4px 8px; display: flex; align-items: center; gap: 6px; font-size: 0.75rem; cursor: pointer; transition: border-color 0.2s ease;"
  title={theme.value === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
>
  {theme.value === 'dark' ? <SunIcon /> : <MoonIcon />}
  <span>{theme.value === 'dark' ? 'Light' : 'Dark'}</span>
</button>
```

For tablet icon-rail: Add a CSS media query approach. At `md` breakpoint (768px), show the full sidebar. At a new `lg` breakpoint threshold (1024px), collapse to icon-only rail. This uses a new `sidebarExpanded` state for tablet.

**NOTE TO IMPLEMENTER:** This is the most complex single component change. Take care to preserve:
- Hash routing (getHashPath function)
- WebSocket status display
- Section headers in nav
- Mobile bottom tab bar
- Guide link in footer

**Step 4: Build and verify**

Run build command. Expected: Build succeeds, sidebar renders with theme-aware colors and toggle works.

**Step 5: Commit**

```bash
git add aria/dashboard/spa/src/components/Sidebar.jsx
git commit -m "feat(dashboard): theme-aware sidebar with dark/light toggle"
```

---

## Phase 2A: Animations (Agent 2 — runs after Phase 1)

### Task 2.1: Expand Animation System

**Files:**
- Modify: `aria/dashboard/spa/src/index.css` (animation section, starting around line 4)

**Step 1: Add new ambient animations**

Add these new keyframes and classes alongside the existing ones (keep all existing animations intact):

```css
/* ── New ambient animations ── */

@keyframes scan-sweep {
  0%   { transform: translateX(-100%); opacity: 0; }
  10%  { opacity: 1; }
  90%  { opacity: 1; }
  100% { transform: translateX(100%); opacity: 0; }
}

@keyframes grid-pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 0.8; }
}

@keyframes border-shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

@keyframes data-stream {
  0% { transform: translateY(8px); opacity: 0; }
  50% { opacity: 1; }
  100% { transform: translateY(-8px); opacity: 0; }
}

@keyframes data-refresh-flash {
  0% { box-shadow: 0 0 0 0 var(--accent-glow); }
  50% { box-shadow: 0 0 0 3px var(--accent-glow); }
  100% { box-shadow: 0 0 0 0 var(--accent-glow); }
}

@keyframes page-enter {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes btn-press {
  0% { transform: scale(1); }
  50% { transform: scale(0.97); }
  100% { transform: scale(1); }
}

@keyframes filter-underline {
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
}

/* Utility classes */
.animate-scan-sweep { animation: scan-sweep 4s ease-in-out infinite; }
.animate-grid-pulse { animation: grid-pulse 8s ease-in-out infinite; }
.animate-border-shimmer {
  background: linear-gradient(90deg, transparent, var(--accent-glow), transparent);
  background-size: 200% 100%;
  animation: border-shimmer 3s ease-in-out infinite;
}
.animate-data-stream { animation: data-stream 1.5s ease-in-out infinite; }
.animate-data-refresh { animation: data-refresh-flash 0.3s ease-out; }
.animate-page-enter { animation: page-enter 0.25s ease-out both; }

/* ── Responsive motion rules ── */
@media (max-width: 767px) {
  /* Phone: disable ambient animations, keep entrance + interactive */
  .animate-scan-sweep,
  .animate-grid-pulse,
  .animate-border-shimmer,
  .animate-data-stream,
  .animate-scan-line,
  .animate-pulse-cyan {
    animation: none;
  }
  .t-section-header::after {
    animation: none;
  }
}

@media (min-width: 768px) and (max-width: 1023px) and (orientation: portrait) {
  /* Tablet portrait: reduce ambient */
  .animate-grid-pulse,
  .animate-border-shimmer {
    animation: none;
  }
}
```

**Step 2: Update reduced-motion media query**

Add the new animation classes to the existing `@media (prefers-reduced-motion: reduce)` block:

```css
@media (prefers-reduced-motion: reduce) {
  /* ...existing list... */
  .animate-scan-sweep,
  .animate-grid-pulse,
  .animate-border-shimmer,
  .animate-data-stream,
  .animate-data-refresh,
  .animate-page-enter {
    animation: none;
  }
  .t-section-header::after {
    animation: none;
  }
}
```

**Step 3: Build and verify**

Run build command. Expected: Build succeeds.

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/index.css
git commit -m "feat(dashboard): rich ambient animation system with responsive motion rules"
```

---

## Phase 2B: Shared Components (Agent 3 — runs after Phase 1, parallel with Agent 2)

Each component gets the same treatment: replace Tailwind color classes with CSS var inline styles, use `t-card` class, apply `data-mono` class to data values, use `var(--radius)` for border radius.

### Task 3.1: Theme-Aware LoadingState

**Files:**
- Modify: `aria/dashboard/spa/src/components/LoadingState.jsx`

**Step 1: Replace component**

Replace all hardcoded Tailwind color/shape classes with theme tokens. Pattern for skeleton blocks:

```jsx
// Before:
<div class="bg-white rounded-md shadow-sm p-4">
  <div class="h-8 w-16 bg-gray-200 rounded animate-pulse mb-2" />

// After:
<div class="t-card" style="padding: 16px;">
  <div class="animate-pulse" style="height: 2rem; width: 4rem; background: var(--bg-inset); border-radius: var(--radius); margin-bottom: 0.5rem;" />
```

Apply this pattern to all skeleton variants: `stats`, `table`, `cards`, `full`.

**Step 2: Build and verify**

Run build command. Expected: Build succeeds.

**Step 3: Commit**

```bash
git add aria/dashboard/spa/src/components/LoadingState.jsx
git commit -m "feat(dashboard): theme-aware LoadingState skeletons"
```

---

### Task 3.2: Theme-Aware ErrorState

**Files:**
- Modify: `aria/dashboard/spa/src/components/ErrorState.jsx`

**Step 1: Replace component**

Replace the `bg-red-50 border border-red-200` pattern with theme-aware error styling:

```jsx
<div style="background: var(--bg-surface); border: 1px solid var(--status-error); border-left-width: 3px; border-radius: var(--radius); padding: 16px;">
  <div class="flex items-start gap-3">
    <svg style="color: var(--status-error);" ...>
    <p style="color: var(--status-error); font-size: 0.875rem;">{message}</p>
    {onRetry && (
      <button class="t-btn" style="padding: 4px 12px; font-size: 0.875rem; color: var(--status-error); background: var(--bg-surface-raised); border-radius: var(--radius);"
        onClick={onRetry}>Retry</button>
    )}
  </div>
</div>
```

**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/components/ErrorState.jsx
git commit -m "feat(dashboard): theme-aware ErrorState"
```

---

### Task 3.3: Theme-Aware StatsGrid

**Files:**
- Modify: `aria/dashboard/spa/src/components/StatsGrid.jsx`

**Step 1: Replace component**

```jsx
<div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
  {items.map((item, i) => (
    <div key={i} class="t-card" style={`padding: 16px;${item.warning ? ` border-color: var(--status-warning); border-width: 2px;` : ''}`}>
      <div class="data-mono" style={`font-size: 1.5rem; font-weight: 700; color: ${item.warning ? 'var(--status-warning)' : 'var(--accent)'};`}>
        {item.value}
      </div>
      <div style="font-size: 0.875rem; color: var(--text-tertiary); margin-top: 4px;">{item.label}</div>
    </div>
  ))}
</div>
```

**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/components/StatsGrid.jsx
git commit -m "feat(dashboard): theme-aware StatsGrid with monospace data"
```

---

### Task 3.4: Theme-Aware DataTable

**Files:**
- Modify: `aria/dashboard/spa/src/components/DataTable.jsx`

**Step 1: Replace all hardcoded colors**

Key replacements:
- `bg-white rounded-md shadow-sm` → `t-card`
- `border-b border-gray-100` → `border-bottom: 1px solid var(--border-subtle);`
- `bg-gray-50` (thead) → `background: var(--bg-surface-raised);`
- `text-gray-500` → `color: var(--text-tertiary);`
- `text-gray-700` → `color: var(--text-secondary);`
- `text-gray-900` → `color: var(--text-primary);`
- `hover:bg-gray-50` → inline `onMouseEnter/Leave` with `var(--bg-surface-raised)`
- Search input: use `t-input` class
- Pagination buttons: use `t-btn t-btn-secondary` classes
- `rounded-md` → remove (t-card handles radius)
- SortIndicator SVG colors: use `var(--text-tertiary)` for inactive, `var(--text-primary)` for active

**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/components/DataTable.jsx
git commit -m "feat(dashboard): theme-aware DataTable"
```

---

### Task 3.5: Theme-Aware StatusBadge

**Files:**
- Modify: `aria/dashboard/spa/src/components/StatusBadge.jsx`

**Step 1: Replace with monospace terminal-style badges**

Replace colored pill badges with the new `t-status` pattern:

```jsx
export default function StatusBadge({ state }) {
  const s = (state || '').toLowerCase();

  let statusClass;
  if (s === 'on' || s === 'home') statusClass = 't-status-healthy';
  else if (s === 'unavailable' || s === 'unknown') statusClass = 't-status-error';
  else if (state != null && state !== '' && !isNaN(Number(state))) statusClass = 't-status-healthy';
  else statusClass = 't-status-waiting';

  return (
    <span class={`t-status ${statusClass}`}>
      <span style={`width: 5px; height: 5px; border-radius: 50%; background: currentColor;`} />
      {state}
    </span>
  );
}
```

**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/components/StatusBadge.jsx
git commit -m "feat(dashboard): monospace terminal-style StatusBadge"
```

---

### Task 3.6: Theme-Aware DomainChart

**Files:**
- Modify: `aria/dashboard/spa/src/components/DomainChart.jsx`

**Step 1: Replace colors**

- Bar background: `var(--bg-inset)` instead of `bg-gray-100`
- Bar fill: `var(--accent)` instead of `bg-blue-500`
- Label text: `var(--text-secondary)` instead of `text-gray-600`
- Count text: `var(--text-tertiary)` + `data-mono` class
- Rounded: `var(--radius)`

**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/components/DomainChart.jsx
git commit -m "feat(dashboard): theme-aware DomainChart"
```

---

### Task 3.7: Theme-Aware AriaLogo

**Files:**
- Modify: `aria/dashboard/spa/src/components/AriaLogo.jsx`

**Step 1: Minimal change**

AriaLogo already accepts a `color` prop. No change needed to the component itself — callers will pass `var(--accent)` or specific colors. This task is just verification.

Verify the component works when color is `var(--accent)` by checking that SVG `fill` accepts CSS custom properties (it does).

**Step 2: Commit (if any changes)**

No commit needed if unchanged.

---

## Phase 3: Pages (Agent 4 — runs after Phase 2B)

Every page gets the same treatment pattern. I'll detail the complex pages fully and provide the pattern for simpler pages.

### Migration Pattern for All Pages

Every page follows this mechanical replacement:

| Find | Replace With |
|------|-------------|
| `class="bg-white rounded-md shadow-sm p-4"` | `class="t-card" style="padding: 16px;"` |
| `class="bg-white rounded-md shadow-sm p-5"` | `class="t-card" style="padding: 20px;"` |
| `class="bg-white rounded-md shadow-sm p-3"` | `class="t-card" style="padding: 12px;"` |
| `text-2xl font-bold text-gray-900` | `style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);"` |
| `text-lg font-bold text-gray-900` | `style="font-size: 1.125rem; font-weight: 700; color: var(--text-primary);"` |
| `text-lg font-semibold text-gray-900` | `style="font-size: 1.125rem; font-weight: 600; color: var(--text-primary);"` |
| `text-sm text-gray-500` | `style="font-size: 0.875rem; color: var(--text-tertiary);"` |
| `text-xs text-gray-400` | `style="font-size: 0.75rem; color: var(--text-tertiary);"` |
| `text-xs text-gray-500` | `style="font-size: 0.75rem; color: var(--text-tertiary);"` |
| `text-sm text-gray-600` | `style="font-size: 0.875rem; color: var(--text-secondary);"` |
| `text-sm text-gray-700` | `style="font-size: 0.875rem; color: var(--text-secondary);"` |
| `text-gray-900` | `color: var(--text-primary)` |
| `bg-blue-50 border border-blue-200` | `class="t-callout"` |
| `text-blue-800` | `color: var(--text-secondary)` |
| `bg-gray-100` (badges/tags) | `background: var(--bg-surface-raised)` |
| `bg-gray-200` (progress bars) | `background: var(--bg-inset)` |
| `bg-blue-500` (progress fills) | `background: var(--accent)` |
| `text-blue-600 hover:text-blue-800` (links) | `color: var(--accent); hover: var(--accent-dim)` |
| `rounded-md` | `border-radius: var(--radius)` |
| `rounded-full` (progress bars) | `border-radius: var(--radius-lg)` |
| `border-gray-100` | `border-color: var(--border-subtle)` |
| `border-gray-200` | `border-color: var(--border-primary)` |
| Numeric values in JSX | Add `class="data-mono"` |

For colored status badges like `bg-green-100 text-green-700`:
- Use inline styles: `style="background: color-mix(in srgb, var(--status-healthy) 15%, transparent); color: var(--status-healthy);"`
- Or simpler: `style="color: var(--status-healthy); font-family: var(--font-mono);"`

### Task 4.1: Home Page

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Home.jsx`

This is the most complex page. Key changes:

1. **STATUS object colors** — Replace hardcoded Tailwind classes with CSS var references
2. **StatusChip** — Use `t-status` classes
3. **LaneHeader** — Use accent colors via CSS vars
4. **PipelineNode** — Use `t-card` class, accent glow on healthy
5. **YouNode** — Use `t-callout` pattern with accent border
6. **JourneyProgress** — Progress bars use `var(--accent)`, `var(--status-healthy)`
7. **RightNowStrip** — Data values get `data-mono` class
8. **AriaLogo color** — Use `var(--text-primary)` instead of `#1f2937`

Replace the STATUS object color classes with inline style functions that return theme-aware styles. Replace all card containers with `t-card`. Add `data-mono` class to all metric values.

**Step 1: Implement all changes**
**Step 2: Build and verify**
**Step 3: Commit**

```bash
git add aria/dashboard/spa/src/pages/Home.jsx
git commit -m "feat(dashboard): theme-aware Home page with monospace data values"
```

---

### Task 4.2: Guide Page

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Guide.jsx`

The Guide page already uses a dark theme with hardcoded color constants (`DARK`, `DARK_CARD`, etc.). Replace these constants with CSS vars:

```jsx
// Remove these constants:
// const DARK = '#111827';
// const DARK_CARD = '#1a2332';
// etc.

// Replace all references:
// style={`background: ${DARK}`} → style="background: var(--bg-surface);"
// style={`background: ${DARK_CARD}`} → style="background: var(--bg-surface-raised);"
// style={`color: ${TEXT_PRIMARY}`} → style="color: var(--text-primary);"
// style={`color: ${CYAN}`} → style="color: var(--accent);"
// style={`color: ${TEXT_DIM}`} → style="color: var(--text-tertiary);"
// style={`color: ${TEXT_SECONDARY}`} → style="color: var(--text-secondary);"
// style={`border-bottom: 1px solid ${DARK_BORDER}`} → style="border-bottom: 1px solid var(--border-subtle);"
```

The Guide page will now automatically match whatever theme is active instead of being permanently dark.

**Step 1: Replace all constant references with CSS vars**
**Step 2: Remove the constant definitions (DARK, DARK_CARD, etc.)**
**Step 3: Build and verify**
**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/pages/Guide.jsx
git commit -m "feat(dashboard): theme-aware Guide page (replaces hardcoded dark constants)"
```

---

### Task 4.3: Discovery Page

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Discovery.jsx`

Apply the standard migration pattern. Key specific changes:
- Page heading: theme-aware
- StatsGrid already themed (Task 3.3)
- DomainChart already themed (Task 3.6)
- Area grid cards: `t-card`
- Entity table: DataTable already themed (Task 3.4)
- Filter dropdowns: use `t-input` class
- Domain tags (`bg-gray-800 text-white`): use `style="background: var(--bg-inset); color: var(--text-primary);"`
- Entity names in table: keep `font-mono` class for entity IDs

**Step 1: Implement changes**
**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/pages/Discovery.jsx
git commit -m "feat(dashboard): theme-aware Discovery page"
```

---

### Task 4.4: Intelligence Page + Utils

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Intelligence.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/utils.jsx`

**Intelligence.jsx:** Apply standard pattern to ShadowBrief component and page layout.

**utils.jsx — Section component:**
```jsx
export function Section({ title, subtitle, children }) {
  return (
    <section class="space-y-3">
      <div class="t-section-header">
        <h2 style="font-size: 1.125rem; font-weight: 700; color: var(--text-primary);">{title}</h2>
        {subtitle && <p style="font-size: 0.875rem; color: var(--text-tertiary);">{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}
```

**utils.jsx — Callout component:**
```jsx
export function Callout({ children }) {
  return (
    <div class="t-callout" style="padding: 12px; font-size: 0.875rem;">
      {children}
    </div>
  );
}
```

Remove the `color` parameter from Callout — all callouts use the same theme-aware style now.

**utils.jsx — confidenceColor:**
Replace Tailwind classes with inline styles:
```jsx
export function confidenceColor(conf) {
  if (conf === 'high') return 'color: var(--status-healthy);';
  if (conf === 'medium') return 'color: var(--status-warning);';
  return 'color: var(--status-error);';
}
```

**Step 1: Implement changes**
**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/pages/Intelligence.jsx aria/dashboard/spa/src/pages/intelligence/utils.jsx
git commit -m "feat(dashboard): theme-aware Intelligence page + shared utils"
```

---

### Task 4.5: Intelligence Sub-Components (Batch 1)

**Files:**
- Modify: `aria/dashboard/spa/src/pages/intelligence/LearningProgress.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/HomeRightNow.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/ActivitySection.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/DailyInsight.jsx`

Apply standard migration pattern to each. Key specific changes:
- Progress bars: `var(--accent)` fill, `var(--bg-inset)` track
- Metric values: `data-mono` class
- Phase labels: `var(--text-tertiary)` for inactive, `var(--accent)` for active
- Activity timeline event items: theme-aware backgrounds
- Activity rate values: `data-mono`

**Step 1: Implement changes across all 4 files**
**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/pages/intelligence/LearningProgress.jsx \
  aria/dashboard/spa/src/pages/intelligence/HomeRightNow.jsx \
  aria/dashboard/spa/src/pages/intelligence/ActivitySection.jsx \
  aria/dashboard/spa/src/pages/intelligence/DailyInsight.jsx
git commit -m "feat(dashboard): theme-aware Intelligence sub-components batch 1"
```

---

### Task 4.6: Intelligence Sub-Components (Batch 2)

**Files:**
- Modify: `aria/dashboard/spa/src/pages/intelligence/TrendsOverTime.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/PredictionsVsActuals.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/Baselines.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/Correlations.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/SystemStatus.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/Configuration.jsx`

Same pattern. Key specific changes:
- Sparkline/chart colors: use `var(--accent)` for fills
- Correlation matrix cells: theme-aware backgrounds with opacity
- ML model scores: `data-mono` class
- Code blocks (meta-learning JSON): `background: var(--bg-inset); color: var(--text-primary);`

**Step 1: Implement changes across all 6 files**
**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/pages/intelligence/TrendsOverTime.jsx \
  aria/dashboard/spa/src/pages/intelligence/PredictionsVsActuals.jsx \
  aria/dashboard/spa/src/pages/intelligence/Baselines.jsx \
  aria/dashboard/spa/src/pages/intelligence/Correlations.jsx \
  aria/dashboard/spa/src/pages/intelligence/SystemStatus.jsx \
  aria/dashboard/spa/src/pages/intelligence/Configuration.jsx
git commit -m "feat(dashboard): theme-aware Intelligence sub-components batch 2"
```

---

### Task 4.7: Shadow Page

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Shadow.jsx`

Key specific changes:
- Pipeline stage progress bar: `var(--accent)` fill
- Stage labels: `var(--accent)` for active, `var(--text-tertiary)` for inactive
- Accuracy numbers: `data-mono` class + color coded via `var(--status-*)`
- Daily trend bars: use status colors via CSS vars
- Prediction feed rows: `var(--bg-surface)` background, `var(--border-subtle)` dividers
- Disagreement cards: `t-card` with `border-left: 3px solid var(--status-warning)`
- TYPE_COLORS and OUTCOME_COLORS: replace Tailwind classes with inline styles using CSS vars
- Advance/Retreat buttons: `t-btn t-btn-primary` / `t-btn t-btn-secondary`

**Step 1: Implement changes**
**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/pages/Shadow.jsx
git commit -m "feat(dashboard): theme-aware Shadow Mode page"
```

---

### Task 4.8: Simple Pages (Batch)

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Capabilities.jsx`
- Modify: `aria/dashboard/spa/src/pages/Predictions.jsx`
- Modify: `aria/dashboard/spa/src/pages/Patterns.jsx`
- Modify: `aria/dashboard/spa/src/pages/Automations.jsx`

All follow the same card-grid pattern. Apply standard migration:
- All card containers: `t-card`
- Headings: CSS var text colors
- Badge pills: inline styles with CSS var status colors
- Confidence bars: `var(--accent)` fill, `var(--bg-inset)` track
- Expand/collapse buttons: `var(--accent)` text
- Code blocks: `background: var(--bg-inset); color: var(--text-primary);`
- Entity ID displays: keep `font-mono`, add theme-aware color
- Approve/Reject buttons (Automations): `var(--status-healthy)` / `var(--status-error)` backgrounds

**Step 1: Implement changes across all 4 files**
**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/pages/Capabilities.jsx \
  aria/dashboard/spa/src/pages/Predictions.jsx \
  aria/dashboard/spa/src/pages/Patterns.jsx \
  aria/dashboard/spa/src/pages/Automations.jsx
git commit -m "feat(dashboard): theme-aware Capabilities, Predictions, Patterns, Automations pages"
```

---

### Task 4.9: Settings + Data Curation Pages

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Settings.jsx`
- Modify: `aria/dashboard/spa/src/pages/DataCuration.jsx`

**Settings:** Form controls need special attention:
- Range inputs: `accent-color: var(--accent)` (CSS property)
- Toggle switches: `var(--accent)` for active state, `var(--bg-inset)` for inactive
- Text inputs: `t-input` class
- Select dropdowns: `t-input` class
- Reset buttons: `var(--accent)` text
- Category section headers: `t-card`, collapsible with theme-aware styling

**DataCuration:**
- Summary bar stats: `t-card` with theme-aware colored values
- STATUS_COLORS map: replace Tailwind classes with inline CSS var styles
- Tier section headers: `t-card`, theme-aware
- Entity rows: theme-aware borders and text
- Bulk action buttons: use `var(--status-healthy)` / `var(--status-error)` with CSS vars
- Search inputs: `t-input` class

**Step 1: Implement changes across both files**
**Step 2: Build, commit**

```bash
git add aria/dashboard/spa/src/pages/Settings.jsx aria/dashboard/spa/src/pages/DataCuration.jsx
git commit -m "feat(dashboard): theme-aware Settings and Data Curation pages"
```

---

## Phase 4: QA/Integration (Agent 5 — runs after all others)

### Task 5.1: Audit for Hardcoded Colors

**Step 1: Search for remaining hardcoded Tailwind color classes**

Run these searches and fix any hits:
```bash
cd aria/dashboard/spa/src
grep -rn 'bg-white\|bg-gray-\|bg-blue-\|bg-red-\|bg-green-\|bg-amber-\|bg-purple-\|text-gray-\|text-blue-\|text-red-\|text-green-\|text-amber-\|border-gray-\|border-blue-' --include='*.jsx' .
```

Any remaining hardcoded colors should be converted to CSS vars.

**Exceptions (keep as-is):**
- Tailwind utility classes that ARE in the pre-built bundle and don't affect theming (e.g., `w-5 h-5`, `flex`, `grid`, layout classes)
- Classes used inside `class` that don't involve colors (spacing, display, positioning)

**Step 2: Search for remaining hardcoded hex colors**

```bash
grep -rn '#[0-9a-fA-F]\{3,8\}' --include='*.jsx' . | grep -v 'import\|from\|//'
```

Any hex colors that aren't in SVG viewBox/path data should be CSS vars.

---

### Task 5.2: Build Verification

**Step 1: Clean build**

```bash
cd aria/dashboard/spa
rm -f dist/bundle.js
npx esbuild src/index.jsx --bundle --outfile=dist/bundle.js --jsx-factory=h --jsx-fragment=Fragment --inject:src/preact-shim.js --loader:.jsx=jsx --minify
```

Expected: Build succeeds with no errors or warnings.

**Step 2: Verify bundle size is reasonable**

```bash
ls -la dist/bundle.js
```

Expected: Bundle size should be within ~20% of previous size (CSS vars don't add much weight).

---

### Task 5.3: Final Commit + Summary

**Step 1: Run lint**

```bash
cd /home/justin/Documents/projects/ha-aria
python3 -m ruff check aria/ --fix && python3 -m ruff format aria/
```

**Step 2: Verify tests still pass**

```bash
.venv/bin/python -m pytest tests/ -x -q
```

(JSX changes shouldn't affect Python tests, but verify nothing broke.)

**Step 3: Final build**

```bash
cd aria/dashboard/spa && npx esbuild src/index.jsx --bundle --outfile=dist/bundle.js --jsx-factory=h --jsx-fragment=Fragment --inject:src/preact-shim.js --loader:.jsx=jsx --minify
```

**Step 4: Stage any remaining fixes and commit**

```bash
git add -A aria/dashboard/spa/
git commit -m "fix(dashboard): QA pass — remaining hardcoded colors and build verification"
```

---

## Execution Order Summary

```
Phase 1 (Agent 1: Infrastructure)
  Task 1.1: CSS tokens
  Task 1.2: Theme store
  Task 1.3: App shell
  Task 1.4: Sidebar
  ↓
Phase 2A (Agent 2: Animations)     Phase 2B (Agent 3: Shared Components)
  Task 2.1: Animation system         Task 3.1: LoadingState
                                      Task 3.2: ErrorState
                                      Task 3.3: StatsGrid
                                      Task 3.4: DataTable
                                      Task 3.5: StatusBadge
                                      Task 3.6: DomainChart
                                      Task 3.7: AriaLogo
  ↓                                   ↓
Phase 3 (Agent 4: Pages)
  Task 4.1: Home
  Task 4.2: Guide
  Task 4.3: Discovery
  Task 4.4: Intelligence + utils
  Task 4.5: Intel sub-components batch 1
  Task 4.6: Intel sub-components batch 2
  Task 4.7: Shadow
  Task 4.8: Simple pages batch
  Task 4.9: Settings + DataCuration
  ↓
Phase 4 (Agent 5: QA)
  Task 5.1: Audit hardcoded colors
  Task 5.2: Build verification
  Task 5.3: Final lint + test + commit
```

Total: 19 tasks across 5 agents.
