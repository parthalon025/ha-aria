# ARIA UI Redesign — "Living Terminal Intelligence" Implementation Plan

## In Plain English

This is the step-by-step construction plan for rebuilding ARIA's dashboard appearance. It is organized like a home renovation -- foundation work first (color system, navigation), then structural changes (reusable components, chart upgrades), then room-by-room finishing (each page migrated to the new look), and finally a walkthrough inspection.

## Why This Exists

A design document describes what the finished product should look like; an implementation plan describes how to build it without breaking anything along the way. The ARIA dashboard has 13 pages and dozens of shared components, all currently using hardcoded colors and inconsistent styling. Changing everything at once would be chaotic. This plan sequences the work so each phase builds on the last, every step is independently verifiable, and the dashboard stays functional throughout the transition.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform ARIA's dashboard from a generic cyan-on-gray Preact SPA into a living terminal intelligence UI with HA-aligned colors, ASCII bracket aesthetic, context-based typography, three-tier animations, and purpose-built phone/tablet/desktop layouts.

**Architecture:** All changes are frontend-only (Preact SPA at `aria/dashboard/spa/`). CSS custom properties are the primary styling mechanism — Tailwind v4 is used for layout utilities only. No backend changes needed. The build produces `dist/bundle.css` (Tailwind + custom CSS) and `dist/bundle.js` (esbuild-bundled Preact).

**Tech Stack:** Preact 10 + @preact/signals, Tailwind CSS v4 (@tailwindcss/cli), esbuild, uPlot (new dependency for charts)

**Design doc:** `docs/plans/2026-02-13-aria-ui-redesign-design.md`

**Testing note:** This project has no frontend test suite. Verification is: (1) build succeeds, (2) visual check in browser at `http://127.0.0.1:8001/ui/`, (3) no console errors. Each task includes a build + verify step.

**Build commands:**
```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa
npm run build
# Then restart to serve new files:
systemctl --user restart aria-hub
```

---

## Phase 1: Design Foundation (CSS Tokens)

No component changes. Pure CSS token updates in `index.css`.

### Task 1: Update color palette to HA-aligned tokens

**Files:**
- Modify: `aria/dashboard/spa/src/index.css:3-82`

**Step 1: Replace `:root` light theme tokens**

Replace the existing `:root` block (lines 4-49) with:

```css
:root {
  /* Surfaces — HA-aligned neutral grays */
  --bg-base: #FAFAFA;
  --bg-surface: #FFFFFF;
  --bg-surface-raised: #F5F5F5;
  --bg-inset: #F0F0F0;
  --bg-terminal: #F5F5F0;

  /* Borders */
  --border-primary: #9E9E9E;
  --border-subtle: #E0E0E0;
  --border-accent: #0891B2;

  /* Text — HA-aligned */
  --text-primary: #212121;
  --text-secondary: #727272;
  --text-tertiary: #9E9E9E;
  --text-inverse: #FAFAFA;
  --text-accent: #0891B2;

  /* Accent — ARIA cyan (brand identity) */
  --accent: #0891B2;
  --accent-dim: #0E7490;
  --accent-glow: rgba(8, 145, 178, 0.10);
  --accent-text: #FFFFFF;

  /* Accent warm — HA orange (human action / attention) */
  --accent-warm: #FF9800;
  --accent-warm-dim: #F57C00;
  --accent-warm-glow: rgba(255, 152, 0, 0.10);
  --accent-warm-text: #FFFFFF;

  /* Status — HA exact values */
  --status-healthy: #0F9D58;
  --status-warning: #FF9800;
  --status-error: #DB4437;
  --status-active: #FDD835;
  --status-waiting: #9E9E9E;

  /* Ambient */
  --grid-line: rgba(33, 33, 33, 0.04);
  --scan-line: rgba(8, 145, 178, 0.08);
  --glow-spread: rgba(8, 145, 178, 0.06);

  /* Depth */
  --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04);
  --card-shadow-hover: 0 4px 12px rgba(0, 0, 0, 0.10), 0 2px 4px rgba(0, 0, 0, 0.06);

  /* Typography */
  --font-mono: ui-monospace, 'Cascadia Code', 'Fira Code', 'Menlo', monospace;
  --type-hero: 2.5rem;
  --type-headline: 1.25rem;
  --type-body: 0.9375rem;
  --type-data: 1rem;
  --type-label: 0.6875rem;
  --type-micro: 0.625rem;

  /* Shape */
  --radius: 4px;
  --radius-lg: 6px;
}
```

**Step 2: Replace `[data-theme="dark"]` block**

Replace lines 51-82 with:

```css
[data-theme="dark"] {
  --bg-base: #111318;
  --bg-surface: #1C1C1E;
  --bg-surface-raised: #2C2C2E;
  --bg-inset: #0D0D0F;
  --bg-terminal: #0A0E14;

  --border-primary: #48484A;
  --border-subtle: #2C2C2E;
  --border-accent: #22D3EE;

  --text-primary: #E8EAED;
  --text-secondary: #9AA0A6;
  --text-tertiary: #5F6368;
  --text-inverse: #111318;
  --text-accent: #22D3EE;

  --accent: #22D3EE;
  --accent-dim: #0891B2;
  --accent-glow: rgba(34, 211, 238, 0.12);
  --accent-text: #111318;

  --accent-warm: #FFB74D;
  --accent-warm-dim: #FF9800;
  --accent-warm-glow: rgba(255, 183, 77, 0.10);
  --accent-warm-text: #111318;

  --status-healthy: #4ADE80;
  --status-warning: #FFC107;
  --status-error: #F87171;
  --status-active: #FDE68A;
  --status-waiting: #5F6368;

  --grid-line: rgba(34, 211, 238, 0.03);
  --scan-line: rgba(34, 211, 238, 0.05);
  --glow-spread: rgba(34, 211, 238, 0.06);

  --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.3), 0 1px 2px rgba(0, 0, 0, 0.2);
  --card-shadow-hover: 0 4px 16px rgba(0, 0, 0, 0.4), 0 2px 6px rgba(0, 0, 0, 0.2);
}
```

**Step 3: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

Expected: Build succeeds. Open dashboard — colors shift to HA-neutral grays with same cyan accent.

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/index.css
git commit -m "feat(dashboard): migrate color tokens to HA-aligned palette with warm accent"
```

### Task 2: Fix hardcoded colors in components

**Files:**
- Modify: `aria/dashboard/spa/src/pages/intelligence/TrendsOverTime.jsx:76-85`
- Modify: `aria/dashboard/spa/src/pages/intelligence/utils.jsx:4-6`

**Step 1: Replace hardcoded bar chart colors in TrendsOverTime.jsx**

In `TrendsOverTime.jsx`, replace the hardcoded hex colors with CSS variable references:

- Line 76: `color="#3b82f6"` → `color="var(--accent)"`
- Line 77: `color="#f59e0b"` → `color="var(--accent-warm)"`
- Line 78: `color="#ef4444"` → `color="var(--status-error)"`
- Line 84: `color="#6366f1"` → `color="var(--accent-dim)"`
- Line 85: `color="#f43f5e"` → `color="var(--status-error)"`

**Step 2: Replace hardcoded rgba in utils.jsx**

In `utils.jsx`, replace `confidenceColor()` (lines 3-7):

```javascript
export function confidenceColor(conf) {
  if (conf === 'high') return 'background: var(--accent-glow); color: var(--status-healthy);';
  if (conf === 'medium') return 'background: var(--accent-warm-glow); color: var(--status-warning);';
  return 'background: rgba(var(--status-error), 0.15); color: var(--status-error);';
}
```

Note: The `rgba()` with CSS variable won't work directly. Use a new token `--status-error-glow: rgba(219, 68, 55, 0.15)` in `:root` and `--status-error-glow: rgba(248, 113, 113, 0.15)` in dark theme. Then reference `background: var(--status-error-glow)`.

**Step 3: Grep for any remaining hardcoded colors**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa/src
grep -rn '#[0-9a-fA-F]\{6\}' --include='*.jsx' --include='*.js' | grep -v 'node_modules'
```

Fix any found. Common ones may be in `Shadow.jsx` (purple gradient).

**Step 4: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 5: Commit**

```bash
git add aria/dashboard/spa/src/
git commit -m "fix(dashboard): replace all hardcoded colors with CSS variable references"
```

---

## Phase 2: ASCII Frame System

### Task 3: Add ASCII bracket frame CSS classes

**Files:**
- Modify: `aria/dashboard/spa/src/index.css` (append after `.t-callout` block, ~line 200)

**Step 1: Add terminal frame CSS**

Append to `index.css` before the animations section:

```css
/* ── ASCII terminal frame system ── */

/* Card with ASCII bracket corners */
.t-frame {
  position: relative;
  background: var(--bg-surface);
  border: none;
  padding: 16px 20px;
  box-shadow: var(--card-shadow);
  transition: box-shadow 0.2s ease;
}
.t-frame::before {
  content: attr(data-label);
  display: block;
  font-family: var(--font-mono);
  font-size: var(--type-label);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-tertiary);
  padding-bottom: 12px;
  margin-bottom: 12px;
  border-bottom: 1px solid var(--border-subtle);
}
.t-frame::after {
  content: attr(data-footer);
  display: block;
  font-family: var(--font-mono);
  font-size: var(--type-micro);
  color: var(--text-tertiary);
  padding-top: 12px;
  margin-top: 12px;
  border-top: 1px solid var(--border-subtle);
  text-align: right;
}
/* Hide pseudo-elements when no data attribute set */
.t-frame:not([data-label])::before { display: none; }
.t-frame:not([data-footer])::after { display: none; }

.t-frame:hover {
  box-shadow: var(--card-shadow-hover);
}

/* Inline bracket labels */
.t-bracket {
  font-family: var(--font-mono);
  font-size: var(--type-label);
  font-weight: 500;
  color: var(--text-secondary);
  letter-spacing: 0.02em;
}
.t-bracket::before { content: '['; color: var(--text-tertiary); }
.t-bracket::after  { content: ']'; color: var(--text-tertiary); }

/* Angle bracket nav */
.t-nav-bracket {
  font-family: var(--font-mono);
  font-size: var(--type-label);
  color: var(--text-tertiary);
  letter-spacing: 0.03em;
}
.t-nav-bracket a {
  color: var(--text-accent);
  text-decoration: none;
}
.t-nav-bracket a:hover {
  text-decoration: underline;
}

/* Terminal background texture */
.t-terminal-bg {
  background-color: var(--bg-terminal);
  background-image:
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      var(--scan-line) 2px,
      var(--scan-line) 4px
    );
  background-size: 100% 4px;
}

/* ── Phone: simplified frames ── */
@media (max-width: 639px) {
  .t-frame {
    padding: 12px 16px;
    border-radius: 0;
  }
  /* Top/bottom borders only — no side styling waste on narrow screens */
  .t-frame::before {
    padding-bottom: 8px;
    margin-bottom: 8px;
  }
  .t-frame::after {
    padding-top: 8px;
    margin-top: 8px;
  }
  .t-terminal-bg {
    background-image: none; /* No scan lines on phone */
  }
}

/* ── Tablet: full frames, no terminal texture except headers ── */
@media (min-width: 640px) and (max-width: 1023px) {
  .t-terminal-bg:not(.t-section-header) {
    background-image: none;
  }
}
```

**Step 2: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

Expected: Build succeeds. No visual changes yet (classes not applied to components).

**Step 3: Commit**

```bash
git add aria/dashboard/spa/src/index.css
git commit -m "feat(dashboard): add ASCII bracket frame CSS system"
```

### Task 4: Add cursor animation CSS

**Files:**
- Modify: `aria/dashboard/spa/src/index.css` (append after frame system)

**Step 1: Add cursor keyframes and classes**

Append to `index.css`:

```css
/* ── Cursor system — state indicators + expand/collapse affordance ── */

@keyframes cursor-blink {
  0%, 49.9% { opacity: 1; }
  50%, 100% { opacity: 0; }
}

/* Block cursor: active/expanded state (1s steady blink) */
.cursor-active::after {
  content: '\2588';  /* █ */
  font-family: var(--font-mono);
  color: var(--accent);
  margin-left: 8px;
  animation: cursor-blink 1s step-end infinite;
}

/* Half cursor: processing/loading state (0.5s fast blink) */
.cursor-working::after {
  content: '\258A';  /* ▊ */
  font-family: var(--font-mono);
  color: var(--accent-warm);
  margin-left: 8px;
  animation: cursor-blink 0.5s step-end infinite;
}

/* Underscore cursor: idle/collapsed state (2s slow blink) */
.cursor-idle::after {
  content: '_';
  font-family: var(--font-mono);
  color: var(--text-tertiary);
  margin-left: 8px;
  animation: cursor-blink 2s step-end infinite;
}

/* Phone: smaller cursors, only active + working */
@media (max-width: 639px) {
  .cursor-active::after,
  .cursor-working::after,
  .cursor-idle::after {
    font-size: 0.75em;
  }
  .cursor-idle::after {
    display: none;
  }
}

/* Reduced motion: static symbols, no blink */
@media (prefers-reduced-motion: reduce) {
  .cursor-active::after,
  .cursor-working::after,
  .cursor-idle::after {
    animation: none;
    opacity: 1;
  }
}
```

**Step 2: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 3: Commit**

```bash
git add aria/dashboard/spa/src/index.css
git commit -m "feat(dashboard): add cursor animation system (active/working/idle)"
```

---

## Phase 3: Core Components

### Task 5: Create CollapsibleSection component

**Files:**
- Create: `aria/dashboard/spa/src/components/CollapsibleSection.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/utils.jsx` (update `Section` export)

**Step 1: Create the CollapsibleSection component**

Create `aria/dashboard/spa/src/components/CollapsibleSection.jsx`:

```jsx
import { useState, useEffect, useRef } from 'preact/hooks';

/**
 * ASCII-framed collapsible section with cursor state affordance.
 *
 * @param {Object} props
 * @param {string} props.title - Section header text
 * @param {string} [props.subtitle] - Optional subtitle
 * @param {string} [props.summary] - Text shown when collapsed (e.g. "12 pairs")
 * @param {boolean} [props.defaultOpen=true] - Initial expanded state
 * @param {boolean} [props.loading=false] - Show working cursor
 * @param {'phone'|'tablet'|'desktop'} [props.collapseBelow] - Auto-collapse on smaller devices
 * @param {import('preact').ComponentChildren} props.children
 */
export default function CollapsibleSection({
  title,
  subtitle,
  summary,
  defaultOpen = true,
  loading = false,
  children,
}) {
  const [open, setOpen] = useState(defaultOpen);
  const contentRef = useRef(null);

  // Determine cursor class
  let cursorClass = 'cursor-idle';
  if (loading) cursorClass = 'cursor-working';
  else if (open) cursorClass = 'cursor-active';

  function toggle() {
    if (!loading) setOpen(!open);
  }

  return (
    <section class="space-y-0">
      {/* Header — clickable to toggle */}
      <button
        type="button"
        onClick={toggle}
        class={`t-section-header ${cursorClass} w-full text-left flex items-center justify-between`}
        style="padding: 8px 0; cursor: pointer; background: none; border: none; border-bottom: 1px solid var(--border-subtle);"
        aria-expanded={open}
      >
        <div class="flex-1">
          <h2
            class="font-bold"
            style={`font-size: var(--type-headline); color: var(--text-primary); font-family: var(--font-mono);`}
          >
            {title}
          </h2>
          {subtitle && !open && (
            <span
              class="t-bracket"
              style="margin-left: 8px;"
            >
              {summary || subtitle}
            </span>
          )}
          {subtitle && open && (
            <p style={`font-size: var(--type-label); color: var(--text-tertiary); margin-top: 2px;`}>
              {subtitle}
            </p>
          )}
        </div>
      </button>

      {/* Content — collapses */}
      {open && (
        <div
          ref={contentRef}
          class="animate-page-enter"
          style="padding-top: 12px;"
        >
          {children}
        </div>
      )}
    </section>
  );
}
```

**Step 2: Update utils.jsx Section to use CollapsibleSection**

In `aria/dashboard/spa/src/pages/intelligence/utils.jsx`, replace the `Section` function (lines 31-41):

```jsx
import CollapsibleSection from '../../components/CollapsibleSection.jsx';

export function Section({ title, subtitle, summary, defaultOpen = true, loading, children }) {
  return (
    <CollapsibleSection
      title={title}
      subtitle={subtitle}
      summary={summary}
      defaultOpen={defaultOpen}
      loading={loading}
    >
      {children}
    </CollapsibleSection>
  );
}
```

**Step 3: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

Open Intelligence page — sections should now have monospace headers with blinking cursors. Click a header to collapse/expand.

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/components/CollapsibleSection.jsx aria/dashboard/spa/src/pages/intelligence/utils.jsx
git commit -m "feat(dashboard): add CollapsibleSection with cursor state affordance"
```

### Task 6: Create HeroCard component and refactor StatsGrid

**Files:**
- Create: `aria/dashboard/spa/src/components/HeroCard.jsx`
- Modify: `aria/dashboard/spa/src/components/StatsGrid.jsx`

**Step 1: Create HeroCard component**

Create `aria/dashboard/spa/src/components/HeroCard.jsx`:

```jsx
/**
 * Hero metric card — the single most important number on the page.
 * Large monospace value, label, optional delta and sparkline data.
 *
 * @param {Object} props
 * @param {string|number} props.value - The primary metric value
 * @param {string} props.label - What the metric represents
 * @param {string} [props.unit] - Unit suffix (e.g. "W", "%")
 * @param {string} [props.delta] - Change description (e.g. "▾ 12% from yesterday")
 * @param {boolean} [props.warning] - Apply warning styling
 * @param {boolean} [props.loading] - Show working cursor
 */
export default function HeroCard({ value, label, unit, delta, warning, loading }) {
  const cursorClass = loading ? 'cursor-working' : 'cursor-active';

  return (
    <div
      class={`t-frame ${cursorClass}`}
      data-label={label}
      style={warning ? `border-left: 3px solid var(--status-warning);` : ''}
    >
      <div class="flex items-baseline gap-2">
        <span
          class="data-mono"
          style={`
            font-size: var(--type-hero);
            font-weight: 600;
            color: ${warning ? 'var(--status-warning)' : 'var(--accent)'};
            line-height: 1;
          `}
        >
          {value ?? '—'}
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
      {delta && (
        <div
          style={`
            font-size: var(--type-label);
            color: var(--text-secondary);
            margin-top: 8px;
            font-family: var(--font-mono);
          `}
        >
          {delta}
        </div>
      )}
    </div>
  );
}
```

**Step 2: Refactor StatsGrid to use t-frame and bracket labels**

Replace `aria/dashboard/spa/src/components/StatsGrid.jsx`:

```jsx
/**
 * Responsive stats grid with ASCII bracket labels.
 * Phone: 2-col. Tablet: 3-col. Desktop: follows parent grid.
 *
 * @param {{ items: Array<{ label: string, value: string|number, warning?: boolean }> }} props
 */
export default function StatsGrid({ items }) {
  if (!items || items.length === 0) return null;

  return (
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {items.map((item, i) => (
        <div
          key={i}
          class="t-frame"
          style={`padding: 12px 16px;${item.warning ? ' border-left: 3px solid var(--status-warning);' : ''}`}
        >
          <div
            class="data-mono"
            style={`font-size: var(--type-data); font-weight: 600; color: ${item.warning ? 'var(--status-warning)' : 'var(--accent)'};`}
          >
            {item.value}
          </div>
          <div
            class="t-bracket"
            style="margin-top: 4px;"
          >
            {item.label}
          </div>
        </div>
      ))}
    </div>
  );
}
```

**Step 3: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/components/HeroCard.jsx aria/dashboard/spa/src/components/StatsGrid.jsx
git commit -m "feat(dashboard): add HeroCard component, refactor StatsGrid with bracket labels"
```

---

## Phase 4: Responsive Navigation

### Task 7: Rewrite Sidebar with phone/tablet/desktop layouts

**Files:**
- Modify: `aria/dashboard/spa/src/components/Sidebar.jsx`
- Modify: `aria/dashboard/spa/src/app.jsx:100` (adjust content margins)

**Step 1: Define phone tab items (5 primary + "More")**

At the top of `Sidebar.jsx`, add a constant for the 5 phone tabs:

```javascript
const PHONE_TABS = [
  { path: '/', label: 'Home', icon: GridIcon },
  { path: '/intelligence', label: 'Intel', icon: BrainIcon },
  { path: '/predictions', label: 'Predict', icon: TrendingUpIcon },
  { path: '/shadow', label: 'Shadow', icon: EyeIcon },
  { path: '/more', label: 'More', icon: MoreIcon },
];
```

Add a `MoreIcon` SVG (three dots / ellipsis).

**Step 2: Rewrite the Sidebar component**

The sidebar renders three variants based on screen width:

1. **Phone (`< 640px`):** Bottom tab bar with 5 tabs + slide-up "More" sheet
2. **Tablet (`640px - 1023px`):** 56px icon rail on left, expandable to 240px
3. **Desktop (`1024px+`):** Full 240px sidebar (similar to current, with terminal texture)

This is a large component rewrite. Key changes:
- Use `window.matchMedia` or CSS classes to show/hide the right variant
- Phone "More" button opens a slide-up panel listing all remaining nav items
- Tablet rail uses icon-only with tooltip, tap to expand
- Desktop keeps current layout but adds `t-terminal-bg` to sidebar background
- Add ARIA labels to all icon-only buttons

Full implementation: Replace entire Sidebar component body. Use CSS classes `hidden sm:hidden md:hidden lg:flex` etc. to show the right variant at each breakpoint.

Mobile tab bar needs 44x44px minimum touch targets. Add `min-w-[44px] min-h-[44px]` to tab items.

**Step 3: Update app.jsx content margins**

In `app.jsx` line 100, update the main content offset:

```jsx
<main class="lg:ml-60 sm:ml-14 pb-16 sm:pb-0 min-h-screen">
```

- `pb-16` — bottom padding for phone tab bar
- `sm:ml-14` — offset for tablet 56px rail
- `lg:ml-60` — offset for desktop 240px sidebar
- `sm:pb-0` — no bottom padding on tablet+ (no bottom bar)

**Step 4: Build and verify at all three widths**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

Open dashboard. Resize browser to check:
- `< 640px`: bottom tab bar, 5 icons + labels
- `640-1023px`: left icon rail
- `1024px+`: full sidebar with labels

**Step 5: Commit**

```bash
git add aria/dashboard/spa/src/components/Sidebar.jsx aria/dashboard/spa/src/app.jsx
git commit -m "feat(dashboard): responsive navigation — phone tabs, tablet rail, desktop sidebar"
```

---

## Phase 5: Animation Tier System

### Task 8: Refactor animations into three tiers

**Files:**
- Modify: `aria/dashboard/spa/src/index.css` (animation section, lines 202-394)

**Step 1: Reorganize keyframes by tier**

Replace the entire animation section (from `/* ── New ambient animations ──` to end of file) with a restructured version organized by tier:

```css
/* ═══════════════════════════════════════════════════════
   TIER 1 — Ambient (Processing Hum)
   "ARIA is alive and working." Peripheral vision only.
   GPU-composited: opacity + background-position only.
   ═══════════════════════════════════════════════════════ */

@keyframes scan-sweep { /* existing */ }
@keyframes grid-pulse { /* existing */ }
@keyframes border-shimmer { /* existing */ }
@keyframes data-stream { /* existing */ }
@keyframes scan-line { /* existing */ }
@keyframes pulse-cyan { /* existing */ }

/* Tier 1 classes */
.t1-scan-sweep { animation: scan-sweep 6s ease-in-out infinite; }
.t1-grid-pulse { animation: grid-pulse 8s ease-in-out infinite; }
.t1-border-shimmer { /* existing definition */ }
.t1-data-stream { animation: data-stream 2s ease-in-out infinite; }
.t1-scan-line { animation: scan-line 3s linear infinite; }
.t1-pulse-ring { animation: pulse-cyan 3s ease-in-out infinite; }

/* ═══════════════════════════════════════════════════════
   TIER 2 — Metric Refresh (Data is Alive)
   "Values are real-time, not screenshots."
   Triggered by data updates.
   ═══════════════════════════════════════════════════════ */

@keyframes typewriter-in {
  0% { opacity: 0; transform: translateY(-4px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes tick-flash {
  0% { background-color: var(--accent-glow); }
  100% { background-color: transparent; }
}
@keyframes bar-grow {
  from { transform: scaleY(0); }
  to { transform: scaleY(1); }
}

.t2-typewriter { animation: typewriter-in 0.3s ease-out; }
.t2-tick-flash { animation: tick-flash 0.4s ease-out; }
.t2-bar-grow { animation: bar-grow 0.5s ease-out; transform-origin: bottom; }

/* ═══════════════════════════════════════════════════════
   TIER 3 — Status Alert (Human Attention Needed)
   "Something changed that matters." Strongest animation.
   Auto-expires after 30s (JS handles removal).
   ═══════════════════════════════════════════════════════ */

@keyframes orange-pulse {
  0%, 100% { box-shadow: 0 0 0 0 var(--accent-warm-glow); }
  50% { box-shadow: 0 0 0 6px var(--accent-warm-glow); }
}
@keyframes border-alert {
  0% { border-left-color: var(--accent-warm); }
  50% { border-left-color: var(--accent); }
  100% { border-left-color: var(--border-subtle); }
}
@keyframes badge-appear {
  0% { opacity: 0; transform: translateX(12px); }
  70% { transform: translateX(-2px); }
  100% { opacity: 1; transform: translateX(0); }
}
@keyframes counter-bump {
  0% { transform: scale(1); }
  50% { transform: scale(1.2); }
  100% { transform: scale(1); }
}

.t3-orange-pulse { animation: orange-pulse 0.6s ease-out 3; }
.t3-border-alert { animation: border-alert 1s ease-out forwards; border-left: 3px solid var(--accent-warm); }
.t3-badge-appear { animation: badge-appear 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) both; }
.t3-counter-bump { animation: counter-bump 0.3s ease-out; }

/* ═══════════════════════════════════════════════════════
   Shared: page transitions, stagger, delays
   ═══════════════════════════════════════════════════════ */

@keyframes page-enter { /* existing */ }
@keyframes fade-in-up { /* existing */ }
@keyframes fade-in { /* existing */ }

.animate-page-enter { animation: page-enter 0.25s ease-out both; }
.animate-fade-in-up { animation: fade-in-up 0.4s ease-out both; }
.animate-fade-in { animation: fade-in 0.5s ease-out both; }

/* Stagger children */
.stagger-children > * { opacity: 0; animation: fade-in-up 0.4s ease-out forwards; }
/* ... existing nth-child delays ... */

/* Delay utilities */
.delay-100 { animation-delay: 0.1s; }
/* ... etc ... */

/* ═══════════════════════════════════════════════════════
   Responsive animation budget
   ═══════════════════════════════════════════════════════ */

/* Phone: Tier 1 OFF, Tier 2 partial, Tier 3 partial */
@media (max-width: 639px) {
  .t1-scan-sweep, .t1-grid-pulse, .t1-border-shimmer,
  .t1-data-stream, .t1-scan-line, .t1-pulse-ring {
    animation: none;
  }
  .t-section-header::after { animation: none; }
  .t3-border-alert { animation: none; border-left-color: var(--accent-warm); }
}

/* Tablet: Tier 1 reduced */
@media (min-width: 640px) and (max-width: 1023px) {
  .t1-grid-pulse, .t1-border-shimmer, .t1-data-stream {
    animation: none;
  }
}

/* Reduced motion: ALL tiers off except Tier 3 color-only */
@media (prefers-reduced-motion: reduce) {
  [class*="t1-"], [class*="t2-"],
  .animate-page-enter, .animate-fade-in-up, .animate-fade-in,
  .stagger-children > * {
    animation: none !important;
    opacity: 1 !important;
  }
  .t3-orange-pulse { animation: none; box-shadow: 0 0 0 4px var(--accent-warm-glow); }
  .t3-border-alert { animation: none; border-left-color: var(--accent-warm); }
  .t3-badge-appear { animation: none; opacity: 1; }
  .t-section-header::after { animation: none; }
}
```

Keep all existing keyframe definitions (scan-sweep, grid-pulse, etc.) — just rename the utility classes from `.animate-*` to `.t1-*`, `.t2-*`, `.t3-*` and add the new tier-specific keyframes.

**Step 2: Update any component references to old animation class names**

Search for `animate-scan-sweep`, `animate-grid-pulse`, etc. and replace with the new tier-prefixed names. Components that use these:
- Section header (`.t-section-header::after`) — unchanged, still CSS-based
- Home page pipeline nodes — update class names
- Various decorative elements

```bash
grep -rn 'animate-scan-sweep\|animate-grid-pulse\|animate-border-shimmer\|animate-data-stream\|animate-pulse-cyan\|animate-pulse-amber' aria/dashboard/spa/src/
```

Update each reference to use `t1-` prefix.

**Step 3: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/
git commit -m "feat(dashboard): restructure animations into 3-tier system with responsive budget"
```

---

## Phase 6: Chart Upgrade

### Task 9: Install uPlot and create chart wrapper

**Files:**
- Modify: `aria/dashboard/spa/package.json` (add uPlot dependency)
- Create: `aria/dashboard/spa/src/components/TimeChart.jsx`

**Step 1: Install uPlot**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm install uplot
```

**Step 2: Create TimeChart wrapper component**

Create `aria/dashboard/spa/src/components/TimeChart.jsx`:

```jsx
import { useRef, useEffect } from 'preact/hooks';

/**
 * uPlot wrapper for ARIA time-series charts.
 * Uses CSS variables for theming.
 *
 * @param {Object} props
 * @param {Array} props.data - uPlot data format: [timestamps[], series1[], series2[], ...]
 * @param {Array<{label: string, color: string, width?: number}>} props.series - Series config
 * @param {number} [props.height=120] - Chart height in px
 * @param {string} [props.className] - Additional CSS classes
 */
export default function TimeChart({ data, series, height = 120, className }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !data || data.length === 0) return;

    // Lazy-load uPlot to keep initial bundle small
    import('uplot').then(({ default: uPlot }) => {
      // Get computed CSS variables for theme-aware colors
      const styles = getComputedStyle(document.documentElement);
      const textColor = styles.getPropertyValue('--text-tertiary').trim();
      const gridColor = styles.getPropertyValue('--border-subtle').trim();

      const opts = {
        width: containerRef.current.clientWidth,
        height,
        cursor: { show: true, drag: { x: false, y: false } },
        legend: { show: false },
        axes: [
          {
            stroke: textColor,
            grid: { stroke: gridColor, width: 1 },
            font: `${10}px ${styles.getPropertyValue('--font-mono').trim()}`,
            ticks: { stroke: gridColor, width: 1 },
          },
          {
            stroke: textColor,
            grid: { stroke: gridColor, width: 1 },
            font: `${10}px ${styles.getPropertyValue('--font-mono').trim()}`,
            ticks: { stroke: gridColor, width: 1 },
            size: 50,
          },
        ],
        series: [
          {}, // x-axis (timestamps)
          ...series.map((s) => ({
            label: s.label,
            stroke: s.color,
            width: s.width || 2,
            fill: s.color + '15', // 15 = ~8% opacity hex
          })),
        ],
      };

      // Destroy previous chart if exists
      if (chartRef.current) {
        chartRef.current.destroy();
      }

      chartRef.current = new uPlot(opts, data, containerRef.current);
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [data, series, height]);

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(() => {
      if (chartRef.current && containerRef.current) {
        chartRef.current.setSize({
          width: containerRef.current.clientWidth,
          height,
        });
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [height]);

  return <div ref={containerRef} class={className || ''} />;
}
```

**Step 3: Add uPlot CSS import**

In `index.css`, add at the top (after `@import "tailwindcss";`):

```css
@import "uplot/dist/uPlot.min.css";
```

Note: If the CSS import doesn't resolve through Tailwind CLI, copy the uPlot CSS file to `src/` and import locally, or inline the minimal styles needed (uPlot CSS is small).

**Step 4: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 5: Commit**

```bash
git add aria/dashboard/spa/package.json aria/dashboard/spa/package-lock.json aria/dashboard/spa/src/components/TimeChart.jsx aria/dashboard/spa/src/index.css
git commit -m "feat(dashboard): add uPlot chart wrapper with theme-aware styling"
```

### Task 10: Migrate TrendsOverTime to use uPlot

**Files:**
- Modify: `aria/dashboard/spa/src/pages/intelligence/TrendsOverTime.jsx`

**Step 1: Replace BarChart with TimeChart**

Rewrite `TrendsOverTime.jsx` to use the new `TimeChart` component for daily and intraday trends. Keep the existing `Section` (now `CollapsibleSection`) wrapper.

Convert the existing data format (array of objects with `date` and metric keys) to uPlot format (array of arrays: `[timestamps[], values1[], values2[]]`).

Keep the CSS bar approach as a fallback for browsers that don't support canvas well, but primary rendering uses uPlot.

**Step 2: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

Open Intelligence page → Trends section should show canvas-rendered charts.

**Step 3: Commit**

```bash
git add aria/dashboard/spa/src/pages/intelligence/TrendsOverTime.jsx
git commit -m "feat(dashboard): migrate TrendsOverTime to uPlot canvas charts"
```

---

## Phase 7: Page Migration

Apply the new design system to each page. These tasks can be parallelized — each page is independent.

### Task 11: Migrate Intelligence page to new design system

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Intelligence.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/HomeRightNow.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/ActivitySection.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/LearningProgress.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/SystemStatus.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/Baselines.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/Correlations.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/DailyInsight.jsx`
- Modify: `aria/dashboard/spa/src/pages/intelligence/PredictionsVsActuals.jsx`

**Step 1: Add HeroCard to Intelligence page**

Import `HeroCard` and add it at the top of the Intelligence page as the primary metric — show overall home status (e.g., "All Clear" / entity count / occupancy).

**Step 2: Convert all Section calls to use collapsible props**

Each sub-component already uses `Section` from utils.jsx (which now delegates to CollapsibleSection). Add `summary` props so collapsed state shows useful info:

- HomeRightNow: `summary="12 metrics tracked"`
- TrendsOverTime: `summary="30 days"`
- Correlations: `summary="24 pairs"`
- Baselines: `summary="7 metrics"`
- SystemStatus: `summary="3 models active"`

**Step 3: Apply responsive grid**

Wrap the sub-components in a responsive grid:
- Phone: single column, sections collapsed below fold
- Tablet: 2-column grid
- Desktop: 2-column grid, all expanded

```jsx
<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
```

**Step 4: Build and verify at all three widths**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 5: Commit**

```bash
git add aria/dashboard/spa/src/pages/Intelligence.jsx aria/dashboard/spa/src/pages/intelligence/
git commit -m "feat(dashboard): migrate Intelligence page to new design system"
```

### Task 12: Migrate Home page pipeline to new design system

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Home.jsx`

**Step 1: Apply t-frame to pipeline lane cards**

Replace `.t-card` with `.t-frame` on pipeline nodes. Add `data-label` for section names.

**Step 2: Add terminal texture to pipeline background**

Add `t-terminal-bg` class to the pipeline section background (desktop only — CSS handles responsive).

**Step 3: Update pipeline node status badges with Tier 3 animation hooks**

Node status changes should trigger `t3-orange-pulse` class temporarily. Add a `useEffect` that watches for status prop changes and applies the class for 30 seconds.

**Step 4: Responsive pipeline layout**

- Phone: vertical stack with down-arrow connectors (existing behavior, refine)
- Tablet: 3-column grid (landscape) or vertical (portrait)
- Desktop: 3-column with arrow connectors + guidance nodes

**Step 5: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 6: Commit**

```bash
git add aria/dashboard/spa/src/pages/Home.jsx
git commit -m "feat(dashboard): migrate Home pipeline to ASCII frames with responsive layout"
```

### Task 13: Migrate remaining pages (batch)

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Discovery.jsx`
- Modify: `aria/dashboard/spa/src/pages/Capabilities.jsx`
- Modify: `aria/dashboard/spa/src/pages/DataCuration.jsx`
- Modify: `aria/dashboard/spa/src/pages/Predictions.jsx`
- Modify: `aria/dashboard/spa/src/pages/Patterns.jsx`
- Modify: `aria/dashboard/spa/src/pages/Shadow.jsx`
- Modify: `aria/dashboard/spa/src/pages/Automations.jsx`
- Modify: `aria/dashboard/spa/src/pages/Settings.jsx`
- Modify: `aria/dashboard/spa/src/pages/Guide.jsx`

For each page, apply:

1. Replace `.t-card` with `.t-frame` where appropriate
2. Add `data-label` attributes for section titles
3. Use `CollapsibleSection` for major content blocks
4. Apply responsive grid patterns (1-col phone, 2-col tablet, 2-3 col desktop)
5. Use `HeroCard` for the primary metric on pages that have one:
   - Predictions: next predicted event
   - Shadow: accuracy score
   - Discovery: entity count
   - DataCuration: curation progress
6. Replace old animation classes (`animate-*`) with tier-prefixed (`t1-*`)
7. Ensure all interactive elements have 44px min touch targets on phone

**Step N: Build and verify after each page**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step N+1: Commit after each page (or batch 2-3 related pages)**

```bash
git commit -m "feat(dashboard): migrate [PageName] to new design system"
```

---

## Phase 8: Accessibility Fixes

### Task 14: Add ARIA labels and keyboard navigation

**Files:**
- Modify: `aria/dashboard/spa/src/components/Sidebar.jsx` (add aria-labels to icon buttons)
- Modify: `aria/dashboard/spa/src/components/CollapsibleSection.jsx` (already has aria-expanded)
- Modify: `aria/dashboard/spa/src/index.css` (focus ring styles)

**Step 1: Add ARIA labels to all icon-only buttons**

In the phone tab bar and tablet rail, every icon-only element needs `aria-label`:

```jsx
<a href="#/" aria-label="Home" ...>
```

**Step 2: Fix focus styles**

In `index.css`, update `.t-input:focus` and add a global focus-visible rule:

```css
:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}
.t-input:focus {
  outline: none; /* override for custom styling */
  border-color: var(--accent);
  box-shadow: 0 0 0 2px var(--accent-glow);
}
```

**Step 3: Add screen-reader-only data table fallback for charts**

Create a `.sr-only` utility class and add hidden data tables alongside uPlot charts.

**Step 4: Build and verify**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

Tab through the dashboard — verify focus rings are visible on all interactive elements.

**Step 5: Commit**

```bash
git add aria/dashboard/spa/src/
git commit -m "fix(dashboard): add ARIA labels, focus rings, and screen-reader chart fallbacks"
```

---

## Phase 9: Final Build and Verification

### Task 15: Full build, smoke test, restart service

**Step 1: Full build**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 2: Restart service**

```bash
systemctl --user restart aria-hub
```

**Step 3: Smoke test at all three sizes**

Open `http://127.0.0.1:8001/ui/` and verify:

- [ ] Light theme: HA-aligned neutral grays, cyan accent, orange for warnings
- [ ] Dark theme: dark surfaces, bright cyan accent, warm orange
- [ ] Phone (< 640px): bottom tab bar, hero cards full-width, sections collapsed
- [ ] Tablet (640-1023px): icon rail sidebar, 2-col grids, sections expanded
- [ ] Desktop (1024px+): full sidebar, 3-col grids, terminal texture, all animations
- [ ] Cursors: block blinks on active sections, underscore on collapsed
- [ ] Collapsible sections: tap to toggle, cursor changes state
- [ ] Charts: uPlot renders in Trends section
- [ ] No console errors
- [ ] WebSocket reconnects normally
- [ ] Reduced motion: animations disabled, cursors static

**Step 4: Final commit if any fixes needed**

```bash
git add aria/dashboard/spa/
git commit -m "fix(dashboard): smoke test fixes for UI redesign"
```

---

## Task Dependency Graph

```
Task 1 (palette) ──→ Task 2 (hardcoded colors) ──→ Task 3 (frames) ──→ Task 5 (collapsible)
                                                  ↘                   ↗
                                                    Task 4 (cursors) ─┘
                                                                        ↘
Task 7 (navigation) ─────────────────────────────────────────────────────→ Task 11-13 (pages)
                                                                        ↗
Task 6 (hero card) ────────────────────────────────────────────────────┘
                                                                        ↗
Task 8 (animation tiers) ─────────────────────────────────────────────┘
                                                                        ↗
Task 9 (uPlot) ──→ Task 10 (TrendsOverTime) ─────────────────────────┘

Task 14 (a11y) ──→ Task 15 (final verification)
```

**Parallelizable:** Tasks 11, 12, 13 (page migrations) can run in parallel once Tasks 1-10 are complete.
