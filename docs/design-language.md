# ARIA Dashboard Design Language

## In Plain English

This document defines the visual rules for ARIA's dashboard — the colors, fonts, spacing, and component styles that make every page look and feel consistent. Think of it as the "brand guidelines" for the UI. It enforces a retro ASCII terminal aesthetic: monospace fonts, bracket-framed cards, and a data-dense layout designed for someone who wants information at a glance, not eye candy.

## Why This Exists

Without a shared design language, every new page or feature ends up looking slightly different — different colors, different spacing, different button styles. Over time the UI becomes a patchwork. This document prevents that by establishing a single source of truth for every visual decision. It also encodes data visualization principles (from Edward Tufte and others) so that charts and metrics communicate clearly instead of just looking pretty.

Reference for creating and modifying UI components. Read this before touching any JSX or CSS.

## Visual Identity

**Aesthetic:** ASCII terminal — monospace typography, bracket-framed cards, blinking cursor affordances, scan-line textures. Inspired by Home Assistant's design system but with a technical, data-forward personality.

**Philosophy:** Each page tells its story within the ARIA pipeline (Data Collection → Learning → Actions). Components should reinforce where the user is in that flow.

### Thematic Influences: SUPERHOT

The dashboard borrows visual motifs from SUPERHOT's aesthetic language — minimal geometry, stark contrast, and time-as-information.

| SUPERHOT Motif | ARIA Application |
|----------------|------------------|
| **Crystalline red on white** | Anomalies and critical alerts use `--sh-threat` (crystalline red) against neutral surfaces — maximum contrast for maximum urgency |
| **Time freeze** | Stale data dims and desaturates (`.sh-frozen`). Fresh data is crisp and vivid. The passage of time is always visible. |
| **Shatter/fragment** | Dismissed alerts and resolved anomalies break apart via `.sh-shatter` transition before removal |
| **Glitch corruption** | Error states and data gaps use `.sh-glitch` — horizontal offset + chromatic aberration on text |
| **Typographic repetition** | Critical system states repeat their label (e.g., "OFFLINE OFFLINE OFFLINE") as background watermark via `.sh-mantra` |
| **Geometric minimalism** | Data visualizations favor sharp angles and clean geometry over rounded or organic shapes |

These motifs layer on top of the terminal aesthetic — SUPERHOT provides the emotional register (tension, urgency, time-awareness), the terminal provides the structural register (monospace, brackets, cursors).

## Science-Backed Visualization Framework

All dashboard visual design must be grounded in peer-reviewed research. This is not optional — it applies to every chart, diagram, layout, and interaction. When in doubt, cite the principle.

### Foundational Research (always apply)

| # | Principle | Source | Rule |
|---|-----------|--------|------|
| 1 | **Data-ink ratio** | Tufte, *The Visual Display of Quantitative Information*, 1983 | Maximize data-carrying pixels. Remove chartjunk: decorative gridlines, redundant labels, ornamental elements. Every pixel earns its place or gets deleted. |
| 2 | **Perceptual accuracy hierarchy** | Cleveland & McGill, *JASA*, 1984 | Position > Length > Angle > Area > Color saturation > Color hue. Encode the most important variable in position (e.g., node placement = module role), use color for secondary dimensions only (e.g., status). Never reverse this. |
| 3 | **Preattentive processing** | Treisman & Gelade, *Cognitive Psychology*, 1980 | Color, size, orientation, and motion are processed in <250ms without conscious effort. Use max 3-4 color channels. Combine color + line style for colorblind safety (don't rely on hue alone). |
| 4 | **Gestalt grouping** | Wertheimer, *Psychologische Forschung*, 1923 | Enclosure > Connection > Proximity > Similarity. Swim lane backgrounds (enclosure) are stronger than whitespace (proximity) for grouping. Use enclosure for primary grouping, proximity for secondary. |
| 5 | **Cognitive load** | Sweller, *Cognitive Science*, 1988; Miller, *Psych Review*, 1956 | Working memory holds 7±2 chunks. A diagram with 40 labels exceeds capacity. Show ≤15 primary elements; use progressive disclosure for detail. |
| 6 | **Progressive disclosure** | Shneiderman, *IEEE Software*, 1996 | "Overview first, zoom and filter, then details on demand." Default view = structural understanding in 10 seconds. Hover/click = specific data paths in 60 seconds. |
| 7 | **Small multiples** | Tufte, *Envisioning Information*, 1990 | When metrics have different units or scales, NEVER overlay them on the same y-axis. Stack small multiples sharing the x-axis. |
| 8 | **Sparklines** | Tufte, *Beautiful Evidence*, 2006 | A number without trend is half a story. Every HeroCard KPI should have a sparkline when historical data is available. |

### Application Rules

1. **Diverging scales for correlation.** Use one hue for positive, another for negative, neutral for near-zero. Filter weak signals (|r| ≤ 0.3).
2. **Temporal swim-lanes.** Group events by category along a shared time axis. Reveals clustering that flat lists hide.
3. **Rolling averages over raw points.** 7-day rolling average for trend, raw points for detail. Show both when space allows.
4. **Explain like I'm 5.** Every visualization MUST include a plain-English explanation and legend. If a user needs domain expertise to interpret a chart, the chart has failed.
5. **Color-coding protocol.** Max 3 semantic colors per diagram. Each color must also have a non-color discriminator (dash pattern, shape, label). Include a visible legend.
6. **Feedback loops are visually distinct.** Data that flows backward (writes back to a source it reads from) must use a different visual treatment — curved arcs, distinct color, dashed stroke. Feedback is the exception path; it should look like one.
7. **Hover detail for density.** When a diagram or chart contains more than 15 data points, show structural overview by default. Detail appears on hover via SVG tooltip panel or HTML overlay.

### Choosing the Right Visualization

| Data Shape | Recommended Viz | Component |
|------------|----------------|-----------|
| Single KPI with trend | HeroCard + sparkline | `HeroCard` with `sparkData` |
| Time series (single metric) | Line chart, 80-140px | `TimeChart` |
| Time series (multiple metrics, same unit) | Overlaid lines | `TimeChart` with multiple series |
| Time series (multiple metrics, different units) | Small multiples (stacked) | Multiple `TimeChart`s in `space-y-2` |
| Matrix (day × metric, value intensity) | Heatmap grid | CSS grid + opacity interpolation |
| Pairwise relationships | Correlation matrix | CSS grid + `color-mix()` diverging scale |
| Events over time by category | Swim-lane timeline | CSS grid + positioned dots |
| Accuracy over time + volume | Dual-series chart (line + bars) | `TimeChart` with 2 series |
| Single-value distribution | Not yet implemented | Consider box plot or histogram |

## Color System

All colors are CSS custom properties in `index.css`. NEVER hardcode hex values in JSX.

### Surfaces

| Token | Light | Dark | Use |
|-------|-------|------|-----|
| `--bg-base` | #FAFAFA | #111318 | Page background |
| `--bg-surface` | #FFFFFF | #1C1C1E | Card/panel background |
| `--bg-surface-raised` | #F5F5F5 | #2C2C2E | Hover states, active items |
| `--bg-inset` | #F0F0F0 | #0D0D0F | Sunken areas |
| `--bg-terminal` | #F5F5F0 | #0A0E14 | Terminal texture background |

### Text

| Token | Light | Dark | Use |
|-------|-------|------|-----|
| `--text-primary` | #212121 | #E8EAED | Headings, primary content |
| `--text-secondary` | #727272 | #9AA0A6 | Body text, labels |
| `--text-tertiary` | #9E9E9E | #5F6368 | Hints, timestamps, bracket labels |

### Accent Colors

| Token | Light | Dark | Use |
|-------|-------|------|-----|
| `--accent` | #0891B2 (cyan) | #22D3EE | ARIA brand, links, focus rings, active states |
| `--accent-warm` | #FF9800 (orange) | #FFB74D | Human action, warnings, attention items |
| `--accent-purple` | #9333EA | #C084FC | Correlations, room occupancy |

### Status Colors (HA-aligned)

| Token | Meaning |
|-------|---------|
| `--status-healthy` | Green — operational, connected, passing |
| `--status-warning` | Orange — degraded, medium confidence |
| `--status-error` | Red — failed, offline, low confidence |
| `--status-active` | Yellow — in-progress, currently running |
| `--status-waiting` | Gray — pending, not yet started |

Each status has a `--status-*-glow` variant (15% opacity) for background tints.

### SUPERHOT Palette

Dedicated tokens for SUPERHOT-influenced visual treatments. These supplement — not replace — the core palette.

| Token | Light | Dark | Use |
|-------|-------|------|-----|
| `--sh-threat` | #DC2626 (crystalline red) | #EF4444 | Anomaly highlights, critical alerts, shatter fragments |
| `--sh-threat-glow` | rgba(220,38,38,0.12) | rgba(239,68,68,0.15) | Background tint behind threat elements |
| `--sh-frozen` | #94A3B8 (slate) | #475569 | Stale/frozen data overlay, desaturated state |
| `--sh-glass` | rgba(255,255,255,0.85) | rgba(255,255,255,0.06) | Crystalline overlay on shatter fragments |
| `--sh-void` | #0F172A | #020617 | Deep background for mantra watermarks |

## Typography

All text is monospace (`--font-mono`). Size scale:

| Token | Size | Use |
|-------|------|-----|
| `--type-hero` | 2.5rem | HeroCard values |
| `--type-headline` | 1.25rem | Section titles |
| `--type-body` | 0.9375rem | Body text |
| `--type-data` | 1rem | Data values in tables/grids |
| `--type-label` | 0.6875rem | Frame labels, bracket text, timestamps |
| `--type-micro` | 0.625rem | Frame footers, fine print |

## Component Patterns

### Page Banner: `PageBanner`

Every page starts with an ASCII pixel-art banner for visual consistency. Renders "ARIA + PAGE_NAME" in the same SVG pixel style as `AriaLogo`.

```jsx
<PageBanner page="SHADOW" subtitle="Predict-compare-score validation loop." />
```

- **ARIA** text in `--accent` (brand color)
- **Separator** (plus/cross) in `--text-tertiary`
- **Page name** in `--text-primary`
- Optional `subtitle` renders as `--type-label` text below the banner
- SVG uses `useMemo` — layout computed once per page name
- Height: 2rem, scales proportionally with `max-width: 100%`
- Font: 5-row pixel grid matching AriaLogo proportions (26 uppercase characters)
- MUST be the first element in every page's return JSX, before HeroCard

### Content Cards: `.t-frame`

The primary container. Replaces the legacy `.t-card` class.

```jsx
<div class="t-frame" data-label="section name" data-footer="optional footer">
  {/* content */}
</div>
```

- `data-label` renders as uppercase mono label above content (via `::before`)
- `data-footer` renders as right-aligned mono text below content (via `::after`)
- Has `border-radius: var(--radius)` and `box-shadow: var(--card-shadow)`
- Phone (<640px): padding shrinks, border-radius removed
- Do NOT use `.t-card` for new components

### Inline Labels: `.t-bracket`

For inline metadata labels with bracket decoration:

```jsx
<span class="t-bracket">12 items</span>
```

Renders as `[12 items]` with brackets in `--text-tertiary`.

### Collapsible Sections

Use `CollapsibleSection` from `components/CollapsibleSection.jsx`:

```jsx
<CollapsibleSection
  title="Section Title"
  subtitle="Description when expanded"
  summary="Brief when collapsed"
  defaultOpen={true}
  loading={false}
>
  {children}
</CollapsibleSection>
```

The **cursor IS the expand/collapse indicator** — no chevrons or arrows:
- `cursor-active` (block cursor, 1s blink): expanded
- `cursor-working` (half cursor, 0.5s blink): loading
- `cursor-idle` (underscore, 2s blink): collapsed

### Hero Metrics

Use `HeroCard` from `components/HeroCard.jsx` for prominent KPIs:

```jsx
<HeroCard label="Power Draw" value="847" unit="W" delta="+12%" />

// With sparkline trend (uPlot data format)
<HeroCard
  label="Shadow Accuracy"
  value={65}
  unit="%"
  sparkData={[timestamps, values]}
  sparkColor="var(--accent)"
/>
```

- Renders large monospace value (2.5rem) inside a `.t-frame`
- Optional `delta` shows change text
- Optional `warning` shows orange alert border + text
- Optional `sparkData` renders an 80×32px inline sparkline next to the value
- Optional `sparkColor` sets sparkline color (defaults to `--accent`)
- One HeroCard per page at the top — the page's primary metric

### Charts: `TimeChart`

Use `TimeChart` from `components/TimeChart.jsx` for time-series:

```jsx
// Full chart with axes, grid, cursor
<TimeChart
  data={[timestamps, values1, values2]}
  series={[
    { label: 'Power', color: 'var(--accent)' },
    { label: 'Errors', color: 'var(--status-error)' },
  ]}
  height={140}
/>

// Compact sparkline mode (no axes, no grid, no cursor)
<TimeChart
  data={[timestamps, values]}
  series={[{ label: 'trend', color: 'var(--accent)', width: 1.5 }]}
  compact
/>
```

- CSS variables are resolved automatically via `getComputedStyle()`
- Full mode: wrapped in semantic `<figure>` with `<figcaption class="sr-only">`
- Compact mode: bare `<div>` with `role="img"`, defaults to 32px height
- Responds to container width via ResizeObserver
- Theme changes require data update to re-render (known limitation)

### Stats Grid

Use `StatsGrid` from `components/StatsGrid.jsx` for labeled value grids:

```jsx
<StatsGrid items={[{ label: 'Entities', value: '3,058' }, ...]} />
```

## Data Visualization Patterns

### Small Multiples (TrendsOverTime)

Show each metric in its own chart stacked vertically, sharing the x-axis but with independent y-axes. Better than overlaid series because each metric has its own scale.

```jsx
// One chart per metric, 80px each, stacked in space-y-2
<MetricChart label="Power (W)" data={...} color="var(--accent)" height={80} />
<MetricChart label="Lights On" data={...} color="var(--accent-warm)" height={80} />
```

### Heatmap Grid (Baselines)

CSS grid with color intensity mapping. Rows = categories, columns = metrics. Each cell uses opacity interpolation against a base color.

```jsx
// Color intensity: opacity scales from 0.12 (low) to 0.67 (high)
<div style={`background: var(--accent); opacity: ${0.12 + intensity * 0.55}`} />
```

- Positive metrics (Power, Lights, Devices): scale against `--accent`
- Negative metrics (Unavailable): scale against `--status-error`
- Always include a color scale legend below the heatmap

### Correlation Matrix (Correlations)

Diverging color heatmap for pairwise relationships. Uses `color-mix()` CSS function for dynamic opacity.

```jsx
// Positive correlation → accent, negative → accent-purple
background: color-mix(in srgb, var(--accent) 70%, transparent)
background: color-mix(in srgb, var(--accent-purple) 70%, transparent)
```

- Filter weak correlations (|r| ≤ 0.3) with gray
- Sort entities by strongest average |correlation|
- Rotate column headers 45° to save space
- Always include a diverging legend (Negative / Weak / Positive)

### Swim-Lane Timeline (ActivitySection)

Horizontal lanes per domain, events as positioned dots along a 60-minute time axis.

```jsx
// Domain colors
light → var(--accent-warm)
switch → var(--accent)
binary_sensor → var(--accent-purple)
person/device_tracker → var(--status-healthy)
other → var(--text-tertiary)
```

- Each lane: 20px tall, 2px gap
- Domain labels on left (60px), timeline on right
- Events: 8px circles at `((eventTime - startTime) / 60min) * 100%`
- Vertical dashed "now" line at right edge
- Hide domains with no events in the window
- Always include a domain color legend

### Dual-Axis Chart (Shadow DailyTrend)

Rolling average line + volume bars sharing the same time axis. Use TimeChart with 2 series.

```jsx
// Series 1: 7-day rolling accuracy line
{ label: '7-day Accuracy', color: 'var(--accent)', width: 2 }
// Series 2: Daily prediction count bars
{ label: 'Predictions', color: 'var(--text-tertiary)', width: 1 }
```

Include gate threshold annotation below the chart.

### SUPERHOT Data States

Apply SUPERHOT treatments to communicate data freshness and anomaly status at a glance.

**Freshness spectrum** (applies to any data-bearing component):

| Age | Treatment | Class |
|-----|-----------|-------|
| < 5 min | Full color, crisp rendering | (default) |
| 5-30 min | Slight desaturation (filter: saturate(0.7)) | `.sh-cooling` |
| 30-60 min | Visible desaturation + reduced opacity | `.sh-frozen` |
| > 60 min | Full grayscale + mantra watermark "STALE" | `.sh-frozen .sh-mantra` |

**Anomaly detection** (applies to HeroCards, chart annotations, timeline events):

| Severity | Treatment |
|----------|-----------|
| Info | Standard accent color, no SUPERHOT treatment |
| Warning | `--sh-threat-glow` background tint |
| Critical | `--sh-threat` border + `.sh-threat-pulse` + `.sh-glitch` on value text |
| Resolved | `.sh-shatter` exit transition, then remove from DOM |

**Time-awareness principle:** Borrowed directly from SUPERHOT's core mechanic — the visual intensity of every element should communicate its temporal relevance. Recent = vivid. Stale = frozen. This makes the dashboard self-documenting: you can assess system health from across the room.

### Visualization Rules

1. **Every visualization MUST include:**
   - A 1-2 sentence layman explanation (what it shows, why it matters)
   - A color legend explaining all visual encodings
2. **Prefer small multiples** over overlaid series when metrics have different scales
3. **Use `color-mix()` in CSS** for dynamic opacity against CSS variable colors
4. **Inline styles for dynamic values** — Tailwind can't handle computed colors/positions
5. **Screen reader fallback:** When replacing text with graphics, keep the text in `sr-only`

## Cursor State System

Cursors replace traditional expand/collapse chevrons. Three states:

| Class | Symbol | Speed | Color | Meaning |
|-------|--------|-------|-------|---------|
| `.cursor-active` | Block (█) | 1s | `--text-primary` | Active, expanded, operational |
| `.cursor-working` | Half (▊) | 0.5s | `--text-primary` | Loading, processing |
| `.cursor-idle` | Underscore (_) | 2s | `--text-tertiary` | Collapsed, waiting |

On Home page pipeline nodes, cursor states map to node health:
- HEALTHY → `cursor-active`
- REVIEW → `cursor-working`
- WAITING → `cursor-idle`

Phone: cursors shrink to 0.75em, idle cursors hidden.
Reduced motion: all cursors static (no animation).

## Animation Tiers

Three tiers with responsive budgets:

### Tier 1: Ambient (decorative, always-on on desktop)

Classes: `t1-scan-sweep`, `t1-grid-pulse`, `t1-border-shimmer`, `t1-data-stream`, `t1-scan-line`, `t1-pulse-ring`

- Desktop: all active
- Tablet: `t1-scan-sweep` and `t1-scan-line` only
- Phone: all disabled
- Reduced motion: all disabled

### Tier 2: Data Refresh (triggered by data updates)

Classes: `t2-typewriter`, `t2-tick-flash`, `t2-bar-grow`

- Apply when data changes, runs once
- Active on all screen sizes
- Reduced motion: disabled

### Tier 3: Status Alert (strongest attention, auto-expire)

Classes: `t3-orange-pulse`, `t3-border-alert`, `t3-badge-appear`, `t3-counter-bump`

- Active on all screen sizes (phone: simplified)
- Reduced motion: static color indicators only

### SUPERHOT Effects (layered on tiers)

These classes implement the SUPERHOT thematic motifs. They compose with the tier system — each effect has a tier-appropriate energy level.

| Class | Tier | Effect | Use |
|-------|------|--------|-----|
| `.sh-frozen` | T1 | Desaturate to grayscale + 60% opacity, subtle frost overlay | Data older than staleness threshold |
| `.sh-glitch` | T2 | 2-frame horizontal jitter + red/cyan chromatic split on text | Error states, data parse failures |
| `.sh-shatter` | T2 | Element breaks into 4-6 triangular fragments that drift outward and fade | Dismissed alerts, resolved anomalies |
| `.sh-mantra` | T1 | Repeating label text as faded watermark behind content | Critical persistent states (OFFLINE, STALE, ERROR) |
| `.sh-threat-pulse` | T3 | `--sh-threat` border glow, 2-cycle pulse then static | New anomaly detection, threshold breach |

**Responsive rules:**
- Phone: `.sh-shatter` reduced to simple fade-out, `.sh-mantra` disabled
- Tablet: `.sh-glitch` simplified to color shift only (no jitter)
- Reduced motion: all effects replaced with instant state change (no animation)

## Responsive Breakpoints

| Breakpoint | Width | Layout | Nav |
|------------|-------|--------|-----|
| Phone | <640px | Single column, full-width cards | Bottom tab bar (5 tabs + More sheet) |
| Tablet | 640-1023px | 2-column grids (`sm:grid-cols-2`) | Collapsible icon rail (56px/240px) |
| Desktop | 1024px+ | 3-column grids (`lg:grid-cols-3`) | Full sidebar (240px fixed) |

Use `sm:` for tablet, `lg:` for desktop. Phone is the default (mobile-first).

Content area padding accounts for nav:
- Phone: `pb-16` (bottom tab bar)
- Tablet: `pl-14` (icon rail)
- Desktop: `pl-60` (full sidebar)

## Terminal Texture

`.t-terminal-bg` adds horizontal scan-line overlay via repeating gradient:

- Desktop: full texture
- Tablet: only on section headers
- Phone: disabled (flat background)

## Accessibility

- `:focus-visible` with 2px `--accent` outline on all focusable elements
- `.sr-only` utility for screen-reader-only content
- `aria-hidden="true"` on all decorative SVG icons
- `aria-expanded` on all collapsible triggers
- `aria-label` on all icon-only buttons
- `role="img"` on chart canvas containers
- `env(safe-area-inset-bottom)` on phone tab bar for notched devices
- `prefers-reduced-motion`: all animations disabled, cursors static

## Cross-Dashboard Navigation

All ARIA-aesthetic dashboards share a unified navigation pattern via the Project Hub.

### Hub Nav Script

Every dashboard (except the hub itself) includes a shared script that injects a floating "← Hub" button:

```html
<script src="/hub/hub-nav.js"></script>
```

- **Position:** Fixed, top-left, `z-index: 9999`
- **Touch target:** 48px minimum (mobile-safe)
- **Theme-aware:** Reads `data-theme` attribute on `<html>`, falls back to `prefers-color-scheme`
- **Self-skipping:** Does not inject on `/hub/*` paths
- **No dependencies:** Vanilla JS, inline styles, ~1.8KB

Currently active on:
- ARIA dashboard (`/aria/` via Tailscale Serve)
- Ollama Queue dashboard (`/queue/` via Tailscale Serve)

### Internal Navigation (Project Hub)

The Project Hub uses History API routing for its own views:
- `/hub/` — project list (summary cards with status badges)
- `/hub/project/:id` — project detail page (services, timers, git, actions)

Navigation components:
- `BackLink` — `← All Projects` link, 48px touch target, uses `navigate()` from `useRoute` hook
- `PageBanner` — reused with project name: `HUB ÷ {PROJECT_NAME}`

### Adding Hub Nav to a New Dashboard

1. Add `<script src="/hub/hub-nav.js"></script>` before `</body>` in the dashboard HTML
2. Ensure the dashboard sets `data-theme` on `<html>` for correct styling (falls back to OS preference)
3. The script auto-detects dark/light and watches for theme changes via MutationObserver
4. Rebuild the dashboard if HTML is a build artifact

## Adding a New Page

1. Create `aria/dashboard/spa/src/pages/YourPage.jsx`
2. Add route in `app.jsx`
3. Add nav entry in `Sidebar.jsx` (NAV_ITEMS array, and PHONE_TABS if primary)
4. Use `PageBanner` as the first element with page name and subtitle
5. Use `HeroCard` below the banner with the page's primary metric (if applicable)
6. Use `.t-frame` with `data-label` for all content sections
6. Use `CollapsibleSection` for expandable content
7. Use `sm:grid-cols-2` / `lg:grid-cols-3` for responsive grids
8. NEVER hardcode colors — use CSS custom properties
9. Rebuild: `cd aria/dashboard/spa && npm run build`
