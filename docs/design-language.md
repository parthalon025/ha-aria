# ARIA Dashboard Design Language

Reference for creating and modifying UI components. Read this before touching any JSX or CSS.

## Visual Identity

**Aesthetic:** ASCII terminal — monospace typography, bracket-framed cards, blinking cursor affordances, scan-line textures. Inspired by Home Assistant's design system but with a technical, data-forward personality.

**Philosophy:** Each page tells its story within the ARIA pipeline (Data Collection → Learning → Actions). Components should reinforce where the user is in that flow.

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
```

- Renders large monospace value (2.5rem) inside a `.t-frame`
- Optional `delta` shows change (styled green/red automatically)
- Optional `warning` shows orange alert text
- One HeroCard per page at the top — the page's primary metric

### Charts: `TimeChart`

Use `TimeChart` from `components/TimeChart.jsx` for time-series:

```jsx
<TimeChart
  data={[timestamps, values1, values2]}
  series={[
    { label: 'Power', color: 'var(--accent)' },
    { label: 'Errors', color: 'var(--status-error)' },
  ]}
  height={140}
/>
```

- CSS variables are resolved automatically via `getComputedStyle()`
- Wrapped in semantic `<figure>` with `<figcaption class="sr-only">`
- Responds to container width via ResizeObserver
- Theme changes require data update to re-render (known limitation)

### Stats Grid

Use `StatsGrid` from `components/StatsGrid.jsx` for labeled value grids:

```jsx
<StatsGrid items={[{ label: 'Entities', value: '3,058' }, ...]} />
```

## Cursor State System

Cursors replace traditional expand/collapse chevrons. Three states:

| Class | Symbol | Speed | Color | Meaning |
|-------|--------|-------|-------|---------|
| `.cursor-active` | Block (█) | 1s | `--accent` | Active, expanded, operational |
| `.cursor-working` | Half (▊) | 0.5s | `--accent-warm` | Loading, processing |
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

## Adding a New Page

1. Create `aria/dashboard/spa/src/pages/YourPage.jsx`
2. Add route in `app.jsx`
3. Add nav entry in `Sidebar.jsx` (NAV_ITEMS array, and PHONE_TABS if primary)
4. Use `HeroCard` at the top with the page's primary metric
5. Use `.t-frame` with `data-label` for all content sections
6. Use `CollapsibleSection` for expandable content
7. Use `sm:grid-cols-2` / `lg:grid-cols-3` for responsive grids
8. NEVER hardcode colors — use CSS custom properties
9. Rebuild: `cd aria/dashboard/spa && npm run build`
