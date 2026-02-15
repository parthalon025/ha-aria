# ARIA UI Redesign — "Living Terminal Intelligence"

**Date:** 2026-02-13
**Status:** Design approved, pending implementation plan
**Scope:** Full visual redesign of ARIA dashboard SPA

## In Plain English

This is the design plan for giving ARIA's dashboard a distinctive visual identity -- moving from a generic-looking web app to something that feels like a living command center with subtle animations, a retro-terminal aesthetic, and layouts purpose-built for phones, tablets, and desktops.

## Why This Exists

ARIA's dashboard looked like every other template-based web app: white cards, blue accents, rounded corners. It did not feel like an intelligence system, and it was not designed for the three different screen sizes people actually use it on. The phone layout was an afterthought, tablet was ignored entirely, and the dark mode only existed on one page. This redesign creates a unified visual language -- inspired by the pixel-art logo -- that works across all devices and makes the system feel alive through purposeful animation, not just functional.

---

## Design Goals

1. **Color feels flat/generic** — Replace mono-cyan with HA-aligned dual-accent palette
2. **Information density** — Hero metric pattern replaces equal-weight grids
3. **Charts feel basic** — Add uPlot for trends, inline sparklines, animated bars
4. **Living system feel** — Three-tier animation system with ASCII terminal aesthetic
5. **Three first-class experiences** — Phone, tablet, desktop each purpose-built

## Design Direction

**Mood:** Calm intelligence (Linear, Vercel, Apple Health)
**Aesthetic:** ASCII terminal — bracket frames, blinking cursors, monospace data
**Palette basis:** HA ecosystem colors (backgrounds, status, text) + ARIA's own accent

---

## 1. Color Palette

### Light Theme

```
CANVAS                          ACCENT & BRAND
--bg-base:       #FAFAFA        --accent:          #0891B2   (ARIA cyan)
--bg-surface:    #FFFFFF        --accent-dim:      #0E7490
--bg-inset:      #F0F0F0        --accent-glow:     rgba(8,145,178,0.10)
--bg-terminal:   #F5F5F0        --accent-warm:     #FF9800   (HA orange)
                                --accent-warm-glow: rgba(255,152,0,0.10)

TEXT                            STATUS (from HA)
--text-primary:   #212121       --status-healthy:  #0F9D58
--text-secondary: #727272       --status-warning:  #FF9800
--text-tertiary:  #9E9E9E       --status-error:    #DB4437
--text-accent:    #0891B2       --status-active:   #FDD835   (HA yellow)
--text-inverse:   #FAFAFA       --status-waiting:  #9E9E9E
```

### Dark Theme

```
CANVAS                          ACCENT & BRAND
--bg-base:       #111318        --accent:          #22D3EE   (bright cyan)
--bg-surface:    #1C1C1E        --accent-dim:      #0891B2
--bg-inset:      #0D0D0F        --accent-glow:     rgba(34,211,238,0.12)
--bg-terminal:   #0A0E14        --accent-warm:     #FFB74D   (soft orange)
                                --accent-warm-glow: rgba(255,183,77,0.10)

TEXT                            STATUS
--text-primary:   #E8EAED       --status-healthy:  #4ADE80
--text-secondary: #9AA0A6       --status-warning:  #FFC107
--text-tertiary:  #5F6368       --status-error:    #F87171
--text-accent:    #22D3EE       --status-active:   #FDE68A
```

### Color Hierarchy

- **Cyan** = system/data/ARIA is working (ambient, metrics, data visualization)
- **Orange** = human/action/attention needed (CTAs, status changes, alerts)
- **Status colors** = HA-exact semantics (green/amber/red/yellow)

---

## 2. ASCII Terminal Aesthetic

### Bracket Framing System

Three frame styles with CSS pseudo-elements (not `<pre>` blocks):

| Style | When | Example |
|-------|------|---------|
| `[ square ]` | Inline data labels, counters | `[3] entities` `[active]` |
| `corner` | Section/card frames | Full card borders with monospace corners |
| `< angle >` | Navigation, breadcrumbs | `< Intelligence > / Predictions` |

Card example:
```
-- Predictions -----------------------------------------

  [3] upcoming  .  [1] high confidence  .  [0] missed

  > Living room lights ON          94% ########..  21:30
  > Front door lock                87% #######...  22:00
  > Thermostat night mode          71% ######....  22:30

----------------------------------------------- 3 of 12
```

### Blinking Cursor System

Cursors serve as **state indicators AND expand/collapse affordances**:

| Cursor | Speed | Meaning | Interactive Role |
|--------|-------|---------|-----------------|
| `block` | 1s steady blink | Active/expanded | Click to collapse |
| `half` | 0.5s fast blink | Processing/loading | Shown during data fetch |
| `underscore` | 2s slow blink | Idle/collapsed | Click to expand |

Cursor replaces traditional chevron for collapsible sections:
- Collapsed: `-- Section Title ------------ [12 items] -- _` (slow blink)
- Expanding: `-- Section Title ----------------------------- half` (fast blink, brief)
- Expanded: `-- Section Title ----------------------------- block` (steady blink)

### Terminal Background Texture

Subtle scan-line overlay on `--bg-terminal` areas:
- Sidebar background
- Section headers
- Status bar / footer
- NOT on card content areas (keeps data readable)

---

## 3. Context-Based Typography

### Font Scale

| Token | Size | Weight | Use |
|-------|------|--------|-----|
| `--type-hero` | 2.5rem (40px) | 600 | Primary metric on a page/card |
| `--type-headline` | 1.25rem (20px) | 600 | Section headers (inside brackets) |
| `--type-body` | 0.9375rem (15px) | 400 | List items, descriptions |
| `--type-data` | 1rem (16px) | 500 | Tabular data, secondary metrics |
| `--type-label` | 0.6875rem (11px) | 500 | Uppercase labels, column headers |
| `--type-micro` | 0.625rem (10px) | 400 | Timestamps, footnotes |

### Rules

- All data VALUES render in `--font-mono` (monospace)
- All labels and body text render in system sans-serif
- Font contrast between families reinforces "value" vs. "description"

### Mobile Scaling

- Hero: 2.5rem -> 1.75rem
- Headline: 1.25rem -> 1.125rem
- All other sizes unchanged

---

## 4. Three-Tier Animation System

### Tier 1 — Ambient (Processing Hum)

"ARIA is alive and working." Peripheral vision only.

| Animation | Element | Behavior |
|-----------|---------|----------|
| Scan sweep | Sidebar accent line | 6s horizontal gradient crawl, cyan |
| Data stream | Pipeline connection lines | Tiny dots flowing along lines, 3s loop |
| Pulse ring | Hub status indicator | Slow concentric rings, 4s, low opacity |
| Grid breathe | Terminal-textured areas | 8s subtle opacity shift |

CSS properties: `opacity` and `background-position` only (GPU-composited).

### Tier 2 — Metric Refresh (Data is Alive)

"Values are real-time, not screenshots." Medium attention.

| Animation | Element | Behavior |
|-----------|---------|----------|
| Typewriter count | Hero metrics on value change | Old fades, new types in char-by-char (0.3s) |
| Bar grow | Chart bars on data refresh | Width/height animates old -> new (0.5s ease-out) |
| Tick flash | Any value that just updated | Brief accent-glow background (0.3s), fade |
| Cursor burst | Working cursor on data fetch | Fast blink 2s during fetch, returns to normal |

Trigger: WebSocket `data_update` events.

### Tier 3 — Status Alert (Human Attention Needed)

"Something changed that matters." Strongest animation — pulls the eye.

| Animation | Element | Behavior |
|-----------|---------|----------|
| Orange pulse | Status badge on state change | 3 pulses of accent-warm-glow, then steady |
| Border flash | Card with changed entity | Left border flashes orange -> cyan -> steady (1s) |
| Counter increment | Notification count in sidebar | Number rolls up with bounce easing |
| Badge appear | New anomaly/prediction | Slides in from right with overshoot (0.4s spring) |

**Decay:** All Tier 3 animations auto-expire after 30 seconds.

### Animation Budget

| Device | Tier 1 | Tier 2 | Tier 3 | Cursors |
|--------|--------|--------|--------|---------|
| Phone | OFF | Typewriter + tick flash only | Pulse + badge (no border flash) | Active + Working only |
| Tablet | Reduced (scan sweep + pulse ring) | Full | Full | All three |
| Desktop | Full | Full | Full | All three |
| Reduced motion | OFF | OFF | Color only (no movement) | Static symbols |

Constraints:
- Max concurrent Tier 3: 5 (oldest expires first)
- Tier 2 debounce: 1 value change per 2s per element

---

## 5. Information Density — Hero Metric Pattern

### Card Anatomy

```
-- Power Consumption -------------------------------- block

              24.5 W                    (hero, 2.5rem mono)
         avg consumption                (label, 0.6875rem)

   v 12% from yesterday . > 18% below baseline   (context)

   ################.......              (inline sparkline)
   06    12    18    00                 (time axis, micro)

---------------------------------------------------------
```

### Collapsible Sections

- Cursor IS the affordance (block = expanded, underscore = collapsed)
- Collapsed state shows summary count: `-- Correlations -- [12 pairs] -- _`
- Phone: sections below fold start collapsed by default
- Tablet: all visible sections expanded
- Desktop: all sections expanded

---

## 6. Responsive Strategy

### Breakpoints (Mobile-First CSS)

| Token | Width | Device |
|-------|-------|--------|
| Base | < 640px | Phone |
| `min-width: 640px` | 640px - 1023px | Tablet |
| `min-width: 1024px` | 1024px+ | Desktop |

### Navigation

| Device | Pattern | Details |
|--------|---------|---------|
| Phone | Bottom tab bar (5 tabs) | 56px, icon + 10px label. "More" opens slide-up sheet. Swipe between tabs. |
| Tablet | Collapsible side rail | 56px icon-only, expands to 240px on tap. Auto-collapses on content tap. |
| Desktop | Full sidebar | 240px fixed, all labels visible. |

### Grid Layouts

**Home Page Pipeline:**
- Phone: Vertical stack, full-width lane cards, down-arrow connectors
- Tablet: 3-column pipeline (landscape-friendly)
- Desktop: 3-column with arrow connectors + "YOU" guidance nodes

**Intelligence Page:**
- Phone: Hero full-width, metrics 2x2, sections stacked + collapsed
- Tablet: Hero + 4 metrics in row, sections 2-column, expanded
- Desktop: Hero 2-col span + 4 metrics, sections 2-column, all expanded

### Touch Targets

| Device | Min Target | Spacing | Notes |
|--------|-----------|---------|-------|
| Phone | 44x44px | 8px min | Bottom tab icons 48x48px hit area |
| Tablet | 40x40px | 6px min | Section headers are full-width tap targets |
| Desktop | No minimum | — | Hover states + 2px accent focus rings |

### ASCII Scaling

| Device | Bracket Style | Cursors | Terminal Texture |
|--------|--------------|---------|-----------------|
| Phone | Top/bottom borders only (no side frames) | Smaller (0.75em), active + working only | OFF |
| Tablet | Full bracket frames | Normal size, all types | Headers only |
| Desktop | Full frames + terminal texture backgrounds | Normal size, all types | Sidebar + headers + footer |

---

## 7. Chart Upgrade

### Library: uPlot

- 35KB, GPU-accelerated, canvas-based
- Used for: TrendsOverTime (30-day bars), inline sparklines, any time-series
- NOT used for: simple domain breakdowns (keep CSS horizontal bars)

### Style Integration

- Primary data color: `--accent` (cyan)
- Secondary data color: `--accent-warm` (orange) for comparisons/baselines
- Grid lines: `--border-subtle`
- Axis labels: `--type-micro` in `--text-tertiary`
- Background: transparent (card surface shows through)

---

## 8. Migration Notes

### Hardcoded Colors to Fix

- TrendsOverTime.jsx: `#3b82f6`, `#f59e0b`, `#ef4444` -> CSS variables
- Shadow.jsx: `rgba(168,85,247,0.15)` -> CSS variable

### Accessibility Improvements

- Add ARIA labels to all icon-only buttons (sidebar tabs, mobile nav)
- Chart visualizations need data table fallback (hidden, screen-reader accessible)
- Focus states: add visible 2px outline ring (not just border-color change)
- Status badges need text labels alongside color indicators

### Tailwind Bundle

The static Tailwind bundle means new utility classes require rebuilding.
Options:
1. Rebuild bundle with new classes as needed
2. Lean harder into CSS custom properties (already working well for themes)
3. Hybrid: CSS variables for design tokens, Tailwind for layout utilities

Recommendation: Option 2 — extend the CSS variable system. Less build tooling friction.
