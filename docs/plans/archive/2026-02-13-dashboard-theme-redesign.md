# ARIA Dashboard Theme Redesign

**Date:** 2026-02-13
**Status:** Approved
**Scope:** Full visual overhaul of ARIA Preact SPA — dark/light mode, animation system, pixel-art logo aesthetic

## In Plain English

This is the design blueprint for making ARIA's web interface look and feel like a unified system instead of a patchwork of mismatched styles. It defines the exact colors, fonts, animations, and layout rules that every page will follow -- like a brand guide for a company, but for a home intelligence dashboard.

## Why This Exists

When every dashboard page picks its own colors and styles independently, the result feels disjointed and amateurish. More practically, hardcoded colors made dark mode impossible and the lack of a responsive strategy meant the tablet experience was broken. This design document establishes a single source of truth for the visual system -- a shared color palette, animation catalog, responsive breakpoints, and component patterns -- so that every page feels like it belongs to the same product and both light and dark themes work everywhere.

## Problem

The dashboard has two visual languages:
- **Guide page:** Dark theme, cyan accents, scan-line animations, monospace touches — matches the pixel-art logo
- **Everything else:** Generic light-mode SaaS (white cards, shadow-sm, blue accents, rounded-md) — looks AI-generated

The UI needs a unified technical aesthetic that follows the logo's personality across both a dark and light theme.

## Design Decisions

### Theme System: CSS Custom Properties

All colors defined as CSS custom properties in `:root` (light) and `[data-theme="dark"]` (dark). Components reference variables, not hardcoded values. This sidesteps the Tailwind pre-built CSS limitation — new utility classes won't work, but inline `style` with `var()` always works.

Toggle stored in `localStorage`, defaults to `prefers-color-scheme` media query.

### Color Tokens

```css
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

  /* Accent (from logo) */
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
```

### Typography

- **Headings:** System sans-serif, tight letter-spacing
- **Data values:** Monospace (`ui-monospace, 'Cascadia Code', 'Fira Code', monospace`) — entity counts, rates, percentages, timestamps
- **Body text:** System sans-serif, `text-sm` default
- **Section labels:** Uppercase, wider letter-spacing, smaller size

### Spacing & Shape

- **Border radius:** `2px` globally (pixel-art = sharp edges)
- **Card padding:** `16px` standard, `12px` compact
- **Section gaps:** `24px` between sections, `8px` between related items

### Animation Catalog

#### Ambient (Always Running)
| Animation | Where | Purpose |
|-----------|-------|---------|
| Grid pulse | Page background | Faint dot grid with slow radial pulse. Radar sweep feel. |
| Scan line | Section headers | Thin accent line sweeps left-to-right. Extended from Guide page. |
| Status glow | Healthy indicators | Gentle cyan glow pulse. Already exists as `animate-pulse-cyan`. |
| Data stream | Sidebar connection badge | Tiny dots flow upward when connected. "Data is flowing." |
| Border shimmer | Active nav item | Accent-colored border with traveling highlight. |

#### Transitional (On Events)
| Animation | Where | Purpose |
|-----------|-------|---------|
| Card entrance | All cards on page load | Staggered fade-in-up, 0.3s per card, 30ms stagger. |
| Page transition | Route changes | Fade out (100ms), fade in + slide-up (200ms). |
| Data refresh | Cards receiving WS data | Brief cyan border flash (200ms). Shows liveness. |
| Theme toggle | Entire page | 300ms crossfade. Colors transition smoothly. |

#### Interactive (On User Action)
| Animation | Where | Purpose |
|-----------|-------|---------|
| Card hover | Clickable cards | Lift + glow: translateY(-2px) + accent-glow shadow. |
| Button press | All buttons | Scale to 0.97 on click, spring back. |
| Nav hover | Sidebar items | Accent border slides in from left (100ms). |
| Filter activation | Table filters | Accent underline grows from center outward. |

### Responsive Strategy

**Phone (< 768px):**
- Bottom tab bar (existing), theme-aware
- Single-column, full-width cards
- 12px padding
- No ambient animations (battery savings)
- Entrance + interactive animations only

**Tablet (768px–1024px):**
- Sidebar collapses to 48px icon rail, expandable on tap
- 2-column card grids
- Full animations in landscape, reduced in portrait

**Desktop (> 1024px):**
- Full 240px sidebar (existing)
- 3-column grids
- Full animation catalog

### Component Changes

**Cards:** `bg-white rounded-md shadow-sm` → `var(--bg-surface)` + `1px var(--border-subtle)` + `2px radius`
**Section headers:** Add scan-line animation, use `var(--text-primary)`
**Status chips:** Colored pills → monospace uppercase text + left-border accent + status glow
**Data values:** All numeric/data values rendered in monospace font
**Loading skeletons:** Theme-aware pulse colors
**Error states:** Theme-aware error styling
**Form inputs:** Theme-aware borders, backgrounds, focus rings

### Theme Toggle

Location: Sidebar footer, between version info and connection status.
Icon: Sun (light) / Moon (dark) SVG toggle.
Behavior: Persists to `localStorage`, defaults to `prefers-color-scheme`.
Transition: 300ms ease-out on all color properties.

## Agent Team

| # | Agent | Owns | Depends On |
|---|-------|------|-----------|
| 1 | Infrastructure | `index.css` (tokens), `store.js` (theme state), `app.jsx` (theme wrapper), `Sidebar.jsx` (toggle + tablet rail) | Nothing |
| 2 | Animations | `index.css` (animation section), motion utility CSS | #1 |
| 3 | Shared Components | `LoadingState`, `ErrorState`, `StatsGrid`, `DataTable`, `StatusBadge`, `DomainChart`, `AriaLogo` | #1 |
| 4 | Pages | All 12 page files + Intelligence sub-components | #1 + #3 |
| 5 | QA/Integration | No file ownership — reads + reports | #2 + #3 + #4 |

Work order: `1 → (2 + 3 in parallel) → 4 → 5`

## Files Affected

### Infrastructure (Agent 1)
- `aria/dashboard/spa/src/index.css`
- `aria/dashboard/spa/src/store.js`
- `aria/dashboard/spa/src/app.jsx`
- `aria/dashboard/spa/src/components/Sidebar.jsx`

### Animations (Agent 2)
- `aria/dashboard/spa/src/index.css` (animation section only)

### Shared Components (Agent 3)
- `aria/dashboard/spa/src/components/LoadingState.jsx`
- `aria/dashboard/spa/src/components/ErrorState.jsx`
- `aria/dashboard/spa/src/components/StatsGrid.jsx`
- `aria/dashboard/spa/src/components/DataTable.jsx`
- `aria/dashboard/spa/src/components/StatusBadge.jsx`
- `aria/dashboard/spa/src/components/DomainChart.jsx`
- `aria/dashboard/spa/src/components/AriaLogo.jsx`

### Pages (Agent 4)
- `aria/dashboard/spa/src/pages/Home.jsx`
- `aria/dashboard/spa/src/pages/Guide.jsx`
- `aria/dashboard/spa/src/pages/Discovery.jsx`
- `aria/dashboard/spa/src/pages/Intelligence.jsx`
- `aria/dashboard/spa/src/pages/intelligence/*.jsx` (10 sub-components)
- `aria/dashboard/spa/src/pages/Shadow.jsx`
- `aria/dashboard/spa/src/pages/Capabilities.jsx`
- `aria/dashboard/spa/src/pages/DataCuration.jsx`
- `aria/dashboard/spa/src/pages/Predictions.jsx`
- `aria/dashboard/spa/src/pages/Patterns.jsx`
- `aria/dashboard/spa/src/pages/Automations.jsx`
- `aria/dashboard/spa/src/pages/Settings.jsx`

## Success Criteria

1. Both themes work — every page renders correctly in dark and light mode
2. No hardcoded color values in any component (all use CSS vars)
3. Monospace font on all data values
4. 2px border radius globally
5. All ambient animations running in both themes
6. Theme persists across page reloads
7. Tablet icon-rail sidebar works
8. `prefers-reduced-motion` disables all animations
9. esbuild bundle builds without errors
10. Visual consistency — no page looks like it belongs to a different app
