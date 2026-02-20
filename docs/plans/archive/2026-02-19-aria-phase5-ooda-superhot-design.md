# ARIA Phase 5: OODA Dashboard Redesign + SUPERHOT Visual Layer

**Date:** 2026-02-19
**Status:** Approved
**Parent:** `2026-02-19-lean-audit-roadmap.md` (Phase 5)
**Location:** `~/Documents/projects/ha-aria/aria/dashboard/spa/`

## Summary

Restructure the ARIA dashboard from a 14-page pipeline-oriented status display into a 5-destination OODA decision tool, layered with superhot-ui visual effects for data freshness, anomaly urgency, and alert dismissal.

## Motivation

ARIA exists to produce two outputs: **automation recommendations** and **anomaly detection**. The current dashboard exposes system internals (Discovery, Capabilities, ML Engine, Data Curation, Validation) that don't serve either output. Phase 5 reorganizes around the user's decision process, not the system's architecture.

## Navigation Restructure

**Current:** 14 pages in 3 sections (Data Collection / Learning / Actions) — organized by system internals.

**New:** 5 primary destinations + 1 expandable System menu — organized by OODA.

| Nav Item | Route | Content | OODA Stage |
|----------|-------|---------|------------|
| **Home** | `/` | Hero KPIs, OODA summary cards, pipeline status bar | Overview |
| **Observe** | `/observe` | Presence, activity stream, live metrics, entity states | Observe |
| **Understand** | `/understand` | Anomalies, patterns, predictions, drift, correlations, shadow accuracy | Orient+Understand |
| **Decide** | `/decide` | Automation recommendations (approve/reject/defer), history, generated YAML | Decide |
| **System** | expandable section | Discovery, Capabilities, ML Engine, Data Curation, Validation, Settings, Guide | Plumbing |

**Phone bottom bar:** Home / Observe / Understand / Decide / More

**Route redirects:** `/intelligence` → `/understand`, `/predictions` → `/understand`, `/automations` → `/decide`, `/presence` → `/observe`, `/patterns` → `/understand`, `/shadow` → `/understand`

## Page Layouts

### Home (OODA Summary)

Three hero cards — the two outputs + meta-trust:

| Card | Normal State | Active State |
|------|-------------|-------------|
| **Anomalies** | "Clear — last anomaly 3d ago" `[fresh]` | "2 detected (1 critical)" `[threat-pulse]` |
| **Recommendations** | "None pending — 4 approved this week" `[fresh]` | "3 pending review" `[fresh]` |
| **Accuracy** | "94% (7-day avg)" `[fresh]` | Same — always informational |

Accuracy uses **trailing 7-day average** from shadow engine daily_trend, not the cumulative overall_accuracy (which is poisoned by historical outlier days).

Below the heroes:

1. **Pipeline status bar** — one line: `Pipeline: ml-active · Shadow: backtest · WebSocket: connected`. ShGlitch on failures.
2. **OODA summary cards** — clickable teasers for Observe/Understand/Decide with key metrics. Entry points when nothing urgent.
3. **Pipeline Sankey** (compact) — existing component with ShFrozen per module node.

### Observe

"What's happening in your home right now."

Reuses existing components:
- `PresenceCard` (wrapped in ShFrozen)
- Live metrics strip: Power, Lights, Events/min, WebSocket (ShFrozen per metric, ShMantra "OFFLINE" if WS down)
- `HomeRightNow` cards (ShFrozen per card)
- `ActivitySection` (activity stream)

### Understand

"What's unusual, what's repeating, and why."

Anomalies at top — primary output. Everything else is context/explanation.

Reuses existing components:
- `AnomalyAlerts` (ShThreatPulse on critical, ShShatter on dismiss, ShGlitch on score text)
- Patterns page content (inline)
- `PredictionsVsActuals`
- `DriftStatus` + `ShapAttributions` (combined section)
- `Baselines` + `TrendsOverTime` + `Correlations` (combined section)
- `ShadowBrief` (shadow accuracy with daily trend)

### Decide

"Recommendations ARIA has generated. Approve, reject, or defer."

- Pending recommendation cards with pattern description, confidence, occurrence count, approve/reject/defer buttons (ShShatter on action)
- History section: approved/rejected/deferred counts, acceptance rate
- Generated YAML section (existing Automations content)

## SUPERHOT Effect Mapping

| Effect | Component / Location | Data Source | Trigger |
|--------|---------------------|-------------|---------|
| **ShFrozen** | Every data card (HeroCard, HomeRightNow items, OODA summary cards, Sankey nodes) | `cache.last_updated` per category | Age: 5min cooling, 30min frozen, 60min stale |
| **ShThreatPulse** | Anomaly hero card, individual anomaly items | Anomaly count with severity critical | New critical anomaly via WebSocket |
| **ShGlitch** | ErrorState, pipeline status bar | `health.modules[x] === 'failed'`, WS disconnect | Module failure or WS drop |
| **ShMantra** | Pipeline status bar, offline sections | `wsConnected === false`, module failed | "OFFLINE" when WS down, "STALE" when data > 60min |
| **ShShatter** | Anomaly dismiss, recommendation approve/reject | User click | Button click → shatter animation → callback removes item |

### CSS Integration

```css
/* In index.css, after Tailwind import */
@import "superhot-ui/css";
```

ARIA's existing `--sh-*` tokens (defined in design-language.md) override superhot-ui defaults. No conflicts.

### Preact Integration

```jsx
import { ShFrozen, ShThreatPulse } from 'superhot-ui/preact';
```

Install: `npm install file:../../superhot-ui` (sibling project).

### Freshness Data Flow

```
useCache('intelligence') → data.last_updated
                         → <ShFrozen timestamp={data.last_updated}>
                         → applyFreshness runs every 30s
                         → card desaturates as data ages
```

## File Changes

### New Files (~5)

| File | Purpose |
|------|---------|
| `src/pages/Observe.jsx` | Assembles Presence + LiveMetrics + HomeRightNow + ActivitySection |
| `src/pages/Understand.jsx` | Assembles Anomalies + Patterns + Predictions + Drift + Shadow |
| `src/pages/Decide.jsx` | Recommendation cards + history + generated YAML |
| `src/components/OodaSummaryCard.jsx` | Clickable summary card for Home page |
| `src/components/PipelineStatusBar.jsx` | Compact one-line system status with ShGlitch/ShMantra |

### Modified Files (~12)

| File | Change |
|------|--------|
| `package.json` | Add superhot-ui dependency |
| `src/index.css` | Import superhot-ui CSS |
| `src/app.jsx` | New routes, remove old routes, add redirects |
| `src/components/Sidebar.jsx` | OODA nav structure + System expandable |
| `src/components/HeroCard.jsx` | Wrap with ShFrozen, add ShThreatPulse variant |
| `src/components/ErrorState.jsx` | Wrap with ShGlitch |
| `src/pages/Home.jsx` | Rebuild as OODA summary |
| `src/pages/intelligence/AnomalyAlerts.jsx` | Add ShThreatPulse + ShShatter on dismiss |
| `src/pages/Automations.jsx` | Embedded into Decide page |
| `src/pages/Patterns.jsx` | Embedded into Understand page |
| `src/pages/intelligence/utils.jsx` | Add freshness helper using superhot-ui |
| `esbuild.config.mjs` | Alias for superhot-ui resolution if needed |

### Archived (removed from primary nav, accessible under System)

Discovery, Capabilities, DataCuration, MLEngine, Shadow, Validation — files unchanged, just moved in nav.

## Testing & Verification

- All existing ~1543 Python tests unaffected (backend unchanged)
- SPA rebuild: `cd aria/dashboard/spa && npm run build`
- Manual verification:
  - Each OODA page loads without errors
  - Freshness states visibly change on cards (wait 5min or mock timestamp)
  - Threat pulse fires when anomalies exist
  - Old hash routes redirect correctly
  - Phone/tablet/desktop nav all work
  - Dark mode: superhot tokens switch correctly
- Vertical trace: trigger snapshot → cache update → Home hero cards refresh → freshness "fresh"

## Out of Scope

- Backend API changes (all data already available via existing endpoints)
- New ML models or prediction logic
- Fixing the 18 remaining open issues (separate from Phase 5)
- Mobile app or push notifications

## Success Criteria

1. Navigation reduced from 14 primary pages to 5 (Home, Observe, Understand, Decide, System)
2. Recommendations and anomalies are the primary views (hero cards on Home)
3. All 5 superhot-ui effects visible in the dashboard
4. Every data card shows freshness state based on cache age
5. Old routes redirect without 404
6. SPA builds and loads without errors
7. Dark mode works with superhot tokens
