# Interactive Dashboard — Screenshot Audit, Clickable Detail Pages, Configurable Data Sources

**Date:** 2026-02-19
**Status:** Approved
**Approach:** A — Screenshot-First Audit, then Incremental Enhancement

## Summary

Three-phase enhancement to the ARIA dashboard:

1. **Phase 1 — Screenshot Audit:** Headless browser captures every page, compares rendered content against backend API responses, produces mismatch report
2. **Phase 2 — Clickable Detail Sub-Pages:** Every data point on all pages navigates to a dedicated detail view (`#/detail/:type/:id`) with full API data, SUPERHOT visual treatment
3. **Phase 3 — Configurable Data Sources:** Per-module UI for toggling which signals/entities feed each module, using terminal-style `[ON ]/[OFF]` toggles

## Phase 1: Screenshot Audit

### Tool

Puppeteer (headless Chromium) via Node.js script at `aria/dashboard/spa/scripts/screenshot-audit.js`.

### Pages to Capture

All 11 routes: `/`, `/observe`, `/understand`, `/decide`, `/discovery`, `/capabilities`, `/ml-engine`, `/data-curation`, `/validation`, `/settings`, `/guide`.

### Process Per Page

1. Navigate to `http://127.0.0.1:8001/ui/#/<route>`
2. Wait for loading states to resolve (no `LoadingState` components visible)
3. Capture full-page screenshot to `docs/audit/screenshots/<route>.png`
4. Extract rendered text content from the DOM
5. Hit the corresponding backend API endpoints and compare key data points
6. Generate mismatch report to `docs/audit/audit-report.md`

### API Mapping

| Page | API Endpoints |
|------|--------------|
| Home | `/health`, `/api/ml/anomalies`, `/api/shadow/accuracy`, `/api/pipeline`, cache: `intelligence`, `activity_summary`, `automation_suggestions`, `entities` |
| Observe | cache: `intelligence`, `activity_summary`, `presence` |
| Understand | `/api/ml/anomalies`, `/api/shadow/accuracy`, `/api/ml/drift`, `/api/ml/shap`, `/api/patterns` |
| Decide | cache: `automation_suggestions`, `/api/automations/feedback` |
| Discovery | `/api/discovery/status`, `/api/settings/discovery` |
| Capabilities | `/api/capabilities/registry`, `/api/capabilities/candidates` |
| ML Engine | `/api/ml/models`, `/api/ml/drift`, `/api/ml/features`, `/api/ml/hardware`, `/api/ml/online` |
| Data Curation | `/api/curation`, `/api/curation/summary` |
| Validation | `/api/validation/latest` |
| Settings | `/api/config` |
| Guide | Static content — screenshot only |

## Phase 2: Clickable Detail Sub-Pages

### Architecture

New route: `#/detail/:type/:id` handled by a generic `DetailPage.jsx` that dispatches to type-specific renderers.

### Detail Types

| Click Target | Detail Type | API Source | Detail Content |
|---|---|---|---|
| Anomaly card | `anomaly` | `/api/ml/anomalies` + `/api/anomalies/explain` | Full anomaly data, SHAP attribution, path trace, severity history |
| Prediction row | `prediction` | `/api/shadow/predictions` | Prediction vs actual, confidence, Thompson Sampling stats |
| Room in presence | `room` | cache: `presence` | All signals, Bayesian probability breakdown, person history, recent detections |
| Entity anywhere | `entity` | `/api/cache/entities` + `/api/curation` | Entity state, area, device, curation tier, domain, last changed |
| Automation suggestion | `suggestion` | cache: `automation_suggestions` + `/api/automations/feedback` | Full suggestion detail, feedback history, approval/rejection actions |
| Capability | `capability` | `/api/capabilities/registry/:id` | Status, dependencies, health, can_predict flag, discovery history |
| ML model | `model` | `/api/ml/models` | Model type, accuracy, last trained, drift status, feature importance |
| Drift indicator | `drift` | `/api/ml/drift` | Page-Hinkley score, ADWIN windows, rolling MAE, trend chart |
| Pipeline node (Sankey) | `module` | `/api/modules/:id` | Module status, uptime, last error, cache categories owned |
| Config key | `config` | `/api/config/:key` + `/api/config-history` | Current value, change history, default, description |
| Curation entity | `curation` | `/api/curation/:entity_id` + `/api/audit/curation/:entity_id` | Classification, override history, audit trail |
| Correlation pair | `correlation` | cache: `intelligence` | Correlation coefficient, entity pair, time range, trend |
| Baseline metric | `baseline` | cache: `intelligence` | Current vs baseline value, deviation %, history chart |

### SUPERHOT Visual Treatment

- **Page banner:** `PageBanner` with dynamic name — e.g., `DETAIL + ROOM` or `DETAIL + ANOMALY`
- **Breadcrumb:** Terminal-style path: `HOME / OBSERVE / ROOM: LIVING ROOM` — each segment clickable, `--text-tertiary` with `--accent` for current
- **Data freshness:** Full SUPERHOT spectrum — `.sh-cooling` / `.sh-frozen` / `.sh-mantra` based on data age
- **Anomaly details:** Critical anomalies get `.sh-threat-pulse` border + `.sh-glitch` on severity text
- **Click affordance on parent pages:** `.clickable-data` class — `--bg-surface-raised` on hover, `cursor: pointer`, left accent border appears on hover
- **Transition:** `.sh-card-shatter` on the card before navigating (consistent with OODA summary card behavior)
- **Back navigation:** Terminal-style `<- BACK` link below breadcrumb, `--accent` colored, 48px touch target

### Detail Page Layout Pattern

Every detail page follows: **Summary -> Explanation -> History**

```
PageBanner: DETAIL + ANOMALY
HOME / UNDERSTAND / ANOMALY: light.kitchen
<- BACK

t-frame [summary]
  HeroCard: severity + score
  StatsGrid: entity, area, detected_at, duration

t-frame [explanation]
  SHAP attribution bars
  Path trace visualization

t-frame [history]
  TimeChart: anomaly score over time
```

Each detail type adapts this three-section pattern for its data.

## Phase 3: Configurable Data Sources

### Modules and Sources

| Module | Configurable Sources | Default State |
|--------|---------------------|---------------|
| Presence | camera_person, camera_face, motion, light_interaction, dimmer_press, door, media_active, device_tracker | All enabled |
| Activity | Entity domains: light, switch, binary_sensor, media_player, climate, cover | All enabled |
| Anomaly Detection | Which capabilities/entities feed autoencoder + isolation forest | All discovered entities |
| Shadow Engine | Which capabilities are shadow-predicted | All with `can_predict=true` |
| Discovery | Entity domain filter, minimum state-change threshold | Current defaults |

### Backend

- `GET /api/config/modules/{module}/sources` — current source config
- `PUT /api/config/modules/{module}/sources` — update source config
- Storage via existing config system with keys like `presence.enabled_signals`, `activity.enabled_domains`
- Modules read source config at startup and on config change events

### Frontend

Extends existing `InlineSettings` component with "Data Sources" sections:

- Toggle list per source: name, description, `[ON ]/[OFF]` terminal-style toggle
- Changes save immediately via `PUT` with `t2-tick-flash` confirmation
- Disabling a source shows `.sh-frozen` preview of affected data

### Placement

- **Observe** -> "Presence Sources" + "Activity Sources" in InlineSettings
- **Understand** -> "Anomaly Sources" + "Shadow Sources" in InlineSettings
- **Discovery** -> Extend existing settings with domain filter toggles
- **Detail sub-pages** -> Show module source config inline

### Safety

- At least one source must remain enabled per module — UI prevents last toggle-off
- Warning banner when disabling a source contributing >50% of signals
- Config changes logged to audit trail (`/api/audit/events`)

## File Inventory (New/Modified)

### New Files

| File | Purpose |
|------|---------|
| `spa/scripts/screenshot-audit.js` | Puppeteer screenshot + API comparison script |
| `docs/audit/audit-report.md` | Generated mismatch report |
| `docs/audit/screenshots/*.png` | Page screenshots |
| `spa/src/pages/DetailPage.jsx` | Generic detail route dispatcher |
| `spa/src/pages/details/AnomalyDetail.jsx` | Anomaly detail renderer |
| `spa/src/pages/details/RoomDetail.jsx` | Room/presence detail renderer |
| `spa/src/pages/details/EntityDetail.jsx` | Entity detail renderer |
| `spa/src/pages/details/SuggestionDetail.jsx` | Automation suggestion detail |
| `spa/src/pages/details/CapabilityDetail.jsx` | Refactored from existing component |
| `spa/src/pages/details/ModelDetail.jsx` | ML model detail |
| `spa/src/pages/details/DriftDetail.jsx` | Drift detail |
| `spa/src/pages/details/ModuleDetail.jsx` | Pipeline module detail |
| `spa/src/pages/details/ConfigDetail.jsx` | Config key detail |
| `spa/src/pages/details/CurationDetail.jsx` | Curation entity detail |
| `spa/src/pages/details/CorrelationDetail.jsx` | Correlation pair detail |
| `spa/src/pages/details/BaselineDetail.jsx` | Baseline metric detail |
| `spa/src/pages/details/PredictionDetail.jsx` | Prediction detail |
| `spa/src/components/Breadcrumb.jsx` | Terminal-style breadcrumb nav |
| `spa/src/components/TerminalToggle.jsx` | `[ON ]/[OFF]` toggle component |
| `spa/src/components/DataSourceConfig.jsx` | Source toggle list per module |
| `aria/hub/routes_module_config.py` | Backend routes for module source config |

### Modified Files

| File | Change |
|------|--------|
| `spa/src/app.jsx` | Add `DetailPage` route |
| `spa/src/index.css` | Add `.clickable-data` hover styles |
| `spa/src/pages/Home.jsx` | Add click handlers to HeroCards, OodaSummaryCards, Sankey nodes |
| `spa/src/pages/Observe.jsx` | Add click handlers to presence rooms, metrics, activity items |
| `spa/src/pages/Understand.jsx` | Add click handlers to anomalies, predictions, drift, correlations, baselines |
| `spa/src/pages/Decide.jsx` | Add click handlers to suggestions |
| `spa/src/pages/Discovery.jsx` | Add click handlers to discovery items |
| `spa/src/pages/Capabilities.jsx` | Refactor to use detail routing |
| `spa/src/pages/MLEngine.jsx` | Add click handlers to models, drift indicators |
| `spa/src/pages/DataCuration.jsx` | Add click handlers to curation entities |
| `spa/src/pages/Settings.jsx` | Add click handlers to config keys |
| `spa/src/components/InlineSettings.jsx` | Add DataSourceConfig section support |
| `spa/src/components/PresenceCard.jsx` | Add click handlers to rooms |
| `spa/src/components/PipelineSankey.jsx` | Add click handlers to Sankey nodes |
| `spa/src/components/HeroCard.jsx` | Add optional `href` prop for click navigation |
| `spa/src/components/OodaSummaryCard.jsx` | Already has `href` — verify working |
| `aria/hub/server.py` | Register module config routes |
| `aria/modules/presence.py` | Read enabled_signals from config |
| `aria/modules/activity.py` | Read enabled_domains from config |
| `aria/modules/intelligence.py` | Read anomaly source config |
| `aria/hub/config_defaults.py` | Add default source configs |
| `package.json` | Add puppeteer dev dependency |
