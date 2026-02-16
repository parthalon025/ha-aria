# ARIA Dashboard Components

## In Plain English

This is the parts catalog for ARIA's web interface. Each piece listed here is a reusable building block -- like LEGO bricks that snap together to form the pages you see in your browser. Some show charts, some show numbers, some handle navigation.

## Why This Exists

The dashboard has 13 pages built from dozens of shared components. Without a single reference listing what each component does, where it lives, and what data it needs, anyone modifying the UI risks duplicating work or breaking an existing page. This document is the source of truth for the dashboard's visual vocabulary, so every page stays consistent and new features reuse what already exists.

**Stack:** Preact 10 + @preact/signals + Tailwind CSS v4 + uPlot, bundled with esbuild
**Location:** `aria/dashboard/spa/`
**Design language:** `docs/design-language.md` — MUST READ before creating or modifying UI components
**Design doc:** `docs/plans/2026-02-13-aria-ui-redesign-design.md`

## Pages (15)

Home (pipeline flowchart), Discovery, Capabilities, Data Curation, Presence, Intelligence, Predictions, Patterns, Shadow Mode, ML Engine (feature selection, model health), Automations, Settings, Validation (on-demand test suite), Guide (onboarding)

## Sidebar

3 responsive variants — phone bottom tab bar (<640px), tablet icon rail (640-1023px), desktop full sidebar (1024px+). Organized by pipeline stage.

## Reusable Components

| Component | File | Purpose |
|-----------|------|---------|
| `PageBanner` | `components/PageBanner.jsx` | ASCII pixel-art "ARIA + PAGE_NAME" header — first element on every page |
| `CollapsibleSection` | `components/CollapsibleSection.jsx` | Expand/collapse with cursor-as-affordance (cursor-active/working/idle) |
| `HeroCard` | `components/HeroCard.jsx` | Large monospace KPI with optional sparkline (`sparkData`/`sparkColor` props) |
| `TimeChart` | `components/TimeChart.jsx` | uPlot wrapper — full mode (`<figure>`) or `compact` sparkline mode (no axes) |
| `StatsGrid` | `components/StatsGrid.jsx` | Grid of labeled values with `.t-bracket` labels |
| `AriaLogo` | `components/AriaLogo.jsx` | SVG pixel-art logo |
| `UsefulnessBar` | `components/UsefulnessBar.jsx` | Horizontal percentage bar with color thresholds (green/orange/red) |
| `CapabilityDetail` | `components/CapabilityDetail.jsx` | Expanded capability view: 5 usefulness bars, metadata, temporal patterns, entity list |
| `DiscoverySettings` | `components/DiscoverySettings.jsx` | Settings panel: autonomy mode, naming backend, thresholds, Save/Run Now |

## Home Page Data Sources

Interactive 3-lane pipeline dashboard (Data Collection → Learning → Actions) with 9 module nodes, cursor-state indicators, "YOU" guidance nodes, journey progress bar, and live metrics strip.

Data sources (7, fetched in parallel): `/health`, `/api/cache/intelligence`, `/api/cache/activity_summary`, `/api/cache/entities`, `/api/shadow/accuracy`, `/api/pipeline`, `/api/curation/summary`

## Intelligence Sub-Components

Located in `aria/dashboard/spa/src/pages/intelligence/`:

| Component | What it shows |
|-----------|---------------|
| `LearningProgress` | Data maturity bar (collecting → baselines → ML training → ML active) |
| `HomeRightNow` | Current intraday metrics vs baselines with color-coded deltas |
| `ActivitySection` | Activity monitor: swim-lane timeline, occupancy, event rates, patterns, anomalies, WS health |
| `TrendsOverTime` | 30-day small multiples (one chart per metric) + intraday charts |
| `PredictionsVsActuals` | Predicted vs actual metric comparison |
| `Baselines` | Day × metric heatmap grid with color intensity = value |
| `DailyInsight` | LLM-generated daily insight text |
| `Correlations` | Diverging-color correlation matrix heatmap (positive=accent, negative=purple) |
| `SystemStatus` | Run log, ML model scores (R2/MAE), meta-learning applied suggestions |
| `Configuration` | Current intelligence engine config (deprecated — replaced by Settings page) |
| `DriftStatus` | Per-metric drift detection status (Page-Hinkley + ADWIN scores) |
| `AnomalyAlerts` | IsolationForest + autoencoder anomaly alerts |
| `ShapAttributions` | SHAP feature attribution horizontal bar chart |
| `utils.jsx` | Shared helpers: Section, Callout, durationSince, describeEvent, EVENT_ICONS, DOMAIN_LABELS |
