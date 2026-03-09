# ARIA Dashboard -- LLM Design System Guide

**Read this document** alongside the base design guides before modifying the ARIA dashboard.

**Pipeline position:** `ui-template` (base tokens + React components) -> `expedition33-ui` (theme CSS) -> **ha-aria dashboard** (project-specific).

**Dashboard tech:** Preact SPA (not React), esbuild bundler, Tailwind v4 via `@import "tailwindcss"`. No Next.js, no Framer Motion, no TypeScript. All components are `.jsx` files using Preact's `h()` factory. Hash-based routing via `preact-router`.

**Design system imports (index.css):**
```css
@import "tailwindcss";
@import "expedition33-ui";        /* theme tokens + component CSS */
@import "uplot/dist/uPlot.min.css";
@import "../node_modules/superhot-ui/css/superhot.css";
```

**Current state:** The dashboard uses a hybrid of ARIA-specific CSS classes (`t-frame`, `t-status`, `t-card`, `t-bracket`, cursor system, animation tiers) and expedition33-ui tokens (via CSS custom properties). SUPERHOT effects layer on top for data freshness and alert urgency. The migration target is to progressively adopt expedition33-ui component classes (`exp-*`) where they provide richer visual treatment than the current `t-*` classes, while preserving ARIA's terminal-diagnostic personality.

---

## 1. Architecture Context

### Pipeline Layers

| Layer | Package | What It Provides | ARIA's Relationship |
|-------|---------|-----------------|---------------------|
| Base | `ui-template` | Token names, React component API, layout primitives, shadcn/ui wrappers | ARIA does NOT consume ui-template directly (Preact, not React). Token names flow through expedition33-ui. |
| Theme | `expedition33-ui` | 688 CSS custom properties, 27 CSS files + themes/, character Chroma, mood/location/time overlays, atmosphere layers, battle UI, monolith, journal, paint states, animations | ARIA imports this via `@import "expedition33-ui"` in `index.css`. All token values come from here. |
| Effects | `superhot-ui` | Data freshness states, shatter/glitch/mantra effects, threat palette | ARIA imports and force-enables on mobile via `!important` overrides in `index.css`. |
| App | `ha-aria/index.css` | App-specific layout tokens, `t-*` component classes, cursor system, 3-tier animation system, pipeline-specific styles | The file you are editing. |

### Current Design System Coverage

| Area | Coverage | Notes |
|------|----------|-------|
| Color tokens | Full | All via CSS custom properties from expedition33-ui + SUPERHOT palette |
| Typography | Full | Monospace-first (`--font-mono`), 6-step type scale |
| Cards/Frames | Full | `.t-frame` with `data-label`/`data-footer` pseudo-elements |
| Status indicators | Full | `.t-status` with 4 semantic states + cursor system |
| Hero metrics | Full | `HeroCard` component with sparkline support |
| Charts | Full | `TimeChart` (uPlot), `DomainChart`, correlation matrices |
| Data tables | Full | `DataTable` with sort, search, pagination, filter slots |
| Animations | Full | 3-tier system (ambient/data-refresh/status-alert) + SUPERHOT |
| Atmosphere | Partial | CRT overlay, scan lines -- missing: petals, dust, vignette, light beams |
| Game components | Minimal | No monolith, journal, battle HUD, portrait frames, paint states |
| Chroma system | Not started | expedition33-ui has 6 character chromas; ARIA uses only generic accent colors |
| Mood system | Not started | expedition33-ui has 4 moods; ARIA uses none |
| Location/time | Not started | expedition33-ui has location and time-period overlays; ARIA uses none |

### Gap Summary

The deepest gaps are in **narrative-driven components** (journal, monolith, portraits, battle HUD), **atmosphere layers** (petals, dust, vignette, light beams, star field), and **contextual theming** (chroma, mood, location, time-of-day). These are exactly what give the Expedition 33 aesthetic its emotional depth. The migration path is incremental: adopt expedition33-ui classes page-by-page, replacing `t-*` classes where the `exp-*` equivalent provides richer treatment.

---

## 1.5 Strategy Stack

ARIA is an AI-powered home intelligence system. The user's core question: **"Is my home OK?"** Every design decision serves that question.

### UX Strategies (Mission Plan)

| Strategy | Priority | Application |
|----------|----------|-------------|
| **Outcome-Driven** | Primary | The outcome is "home status confidence." Design backward from that. Every page answers a sub-question: Observe="what's happening now?", Understand="what's unusual?", Decide="what should I do?" |
| **Trust & Predictability** | Primary | AI predictions must show confidence levels, not just results. ML training status visible. Anomaly explanations transparent. Users forgive slow AI — not wrong AI that hides its reasoning. |
| **Context-Aware** | High | Mobile quick-check (glance at home status) vs desktop deep-analysis (anomaly drill-down). High-stress (alert received) vs casual (morning check). `data-density="compact"` on mobile, full atmosphere on desktop. |
| **Emotional Journey** | Medium | Normal ops=nostalgic (warm, safe). Anomaly detected=dread (tension, attention). Recovery=dawn (relief, resolution). The mood system maps directly to the user's emotional state. |
| **Systems Thinking** | Medium | ARIA extends beyond the dashboard: Telegram alerts (telegram-brief), watchdog notifications, automation suggestions. The dashboard is one surface of a multi-channel experience. |

### UI Strategies (Weapon System)

| Strategy | Priority | Application |
|----------|----------|-------------|
| **Clarity-First** | Primary | One hero metric per page (MonolithDisplay). Status at a glance (GlyphBadge). No decorative elements in data-dense views. |
| **Trust-Centered** | Primary | ML confidence visible. Prediction accuracy history. Transparent anomaly scoring. StatusDot → Banner → Modal escalation for alerts. |
| **Feedback-Rich** | High | Real-time WebSocket updates. CanvasSaved on automation approval. CrossingOut on anomaly resolution. SUPERHOT freshness states on all data cards. |
| **Progressive Disclosure** | Medium | Overview (Home) → Category (Observe/Understand/Decide) → Detail pages. AdvancedToggle for ML parameters. |
| **Gamified** | Low | TattooStrip for device achievement marks (first-deploy, veteran, longest-uptime). ExpeditionCounter for entity counts. Battle metaphor only where it genuinely clarifies (not decorative). |

### Behavioral Target

**What behavior are we engineering?** Confident monitoring with low anxiety. The user checks ARIA, confirms home status, and moves on with their day. When something is wrong, ARIA escalates clearly and guides action. The worst outcome is false confidence or alarm fatigue.

---

## 2. HA Entity -> Design System Component Mapping

This table maps every Home Assistant intelligence concept surfaced by ARIA to a specific design system component. The "Current Implementation" column describes what exists today. The "Target Component" column specifies the expedition33-ui class or composite to migrate toward.

### Primary Concepts

| HA Concept | Current Implementation | Target Component | Chroma | Data Attributes | Composition | Page(s) |
|------------|----------------------|------------------|--------|-----------------|-------------|---------|
| Entity health (online/offline) | `.t-status` pill + 5px dot | `exp-status` badge (breathing dot) | gustave (online) / enemy (offline) | `data-status="healthy\|warning\|error"` | Inside `exp-frame` or `exp-card` | Discovery, DetailPage |
| Occupancy (home/away) | Text string in `.t-frame` live metrics | `exp-stat-bar[data-bar="hp"]` with probability as fill | lune (home) / sciel (away) | `data-bar="hp"`, `--stat-value` 0-100 | With `exp-portrait` for person display | Observe, Home |
| Power draw (W) | `HeroCard` with large mono value | `exp-monolith` (painted numeral on stone) | gustave | `data-expedition` | Standalone hero position, first element after banner | Observe, Home |
| Anomaly detected | `AnomalyAlerts` section, colored text severity | `exp-frame` + `exp-crossing-out` on critical | maelle | `data-mood="dread"`, `data-paint-state` by severity | Section wrapper with `exp-crossing-out` overlay on critical items | Understand |
| Automation suggestion | `RecommendationCard` with `.t-frame` | `exp-card` glass card + `exp-btn-paint` for approve/reject | gustave (approved) / maelle (rejected) | `data-chroma` per status | Inside `exp-command-bar` layout on Decide page | Decide |
| Battery levels | Not currently surfaced | `exp-stat-bar[data-bar="ap"]` (purple, instant drop) | sciel | `data-bar="ap"`, `--stat-value` 0-100 | In device detail cards | Discovery detail |
| Presence detection (per-room) | `PresenceCard` with `ProbBar` fills | `exp-portrait[data-frame="tarot"]` per person + `exp-stat-bar` per room | Per-person chroma (see mapping below) | `data-chroma`, `data-state="active\|inactive"` | In Observe page grid | Observe |
| ML training progress | `PipelineFlowBar` with status LEDs | `exp-paint-loading` (brushstroke shimmer bar) | lune | `data-paint-state="fresh\|drying\|dried"` | In ML Engine page, per-pipeline stage | MLEngine |
| Prediction confidence | Numeric percentage in colored text | `exp-expedition-counter` (hand-painted inline number) | gustave (high) / enemy (low) | Size variant `sm\|md\|lg` by importance | Inline within `exp-card` or `exp-frame` | Understand, MLEngine |
| Data freshness | Timestamp text, `data-sh-state` on HeroCards | `exp-tattoo-strip` (earned history marks) | sciel | `data-tattoo`, age-derived variant | Positioned in card margin/footer area | All data cards |
| Sankey pipeline | Custom SVG (`PipelineSankey` component) | Custom SVG + `ChromaProvider` per module | Per-module chroma (see pipeline mapping) | `data-chroma` on SVG groups | Home page, full-width section | Home |
| Weekly intelligence brief | Text blocks in Understand page | `exp-journal` (parchment, ruled lines, marginalia) | lune | `data-expedition` for entry numbering | Understand page, below anomalies | Understand |
| Suggestion recommendation | List of `RecommendationCard` items | `exp-command-bar` action items | gustave | `data-chroma="gustave"` on approved, `maelle` on rejected | Decide page primary section | Decide |
| Discovery entity | Table rows in `DataTable` | `exp-table` rows (chroma row accents, tabular-nums) | verso | `data-variant` by entity domain | Discovery page entity table | Discovery |
| Face/person profile | Face recognition cards with confidence bars | `exp-portrait[data-frame="tarot"]` with `exp-avatar` | Per-person chroma assignment | `data-frame="tarot"`, `data-chroma` | Faces page gallery grid | Faces |
| System health overview | Card grid on Home (HeroCards + OodaSummaryCards) | `exp-battle-hud` layout (4-area CSS Grid) | -- (composite) | `data-side="player"` for healthy, `data-side="enemy"` for failing | Home page, wrapping hero section | Home |
| WebSocket status | 8px colored dot + text | `exp-status` badge + `exp-hover-candle` breathing | gustave (connected) / maelle (disconnected) | `data-status`, `data-state="active\|inactive"` | Layout header / sidebar | All pages (Sidebar) |
| Pipeline module status | `PipelineStatusBar` with cursor-state LEDs | `exp-turn-slot` (portrait circles, `data-active` ring) | Per-module chroma | `data-active`, `data-actions` for multi-step | Home page pipeline bar | Home |
| Shadow accuracy (7d avg) | `HeroCard` with percentage | `exp-monolith-compact` (smaller painted numeral) | gustave (>=70%) / enemy (<40%) | `--monolith-color` driven by accuracy tier | Home hero row, Understand ShadowBrief | Home, Understand |
| Drift detection | `DriftStatus` component with colored badges | `exp-status` badges + `exp-fracture` on drift-flagged | maelle (drifting) / gustave (stable) | `data-paint-state="cracked"` on drifted models | Understand page, ML Engine feedback section | Understand, MLEngine |
| SHAP attributions | Horizontal bar chart | `exp-table` with bar-width encoding + `exp-expedition-counter-ghost` watermarks | lune | Chroma-driven bar fills | Understand page section | Understand |
| Correlation matrix | CSS grid with `color-mix()` diverging scale | `exp-table` grid + `data-chroma` driven by correlation sign | gustave (positive) / verso (negative) | Opacity-mapped cells | Understand page section | Understand |
| Baselines (hourly averages) | Heatmap grid with opacity interpolation | `exp-card` grid + `data-paint-state` by deviation | lune (normal) / maelle (anomalous) | `data-paint-state="fresh"` (normal), `"cracked"` (anomalous) | Understand page section | Understand |
| Capability (predictability) | `UsefulnessBar` + collapsible detail | `exp-card-battle` (chroma flood on hover) + `exp-stat-bar` | Per-capability domain chroma | `data-chroma` per capability type | Capabilities page | Capabilities |
| Data curation tier | Tier sections with entity lists | `exp-log-entry` (journal entries with ghost numeral watermarks) | verso (excluded) / gustave (promoted) | `data-variant` by curation status | DataCuration page | DataCuration |
| Validation scenario | Table with per-metric accuracy | `exp-table` with `exp-expedition-counter` in cells | Color by accuracy tier | Chroma-driven cells | Validation page | Validation |
| Config parameter | Slider/toggle/input controls | `exp-input` / `exp-select` (glass fields, brushstroke borders) | sciel | Standard form attributes | Settings page | Settings |

### Person -> Chroma Mapping

When displaying identified persons (Faces page, Observe presence), assign Expedition 33 character chromas by person index:

| Person Index | Chroma | Rationale |
|-------------|--------|-----------|
| Person 1 (primary resident) | gustave | Guardian of the home -- warm gold |
| Person 2 (partner/co-resident) | maelle | Co-creator of the household -- crimson |
| Person 3+ (guests/family) | lune, verso, sciel (rotating) | Visitors, scholars, transient presences |
| Unknown/unidentified | enemy | Adversarial chroma signals unrecognized presence |

### Pipeline Module -> Chroma Mapping

Each ARIA pipeline module maps to a character chroma based on its narrative role:

| Pipeline Module | Chroma | Rationale |
|-----------------|--------|-----------|
| Discovery | verso | The observer/outsider -- scans without intervening |
| Activity Monitor | lune | The scholar/analyst -- watches and records |
| Intelligence (baselines, trends) | lune | Analytical precision, data understanding |
| Anomaly Detection | maelle | Passionate detection -- state-change, the moment things shift |
| Predictions / Shadow | gustave | Guardian role -- protecting through foresight |
| Automations | gustave | Irreversible commits, the weight of action |
| Feedback Loop | sciel | Transcendent guide -- closing the learning loop |
| Presence | lune | Cartographic -- mapping who is where |
| ML Engine | lune | Technical precision, feature engineering |
| Data Curation | verso | Observing and curating from outside |

---

## 3. Page -> Location / Mood / Atmosphere Mapping

Each ARIA page maps to an Expedition 33 location, mood, and atmosphere configuration. These attributes are set on the page's root `<div>` element and cascade through expedition33-ui's token override system.

| ARIA Page | Route | Location | Mood | Atmosphere Layers | Chroma Context | Rationale |
|-----------|-------|----------|------|-------------------|----------------|-----------|
| Home | `/` | lumiere | nostalgic | `exp-vignette`, `exp-dust-field` (subtle gold motes), CRT overlay | gustave (default) | Lumiere is the home city -- warm, familiar, the place you return to. The Home page is the dashboard's hearth. Nostalgic warmth with CRT scan lines preserves the terminal feel. |
| Observe | `/observe` | continent | wonder | `exp-vignette`, `exp-canvas-texture` (faint), `exp-light-beam` (single diagonal) | lune | The Continent is where you explore and discover. Observe is live data -- the wonder of seeing your home in real-time. A single light beam sweeps like a searchlight. |
| Understand | `/understand` | continent | nostalgic -> dread (dynamic) | `exp-vignette`, `exp-dust-field`, conditionally `exp-crossing-out` on anomaly sections | lune (normal), maelle (anomalies active) | Understanding shifts from calm analysis (nostalgic) to dread when anomalies are detected. The mood should dynamically shift based on anomaly count: 0 = nostalgic, 1-2 = wonder (mild alertness), 3+ = dread. |
| Decide | `/decide` | lumiere | dawn | `exp-vignette`, `exp-canvas-saved` (on approval action) | gustave | Deciding is the moment of hope and action -- dawn mood. Approving a suggestion triggers a `canvas-saved` golden glow pulse. Rejecting triggers a brief `crossing-out` flash. |
| Discovery | `/discovery` | continent | wonder | `exp-vignette`, `exp-canvas-texture` | verso | Discovery is exploration -- cataloging the unknown. Verso's chroma (deep void navy) suits the observer role. Wonder mood for the thrill of mapping your home's 3,000+ entities. |
| Capabilities | `/capabilities` | lumiere | nostalgic | `exp-vignette`, `exp-dust-field` (sparse) | gustave | Capabilities are your home's known strengths -- warm familiarity. Gold chroma for the reliable systems that protect you. |
| ML Engine | `/ml-engine` | monolith | wonder -> dread (dynamic) | `exp-vignette`, `exp-paint-loading` during training, `exp-monolith` for key numbers | lune | The ML Engine is the Monolith -- the massive, inscrutable system that determines outcomes. Wonder when models train successfully; dread when drift is detected. The monolith location brings dark stone surfaces and dramatic numerals. |
| Data Curation | `/data-curation` | continent | nostalgic | `exp-vignette`, `exp-journal` aesthetic for entity lists | verso | Curation is careful, scholarly work -- sorting through expedition findings. Journal-like presentation with marginalia annotations for curation reasons. |
| Validation | `/validation` | monolith | dawn (passing) / dread (failing) | `exp-vignette`, `exp-monolith` for accuracy numbers | gustave (passing) / maelle (failing) | Validation faces the Monolith's judgment -- your system's accuracy. Dawn mood when tests pass (hope, resolution). Dread when they fail. |
| Settings | `/settings` | lumiere | nostalgic | `exp-vignette` only (minimal atmosphere) | sciel | Settings are the guide's domain -- Sciel's silver transcendence. Calm, minimal atmosphere. The guide adjusts dials and knobs without drama. |
| Guide | `/guide` | lumiere | nostalgic | `exp-vignette`, `exp-canvas-texture` (warm parchment feel) | sciel | The Guide page is a journal -- expedition notes explaining how ARIA works. Warm parchment with Sciel's guiding hand. |
| Faces | `/faces` | lumiere | nostalgic | `exp-vignette`, `exp-portrait` frames for each person | Per-person chroma | The Faces page is a portrait gallery. Each person gets a tarot-frame portrait with their assigned chroma. Nostalgic warmth -- these are the people you know. |
| DetailPage | `/detail/:type/:id/:rest*` | Inherited from parent page context | Inherited | Inherited + `exp-frame` tarot card for focused entity | Inherited from entity/module type | Detail pages inherit the atmosphere of their parent concept. An anomaly detail inherits dread; a capability detail inherits nostalgic. |

### Detail Page Types

The ARIA dashboard supports 13 detail page types at `/detail/:type/:id/:rest*`:

| Type | Component | Content |
|------|-----------|---------|
| `anomaly` | AnomalyDetail | Anomaly deep-dive with SHAP attributions |
| `baseline` | BaselineDetail | Entity baseline patterns and deviations |
| `capability` | CapabilityDetail | System capability analysis |
| `config` | ConfigDetail | Configuration item details |
| `correlation` | CorrelationDetail | Cross-entity correlation analysis |
| `curation` | CurationDetail | Data curation workflow status |
| `drift` | DriftDetail | Model drift analysis |
| `entity` | EntityDetail | Individual HA entity detail |
| `model` | ModelDetail | ML model performance metrics |
| `module` | ModuleDetail | Pipeline module status |
| `prediction` | PredictionDetail | Prediction accuracy drill-down |
| `room` | RoomDetail | Room-level aggregation |
| `suggestion` | SuggestionDetail | Automation suggestion details |

### Dynamic Mood Rules

Moods should shift dynamically based on data state:

```
// Pseudo-logic for mood attribute
if (page === 'understand') {
  const anomalyCount = anomalies?.length || 0;
  if (anomalyCount === 0) mood = 'nostalgic';
  else if (anomalyCount <= 2) mood = 'wonder';
  else mood = 'dread';
}

if (page === 'ml-engine') {
  const hasDrift = drift?.flagged > 0;
  mood = hasDrift ? 'dread' : 'wonder';
}

if (page === 'validation') {
  const allPassing = results?.every(r => r.status === 'passed');
  mood = allPassing ? 'dawn' : 'dread';
}
```

Set as: `<div data-mood={mood} data-location={location}>` on the page root element.

---

## 4. Current -> Target Migration Table

Every current CSS pattern and component mapped to its target replacement, with migration steps.

### CSS Classes

| Current Class | Current Behavior | Target Class | Migration Steps |
|---------------|-----------------|-------------|-----------------|
| `.t-frame` | Card with `data-label`/`data-footer` pseudo-elements, box-shadow | `exp-frame` | 1. Add `exp-frame` alongside `t-frame`. 2. Map `data-label` to `exp-frame`'s `data-label` (Cinzel heading). 3. Corner ornaments appear automatically. 4. Remove `t-frame` once all instances converted. |
| `.t-card` | Basic card (deprecated, use `.t-frame`) | `exp-card` | Direct replacement. `.t-card` is already deprecated in design-language.md. |
| `.t-card-hover` | Hover lift + accent border | `exp-card[data-chroma]` | Chroma flood on hover replaces translateY lift. Set `data-chroma` per context. |
| `.t-status` | Monospace pill with left-border color | `exp-status` | `exp-status` adds breathing dot animation. Map status classes: `.t-status-healthy` -> `data-status="healthy"`, etc. |
| `.t-section-header` | Border-bottom + scan-line `::after` sweep | `exp-divider` (ornament variant) | Replace scan-line sweep with Cinzel label divider or `diamond` ornament. Keeps section delineation but gains Expedition 33 character. |
| `.t-btn-primary` | Accent background, brightness hover | `exp-btn-primary` | Gilt-edged gold gradient. Richer than flat accent. |
| `.t-btn-secondary` | Raised surface, subtle border | `exp-btn-secondary` | Ornament border treatment. |
| `.t-btn` (approve/reject in Decide) | Colored background buttons | `exp-btn-paint` | Chroma flood on hover, ink drip, Gommage collapse on press. Significantly more dramatic for irreversible actions. |
| `.t-input` / `select` | Surface bg, accent focus ring | `exp-input` / `exp-select` | Glass fields with brushstroke borders. Focus ring becomes Chroma-driven gold ring. |
| `.t-bracket` | `[text]` with bracket pseudo-elements | Keep `.t-bracket` | Terminal-specific, no expedition33-ui equivalent. Unique to ARIA's diagnostic personality. |
| `.t-terminal-bg` | Scan-line texture overlay | `exp-canvas-texture` | Replace CRT scan lines with oil paint stroke overlay. Both serve as surface texture; canvas texture fits the painted world metaphor. |
| `.t-callout` | Left accent border, surface bg | `exp-journal` (inline variant) | Short callouts become journal margin notes. Longer ones become full journal entries. |
| `.t-nav-bracket` | Angle bracket nav `< a >` | Keep or `exp-tabs` underline variant | Terminal bracket nav is unique to ARIA. Consider keeping for sidebar, using `exp-tabs` for in-page tabbed content. |
| `.clickable-data` | Left border reveal on hover | `exp-hover-candle` | Candle warmth approach replaces mechanical left-border. Subtle warm glow instead of accent stripe. |
| `cursor-active/working/idle` | Blinking cursor expand/collapse | Keep cursor system | Terminal cursors are ARIA's signature. No expedition33-ui equivalent. Keep unchanged. |

### Components

| Current Component | File | Target Treatment | Migration Steps |
|-------------------|------|-----------------|-----------------|
| `HeroCard` | `components/HeroCard.jsx` | `exp-monolith` for primary KPIs, keep HeroCard for secondary | 1. Home page hero: wrap primary metric in `exp-monolith`. 2. Retain HeroCard for grids of 3 (too many monoliths dilute impact). 3. Add `data-paint-state` driven by `computeFreshness()`. |
| `StatusBadge` | `components/StatusBadge.jsx` | `exp-status` pill | 1. Replace `t-status` + dot with `exp-status`. 2. Map `on/home` -> healthy, `unavailable/unknown` -> error, numeric -> healthy, else -> waiting. 3. Breathing dot animation replaces static 5px circle. |
| `OodaSummaryCard` | `components/OodaSummaryCard.jsx` | `exp-card-battle` with `sh-card-shatter` preserved | 1. Replace `.t-frame.t-card-hover` with `exp-card-battle`. 2. Keep SUPERHOT shatter on click-to-navigate. 3. Add `data-chroma` per OODA section (observe=lune, understand=lune, decide=gustave). |
| `PageBanner` | `components/PageBanner.jsx` | Keep PageBanner + add `exp-page-border` | 1. PageBanner SVG pixel art is unique to ARIA; keep it. 2. Wrap page content in `exp-page-border` for corner bracket viewport frame. 3. Set `data-location` and `data-mood` on page root div. |
| `PipelineSankey` | `components/PipelineSankey.jsx` | Custom SVG + `data-chroma` per module group | 1. Apply module chroma mapping (see section 2) to SVG node fills. 2. Use `exp-hover-candle` glow on node hover. 3. Add `exp-paint-loading` to actively-running nodes. |
| `PipelineStatusBar` | `components/PipelineStatusBar.jsx` | `exp-turn-queue` + `exp-turn-slot` per module | 1. Replace LED dots with `exp-turn-slot` portrait circles. 2. Use module chroma for each slot. 3. `data-active` on running modules shows animated ring. 4. Dashed connectors between slots already exist in expedition33-ui. |
| `CollapsibleSection` | `components/CollapsibleSection.jsx` | Keep + add `exp-frame` wrapper | 1. CollapsibleSection's cursor system is unique; keep it. 2. Wrap content in `exp-frame` for corner ornaments. 3. Add `data-label` for Cinzel headings. |
| `PresenceCard` | `components/PresenceCard.jsx` | `exp-portrait` grid + `exp-stat-bar` per room | 1. Replace `ProbBar` with `exp-stat-bar[data-bar="hp"]`. 2. Add `exp-portrait` circles for identified persons. 3. Use per-person chroma mapping. |
| `LoadingState` | `components/LoadingState.jsx` | `exp-skeleton` (chroma sweep shimmer) | 1. Replace pulse animation with `exp-skeleton` shimmer. 2. Skeleton sweep direction and chroma driven by page context. |
| `ErrorState` | `components/ErrorState.jsx` | `exp-empty-state` + `exp-crossing-out` | 1. Use `exp-empty-state` (Monolith-style void) for empty data. 2. Overlay `exp-crossing-out` brushstroke for error states. 3. Red flash + chromatic split for connection failures. |
| `DataTable` | `components/DataTable.jsx` | `exp-table` | 1. Replace inline table styles with `exp-table` classes. 2. Add chroma row accents based on entity status. 3. Use `exp-table__num` for tabular-nums columns (R2, MAE, percentages). |
| `TimeChart` | `components/TimeChart.jsx` | Keep (uPlot) + `exp-frame` wrapper | uPlot is performant and well-integrated. Wrap chart containers in `exp-frame` for visual consistency. Add `data-paint-state` for freshness. |
| `StatsGrid` | `components/StatsGrid.jsx` | `exp-layout-grid` + `exp-card` per stat | 1. Replace custom grid with `exp-layout-grid[data-dense]`. 2. Each stat becomes an `exp-card` with `exp-expedition-counter` for the value. |
| `Sidebar` | `components/Sidebar.jsx` | Keep + add `exp-layout-dashboard` patterns | Sidebar navigation is functional and ARIA-specific. Add `exp-divider` between nav groups. Consider `data-chroma` on active nav item matching the page's chroma context. |

---

## 5. OODA -> Narrative Arc Mapping

ARIA's dashboard follows the OODA loop (Observe -> Orient -> Decide -> Act). This maps naturally onto Expedition 33's story progression, creating a narrative arc that gives data work emotional texture.

| OODA Phase | ARIA Pages | Expedition 33 Parallel | Emotional Register | Design Treatment |
|------------|-----------|----------------------|-------------------|-----------------|
| **Observe** | Home, Observe | **Expedition departure** -- leaving the harbor, scanning the horizon. The moment of setting out to understand what lies ahead. | Wonder, alertness, potential | Continent location, wonder mood. Live data streams feel like new terrain being mapped. `exp-light-beam` sweeps like a searchlight. Fresh data is vivid; stale data freezes (SUPERHOT). |
| **Orient** | Understand, MLEngine, DataCuration | **Traversing the Continent** -- encountering anomalies, building understanding, learning patterns. The expedition's middle act where knowledge accrues and danger clarifies. | Analytical intensity, growing unease when anomalies mount | Continent/monolith location. Mood shifts dynamically: nostalgic (calm analysis) -> wonder (interesting patterns) -> dread (mounting anomalies). `exp-crossing-out` on detected anomalies. `exp-monolith` for key ML numbers. |
| **Decide** | Decide, Capabilities, Validation | **The Choice** -- standing before the Monolith, making irreversible decisions. Approving automations is committing to change. | Moral weight, hope (dawn mood), gravity of commitment | Lumiere location, dawn mood. `exp-btn-paint` for approve/reject (Chroma flood, Gommage collapse). `exp-canvas-saved` glow on approval. Every decision is visible and weighted. |
| **Act** | Home (feedback metrics), Settings | **Resolution** -- the expedition's outcome, returning home changed. Settings are the knobs that tune the system's behavior going forward. | Quiet confidence, maintenance, watchfulness | Lumiere location, nostalgic mood. The cycle completes and begins again. Feedback metrics show whether past decisions improved outcomes. Gold warmth returns. |

### Narrative Transitions Between Phases

When navigating between OODA phases, the visual transition should echo the emotional shift:

- **Observe -> Orient**: Page transition uses `exp-paint-in` (left-to-right reveal) -- entering new understanding.
- **Orient -> Decide**: Page transition uses `exp-crossing-out` briefly then `exp-canvas-saved` -- the moment of judgment.
- **Decide -> Act**: Page transition uses `exp-void-erase` (brief) then fade-in -- commitment, then the dust settles.
- **Act -> Observe**: Standard `animate-page-enter` (existing) -- the cycle quietly resets.

Note: ARIA currently uses `animate-page-enter` (translateY + fade) for all transitions. Narrative transitions are a future enhancement, not a requirement.

---

## 6. Composition Examples

### 6.1 Home Page Hero Section

```
<div data-location="lumiere" data-mood="nostalgic" class="space-y-6 animate-page-enter">
  <!-- Atmosphere -->
  <div class="exp-vignette" />
  <div class="exp-dust-field" style="--dust-count: 12; --dust-color: var(--chroma-gustave);" />

  <!-- Banner (ARIA-specific, keep) -->
  <PageBanner page="HOME" subtitle="Your home at a glance." />

  <!-- Hero metrics as battle HUD layout -->
  <div class="exp-battle-hud" data-side="player">
    <!-- Primary: anomaly count as monolith -->
    <div class="exp-monolith" data-chroma="maelle">
      <span class="exp-monolith__number">{anomalyCount}</span>
      <span class="exp-monolith__label">anomalies</span>
    </div>

    <!-- Secondary: stat bars for recommendations and accuracy -->
    <div class="exp-battle-panel" data-chroma="gustave">
      <div class="exp-stat-bar" data-bar="hp" style="--stat-value: {accuracyPct};">
        <span>Accuracy</span>
      </div>
      <div class="exp-stat-bar" data-bar="ap" style="--stat-value: {reviewedPct};">
        <span>Reviewed</span>
      </div>
    </div>
  </div>

  <!-- OODA summary cards -->
  <div class="exp-layout-grid" data-dense style="--card-min-width: 200px;">
    <a href="#/observe" class="exp-card-battle" data-chroma="lune">
      <span class="exp-card-battle__title">Observe</span>
      <span class="exp-expedition-counter">{occupancy}</span>
    </a>
    <a href="#/understand" class="exp-card-battle" data-chroma="lune">
      <span class="exp-card-battle__title">Understand</span>
      <span class="exp-expedition-counter">{anomalyCount}</span>
    </a>
    <a href="#/decide" class="exp-card-battle" data-chroma="gustave">
      <span class="exp-card-battle__title">Decide</span>
      <span class="exp-expedition-counter">{pendingCount}</span>
    </a>
  </div>

  <!-- Pipeline Sankey (custom, chroma-annotated) -->
  <PipelineSankey moduleStatuses={health} cacheData={cacheData} />
</div>
```

### 6.2 Anomaly Detail Page

```
<div data-location="continent" data-mood="dread" class="space-y-6 animate-page-enter">
  <div class="exp-vignette" />

  <PageBanner page="ANOMALY" subtitle="Detected deviation from learned patterns." />

  <!-- Anomaly as monolith (the threatening number) -->
  <div class="exp-monolith exp-monolith--critical" data-chroma="maelle">
    <span class="exp-monolith__number">{anomaly.score}</span>
    <span class="exp-monolith__label">severity score</span>
  </div>

  <!-- Anomaly detail as journal entry -->
  <div class="exp-journal" data-chroma="maelle">
    <div class="exp-journal__header">
      <span class="exp-journal__date">{anomaly.detected_at}</span>
      <span class="exp-journal__author">ARIA Anomaly Detector</span>
    </div>
    <div class="exp-journal__body">
      <p>{anomaly.description}</p>
    </div>
    <div class="exp-journal__annotation">
      Entity: {anomaly.entity_id}<br/>
      Method: {anomaly.method}<br/>
      Confidence: <span class="exp-expedition-counter exp-expedition-counter--sm">{confidence}%</span>
    </div>
  </div>

  <!-- SHAP attributions as framed table -->
  <div class="exp-frame" data-label="Contributing Factors">
    <table class="exp-table">
      <thead>
        <tr>
          <th>Feature</th>
          <th class="exp-table__num">Impact</th>
        </tr>
      </thead>
      <tbody>
        {shapValues.map(f => (
          <tr data-chroma={f.impact > 0 ? 'maelle' : 'gustave'}>
            <td>{f.feature}</td>
            <td class="exp-table__num">{f.impact.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>

  <!-- Action: crossing-out animation on dismiss -->
  <button class="exp-btn-paint" data-chroma="gustave"
    onClick={() => { /* triggers exp-crossing-out then removes */ }}>
    Dismiss Anomaly
  </button>
</div>
```

### 6.3 ML Engine Training View

```
<div data-location="monolith" data-mood="wonder" class="space-y-6 animate-page-enter">
  <div class="exp-vignette" />

  <PageBanner page="MLENGINE" subtitle="How ARIA learns your home." />

  <!-- Pipeline as turn queue -->
  <div class="exp-turn-queue">
    <div class="exp-turn-slot" data-active data-chroma="verso">
      <span>Data</span>
    </div>
    <div class="exp-turn-slot" data-active data-chroma="lune">
      <span>Features</span>
    </div>
    <div class="exp-turn-slot" data-chroma="lune">
      <span>Models</span>
      <!-- exp-paint-loading when training in progress -->
      <div class="exp-paint-loading" data-paint-state="fresh" />
    </div>
    <div class="exp-turn-slot" data-chroma="gustave">
      <span>Predict</span>
    </div>
    <div class="exp-turn-slot" data-chroma="sciel">
      <span>Feedback</span>
    </div>
  </div>

  <!-- Feature selection as framed list -->
  <div class="exp-frame" data-label="Selected Features">
    <div class="exp-layout-grid" data-dense style="--card-min-width: 120px;">
      {features.map((name, i) => (
        <div class="exp-card" data-chroma="lune" key={name}>
          <span class="exp-expedition-counter-ghost">{i + 1}</span>
          <span>{name}</span>
        </div>
      ))}
    </div>
  </div>

  <!-- Model scores as monolith numbers -->
  <div class="exp-layout-grid" style="--card-min-width: 180px;">
    {Object.entries(scores).map(([name, vals]) => (
      <div class="exp-monolith exp-monolith--compact" data-chroma="lune" key={name}>
        <span class="exp-monolith__number">{vals.r2?.toFixed(2)}</span>
        <span class="exp-monolith__label">{name} R2</span>
      </div>
    ))}
  </div>

  <!-- Training history as journal -->
  <div class="exp-journal" data-chroma="lune">
    <div class="exp-journal__header">
      <span class="exp-journal__date">{lastTrainedDate}</span>
      <span class="exp-journal__author">ARIA Training Pipeline</span>
    </div>
    <div class="exp-journal__body">
      <p>Trained {totalSnapshots} snapshots across {targets.length} prediction targets.
      Validation split: {validationSplit}.</p>
    </div>
  </div>
</div>
```

### 6.4 Observe Live Page

```
<div data-location="continent" data-mood="wonder" class="space-y-6 animate-page-enter">
  <div class="exp-vignette" />
  <div class="exp-light-beam" />

  <PageBanner page="OBSERVE" subtitle="Live view of your home." />

  <!-- Live metrics in framed strip -->
  <div class="exp-frame" data-label="live metrics">
    <div class="flex flex-wrap items-center gap-x-5 gap-y-2">
      <!-- Occupancy as portrait + stat -->
      <div class="flex items-center gap-2">
        <div class="exp-avatar exp-avatar--sm" data-chroma="lune" data-state={anyoneHome ? 'active' : 'inactive'} />
        <span class="exp-expedition-counter exp-expedition-counter--sm">
          {anyoneHome ? 'Home' : 'Away'}
        </span>
      </div>

      <!-- Event rate -->
      <div class="flex items-center gap-2" data-chroma="lune">
        <span class="exp-status" data-status="healthy">
          <span>{eventRate}/min</span>
        </span>
      </div>

      <!-- WebSocket status -->
      <div class="flex items-center gap-2">
        <span class="exp-status" data-status={wsConnected ? 'healthy' : 'error'}>
          {wsConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
    </div>
  </div>

  <!-- Presence as portrait gallery -->
  <div class="exp-frame" data-label="room presence">
    <div class="exp-layout-grid" style="--card-min-width: 160px;">
      {rooms.map(([roomName, roomData]) => (
        <div class="exp-card" data-chroma={roomData.probability > 0.7 ? 'gustave' : 'verso'} key={roomName}>
          <span class="exp-card__title">{roomName}</span>
          <div class="exp-stat-bar" data-bar="hp"
            style={`--stat-value: ${Math.round(roomData.probability * 100)};`}>
            <span>{Math.round(roomData.probability * 100)}%</span>
          </div>
          {roomData.persons?.map(person => (
            <div class="exp-portrait exp-portrait--sm" data-chroma={personChroma(person)} key={person}>
              <span>{person}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  </div>

  <!-- Activity timeline (keep existing custom implementation) -->
  <HomeRightNow intraday={intraday} baselines={baselines} />
  <ActivitySection activity={activity} />
</div>
```

### 6.5 Decide Page (Automation Recommendations)

```
<div data-location="lumiere" data-mood="dawn" class="space-y-6 animate-page-enter">
  <div class="exp-vignette" />

  <PageBanner page="DECIDE" subtitle="Automation suggestions to review." />

  <!-- Hero: pending count as monolith -->
  <div class="exp-monolith" data-chroma="gustave">
    <span class="exp-monolith__number">{pendingCount}</span>
    <span class="exp-monolith__label">pending review</span>
  </div>

  <!-- Decision health bar (replaces HealthBar component) -->
  <div class="exp-frame" data-label="decision health">
    <div class="exp-stat-bar" data-bar="hp"
      style={`--stat-value: ${Math.round((approved / total) * 100)};`}>
      <span>Approved {Math.round((approved / total) * 100)}%</span>
    </div>
  </div>

  <!-- Pending recommendations -->
  <div class="space-y-4">
    {pending.map(suggestion => (
      <div class="exp-frame" data-label={suggestion.name} data-chroma="gustave" key={suggestion.id}>
        <!-- Status badge -->
        <span class="exp-status" data-status="warning">pending</span>

        <!-- Confidence as expedition counter -->
        <span class="exp-expedition-counter exp-expedition-counter--sm"
          data-chroma={suggestion.confidence >= 0.8 ? 'gustave' : suggestion.confidence >= 0.5 ? 'sciel' : 'enemy'}>
          {Math.round(suggestion.confidence * 100)}%
        </span>

        <p>{suggestion.description}</p>

        <!-- Conflict warning (section-level mood override) -->
        {hasConflicts && (
          <div data-mood="dread">
            <span class="exp-status" data-status="error">
              Conflict: {conflict.name} ({conflict.reason})
            </span>
          </div>
        )}

        <!-- Action buttons -->
        <div class="flex gap-2 mt-4">
          <button class="exp-btn-paint" data-chroma="gustave">Approve</button>
          <button class="exp-btn-paint" data-chroma="maelle">Reject</button>
          <button class="exp-btn-secondary">Defer</button>
        </div>
      </div>
    ))}
  </div>
</div>
```

---

## 7. Data Flow Visualization Rules

ARIA handles four categories of data that each require distinct visual treatment: real-time streams, historical records, predictions, and anomalies. These rules define how each category should be represented in the UI.

### 7.1 Real-Time Data (WebSocket Updates)

Real-time data flows from the Hub via WebSocket `cache_updated` messages. The SPA's `@preact/signals` store marks the affected category stale and triggers a background re-fetch. Components subscribe via `useCache(categoryName)`.

**Cache categories:** `intelligence`, `activity_summary`, `entities`, `devices`, `areas`, `capabilities`, `automation_suggestions`, `patterns`, `presence`.

**Visual Rules:**

| Rule | Implementation | Rationale |
|------|---------------|-----------|
| **Fresh data = vivid** | Full color, full opacity. `data-sh-state` not set. `data-paint-state="fresh"` (wet shimmer). | Fresh data is the most reliable, most actionable. |
| **Aging data = degrading** | SUPERHOT freshness states: fresh (0-5min) -> cooling (5-30min) -> frozen (30-60min) -> stale (60min+). Visual: desaturation + dimming. Layered with paint states (see Section 9). | Time since update directly indicates reliability. Tufte: encode information in every visual dimension. |
| **Refresh flash** | `.animate-data-refresh` (box-shadow pulse) on the affected card when new data arrives. `.t2-typewriter` for value changes. | Preattentive processing (Treisman): motion draws attention to changes without requiring scanning. |
| **Connection indicator** | Colored dot (green/red) in sidebar footer and Observe live metrics strip. `exp-status` badge with breathing dot for connected state. | WebSocket = the lifeline. Its state must be always visible (calm technology: status in peripheral chrome). |
| **Stale-while-revalidate** | Existing data stays visible (with `data-sh-state` degradation) while re-fetch runs. Never blank the screen. `loading` flag only triggers spinner on first fetch. | Perceived performance. Users see outdated data (visually marked as stale) rather than a loading spinner. |
| **Update propagation** | Signal update (0ms) -> stale mark (0ms) -> re-fetch (async) -> re-render (next frame) -> t2 animation (next frame) -> freshness reset (0ms). | Non-blocking pipeline. No component ever blocks on a fetch. |

**Anti-Patterns:**
- Never show a loading spinner when cached data exists. Use stale-while-revalidate.
- Never flash the entire page on a single category update. Flash only the affected card/section.
- Never remove freshness indicators during loading. Stale data should look stale, even while refreshing.

### 7.2 Historical Data

Historical data comes from the intelligence engine (daily/intraday snapshots, training history, trend data). It is fetched via REST API on page mount, not pushed via WebSocket.

**Sources:** `/api/ml/features`, `/api/ml/models`, `/api/ml/drift`, `/api/ml/pipeline`, `/api/shadow/accuracy`, plus `intelligence` cache (contains `intraday_trend`, `baselines`, `trend_data`, `entity_correlations`).

**Visual Rules:**

| Rule | Implementation | Rationale |
|------|---------------|-----------|
| **Time series = line charts** | `TimeChart` (uPlot wrapper) for continuous metrics. One series per chart unless same unit. | Cleveland & McGill: position along a common scale is the most accurate perceptual channel for quantitative data. |
| **Small multiples for different units** | Stack multiple TimeCharts sharing x-axis, one per metric. Never overlay different units on one y-axis. | Tufte: small multiples prevent dual-axis confusion. |
| **Sparklines for KPIs** | HeroCard/MonolithDisplay includes sparkline slot for 24h/7d trend. 80x32px. No axis labels. `sparkData` prop. | Tufte: "data graphics should draw the viewer's attention to the substance." Sparklines are data-dense. |
| **Rolling averages over raw** | Show 7-day rolling average as primary line. Raw points as scatter or thin secondary line (when both fit). | Noise reduction. Raw points show variance, rolling average shows trend. |
| **Historical mood = sepia** | Set `data-time="memory"` on sections displaying historical comparisons (e.g., "last week vs this week"). Adds sepia + film-grain treatment. | Expedition 33 narrative: memory/flashback = sepia tone. Distinguishes historical from live data. |
| **Heatmaps for temporal patterns** | CSS grid with opacity interpolation for day-of-week x hour matrices. Used in baselines display. | Reveals temporal clustering without requiring the user to scan a table of numbers. |

**Anti-Patterns:**
- Never overlay more than 3 series on one chart. Use small multiples.
- Never use pie charts. Use bar charts or stacked bars instead (Cleveland & McGill: angle is less accurate than position/length).
- Never omit y-axis labels on full charts (sparklines are the exception).

### 7.3 Predictions

Predictions come from the shadow engine (predict-compare-score loop) and the ML forecaster. Displayed on Understand, ML Engine, and Home pages.

**Sources:** `intelligence` cache (`predictions` key), `/api/shadow/accuracy`, `/api/ml/pipeline` (predictions section).

**Visual Rules:**

| Rule | Implementation | Rationale |
|------|---------------|-----------|
| **Predicted vs Actual** | Dual-line chart: solid line = actual, dashed line = predicted. Both share x/y axis. | Dashed = uncertain/estimated is a near-universal convention. |
| **Confidence interval** | Semi-transparent band around prediction line when confidence data available. Width = confidence range. | Shows uncertainty directly. Users see the prediction's self-assessed reliability. |
| **Accuracy as HP bar** | Shadow accuracy displayed as `exp-stat-bar[data-bar="hp"]`. 100% = full green, degrading through amber to red. Currently shown via colored text in ShadowBrief. | HP metaphor: the model's "health" is its accuracy. Intuitive for the game-inspired UI. |
| **Prediction score** | `exp-expedition-counter` with gustave (>=70%), sciel (40-69%), enemy (<40%) chroma. | Numeral treatment for important values. Color encodes quality tier. |
| **Stage badge** | `exp-status` badge showing prediction stage: backtest / shadow / live. Currently shown as colored pill in ShadowBrief. | Users need to know if predictions are tested-only or affecting real decisions. |
| **Accuracy trend sparkline** | 7-day daily accuracy trend as sparkline in the shadow accuracy card. Data from `daily_trend` array. | A single accuracy number without trend is half a story (Tufte: sparklines). |

**Anti-Patterns:**
- Never show predictions without accuracy context. A prediction without known reliability is noise.
- Never use the same visual treatment for predictions and actuals. The distinction must be immediately apparent (dashed vs solid, reduced opacity vs full).
- Never hide the prediction stage (backtest/shadow/live). It determines trust level.

### 7.4 Anomalies

Anomalies are ML-detected deviations from learned patterns. They range from informational to critical. Displayed on Understand (summary) and anomaly detail pages (drill-down).

**Sources:** `/api/ml/anomalies`, `/api/ml/shap`, plus anomaly items within `intelligence` cache.

**Visual Rules:**

| Rule | Implementation | Rationale |
|------|---------------|-----------|
| **Severity = visual intensity** | Critical: maelle chroma + `exp-crossing-out` animation + `data-mood="dread"`. Warning: enemy chroma + border accent. Info: sciel chroma + subtle highlight. | Preattentive processing: color + motion intensity maps to urgency. Only critical anomalies demand attention. |
| **Anomaly score visualization** | `exp-stat-bar[data-bar="hp"]` showing deviation from baseline. Full bar = within normal, depleting = deviating. | HP depletion metaphor: the entity's "health" (normalcy) is draining. |
| **SHAP attributions** | Bar chart (or table with inline bar widths) showing feature contributions. Positive contributions (increasing anomaly) in maelle color, negative (decreasing) in gustave color. | Diverging color scale for directionality. Maelle = driving the anomaly, gustave = working against it. |
| **Anomaly count threshold for mood** | 0 anomalies = nostalgic/wonder, 1-2 = wonder (mild alertness), 3+ = dread (page-level). | Aggregate state, not individual events, drives mood. One offline sensor is not worth dread. |
| **Resolution = dawn transition** | When an anomaly resolves (clears from API), the card transitions from dread mood to dawn. `exp-canvas-saved` animation on resolution. | Narrative arc: dread (discovery) -> dawn (resolution). |

**Anti-Patterns:**
- Never use maelle (crimson) for informational anomalies. Reserve crimson for critical severity only.
- Never show anomalies without SHAP or baseline context. An anomaly without explanation is just noise.
- Never use void-erase for anomaly dismissal. VoidErase implies permanent deletion (Gommage). Anomaly acknowledgment uses `exp-canvas-saved` (preservation of the record).

### 7.5 Cross-Category Visual Hierarchy

When multiple data categories appear on the same page (common on Home and Understand), maintain this visual hierarchy:

```
1. Anomalies (highest attention)     --> maelle chroma, dread mood, animation, t3 alerts
2. Real-time metrics                 --> full color, gustave/lune chroma, t2 refresh
3. Predictions                       --> dashed treatment, slight opacity, lune chroma
4. Historical data                   --> memory treatment (sepia optional), reduced weight
```

This hierarchy follows the attention principle: the most actionable information (anomalies) gets the most visual weight. Historical context provides backdrop without competing for attention.

**Implementation on the Home page:**
- Hero cards (anomalies, recommendations, accuracy) get MonolithDisplay treatment
- OODA summary cards get card-battle treatment with per-section chroma
- PipelineSankey (historical/live blend) gets standard framing
- All respect the 3-tier animation system (T1 ambient, T2 metric refresh, T3 status alert)

---

## 8. Anti-Patterns for ha-aria

These are mistakes an LLM is likely to make when applying the Expedition 33 design system to ARIA. Each one has been specifically identified from the component inventory and ARIA's data domain.

### DO NOT: Use wasteland theme for normal entity offline states

**Why it's tempting:** Wasteland (Beksinski palette) sounds like the right treatment for "something is down."

**What to do instead:** Entity offline is common and expected. Use `enemy` chroma on an `exp-status` badge. Wasteland theme is reserved for catastrophic system failure (all modules down, HA disconnected, no data for 24h+). An offline light sensor is not a corrupted zone.

```
/* WRONG */
<div data-theme="wasteland"> <!-- just because sensor.hallway is unavailable -->

/* RIGHT */
<span class="exp-status" data-chroma="enemy" data-status="error">unavailable</span>
```

### DO NOT: Use MonolithDisplay for every metric

**Why it's tempting:** Monolith is dramatic and the numbers look impressive.

**What to do instead:** Reserve `exp-monolith` for exactly 1-2 hero numbers per page. Use `exp-expedition-counter` (inline hand-painted numbers) for secondary metrics. Use standard `data-mono` text for tertiary values. The Monolith in the game holds ONE number -- the Gommage count. Its power comes from isolation.

```
/* WRONG -- 6 monoliths on one page */
<div class="exp-monolith">{entities}</div>
<div class="exp-monolith">{devices}</div>
<div class="exp-monolith">{areas}</div>
<div class="exp-monolith">{capabilities}</div>
<div class="exp-monolith">{unavailable}</div>
<div class="exp-monolith">{stale}</div>

/* RIGHT -- 1 monolith, rest are counters */
<div class="exp-monolith">{entities}</div>
<span class="exp-expedition-counter">{devices}</span> devices
<span class="exp-expedition-counter">{areas}</span> areas
```

### DO NOT: Apply dread mood to the whole dashboard when one thing fails

**Why it's tempting:** The mood system is page-level, and a single anomaly might trigger `data-mood="dread"`.

**What to do instead:** Mood should reflect *aggregate* state, not individual events. Use dread only when the system is genuinely concerning (3+ anomalies, drift in multiple models, validation failing). One offline sensor does not warrant dread.

```
/* WRONG */
if (anomalies.length > 0) mood = 'dread';

/* RIGHT */
if (anomalies.length >= 3) mood = 'dread';
else if (anomalies.length >= 1) mood = 'wonder';
else mood = 'nostalgic';
```

### DO NOT: Use Chroma flood (`exp-btn-paint`) for routine actions

**Why it's tempting:** Paint buttons look incredible and feel satisfying.

**What to do instead:** `exp-btn-paint` is for irreversible or high-stakes actions -- approving an automation, dismissing an anomaly, triggering a retrain. Use `exp-btn-primary` for standard navigation and `exp-btn-secondary` for filters, toggles, and preferences. Chroma flood on a "Clear filters" button destroys the visual hierarchy.

```
/* WRONG */
<button class="exp-btn-paint" onClick={clearFilters}>Clear</button>

/* RIGHT */
<button class="exp-btn-secondary" onClick={clearFilters}>Clear</button>
<button class="exp-btn-paint" data-chroma="gustave" onClick={approveAutomation}>Approve</button>
```

### DO NOT: Remove the cursor system in favor of expedition33-ui equivalents

**Why it's tempting:** Expedition33-ui has its own loading/active/idle patterns.

**What to do instead:** The cursor system (`cursor-active`, `cursor-working`, `cursor-idle`) is ARIA's signature UI element -- terminal cursors as state indicators for collapsible sections and pipeline nodes. No expedition33-ui class replaces this. Keep cursors and layer expedition33-ui components *around* them.

### DO NOT: Apply all atmosphere layers simultaneously

**Why it's tempting:** The atmosphere system has vignette, dust, petals, light beams, canvas texture, star field, page border, and more. Stacking them all creates visual drama.

**What to do instead:** Maximum 2-3 atmosphere layers per page. The CRT overlay in `app.jsx` already provides base atmosphere. Add at most: vignette (almost always), plus ONE location-specific layer (dust for lumiere, light beam for continent, star field for monolith). Mobile gets vignette only. Reduced motion gets nothing.

```
/* WRONG */
<div class="exp-vignette exp-dust-field exp-petal-field exp-light-beam exp-canvas-texture exp-star-field exp-page-border" />

/* RIGHT -- lumiere page */
<div class="exp-vignette" />
<div class="exp-dust-field" style="--dust-count: 8;" />
```

### DO NOT: Use `exp-journal` for short status messages

**Why it's tempting:** Journal looks beautiful and thematic.

**What to do instead:** Journal is for substantial narrative content -- weekly intelligence briefs, training history summaries, anomaly explanations. A one-line status message ("No anomalies detected") should use `exp-empty-state` or a simple `t-callout`. Journals need a header (date + author), body text, and ideally annotations. If the content does not have those, it is not a journal entry.

### DO NOT: Mix `exp-*` chroma with hardcoded CSS colors

**Why it's tempting:** Quick fix to get a specific color without setting up `data-chroma`.

**What to do instead:** Always use the `data-chroma` attribute and let cascade handle color values. This ensures theme switches (light/dark, wasteland) propagate correctly. If you need a specific character chroma as a CSS value, reference `var(--chroma-gustave)`, `var(--chroma-maelle)`, etc.

```
/* WRONG */
<div style="border-color: #c5a55a;"> <!-- hardcoded gustave gold -->

/* RIGHT */
<div data-chroma="gustave" class="exp-frame">
```

### DO NOT: Forget that ARIA uses Preact, not React

**Why it's tempting:** expedition33-ui docs reference React components and Framer Motion.

**What to do instead:** ARIA's SPA uses Preact with the `h()` JSX factory. There is no `React.createElement`, no `useId()`, no `React.lazy()`, no Framer Motion. All expedition33-ui integration is via CSS classes, not React component imports. Animation is CSS keyframes and manual DOM manipulation (as with `sh-card-shatter`). Never import from `react` or `framer-motion`.

### DO NOT: Use `Fragment` or `h` as parameter names in callbacks

**Why it's tempting:** Short variable names in `.map()` callbacks.

**What to do instead:** esbuild injects `h` as the JSX factory. Using `h` as a parameter name (e.g., `.map(h => ...)`) shadows the factory and causes silent render crashes. Use descriptive names: `.map(item => ...)`, `.map(entry => ...)`.

```
/* WRONG -- shadows JSX factory */
{items.map(h => <div>{h.name}</div>)}

/* RIGHT */
{items.map(item => <div>{item.name}</div>)}
```

### DO NOT: Nest Canvas elements inside SVG

**Why it's tempting:** Combining SVG diagrams with Canvas-rendered charts.

**What to do instead:** Use native SVG elements for SVG-based visualizations. Canvas and SVG are separate rendering contexts. Nesting Canvas in SVG fails silently. If you need both, use adjacent sibling elements with absolute positioning.

### DO NOT: Use inline `style=` for colors that should respond to theme changes

**Why it's tempting:** Dynamic styles via `style={...}` are the fastest way to set colors in Preact.

**What to do instead:** Use `data-chroma` and CSS custom properties. Inline styles with hardcoded `var(--status-healthy)` are acceptable for dynamic logic (conditional color by threshold). But static colors should be class-driven so theme changes cascade automatically.

---

## 9. Data Freshness -> Paint State Mapping

The existing SUPERHOT freshness states (`sh-cooling`, `sh-frozen`, `data-sh-state="stale"`) map naturally onto expedition33-ui's paint state machine. Both systems model the same concept: data aging as visual entropy.

| Age | SUPERHOT State | Paint State | Visual Treatment | Use Together? |
|-----|---------------|-------------|-----------------|---------------|
| < 5 min | (default, fresh) | `data-paint-state="fresh"` | Full color, wet shimmer sweep | Yes -- shimmer adds life to fresh data |
| 5-30 min | `.sh-cooling` (saturate 0.7) | `data-paint-state="drying"` | Slowing shimmer, brightness reduction | Layer both: desaturation + paint drying |
| 30-60 min | `.sh-frozen` (grayscale 60%) | `data-paint-state="dried"` | Static, no animation | SUPERHOT frozen is stronger; use `.sh-frozen` |
| > 60 min | `.sh-frozen` + `.sh-mantra` | `data-paint-state="cracked"` | Hairline fracture + desaturation + "STALE" watermark | Layer both: cracked paint + STALE mantra |

**Implementation:** The `computeFreshness()` function in `HeroCard.jsx` already computes age tiers. Extend it to set both `data-sh-state` and `data-paint-state` attributes:

```js
function applyFreshness(element, timestamp) {
  const age = (Date.now() - new Date(timestamp).getTime()) / 1000;
  if (age > 3600) {
    element.setAttribute('data-sh-state', 'stale');
    element.setAttribute('data-paint-state', 'cracked');
  } else if (age > 1800) {
    element.setAttribute('data-sh-state', 'frozen');
    element.setAttribute('data-paint-state', 'dried');
  } else if (age > 300) {
    element.setAttribute('data-sh-state', 'cooling');
    element.setAttribute('data-paint-state', 'drying');
  } else {
    element.removeAttribute('data-sh-state');
    element.setAttribute('data-paint-state', 'fresh');
  }
}
```

---

## 10. Accessibility Preservation

All expedition33-ui migrations MUST preserve ARIA's existing accessibility patterns:

| Requirement | Current Implementation | Migration Rule |
|-------------|----------------------|----------------|
| Focus ring | `:focus-visible` 2px `--accent` outline | expedition33-ui uses gold `:focus-visible` ring -- acceptable replacement |
| Screen reader text | `.sr-only` utility class | Keep. expedition33-ui does not override this. |
| Decorative SVG | `aria-hidden="true"` on all ornamental SVGs | Add `aria-hidden="true"` to all `exp-*` ornament elements (corners, dividers, dust) |
| Collapsible sections | `aria-expanded` on triggers | Keep. Cursor system adds visual cue; ARIA attribute is the programmatic cue. |
| Icon-only buttons | `aria-label` | Keep. expedition33-ui buttons have text; icon-only buttons need explicit labels. |
| Charts | `role="img"` on uPlot containers | Keep. expedition33-ui data components use similar patterns. |
| Reduced motion | `@media (prefers-reduced-motion: reduce)` disables all tiers + SUPERHOT | expedition33-ui has its own reduced-motion rules. Both systems cooperate: all atmosphere, paint states, and animations become static. |
| Mobile tab bar | `env(safe-area-inset-bottom)` | Keep. expedition33-ui is not aware of mobile tab bar. |

---

## 11. Quick Reference: Component Decision Tree

When adding or modifying a dashboard element, use this tree to select the right component:

```
Is it a single prominent number?
  Yes -> Is it THE hero metric for this page?
    Yes -> exp-monolith (1 per page max)
    No -> exp-expedition-counter (inline hand-painted)
  No -> continue

Is it a status indicator?
  Yes -> Is it entity health (on/off/unavailable)?
    Yes -> exp-status badge with appropriate data-status
    No -> Is it a probability/percentage?
      Yes -> exp-stat-bar (HP for probabilities, AP for capacities)
      No -> Is it a boolean state (connected/disconnected)?
        Yes -> exp-status badge with breathing dot
        No -> exp-expedition-counter with semantic color
  No -> continue

Is it a container for content?
  Yes -> Is it a substantial text block (description, report)?
    Yes -> exp-journal (with header, body, annotations)
    No -> Is it a data display with a label?
      Yes -> exp-frame with data-label
      No -> Is it a clickable navigation target?
        Yes -> exp-card or exp-card-battle (if high-stakes)
        No -> exp-card (glass, minimal)
  No -> continue

Is it a button/action?
  Yes -> Is it irreversible or high-stakes (approve, delete, retrain)?
    Yes -> exp-btn-paint with appropriate chroma
    No -> Is it a primary action (navigate, submit)?
      Yes -> exp-btn-primary
      No -> exp-btn-secondary
  No -> continue

Is it a table of data?
  Yes -> exp-table with chroma row accents and exp-table__num for numbers
  No -> continue

Is it a loading state?
  Yes -> Is it initial page load?
    Yes -> exp-skeleton (shimmer sweep)
    No -> Is it an in-progress operation?
      Yes -> exp-paint-loading (brushstroke shimmer bar)
      No -> exp-spinner (chroma-driven)
  No -> continue

Is it an error state?
  Yes -> Is it empty data (no error, just missing)?
    Yes -> exp-empty-state (Monolith void)
    No -> Is it a connection/system error?
      Yes -> exp-empty-state + exp-crossing-out overlay
      No -> exp-status[data-status="error"] inline
  No -> continue

Is it a person/identity?
  Yes -> exp-portrait (tarot frame) with per-person chroma
  No -> continue

Is it a form input?
  Yes -> exp-input / exp-select / exp-textarea
  No -> keep existing implementation or consult design-language.md
```

---

## 12. File Reference

| File | Purpose | When to Read |
|------|---------|-------------|
| `aria/dashboard/spa/src/index.css` | App CSS -- all custom classes, animation tiers, SUPERHOT overrides | Before modifying any CSS |
| `aria/dashboard/spa/src/app.jsx` | Root component -- router, sidebar, CRT overlay, error boundary | Before adding pages or changing layout |
| `docs/design-language.md` | Full design language spec -- color tokens, typography, components, animation tiers, visualization rules | Before any visual change |
| `docs/dashboard-components.md` | Component API reference | Before using or modifying components |
| `docs/dashboard-build.md` | Build process (esbuild) | Before changing build config |
| `node_modules/expedition33-ui/CLAUDE.md` | expedition33-ui file structure, class inventory, token list | Before using any `exp-*` class |
| `node_modules/expedition33-ui/src/tokens.css` | All 195 CSS custom properties + chroma + mood + location tokens | Before referencing a token by name |
| `node_modules/expedition33-ui/tests/preview.html` | Visual preview of all expedition33-ui components | To see what a component looks like |

---

## Appendix: File Inventory

### Pages (`aria/dashboard/spa/src/pages/`)

| File | Route | Description |
|------|-------|-------------|
| `Home.jsx` | `/` | Landing page with hero cards and OODA summary |
| `Observe.jsx` | `/observe` | Live entity monitoring |
| `Understand.jsx` | `/understand` | Intelligence analysis (anomalies, patterns) |
| `Decide.jsx` | `/decide` | Recommendations and actions |
| `Discovery.jsx` | `/discovery` | Entity discovery and search |
| `Capabilities.jsx` | `/capabilities` | System capability overview |
| `MLEngine.jsx` | `/ml-engine` | ML pipeline status and training |
| `DataCuration.jsx` | `/data-curation` | Data quality and curation |
| `Validation.jsx` | `/validation` | Model validation results |
| `Settings.jsx` | `/settings` | Configuration |
| `Guide.jsx` | `/guide` | User guide |
| `Faces.jsx` | `/faces` | Face recognition gallery |
| `Intelligence.jsx` | `/intelligence` | Legacy intelligence page (redirects to Understand) |
| `Predictions.jsx` | `/predictions` | Legacy predictions page (redirects to Understand) |
| `Patterns.jsx` | `/patterns` | Legacy patterns page (redirects to Understand) |
| `Shadow.jsx` | `/shadow` | Legacy shadow page (redirects to Understand) |
| `Presence.jsx` | `/presence` | Legacy presence page (redirects to Observe) |
| `Automations.jsx` | `/automations` | Legacy automations page (redirects to Decide) |
| `DetailPage.jsx` | `/detail/:type/:id/:rest*` | Generic detail route dispatcher |

### Detail Pages (`aria/dashboard/spa/src/pages/details/`)

AnomalyDetail, BaselineDetail, CapabilityDetail, ConfigDetail, CorrelationDetail, CurationDetail, DriftDetail, EntityDetail, ModelDetail, ModuleDetail, PredictionDetail, RoomDetail, SuggestionDetail

### Intelligence Sections (`aria/dashboard/spa/src/pages/intelligence/`)

ActivitySection, AnomalyAlerts, Baselines, Configuration, Correlations, DailyInsight, DriftStatus, HomeRightNow, LearningProgress, PredictionsVsActuals, ShapAttributions, SystemStatus, TrendsOverTime, utils

### Shared Components (`aria/dashboard/spa/src/components/`)

AriaLogo, Breadcrumb, CapabilityDetail, CollapsibleSection, DataSourceConfig, DataTable, DiscoverySettings, DomainChart, ErrorState, HeroCard, InlineSettings, LoadingState, OodaSummaryCard, PageBanner, PipelineSankey, PipelineStatusBar, PipelineStepper, PresenceCard, Sidebar, StatsGrid, StatusBadge, TerminalToggle, TimeChart, UsefulnessBar
