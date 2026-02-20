# PRD: Interactive Dashboard — Screenshot Audit, Clickable Detail Pages, Configurable Data Sources

**Design:** `docs/plans/2026-02-19-interactive-dashboard-design.md`
**Created:** 2026-02-19
**Tasks:** 15

## Dependency Graph

```
Phase 1 (Audit):       [1] → [2] → [3] → [4]
Phase 2 (Detail):      [4] → [5] → [6] → [7] → [8,9] → [11]
                        [5] → [10] ──────────────────────→ [11]
Phase 3 (Config):      [4] → [12] → [13] → [15]
                        [12] → [14] ──────→ [15]
Final:                 [11,13,14] → [15]
```

## Tasks

### Phase 1: Screenshot Audit

**T1. Install Puppeteer and create screenshot audit script scaffold**
- Add puppeteer devDependency to `spa/package.json`
- Create `spa/scripts/screenshot-audit.js` — headless browser, navigate 11 routes, capture screenshots to `docs/audit/screenshots/`
- Create `docs/audit/` directory
- **Criteria:** `grep -q 'puppeteer' spa/package.json`, `test -f spa/scripts/screenshot-audit.js`, `test -d docs/audit`

**T2. Add API comparison to screenshot audit script**
- Extend script to hit backend API endpoints per page, extract DOM text, compare data points
- Generate `docs/audit/audit-report.md` with mismatches
- **Criteria:** Script references audit-report, fetch, and key API endpoints (anomalies, shadow, pipeline, curation, capabilities)
- **Blocked by:** T1

**T3. Run screenshot audit and capture all 11 pages**
- Execute against running hub, verify all 11 screenshots + report generated
- **Criteria:** All 11 `.png` files exist + `audit-report.md` exists
- **Blocked by:** T2

**T4. Fix any frontend/backend mismatches found in audit**
- Address mismatches from report, re-run audit to confirm
- **Criteria:** Report shows 0 mismatches / all pages match
- **Blocked by:** T3

### Phase 2: Clickable Detail Sub-Pages

**T5. Create shared detail infrastructure: Breadcrumb, clickable-data CSS, DetailPage router**
- `Breadcrumb.jsx` — terminal-style path nav
- `.clickable-data` CSS — hover affordance with accent border
- `DetailPage.jsx` — route dispatcher for `#/detail/:type/:id`
- Add route to `app.jsx`
- **Criteria:** Files exist, CSS class in index.css, route in app.jsx
- **Blocked by:** T4

**T6. Build core detail renderers: Anomaly, Room, Entity, Prediction**
- 4 most-used detail pages with Summary/Explanation/History layout
- SUPERHOT treatments: sh-threat-pulse for critical, sh-frozen for stale
- **Criteria:** All 4 files exist, use PageBanner and t-frame
- **Blocked by:** T5

**T7. Build remaining detail renderers (9 types)**
- Suggestion, Capability (refactored), Model, Drift, Module, Config, Curation, Correlation, Baseline
- **Criteria:** All 9 files exist in `spa/src/pages/details/`
- **Blocked by:** T6

**T8. Add click handlers to OODA pages (Home, Observe, Understand, Decide)**
- Wire `#/detail/:type/:id` navigation on all data elements across 4 primary pages + PresenceCard
- **Criteria:** All 5 files contain `#/detail/` references
- **Blocked by:** T7

**T9. Add click handlers to system pages (Discovery, Capabilities, ML Engine, Data Curation, Settings)**
- Wire `#/detail/:type/:id` navigation on all data elements across 5 system pages
- **Criteria:** All 5 files contain `#/detail/` references
- **Blocked by:** T7

**T10. Add click handlers to shared components (HeroCard, PipelineSankey)**
- Optional href/onClick on HeroCard, node click on Sankey
- **Criteria:** HeroCard has href/onClick, Sankey has `#/detail/` references
- **Blocked by:** T5

**T11. Build SPA and verify detail routing works end-to-end**
- `npm run build` succeeds, `dist/bundle.js` generated
- **Criteria:** Build completes, bundle exists
- **Blocked by:** T8, T9, T10

### Phase 3: Configurable Data Sources

**T12. Create backend module source config endpoints**
- `routes_module_config.py` with GET/PUT `/api/config/modules/{module}/sources`
- Default configs in `config_defaults.py`
- Register in `server.py`, write tests
- **Criteria:** File exists, defaults added, route registered, tests pass
- **Blocked by:** T4

**T13. Wire modules to read source config at runtime**
- Presence reads `enabled_signals`, activity reads `enabled_domains`, intelligence reads anomaly config
- Re-read on config change events, tests for filtering
- **Criteria:** `enabled_signals` in presence.py, presence tests pass, source_config tests pass
- **Blocked by:** T12

**T14. Build frontend data source config components and integrate into pages**
- `TerminalToggle.jsx` — `[ON ]/[OFF]` toggle
- `DataSourceConfig.jsx` — toggle list per module
- Extend InlineSettings, add to Observe + Understand
- Safety: prevent last toggle-off, high-contribution warning
- **Criteria:** Components exist, referenced in Observe + Understand pages
- **Blocked by:** T12

### Final

**T15. Final build, full test suite, and re-run screenshot audit**
- Rebuild SPA, run full pytest, re-capture screenshots
- **Criteria:** Build succeeds, all tests pass, screenshots exist
- **Blocked by:** T11, T13, T14
