# ARIA Capability Propagation Framework — Design Document

**Date:** 2026-02-14
**Status:** Approved
**Scope:** Capability registry, CI enforcement, auto-generated docs, install verification

## In Plain English

This adds a formal inventory system to ARIA so every feature it has today — predictions, shadow mode, organic discovery, all 22 capabilities — is declared in code, tested by CI, and visible on the dashboard. When the research upgrades (U1-U9), production hardening, autonomy graduation, and open-source packaging happen, nothing silently disappears. It is like a parts manifest for an aircraft: before any maintenance, you know exactly what should be there, and after maintenance, you verify nothing was left out.

## Why This Exists

ARIA has 9 hub modules, 13 engine commands, 13 dashboard pages, 984 tests, and 35+ config parameters. Four upcoming phases will touch all of them: research upgrades (U1-U9 adding 7 files, modifying 18, adding 35 config params), production hardening (CI/CD, linting, type checking), autonomy graduation (shadow→suggest→autonomous), and open-source packaging (PyPI, contributor onboarding). Without a capability registry, regressions are invisible — a linting fix removes an import, a module stops loading, and nobody notices until the dashboard shows empty data 3 days later.

---

## Design

### 1. Capability Declaration Model

Each capability is declared where it lives — in the module or engine command that implements it.

```python
# aria/capabilities.py

@dataclass
class Capability:
    id: str                          # "shadow_predictions", "organic_discovery"
    name: str                        # "Shadow Mode Predictions"
    description: str                 # One-line plain English
    module: str                      # "shadow_engine", "organic_discovery"
    layer: str                       # "hub" | "engine" | "dashboard" | "cross-cutting"
    config_keys: list[str]           # ["shadow.explore_strategy", ...]
    test_paths: list[str]            # ["tests/hub/test_shadow_engine.py"]
    test_markers: list[str]          # ["shadow"] — pytest -k markers
    runtime_deps: list[str]          # ["ha_websocket", "activity_monitor"]
    optional_deps: list[str]         # ["neuralprophet", "river"] — pip extras
    data_paths: list[str]            # ["~/ha-logs/intelligence/predictions/"]
    systemd_units: list[str]         # ["aria-hub.service"]
    pipeline_stage: str | None       # "backtest" | "shadow" | "suggest" | "autonomous"
    status: str                      # "stable" | "experimental" | "planned"
    added_version: str               # "1.0.0"
    depends_on: list[str]            # ["activity_monitoring", "discovery"]
```

Modules declare via a `CAPABILITIES` class attribute:

```python
class ShadowEngine(Module):
    CAPABILITIES = [
        Capability(
            id="shadow_predictions",
            name="Shadow Mode Predictions",
            description="Predict-compare-score loop that learns from implicit feedback",
            module="shadow_engine",
            layer="hub",
            config_keys=["shadow.explore_strategy", "shadow.confidence_threshold"],
            test_paths=["tests/hub/test_shadow_engine.py", "tests/hub/test_cache_shadow.py"],
            test_markers=["shadow"],
            runtime_deps=["activity_monitoring", "discovery"],
            optional_deps=[],
            data_paths=[],
            systemd_units=["aria-hub.service"],
            pipeline_stage="shadow",
            status="stable",
            added_version="1.0.0",
            depends_on=["activity_monitoring", "discovery"],
        ),
    ]
```

Engine commands declare in a central `aria/engine/capabilities.py` file with the same structure.

### 2. Registry

```python
# aria/capabilities.py — continued

class CapabilityRegistry:
    """Discovers and validates all declared capabilities."""

    def collect(self) -> dict[str, Capability]:
        """Walk all hub modules and engine commands, harvest CAPABILITIES lists."""

    def validate(self) -> list[Issue]:
        """Cross-reference declarations against reality.

        Checks:
        1. Every config_key exists in config_defaults.py
        2. Every test_path exists on disk
        3. Every runtime_dep references another registered capability
        4. Every optional_dep is in pyproject.toml extras
        5. No orphan tests (test files not claimed by any capability)
        6. Dependency graph has no cycles
        """

    def diff(self, old_ref: str, new_ref: str) -> CapabilityDiff:
        """Compare capabilities between two git refs.

        Returns: added, removed, modified (config keys changed, deps changed)
        """

    def health(self) -> dict[str, CapabilityHealth]:
        """Runtime health per capability.

        Checks: module status in hub, last successful run, error count
        """

    def install_check(self) -> list[CheckResult]:
        """Verify fresh install has everything needed.

        Checks: deps installed, env vars set, data dirs writable,
        optional deps present/missing, systemd timers installed
        """
```

### 3. CLI Commands

```
aria capabilities list                  # Table: id, name, layer, status, test count
aria capabilities list --verbose        # Full details per capability
aria capabilities list --layer hub      # Filter by layer
aria capabilities list --status planned # Show only planned capabilities
aria capabilities verify                # Run validation checks, report issues
aria capabilities verify --strict       # Fail on any issue (for CI)
aria capabilities verify --quick        # Only check changed files (<2s, for pre-commit)
aria capabilities health                # Runtime: which capabilities are active now
aria capabilities diff main..HEAD       # What changed on this branch
aria capabilities export                # JSON dump for CI/dashboard consumption
aria capabilities install-check         # Fresh install verification
```

### 4. CI Integration

#### GitHub Actions Gate

```yaml
# .github/workflows/ci.yml — after tests pass
- name: Capability regression check
  run: |
    aria capabilities verify --strict
    aria capabilities diff ${{ github.event.pull_request.base.sha }}..HEAD --ci
```

**`verify --strict` fails on:**
- Capability with zero tests
- config_key not in config_defaults.py
- runtime_dep pointing to non-existent capability
- optional_dep not in pyproject.toml
- Orphan test file not claimed by any capability

**`diff --ci` warns on:**
- Capability removed without `# REMOVED: reason` annotation
- Test count decreased for any capability
- depends_on list grew (new coupling)
- New capability declared `stable` with <3 tests

#### pytest Integration

```python
# tests/conftest.py
def pytest_collection_modifyitems(config, items):
    """Auto-tag tests with capability markers from registry declarations."""
    # Match test file paths to capability test_paths
    # Add capability id as pytest marker automatically
```

### 5. Auto-Generated Documentation

Three output formats from the same registry data:

**Markdown** (`docs/capabilities.md`): Generated on release. Full capability reference with descriptions, dependencies, config keys, test counts.

**Dashboard page** (`/ui/capabilities`): Real-time status. Dependency graph visualization. Health badges (green/yellow/red). Config key counts linking to Settings page.

**Install checklist** (`aria capabilities install-check`): For open-source users after `pip install ha-aria`. Shows which capabilities are available, which need optional deps, which need timer setup.

### 6. Dashboard Page Design

New "Capabilities" page in the Preact SPA:

- **Capability list** with status badges (stable/experimental/planned)
- **Dependency graph** — which capabilities feed which (simple directed graph)
- **Health indicators** — green (module loaded, recent activity), yellow (loaded, no recent activity), red (failed to load or erroring)
- **Test coverage** — percentage per capability
- **Config keys** — count with link to Settings page for each

### 7. API Endpoints

```
GET /api/capabilities              # Full capability list with metadata
GET /api/capabilities/{id}         # Single capability details
GET /api/capabilities/health       # Runtime health per capability
GET /api/capabilities/graph        # Dependency graph (nodes + edges)
```

---

## Initial Capability Inventory

### Hub Layer (9)

| ID | Module | Approx Tests | Config Keys | Status |
|----|--------|-------------|-------------|--------|
| `discovery` | discovery.py | 12 | 2 | stable |
| `ml_realtime` | ml_engine.py | 18 | 6 | stable |
| `pattern_recognition` | patterns.py | 8 | 3 | stable |
| `orchestrator` | orchestrator.py | 6 | 2 | stable |
| `shadow_predictions` | shadow_engine.py | 22 | 8 | stable |
| `data_quality` | data_quality.py | 14 | 5 | stable |
| `organic_discovery` | organic_discovery/ | 148 | 7 | stable |
| `intelligence_assembly` | intelligence.py | 10 | 0 | stable |
| `activity_monitoring` | activity_monitor.py | 16 | 4 | stable |

### Engine Layer (13)

| ID | Command | Approx Tests | Config Keys | Status |
|----|---------|-------------|-------------|--------|
| `snapshot` | aria snapshot | 6 | 0 | stable |
| `predictions` | aria predict | 12 | 3 | stable |
| `scoring` | aria score | 8 | 2 | stable |
| `model_training` | aria retrain | 10 | 4 | stable |
| `meta_learning` | aria meta-learn | 6 | 4 | stable |
| `drift_detection` | aria check-drift | 8 | 3 | stable |
| `correlations` | aria correlations | 6 | 0 | stable |
| `sequence_training` | aria sequences train | 4 | 0 | stable |
| `sequence_anomalies` | aria sequences detect | 4 | 0 | stable |
| `automation_suggestions` | aria suggest-automations | 4 | 2 | stable |
| `prophet_forecasting` | aria prophet | 6 | 4 | stable |
| `occupancy_estimation` | aria occupancy | 6 | 0 | stable |
| `power_profiling` | aria power-profiles | 4 | 0 | stable |

### Planned — Research Upgrades (9)

| ID | Source | Depends On | Status |
|----|--------|------------|--------|
| `thompson_fsw` | U1 | shadow_predictions | planned |
| `neural_prophet` | U2 | prophet_forecasting | planned |
| `hybrid_anomaly` | U3 | model_training | planned |
| `adwin_drift` | U4 | drift_detection | planned |
| `reference_model` | U5 | model_training, meta_learning | planned |
| `correction_propagation` | U6 | shadow_predictions | planned |
| `shap_explanations` | U7 | model_training | planned |
| `mrmr_selection` | U8 | model_training | planned |
| `incremental_lgbm` | U9 | adwin_drift, model_training | planned |

**Total:** 22 stable + 9 planned = 31 capabilities

---

## Dependency Graph

```
discovery ──────────┬──────────────────────────────────────────┐
                    │                                          │
activity_monitoring─┼─► shadow_predictions ─► correction_propagation
                    │       │
                    │       └─► thompson_fsw
                    │
                    ├─► ml_realtime ─► model_training ─┬─► reference_model
                    │                                  ├─► hybrid_anomaly
                    │                                  ├─► shap_explanations
                    │                                  ├─► mrmr_selection
                    │                                  └─► incremental_lgbm
                    │                                          │
                    ├─► pattern_recognition ─► orchestrator     │
                    │                                          │
                    ├─► data_quality                            │
                    │                                          │
                    └─► organic_discovery                       │
                                                               │
                    drift_detection ─► adwin_drift ─────────────┘

                    prophet_forecasting ─► neural_prophet

                    intelligence_assembly (reads all hub + engine outputs)
```

---

## Files Changed

| File | Change |
|------|--------|
| New: `aria/capabilities.py` | Capability dataclass, CapabilityRegistry, validation logic |
| New: `aria/engine/capabilities.py` | Engine capability declarations |
| `aria/cli.py` | Add `capabilities` subcommand group |
| `aria/hub/core.py` | Module base class gets CAPABILITIES attribute |
| `aria/hub/api.py` | Add /api/capabilities/* endpoints |
| `aria/modules/*.py` (9 files) | Add CAPABILITIES declarations |
| `tests/conftest.py` | pytest capability marker integration |
| New: `tests/test_capabilities.py` | Registry validation tests |
| New: `aria/dashboard/spa/src/pages/Capabilities.jsx` | Dashboard page |
| `.github/workflows/ci.yml` | Add capability verify step |

**Estimated new code:** ~500-700 lines (registry + CLI + API + declarations)
**Estimated modifications:** ~200 lines across 9 module files (adding declarations)

---

## Success Criteria

1. `aria capabilities list` shows all 22 stable capabilities
2. `aria capabilities verify --strict` passes with zero issues
3. Every module has ≥1 declared capability with ≥1 test
4. Every config key in config_defaults.py is claimed by a capability
5. Dashboard shows capability health page
6. CI blocks PRs that remove capabilities without annotation
7. `aria capabilities install-check` works for fresh pip install

## Exclusions

- No automatic capability *detection* (this is declaration-based, not introspection-based)
- No runtime capability toggling (capabilities are structural, not feature flags)
- Dashboard dependency graph is static visualization, not interactive editor
- No versioned capability schemas (v1 only — revisit if open-source users need migration)
