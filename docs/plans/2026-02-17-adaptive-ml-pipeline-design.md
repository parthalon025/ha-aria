# Adaptive ML Pipeline for ARIA

**Date:** 2026-02-17
**Status:** Phase 4 Completed — all phases done
**Scope:** Hardware-aware tiered model registry with phased capability expansion

## Problem Statement

ARIA's ML pipeline uses fixed hyperparameters, a single train/val split, divergent feature extraction paths, and no hardware awareness. Research comparison (standard ML workflow + V7 Labs pattern recognition framework) reveals gaps in:

- Hyperparameter optimization (fixed params vs. tuned)
- Evaluation rigor (single 80/20 split vs. cross-validation on ~60 samples)
- Feature selection feedback (importance computed but not acted on)
- Documented technical debt (divergent features, delayed feedback loop, missing snapshot validation in hub)
- Model complexity scaling (same ensemble regardless of hardware capability)
- Pattern recognition depth (no sequence modeling, no hierarchical time-scale decomposition, no anomaly explanations)

## Design Principles

1. **Additive by phase** — nothing breaks if you stop after any phase
2. **Hardware-aware with fallback** — auto-detect, recommend, degrade gracefully per-model
3. **Existing behavior preserved** — Tier 2 = today's ARIA, zero migration required
4. **Optional dependencies** — Tier 3-4 packages are `try/except` guarded, never crash on import

---

## Phase 1: Foundation (Close Gaps + Build Tier Infrastructure)

### 1A. Hardware Capability Scanner

New module: `aria/engine/hardware.py`

Runs at hub startup, cached in SQLite (`hw_profile` category), re-scans on request or after 30 days.

**Probes:**
- `psutil.virtual_memory().total` → available RAM
- `os.cpu_count()` → core count
- `torch.cuda.is_available()` / `torch.backends.mps.is_available()` → GPU presence (only if torch installed)
- Optional: quick benchmark (train small LightGBM on synthetic data for baseline speed)

**Tier Assignment:**

| Tier | Name | Hardware Profile | Model Stack |
|------|------|-----------------|-------------|
| 1 | Minimal | <2GB RAM, 1-2 cores | Single LightGBM per target, no lag features, no anomaly detection |
| 2 | Standard | 2-8GB RAM, 2-4 cores | Current ensemble (GBM+RF+LGBM) + IsolationForest, full feature config |
| 3 | Advanced | 8-32GB RAM, 4+ cores, no GPU | Tier 2 + Optuna auto-tuning + `river` online learners + `tslearn` DTW classifiers |
| 4 | GPU-Accelerated | 8+ GB RAM + CUDA/ROCm GPU | Tier 3 + PyTorch small transformer + attention-based anomaly explainer |

```
if gpu_available and ram >= 8GB:  → Tier 4
elif ram >= 8GB and cores >= 4:   → Tier 3
elif ram >= 2GB and cores >= 2:   → Tier 2
else:                              → Tier 1
```

**User override:** `ml.tier_override` config key — `auto` (default), `1`-`4`. Overriding above hardware emits warning but is allowed.

**API:** `/api/ml/hardware` returns detected profile, current tier, recommendation, and active fallbacks.

### 1B. Tiered Model Registry

New module: `aria/engine/models/registry.py`

Each prediction target gets a `ModelStack` — ordered list of model configurations, one per tier.

```python
@dataclass
class ModelEntry:
    name: str                    # "lgbm_power_watts"
    tier: int                    # 1-4
    model_class: str             # "LGBMRegressor" or callable
    params: dict                 # hyperparameters (or "auto" for Optuna)
    weight: float                # ensemble blend weight
    fallback_tier: int | None    # what to drop to on failure
    requires: list[str]          # ["lightgbm"] — pip packages required
    optional_features: list[str] # feature groups this model needs

@dataclass
class ModelStack:
    target: str                  # "power_watts"
    entries: list[ModelEntry]    # sorted by tier ascending
    active_entries: list[str]    # populated at runtime based on current tier
```

**Registration:** Models register via decorator or config file. Default stacks ship with ARIA.

**Runtime resolution:** `registry.resolve(target, current_tier)` returns all entries at or below the current tier. Ensemble blender uses only resolved entries.

**Dependency check:** On startup, registry validates required packages for active tier. Missing → entry excluded with warning, not crash.

### 1C. Graceful Fallback Engine

Per-model, not per-pipeline:

1. Log failure with context (model name, tier, error, memory usage)
2. Mark entry as `fallen_back` with 7-day TTL
3. Exclude from ensemble, re-normalize remaining weights
4. Emit `model_fallback` event on hub event bus
5. On next weekly retrain, retry at original tier

### 1D. Close Existing Pipeline Gaps

**Unify feature extraction:** Hub `MLEngine._extract_features()` delegates entirely to `vector_builder.build_feature_vector()` + appends hub-only rolling window features. No duplicate feature logic.

**Snapshot validation in hub:** `validate_snapshot_batch()` call before `_fit_all_models()` in `ml_engine.py`.

**Fix feedback loop startup delay:** `schedule_periodic_training(run_immediately=True)` when no `ml_training_metadata` exists in cache.

**Wire presence into engine features:** Enable `presence_features` group in `DEFAULT_FEATURE_CONFIG`. Pull from hub cache into snapshot builder. Enabled by default at Tier 2+.

### 1E. Hyperparameter Optimization (Tier 3+)

Optional dependency: `optuna`

- During weekly retrain, run Optuna study (20 trials, time-series CV as objective) per target
- Optimize: ensemble weights, tree max_depth, n_estimators, learning_rate, subsample
- Store best params in SQLite (`ml_hyperparams`) with training date
- Fall back to fixed params if budget exceeded or errors
- Tier 1-2: fixed params (today's behavior)

### 1F. Time-Series Cross-Validation

Replace single 80/20 split with expanding-window CV:

- **Tier 1:** Single train/val split
- **Tier 2:** 3-fold expanding window
- **Tier 3-4:** 5-fold expanding window + Optuna uses CV score as objective

Report per-fold metrics in training metadata.

### 1G. Feature Selection Feedback Loop

After each training cycle:

1. Compute feature importance (existing: RF importances + LGBM gain)
2. Identify features contributing <1% cumulative importance
3. Log as `low_importance_features` in training metadata
4. Tier 3+: auto-disable features <1% for 3 consecutive cycles
5. Emit `feature_pruned` event
6. Re-enable pruned features if drift detection fires

---

## Phase 2: Online Learning Layer

### 2A. River Integration (Tier 3+)

Optional dependency: `river`

**New online models per target:**
- `river.forest.ARFRegressor` (Adaptive Random Forest) — handles concept drift natively
- `river.drift.ADWIN` as wrapper

**Architecture:** Parallel track alongside batch models.

```
State change event
    ↓
Activity Monitor (existing) → rolling stats
    ↓
Online model .learn_one(features, actual_value)
    ↓
Online model .predict_one(features) → shadow comparison
    ↓
Blend: batch_prediction * 0.7 + online_prediction * 0.3 (tunable)
```

Online models produce predictions from first data point — solves 24-48h cold start.

### 2B. Shadow Engine → Online Model Feedback

1. Shadow engine resolves a prediction window (correct/disagreement/nothing)
2. If outcome is measurable, feed `(features, actual)` to online model `.learn_one()`
3. Shadow engine becomes a training signal, not just evaluation

Prediction → observe → learn → predict better, on ~10-minute cycle.

### 2C. Ensemble Weight Auto-Tuning

Track rolling 7-day MAE per model source. Re-weight proportional to inverse MAE every 24h:

```python
weight_i = (1 / mae_i) / sum(1 / mae_j for j in models)
```

Stored in `ml_ensemble_weights` cache.

---

## Phase 3: Pattern Recognition Expansion

### 3A. Sequence Modeling

**Tier 3 — tslearn DTW classifiers:**
- `tslearn.neighbors.KNeighborsTimeSeriesClassifier` with DTW metric
- Input: sliding window of last N snapshots (N=6)
- Classifies trajectory: "winding down," "ramping up," "stable," "anomalous transition"
- Feeds classification as feature `trajectory_class` to main ensemble

**Tier 4 — Small Transformer:**
- 2-layer transformer encoder (PyTorch), ~50K parameters
- Input: last 24 snapshot feature vectors as sequence
- Output: next-snapshot prediction for all targets
- Attention weights = explainability ("which past time steps mattered")
- Falls back to Tier 3 DTW classifier

### 3B. Hierarchical Pattern Recognition

Explicit time-scale decomposition:

| Scale | Window | What It Captures | Method |
|-------|--------|-------------------|--------|
| Micro | Seconds-minutes | Motion → light on | Association rules (existing apriori, enhanced) |
| Meso | Minutes-hours | Morning routine | DTW matching + sequence classifier (3A) |
| Macro | Days-weeks | Seasonal shifts | Batch ensemble + drift detection |

New `PatternScale` enum tags all detected patterns. Shadow engine predictions tagged with scale. Accuracy tracked per-scale.

Organic discovery links related patterns hierarchically: micro patterns nested within meso patterns, named by Ollama.

### 3C. Enhanced Anomaly Classification (Tier 3+)

**Tier 3 — Contribution-based explanation:**
- Per-feature anomaly contribution from IsolationForest isolation path length
- Report top 3 contributing features per anomaly
- Stored in anomaly metadata, surfaced in dashboard and Telegram

**Tier 4 — Attention-based autoencoder:**
- Attention-augmented autoencoder (PyTorch)
- Attention weights = interpretable feature explanations
- Continuous anomaly score (not just binary)

---

## Phase 4: Advanced Expansion

### 4A. Cross-Domain Pattern Transfer

Transfer patterns between similar contexts within the same home:

- **Room-to-room:** Kitchen "motion → lights" hypothesized for bedroom, tested via shadow engine
- **Routine-to-routine:** Weekday morning as template for weekend morning (same stages, shifted timing)

`TransferCandidate` objects generated by organic discovery when new cluster has structural similarity (Jaccard > 0.6) to established capability. Shadow tests at low confidence. Promote after 7-day hit rate threshold.

### 4B. Attention-Based Anomaly Explainer (Tier 4)

Full attention autoencoder plus:
- **Temporal attention:** "Which time steps" contributed, not just which features
- **Contrastive explanation:** "Looks like Tuesday evening but with unusually high power draw"
- **Dashboard integration:** Anomaly timeline with expandable attention heatmaps

---

## Cross-Cutting Concerns

### Configuration

```yaml
ml:
  tier_override: auto
  fallback_ttl_days: 7
  online_blend_weight: 0.3
  optuna_trials: 20
  feature_prune_threshold: 0.01
  feature_prune_cycles: 3
  cv_folds: auto
  pattern_scales: [micro, meso, macro]
```

All defaults match current Tier 2 behavior.

### New Dependencies by Tier

| Tier | New Dependencies |
|------|-----------------|
| 1 | None |
| 2 | None |
| 3 | `optuna`, `river`, `tslearn` |
| 4 | `torch` (CPU or CUDA) |

All optional imports with `try/except` guards.

### Dashboard Integration

- **Phase 1:** Hardware profile, tier indicator, fallback alerts, CV metrics, feature importance with pruning
- **Phase 2:** Online vs batch accuracy, blend weight history, shadow→online feedback visualization
- **Phase 3:** Pattern scale breakdown, trajectory output, anomaly explanations
- **Phase 4:** Transfer candidates, attention heatmaps, contrastive explanations

### Migration Path

- Existing installations start at Tier 2 (identical to today). Scanner logs recommendation. No behavior change until opt-in.
- Existing `.pkl` models wrapped as Tier 2 registry entries. No migration needed.

---

## Phase Summary

| Phase | Key Deliverable | Primary Metric |
|-------|----------------|----------------|
| 1 | Hardware scanner, model registry, fallback, gap fixes, Optuna, CV, feature pruning | Higher R², reliable evaluation |
| 2 | River online learning, shadow→model feedback, auto-weight tuning | Faster adaptation, cold start solved |
| 3 | Sequence modeling, hierarchical patterns, anomaly explanations | New prediction types, explainability |
| 4 | Cross-domain transfer, attention explainer | Faster capability bootstrap, human-readable anomaly stories |
