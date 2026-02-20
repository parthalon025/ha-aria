# ML Training Unification + Pipeline Explanation UI

**Date:** 2026-02-16
**Goal:** Fix divergent training paths, add missing safety checks, close the feedback loop on startup, and enhance the ML Engine page to explain the full pipeline from raw data to predictions.

---

## Problems

### 1. Divergent Feature Extraction
The engine CLI path (`vector_builder.build_feature_vector()`) and hub module path (`MLEngine._extract_features()`) maintain independent feature extraction logic. Adding a feature to one doesn't update the other. This was flagged in `2026-02-14-ml-encoding-inconsistency.md`.

### 2. Hub MLEngine Skips Snapshot Validation
`MLEngine._load_training_data()` loads daily JSON files with no quality checks. The engine CLI path uses `validate_snapshot_batch()` to reject corrupt/incomplete snapshots. Corrupt data during HA restart could poison hub-trained models.

### 3. Feedback Loop Delayed ~7 Days After Restart
`schedule_periodic_training(interval_days=7)` with `run_immediately=False` means the first hub training (and thus the first feedback write-back to capabilities) doesn't fire until 7 days after hub startup. The feedback loop is technically open until then.

### 4. ML Engine Page Shows Pieces, Not the Story
The existing MLEngine.jsx shows FeatureSelection, ModelHealth, and TrainingHistory as separate sections. Nothing connects them into a coherent pipeline narrative: "your home's data comes in here, gets processed like this, and produces predictions like that."

---

## Design

### Backend Fix 1: Unify Feature Extraction

Make `MLEngine._extract_features()` delegate to `vector_builder.build_feature_vector()` for the base feature set, then apply the hub-specific decay weighting and weekday alignment as a post-processing step. This ensures both paths use identical feature definitions.

**Changes:**
- `aria/modules/ml_engine.py` — Replace `_extract_features()` body with a call to `vector_builder.build_feature_vector()`, then apply `recency_decay` and `WEEKDAY_ALIGNMENT_BONUS` to the resulting dict.
- No changes to `vector_builder.py` — it remains the single source of truth.

### Backend Fix 2: Add Snapshot Validation to Hub Training

Wire `validate_snapshot_batch()` into `MLEngine._load_training_data()`.

**Changes:**
- `aria/modules/ml_engine.py` — After loading snapshots in `_load_training_data()`, call `validate_snapshot_batch()` and log rejected count.

### Backend Fix 3: Run Training on Startup If Stale

Check `ml_training_metadata.last_trained` in `schedule_periodic_training()`. If >7 days old or missing, set `run_immediately=True`.

**Changes:**
- `aria/modules/ml_engine.py` — In `schedule_periodic_training()`, check cache staleness before scheduling.

### Backend Fix 4: Add `/api/ml/pipeline` Endpoint

New endpoint that aggregates pipeline state into a single response for the UI.

**Response shape:**
```json
{
  "data_collection": {
    "entity_count": 3066,
    "snapshot_count_intraday": 78,
    "snapshot_count_daily": 5,
    "last_snapshot": "2026-02-16T11:00:00",
    "health_guard": "passing",
    "presence_connected": true
  },
  "feature_engineering": {
    "total_features": 27,
    "feature_categories": {
      "time": 15, "home": 6, "presence": 4, "lag": 2
    },
    "latest_values": { "hour_sin": 0.5, "power_watts": 155, "..." : "..." }
  },
  "model_training": {
    "last_trained": "2026-02-14T02:00:00",
    "model_types": ["GradientBoosting", "RandomForest", "LightGBM"],
    "targets": ["power_watts", "lights_on", "devices_home", "people_home", "motion_active_count"],
    "validation_split": "80/20 chronological",
    "total_snapshots_used": 78,
    "rejected_snapshots": 2
  },
  "predictions": {
    "targets": {
      "power_watts": { "predicted": 155.5, "actual": 162.0, "r2": 0.84, "mae": 12.3 },
      "lights_on": { "predicted": 3.0, "actual": 5.0, "r2": 0.72, "mae": 1.1 }
    }
  },
  "feedback_loop": {
    "ml_feedback_caps": 0,
    "shadow_feedback_caps": 149,
    "drift_flagged": 0,
    "activity_labels": 1,
    "last_feedback_write": null
  }
}
```

**Changes:**
- `aria/hub/api.py` — New `/api/ml/pipeline` endpoint.

### Frontend: Enhance MLEngine.jsx

Add a **Pipeline Overview** section at the top of the existing page, before FeatureSelection. This section shows:

**Pipeline Flow Bar** — A horizontal flow of 5 connected nodes:
```
[Data] → [Features] → [Models] → [Predictions] → [Feedback]
```
Each node shows a live metric and status LED. The flow uses the existing design system (`.t-frame`, status colors, `data-mono`).

**Expandable Narrative Sections** — Below the flow bar, 5 CollapsibleSection blocks that explain each stage in plain language with live data:

1. **Data Collection** — Entity count, snapshot frequency, health guard status, presence connection. Explains: "ARIA snapshots your home's state every hour — sensors, lights, motion, power, presence."

2. **Feature Engineering** — Feature count by category, top features by importance. Explains: "Raw sensor readings are transformed into 27 features the models can learn from — time patterns, home state, and room presence."

3. **Model Training** — Model types, hyperparameters in plain language, train/val split, last trained date. Explains: "Three model types learn your home's patterns independently, then their predictions are blended."

4. **Predictions** — Current predictions vs actuals per target with R²/MAE. Explains: "Each model predicts what your home should look like right now. The difference between prediction and reality reveals anomalies."

5. **Feedback Loop** — Feedback channel status with freshness indicators. Explains: "Model accuracy feeds back into capability scoring, so ARIA focuses on what it can actually predict well."

The existing FeatureSelection, ModelHealth, and TrainingHistory sections remain below, unchanged — they provide the detail drill-down.

**Changes:**
- `aria/dashboard/spa/src/pages/MLEngine.jsx` — Add PipelineOverview component and 5 narrative sections, fetch `/api/ml/pipeline`.
- `aria/dashboard/spa/src/index.css` — Pipeline flow bar styles (minimal, reuses existing tokens).
- Rebuild SPA.

---

## What This Does NOT Change

- No new pages or navigation items — everything on existing MLEngine page.
- No changes to engine CLI training path (`aria retrain`) — it already works correctly.
- No changes to systemd timers.
- No changes to the dashboard design system or other pages.
