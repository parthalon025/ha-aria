# ML Training Unification + Pipeline Explanation UI — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix divergent ML training paths, add snapshot validation to hub training, close the feedback loop on startup, add a pipeline API endpoint, and enhance the MLEngine dashboard page to explain the full pipeline with live data.

**Architecture:** The hub's `MLEngine._extract_features()` delegates to the shared `vector_builder.build_feature_vector()` for base features, then appends hub-only rolling window stats. Snapshot validation gates training data. A new `/api/ml/pipeline` endpoint aggregates pipeline state for a new Pipeline Overview section on the existing MLEngine.jsx page.

**Tech Stack:** Python (scikit-learn, lightgbm, numpy), FastAPI, Preact (JSX), esbuild, CSS custom properties

---

### Task 1: Unify Feature Extraction — Test

**Files:**
- Modify: `tests/hub/test_ml_training.py`

**Step 1: Write the failing test**

Add a test that verifies `_extract_features()` produces the same base features as `vector_builder.build_feature_vector()`:

```python
from aria.engine.features.vector_builder import build_feature_vector, get_feature_names
from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG

@pytest.mark.asyncio
async def test_extract_features_delegates_to_vector_builder(ml_engine):
    """Hub _extract_features must produce same base features as vector_builder."""
    snapshot = {
        "date": "2026-02-16",
        "hour": 14,
        "time_features": {
            "hour_sin": 0.5, "hour_cos": -0.866,
            "dow_sin": 0.0, "dow_cos": 1.0,
            "month_sin": 0.5, "month_cos": 0.866,
            "day_of_year_sin": 0.3, "day_of_year_cos": 0.95,
            "is_weekend": 0, "is_holiday": 0, "is_night": 0, "is_work_hours": 1,
            "minutes_since_sunrise": 450, "minutes_until_sunset": 270,
            "daylight_remaining_pct": 37.5,
        },
        "power": {"total_watts": 155},
        "lights": {"on": 3, "total_brightness": 450},
        "occupancy": {"device_count_home": 2, "people_home_count": 1, "people_home": ["justin"]},
        "motion": {"active_count": 1},
        "media": {"total_active": 0},
        "weather": {"temp_f": 55, "humidity": 60},
        "presence": {"overall_probability": 0.85, "occupied_room_count": 2, "identified_person_count": 1, "camera_signal_count": 0},
    }

    hub_features = await ml_engine._extract_features(snapshot)
    engine_features = build_feature_vector(snapshot)

    # All engine features must appear in hub features with same values
    for key, val in engine_features.items():
        assert key in hub_features, f"Missing engine feature in hub: {key}"
        assert hub_features[key] == pytest.approx(val, abs=1e-6), (
            f"Feature {key} differs: hub={hub_features[key]} engine={val}"
        )
```

**Step 2: Run test to verify it fails**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_ml_training.py::test_extract_features_delegates_to_vector_builder -v`
Expected: FAIL — hub and engine produce different feature sets (hub has rolling window features, may differ on presence features)

---

### Task 2: Unify Feature Extraction — Implement

**Files:**
- Modify: `aria/modules/ml_engine.py:972-1092` (replace `_extract_features` body)

**Step 1: Add import for vector_builder**

At line 39 (after the existing `_ENGINE_FEATURE_CONFIG` import), add:

```python
from aria.engine.features.vector_builder import build_feature_vector as _engine_build_feature_vector  # noqa: E402
```

**Step 2: Replace `_extract_features()` body**

Replace the body of `_extract_features()` (lines 992-1092) with delegation to the shared builder, keeping the hub-only rolling window append:

```python
    async def _extract_features(
        self,
        snapshot: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        prev_snapshot: Optional[Dict[str, Any]] = None,
        rolling_stats: Optional[Dict[str, float]] = None,
        rolling_window_stats: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, float]]:
        """Extract feature vector from snapshot using shared vector_builder.

        Delegates base feature extraction to vector_builder.build_feature_vector()
        (single source of truth), then appends hub-specific rolling window features
        from the live activity log.

        Args:
            snapshot: Snapshot dictionary
            config: Feature configuration (uses default if None)
            prev_snapshot: Previous snapshot for lag features (optional)
            rolling_stats: Rolling statistics dict (optional)
            rolling_window_stats: Rolling window stats from activity log (optional)

        Returns:
            Dictionary of feature_name -> float value
        """
        if config is None:
            config = await self._get_feature_config()

        # If snapshot lacks time_features, compute them (daily snapshots may not have them)
        if "time_features" not in snapshot:
            snapshot = {**snapshot, "time_features": self._compute_time_features(snapshot)}

        # Delegate base feature extraction to shared engine builder
        features = _engine_build_feature_vector(snapshot, config, prev_snapshot, rolling_stats)

        # Hub-only: append rolling window features from live activity log
        rws = rolling_window_stats or {}
        for hours in ROLLING_WINDOWS_HOURS:
            features[f"rolling_{hours}h_event_count"] = rws.get(f"rolling_{hours}h_event_count", 0)
            features[f"rolling_{hours}h_domain_entropy"] = rws.get(f"rolling_{hours}h_domain_entropy", 0)
            features[f"rolling_{hours}h_dominant_domain_pct"] = rws.get(f"rolling_{hours}h_dominant_domain_pct", 0)
            features[f"rolling_{hours}h_trend"] = rws.get(f"rolling_{hours}h_trend", 0)

        return features
```

**Step 3: Run test to verify it passes**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_ml_training.py::test_extract_features_delegates_to_vector_builder -v`
Expected: PASS

**Step 4: Run full ML training tests**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All tests PASS (no regressions)

**Step 5: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "$(cat <<'EOF'
feat: unify feature extraction — hub delegates to vector_builder

MLEngine._extract_features() now calls vector_builder.build_feature_vector()
for base features and only appends hub-specific rolling window stats.
Fixes divergent feature extraction between engine CLI and hub module paths.

Ref: 2026-02-14-ml-encoding-inconsistency.md

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add Snapshot Validation to Hub Training — Test

**Files:**
- Modify: `tests/hub/test_ml_training.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_load_training_data_rejects_corrupt_snapshots(ml_engine, tmp_path):
    """Hub training must reject snapshots with too few entities or high unavailable ratio."""
    training_data_dir = tmp_path / "training_data"
    training_data_dir.mkdir(exist_ok=True)
    ml_engine.training_data_dir = training_data_dir

    today = datetime.now()

    # Good snapshot
    good = {
        "date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
        "entities": {"total": 3050, "unavailable": 10},
        "power": {"total_watts": 150},
    }
    # Bad: too few entities (HA was down)
    bad_low = {
        "date": (today - timedelta(days=2)).strftime("%Y-%m-%d"),
        "entities": {"total": 50, "unavailable": 2},
        "power": {"total_watts": 0},
    }
    # Bad: high unavailable ratio (HA restarting)
    bad_unavail = {
        "date": (today - timedelta(days=3)).strftime("%Y-%m-%d"),
        "entities": {"total": 3050, "unavailable": 2000},
        "power": {"total_watts": 50},
    }

    for snap in [good, bad_low, bad_unavail]:
        path = training_data_dir / f"{snap['date']}.json"
        path.write_text(json.dumps(snap))

    result = await ml_engine._load_training_data(days=5)

    # Only the good snapshot should survive validation
    assert len(result) == 1
    assert result[0]["entities"]["total"] == 3050
    assert result[0]["entities"]["unavailable"] == 10
```

**Step 2: Run test to verify it fails**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_ml_training.py::test_load_training_data_rejects_corrupt_snapshots -v`
Expected: FAIL — currently `_load_training_data` returns all 3 snapshots without validation

---

### Task 4: Add Snapshot Validation to Hub Training — Implement

**Files:**
- Modify: `aria/modules/ml_engine.py:348-372` (enhance `_load_training_data`)

**Step 1: Add import for validation**

After the `_engine_build_feature_vector` import (added in Task 2), add:

```python
from aria.engine.validation import validate_snapshot_batch  # noqa: E402
```

**Step 2: Add validation to `_load_training_data()`**

Replace the method body (lines 357-372) with:

```python
    async def _load_training_data(self, days: int) -> List[Dict[str, Any]]:
        """Load historical snapshots for training, rejecting corrupt data.

        Applies validate_snapshot_batch() to filter out snapshots with too few
        entities or high unavailable ratios (e.g., during HA restarts).

        Args:
            days: Number of days to load

        Returns:
            List of validated snapshot dictionaries
        """
        raw_snapshots = []
        today = datetime.now()

        for i in range(days):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            snapshot_file = self.training_data_dir / f"{date_str}.json"

            if snapshot_file.exists():
                try:
                    with open(snapshot_file) as f:
                        snapshot = json.load(f)
                        raw_snapshots.append(snapshot)
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Failed to load snapshot {snapshot_file}: {e}")

        # Validate snapshots — reject corrupt/incomplete data
        valid, rejected = validate_snapshot_batch(raw_snapshots)

        if rejected:
            self.logger.warning(
                f"Rejected {len(rejected)} of {len(raw_snapshots)} snapshots "
                f"during validation (corrupt or incomplete)"
            )

        return valid
```

**Step 3: Run test to verify it passes**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_ml_training.py::test_load_training_data_rejects_corrupt_snapshots -v`
Expected: PASS

**Step 4: Run full ML training tests**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All tests PASS

**Step 5: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "$(cat <<'EOF'
feat: add snapshot validation to hub ML training

_load_training_data() now calls validate_snapshot_batch() to reject
snapshots with <100 entities or >50% unavailable — same validation
the engine CLI already uses. Prevents corrupt data from HA restarts
poisoning hub-trained models.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Run Training on Startup If Stale — Test

**Files:**
- Modify: `tests/hub/test_ml_training.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_schedule_periodic_training_runs_immediately_when_stale(ml_engine, mock_hub):
    """If last training was >7 days ago, schedule_periodic_training should set run_immediately=True."""
    # Simulate stale training metadata (10 days old)
    stale_date = (datetime.now() - timedelta(days=10)).isoformat()
    mock_hub.get_cache = AsyncMock(return_value={
        "data": {"last_trained": stale_date}
    })
    mock_hub.schedule_task = AsyncMock()

    await ml_engine.schedule_periodic_training(interval_days=7)

    # Verify schedule_task was called with run_immediately=True
    mock_hub.schedule_task.assert_called_once()
    call_kwargs = mock_hub.schedule_task.call_args[1]
    assert call_kwargs["run_immediately"] is True


@pytest.mark.asyncio
async def test_schedule_periodic_training_not_immediate_when_fresh(ml_engine, mock_hub):
    """If last training was recent, schedule_periodic_training should set run_immediately=False."""
    fresh_date = (datetime.now() - timedelta(days=2)).isoformat()
    mock_hub.get_cache = AsyncMock(return_value={
        "data": {"last_trained": fresh_date}
    })
    mock_hub.schedule_task = AsyncMock()

    await ml_engine.schedule_periodic_training(interval_days=7)

    call_kwargs = mock_hub.schedule_task.call_args[1]
    assert call_kwargs["run_immediately"] is False


@pytest.mark.asyncio
async def test_schedule_periodic_training_runs_immediately_when_no_metadata(ml_engine, mock_hub):
    """If no training metadata exists, should run immediately."""
    mock_hub.get_cache = AsyncMock(return_value=None)
    mock_hub.schedule_task = AsyncMock()

    await ml_engine.schedule_periodic_training(interval_days=7)

    call_kwargs = mock_hub.schedule_task.call_args[1]
    assert call_kwargs["run_immediately"] is True
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_ml_training.py -k "schedule_periodic" -v`
Expected: FAIL — current code hardcodes `run_immediately=False`

---

### Task 6: Run Training on Startup If Stale — Implement

**Files:**
- Modify: `aria/modules/ml_engine.py:1397-1418` (enhance `schedule_periodic_training`)

**Step 1: Replace method body**

```python
    async def schedule_periodic_training(self, interval_days: int = 7):
        """Schedule periodic model retraining. Runs immediately if stale.

        Checks ml_training_metadata cache to determine if models are stale
        (>interval_days since last training or no metadata). If stale, the
        first training run fires immediately instead of waiting for the
        next interval.

        Args:
            interval_days: Days between training runs
        """
        # Check if training is stale
        run_immediately = False
        metadata_entry = await self.hub.get_cache("ml_training_metadata")

        if not metadata_entry:
            self.logger.info("No training metadata found — will train immediately on startup")
            run_immediately = True
        else:
            data = metadata_entry if isinstance(metadata_entry, dict) else {}
            if "data" in data:
                data = data["data"]
            last_trained_str = data.get("last_trained")
            if last_trained_str:
                try:
                    last_trained = datetime.fromisoformat(last_trained_str)
                    days_since = (datetime.now() - last_trained).total_seconds() / 86400
                    if days_since > interval_days:
                        self.logger.info(
                            f"Models are {days_since:.1f} days old (threshold: {interval_days}) "
                            f"— will train immediately on startup"
                        )
                        run_immediately = True
                    else:
                        self.logger.info(f"Models are {days_since:.1f} days old — fresh enough, normal schedule")
                except (ValueError, TypeError):
                    self.logger.warning("Invalid last_trained date in metadata — will train immediately")
                    run_immediately = True
            else:
                run_immediately = True

        async def training_task():
            try:
                await self.train_models(days_history=60)
            except Exception as e:
                self.logger.error(f"Scheduled training failed: {e}")

        await self.hub.schedule_task(
            task_id="ml_training_periodic",
            coro=training_task,
            interval=timedelta(days=interval_days),
            run_immediately=run_immediately,
        )

        self.logger.info(
            f"Scheduled periodic training every {interval_days} days "
            f"(run_immediately={run_immediately})"
        )
```

**Step 2: Run tests to verify they pass**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_ml_training.py -k "schedule_periodic" -v`
Expected: All 3 PASS

**Step 3: Run full ML training tests**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All PASS

**Step 4: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "$(cat <<'EOF'
feat: run ML training immediately on startup if stale

schedule_periodic_training() now checks ml_training_metadata cache.
If last_trained is >7 days old or missing, sets run_immediately=True
so the feedback loop closes on first startup instead of waiting 7 days.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Add `/api/ml/pipeline` Endpoint — Test

**Files:**
- Modify: `tests/hub/test_ml_training.py` (or create `tests/hub/test_api_pipeline.py` if test file is >500 lines)

**Step 1: Write the failing test**

```python
import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_api_ml_pipeline_returns_expected_shape():
    """GET /api/ml/pipeline should return all 5 pipeline sections."""
    from aria.hub.api import create_router
    from fastapi import FastAPI

    mock_hub = Mock()
    mock_hub.cache = Mock()

    # Mock cache responses
    async def mock_cache_get(key):
        if key == "intelligence":
            return {
                "entity_count": 3066,
                "snapshot_log": {"intraday_count": 78, "daily_count": 5, "last_snapshot": "2026-02-16T11:00:00"},
                "health_guard": {"status": "passing"},
                "feature_selection": {"selected_features": ["hour_sin", "power_watts"], "total_features": 27},
                "ml_models": {"last_trained": "2026-02-14T02:00:00", "scores": {"power_watts": {"r2": 0.84, "mae": 12.3}}},
                "drift_status": {"needs_retrain": False, "drifted_metrics": []},
            }
        if key == "ml_training_metadata":
            return {"data": {"last_trained": "2026-02-14T02:00:00", "num_snapshots": 78, "targets_trained": ["power_watts"]}}
        if key == "presence":
            return {"data": {"connected": True}}
        if key == "capabilities":
            return {"data": {"power_monitoring": {"available": True}}}
        if key == "feedback_health":
            return {"data": {"ml_feedback": {"capabilities_updated": 0}, "shadow_feedback": {"capabilities_updated": 149}}}
        return None

    mock_hub.cache.get = AsyncMock(side_effect=mock_cache_get)

    app = FastAPI()
    router = create_router(mock_hub)
    app.include_router(router)

    client = TestClient(app)
    resp = client.get("/api/ml/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    # All 5 sections present
    assert "data_collection" in data
    assert "feature_engineering" in data
    assert "model_training" in data
    assert "predictions" in data
    assert "feedback_loop" in data
```

**Step 2: Run test to verify it fails**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_ml_training.py::test_api_ml_pipeline_returns_expected_shape -v`
Expected: FAIL — endpoint doesn't exist yet

---

### Task 8: Add `/api/ml/pipeline` Endpoint — Implement

**Files:**
- Modify: `aria/hub/api.py` (add after line 394, before organic discovery endpoints)

**Step 1: Add the endpoint**

Insert after the `get_ml_shap` endpoint (line 394):

```python
    @router.get("/api/ml/pipeline")
    async def get_ml_pipeline():
        """Aggregate ML pipeline state for the Pipeline Overview UI."""
        try:
            intel = await hub.cache.get("intelligence") or {}
            training_meta = await hub.cache.get("ml_training_metadata")
            training_data = (training_meta or {}).get("data", training_meta or {})
            presence = await hub.cache.get("presence")
            presence_data = (presence or {}).get("data", presence or {})
            caps = await hub.cache.get("capabilities")
            caps_data = (caps or {}).get("data", caps or {})
            feedback_health = await hub.cache.get("feedback_health")
            fb_data = (feedback_health or {}).get("data", feedback_health or {})

            snapshot_log = intel.get("snapshot_log", {})
            feature_sel = intel.get("feature_selection", {})
            ml_models = intel.get("ml_models", {})

            return {
                "data_collection": {
                    "entity_count": intel.get("entity_count", 0),
                    "snapshot_count_intraday": snapshot_log.get("intraday_count", 0),
                    "snapshot_count_daily": snapshot_log.get("daily_count", 0),
                    "last_snapshot": snapshot_log.get("last_snapshot"),
                    "health_guard": intel.get("health_guard", {}).get("status", "unknown"),
                    "presence_connected": bool(presence_data.get("connected", False)),
                },
                "feature_engineering": {
                    "total_features": feature_sel.get("total_features", 0),
                    "selected_features": feature_sel.get("selected_features", []),
                    "method": feature_sel.get("method", "none"),
                },
                "model_training": {
                    "last_trained": training_data.get("last_trained"),
                    "model_types": ["GradientBoosting", "RandomForest", "LightGBM"],
                    "targets": training_data.get("targets_trained", []),
                    "validation_split": "80/20 chronological",
                    "total_snapshots_used": training_data.get("num_snapshots", 0),
                },
                "predictions": {
                    "scores": ml_models.get("scores", {}),
                },
                "feedback_loop": {
                    "ml_feedback_caps": (fb_data.get("ml_feedback", {}) or {}).get("capabilities_updated", 0),
                    "shadow_feedback_caps": (fb_data.get("shadow_feedback", {}) or {}).get("capabilities_updated", 0),
                    "drift_flagged": len(intel.get("drift_status", {}).get("drifted_metrics", [])),
                    "last_feedback_write": training_data.get("last_trained"),
                },
            }
        except Exception:
            logger.exception("Error getting ML pipeline state")
            raise HTTPException(status_code=500, detail="Internal server error")
```

**Step 2: Run test to verify it passes**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_ml_training.py::test_api_ml_pipeline_returns_expected_shape -v`
Expected: PASS

**Step 3: Run all API tests for regressions**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/ -k "api" -v --timeout=120`
Expected: All PASS

**Step 4: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/hub/api.py tests/hub/test_ml_training.py
git commit -m "$(cat <<'EOF'
feat: add /api/ml/pipeline endpoint for Pipeline Overview UI

Aggregates data_collection, feature_engineering, model_training,
predictions, and feedback_loop state from multiple cache categories
into a single response for the dashboard Pipeline Overview section.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Enhance MLEngine.jsx — Pipeline Flow Bar + Narrative Sections

**Files:**
- Modify: `aria/dashboard/spa/src/pages/MLEngine.jsx`
- Modify: `aria/dashboard/spa/src/index.css`

**Step 1: Add Pipeline Overview component to MLEngine.jsx**

Add at the top of the file (after the existing helper functions, before `FeatureSelection`), a new `PipelineOverview` component:

```jsx
// ─── Pipeline Overview ───────────────────────────────────────────────────────

function StatusLed({ status }) {
  const color = status === 'active' || status === 'passing' || status === 'connected'
    ? 'var(--status-healthy)'
    : status === 'stale' || status === 'warning'
    ? 'var(--status-warning)'
    : 'var(--text-tertiary)';
  return (
    <span
      class="pipeline-led"
      style={`background: ${color}; box-shadow: 0 0 6px ${color};`}
    />
  );
}

function PipelineNode({ label, metric, status }) {
  return (
    <div class="pipeline-node">
      <StatusLed status={status} />
      <span class="pipeline-node-label">{label}</span>
      <span class="pipeline-node-metric data-mono">{metric}</span>
    </div>
  );
}

function PipelineFlowBar({ pipeline }) {
  if (!pipeline) return null;

  const dc = pipeline.data_collection || {};
  const fe = pipeline.feature_engineering || {};
  const mt = pipeline.model_training || {};
  const pr = pipeline.predictions || {};
  const fb = pipeline.feedback_loop || {};

  const dataStatus = dc.last_snapshot ? 'active' : 'unknown';
  const featureStatus = fe.total_features > 0 ? 'active' : 'unknown';
  const modelStatus = mt.last_trained ? 'active' : 'unknown';
  const predStatus = pr.scores && Object.keys(pr.scores).length > 0 ? 'active' : 'unknown';
  const fbStatus = (fb.ml_feedback_caps > 0 || fb.shadow_feedback_caps > 0) ? 'active' : 'unknown';

  return (
    <div class="pipeline-flow-bar">
      <PipelineNode label="Data" metric={`${dc.entity_count || 0} entities`} status={dataStatus} />
      <span class="pipeline-arrow">→</span>
      <PipelineNode label="Features" metric={`${fe.total_features || 0} signals`} status={featureStatus} />
      <span class="pipeline-arrow">→</span>
      <PipelineNode label="Models" metric={mt.last_trained ? formatDate(mt.last_trained) : 'untrained'} status={modelStatus} />
      <span class="pipeline-arrow">→</span>
      <PipelineNode label="Predictions" metric={pr.scores ? `${Object.keys(pr.scores).length} targets` : 'none'} status={predStatus} />
      <span class="pipeline-arrow">→</span>
      <PipelineNode label="Feedback" metric={`${(fb.ml_feedback_caps || 0) + (fb.shadow_feedback_caps || 0)} caps`} status={fbStatus} />
    </div>
  );
}

function PipelineOverview({ pipeline, loading }) {
  if (!pipeline && !loading) return null;

  const dc = pipeline?.data_collection || {};
  const fe = pipeline?.feature_engineering || {};
  const mt = pipeline?.model_training || {};
  const pr = pipeline?.predictions || {};
  const fb = pipeline?.feedback_loop || {};

  return (
    <div class="space-y-4">
      <PipelineFlowBar pipeline={pipeline} />

      <CollapsibleSection
        title="1. Data Collection"
        subtitle="Raw sensor readings from Home Assistant"
        summary={dc.last_snapshot ? `${dc.snapshot_count_intraday || 0} snapshots` : 'no data'}
        defaultOpen={false}
        loading={loading}
      >
        <p style="font-size: var(--type-body); color: var(--text-secondary); line-height: 1.6; margin-bottom: 12px;">
          ARIA snapshots your home's state every hour — sensors, lights, motion, power, and presence.
          Each snapshot captures ~{dc.entity_count || '?'} entities into a single data point.
        </p>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div class="t-frame" data-label="Intraday Snapshots">
            <span class="data-mono" style="font-size: var(--type-headline); color: var(--accent);">{dc.snapshot_count_intraday || 0}</span>
          </div>
          <div class="t-frame" data-label="Daily Snapshots">
            <span class="data-mono" style="font-size: var(--type-headline); color: var(--accent);">{dc.snapshot_count_daily || 0}</span>
          </div>
          <div class="t-frame" data-label="Health Guard">
            <Badge label={dc.health_guard || 'unknown'} color={dc.health_guard === 'passing' ? 'var(--status-healthy)' : 'var(--status-warning)'} />
          </div>
        </div>
        {dc.presence_connected && (
          <p style="font-size: var(--type-label); color: var(--text-tertiary); margin-top: 8px;">
            Presence detection connected — camera and sensor signals feeding into snapshots.
          </p>
        )}
      </CollapsibleSection>

      <CollapsibleSection
        title="2. Feature Engineering"
        subtitle="Transforming raw data into learnable signals"
        summary={fe.total_features > 0 ? `${fe.total_features} features` : 'not computed'}
        defaultOpen={false}
        loading={loading}
      >
        <p style="font-size: var(--type-body); color: var(--text-secondary); line-height: 1.6; margin-bottom: 12px;">
          Raw sensor readings are transformed into {fe.total_features || '?'} features the models can learn from —
          time patterns (hour, day-of-week, seasonality), home state (power, lights, occupancy),
          presence signals, and rolling activity trends.
        </p>
        {fe.selected_features && fe.selected_features.length > 0 && (
          <div class="t-frame" data-label={`Top Features (${fe.method || 'ranked'})`}>
            <div style="display: flex; flex-wrap: wrap; gap: 6px;">
              {fe.selected_features.slice(0, 10).map(name => (
                <span key={name} class="data-mono" style="font-size: var(--type-label); padding: 2px 8px; border: 1px solid var(--border-subtle); border-radius: var(--radius);">
                  {name}
                </span>
              ))}
              {fe.selected_features.length > 10 && (
                <span class="data-mono" style="font-size: var(--type-label); color: var(--text-tertiary);">
                  +{fe.selected_features.length - 10} more
                </span>
              )}
            </div>
          </div>
        )}
      </CollapsibleSection>

      <CollapsibleSection
        title="3. Model Training"
        subtitle="Learning your home's patterns"
        summary={mt.last_trained ? `trained ${formatDate(mt.last_trained)}` : 'untrained'}
        defaultOpen={false}
        loading={loading}
      >
        <p style="font-size: var(--type-body); color: var(--text-secondary); line-height: 1.6; margin-bottom: 12px;">
          Three model types learn your home's patterns independently — Gradient Boosting (precise sequential learning),
          Random Forest (robust ensemble averaging), and LightGBM (fast gradient-based). Their predictions are blended
          for better accuracy than any single model.
        </p>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div class="t-frame" data-label="Snapshots Used">
            <span class="data-mono" style="font-size: var(--type-headline); color: var(--accent);">{mt.total_snapshots_used || 0}</span>
          </div>
          <div class="t-frame" data-label="Targets">
            <span class="data-mono" style="font-size: var(--type-headline); color: var(--accent);">{(mt.targets || []).length}</span>
          </div>
          <div class="t-frame" data-label="Validation Split">
            <span class="data-mono" style="font-size: var(--type-body); color: var(--text-primary);">{mt.validation_split || '80/20'}</span>
          </div>
        </div>
        {mt.targets && mt.targets.length > 0 && (
          <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;">
            {mt.targets.map(t => (
              <Badge key={t} label={t} color="var(--accent)" />
            ))}
          </div>
        )}
      </CollapsibleSection>

      <CollapsibleSection
        title="4. Predictions"
        subtitle="What your home should look like right now"
        summary={pr.scores ? `${Object.keys(pr.scores).length} targets` : 'no predictions'}
        defaultOpen={false}
        loading={loading}
      >
        <p style="font-size: var(--type-body); color: var(--text-secondary); line-height: 1.6; margin-bottom: 12px;">
          Each model predicts what your home should look like right now. The difference between prediction and reality
          reveals anomalies — unexpected power spikes, unusual occupancy, or lighting changes that don't match your patterns.
        </p>
        {pr.scores && Object.keys(pr.scores).length > 0 && (
          <div class="t-frame" data-label="Prediction Accuracy">
            <div style="overflow-x: auto;">
              <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: var(--type-body);">
                <thead>
                  <tr style="border-bottom: 2px solid var(--border-subtle);">
                    <th style="text-align: left; padding: 8px 12px; color: var(--text-secondary); font-weight: 600;">Target</th>
                    <th style="text-align: right; padding: 8px 12px; color: var(--text-secondary); font-weight: 600;">R&sup2;</th>
                    <th style="text-align: right; padding: 8px 12px; color: var(--text-secondary); font-weight: 600;">MAE</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(pr.scores).map(([name, vals]) => (
                    <tr key={name} style="border-bottom: 1px solid var(--border-subtle);">
                      <td style="padding: 8px 12px; color: var(--text-primary);">{name}</td>
                      <td style="text-align: right; padding: 8px 12px; color: var(--accent);">
                        {vals?.r2 != null ? vals.r2.toFixed(3) : '\u2014'}
                      </td>
                      <td style="text-align: right; padding: 8px 12px; color: var(--text-primary);">
                        {vals?.mae != null ? vals.mae.toFixed(3) : '\u2014'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </CollapsibleSection>

      <CollapsibleSection
        title="5. Feedback Loop"
        subtitle="Closing the loop — accuracy feeds back into learning"
        summary={
          (fb.ml_feedback_caps || 0) + (fb.shadow_feedback_caps || 0) > 0
            ? `${(fb.ml_feedback_caps || 0) + (fb.shadow_feedback_caps || 0)} capabilities updated`
            : 'awaiting feedback'
        }
        defaultOpen={false}
        loading={loading}
      >
        <p style="font-size: var(--type-body); color: var(--text-secondary); line-height: 1.6; margin-bottom: 12px;">
          Model accuracy feeds back into capability scoring, so ARIA focuses on what it can actually predict well.
          Shadow mode tests new capabilities before they go live. Drift detection flags when your home's patterns change
          and models need retraining.
        </p>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div class="t-frame" data-label="ML Feedback">
            <span class="data-mono" style="font-size: var(--type-headline); color: var(--accent);">{fb.ml_feedback_caps || 0}</span>
            <span style="font-size: var(--type-label); color: var(--text-tertiary); display: block;">capabilities scored</span>
          </div>
          <div class="t-frame" data-label="Shadow Feedback">
            <span class="data-mono" style="font-size: var(--type-headline); color: var(--accent);">{fb.shadow_feedback_caps || 0}</span>
            <span style="font-size: var(--type-label); color: var(--text-tertiary); display: block;">shadow-tested</span>
          </div>
        </div>
        {fb.drift_flagged > 0 && (
          <div style="margin-top: 8px;">
            <Badge label={`${fb.drift_flagged} drift`} color="var(--status-warning)" />
          </div>
        )}
      </CollapsibleSection>
    </div>
  );
}
```

**Step 2: Update the page component to fetch pipeline data**

In the `MLEngine` page component (around line 299), add pipeline state and fetch:

```jsx
export default function MLEngine() {
  const [features, setFeatures] = useState(null);
  const [models, setModels] = useState(null);
  const [drift, setDrift] = useState(null);
  const [pipeline, setPipeline] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  function load() {
    setLoading(true);
    setError(null);
    Promise.all([
      fetchJson('/api/ml/features'),
      fetchJson('/api/ml/models'),
      fetchJson('/api/ml/drift'),
      fetchJson('/api/ml/pipeline'),
    ])
      .then(([f, m, d, p]) => {
        setFeatures(f);
        setModels(m);
        setDrift(d);
        setPipeline(p);
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }

  useEffect(() => { load(); }, []);

  if (error) {
    return (
      <div class="space-y-6">
        <PageBanner page="MLENGINE" />
        <ErrorState error={error} onRetry={load} />
      </div>
    );
  }

  if (loading) {
    return (
      <div class="space-y-6">
        <PageBanner page="MLENGINE" />
        <LoadingState type="full" />
      </div>
    );
  }

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="MLENGINE" subtitle="How ARIA learns your home — from raw data to predictions" />

      <PipelineOverview pipeline={pipeline} loading={false} />

      <FeatureSelection features={features} loading={false} />
      <ModelHealth models={models} loading={false} />
      <TrainingHistory models={models} loading={false} />
    </div>
  );
}
```

**Step 3: Add pipeline flow bar CSS to index.css**

Append to `aria/dashboard/spa/src/index.css`:

```css
/* Pipeline Flow Bar */
.pipeline-flow-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 16px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  overflow-x: auto;
}

.pipeline-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  min-width: 80px;
  padding: 8px;
}

.pipeline-node-label {
  font-family: var(--font-mono);
  font-size: var(--type-body);
  font-weight: 600;
  color: var(--text-primary);
}

.pipeline-node-metric {
  font-size: var(--type-label);
  color: var(--text-tertiary);
  white-space: nowrap;
}

.pipeline-led {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.pipeline-arrow {
  font-family: var(--font-mono);
  font-size: var(--type-headline);
  color: var(--text-tertiary);
  flex-shrink: 0;
}

@media (max-width: 640px) {
  .pipeline-flow-bar {
    gap: 4px;
    padding: 12px 8px;
  }
  .pipeline-node {
    min-width: 56px;
    padding: 4px;
  }
  .pipeline-node-metric {
    font-size: 10px;
  }
}
```

**Step 4: Rebuild SPA**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`
Expected: Build succeeds, `dist/bundle.js` updated

**Step 5: Commit**

```bash
cd /home/justin/Documents/projects/ha-aria
git add aria/dashboard/spa/src/pages/MLEngine.jsx aria/dashboard/spa/src/index.css
git commit -m "$(cat <<'EOF'
feat: add Pipeline Overview to ML Engine page

New Pipeline Flow Bar shows 5-stage pipeline with status LEDs:
Data → Features → Models → Predictions → Feedback.

Five expandable narrative sections explain each stage in plain
language with live data from /api/ml/pipeline endpoint. Existing
FeatureSelection, ModelHealth, and TrainingHistory sections remain
as detail drill-downs below.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Full Test Suite + Verification

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/ -v --timeout=120 -q`
Expected: All pass (1235+ tests), 0 failures

**Step 2: Restart hub service and verify API**

Run: `systemctl --user restart aria-hub && sleep 3 && curl -s http://127.0.0.1:8001/api/ml/pipeline | python3 -m json.tool`
Expected: JSON response with all 5 pipeline sections populated

**Step 3: Verify dashboard loads**

Run: `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/ui/`
Expected: 200

**Step 4: Commit any remaining changes**

```bash
cd /home/justin/Documents/projects/ha-aria
git status
# If clean, skip. Otherwise stage and commit.
```
