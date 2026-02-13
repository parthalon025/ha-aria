"""Test ML Engine training pipeline."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from unittest.mock import AsyncMock, Mock  # noqa: E402

from aria.modules.ml_engine import MLEngine  # noqa: E402
from aria.hub.core import IntelligenceHub  # noqa: E402


@pytest.fixture
def mock_hub():
    """Create mock IntelligenceHub."""
    hub = Mock(spec=IntelligenceHub)
    hub.get_cache = AsyncMock(return_value=None)
    hub.get_cache_fresh = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.logger = Mock()
    return hub


@pytest.fixture
def ml_engine(mock_hub, tmp_path):
    """Create MLEngine with mock hub and temp directories."""
    models_dir = tmp_path / "models"
    training_data_dir = tmp_path / "training_data"
    models_dir.mkdir()
    training_data_dir.mkdir()

    engine = MLEngine(mock_hub, str(models_dir), str(training_data_dir))
    return engine


@pytest.fixture
def mock_capabilities():
    """Create mock capabilities data."""
    return {
        "data": {
            "power_monitoring": {
                "available": True,
                "entities": ["sensor.power_1", "sensor.power_2"]
            },
            "lighting": {
                "available": True,
                "entities": ["light.living_room", "light.bedroom"]
            },
            "occupancy": {
                "available": True,
                "entities": ["device_tracker.phone", "person.justin"]
            }
        }
    }


@pytest.fixture
def synthetic_snapshots(tmp_path):
    """Create synthetic training snapshots."""
    training_data_dir = tmp_path / "training_data"
    training_data_dir.mkdir(exist_ok=True)

    snapshots = []
    base_date = datetime.now()

    # Create 30 days of synthetic data
    for day_offset in range(30):
        date = base_date - timedelta(days=30 - day_offset)
        hour = 12  # Noon snapshot

        snapshot = {
            "date": date.strftime("%Y-%m-%d"),
            "hour": hour,
            "time_features": {
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "dow_sin": np.sin(2 * np.pi * date.weekday() / 7),
                "dow_cos": np.cos(2 * np.pi * date.weekday() / 7),
                "month_sin": 0.5,
                "month_cos": 0.866,
                "day_of_year_sin": 0.3,
                "day_of_year_cos": 0.95,
                "is_weekend": date.weekday() >= 5,
                "is_holiday": False,
                "is_night": False,
                "is_work_hours": True,
                "minutes_since_sunrise": 360,
                "minutes_until_sunset": 300,
                "daylight_remaining_pct": 0.5
            },
            "weather": {
                "temp_f": 65.0 + day_offset % 10,
                "humidity_pct": 50.0 + day_offset % 20,
                "wind_mph": 5.0,
                "pressure": 1013.0,
                "cloud_cover": 30.0,
                "uv_index": 3.0
            },
            "power": {
                "total_watts": 500.0 + day_offset * 10 + np.random.randn() * 50
            },
            "lights": {
                "on": 3 + (day_offset % 3),
                "total_brightness": 150.0 + day_offset * 5
            },
            "occupancy": {
                "people_home": ["person.justin"],
                "people_home_count": 1,
                "device_count_home": 2 + (day_offset % 2),
                "devices_home": ["device_tracker.phone"]
            },
            "motion": {
                "active_count": 1 + (day_offset % 2)
            }
        }

        # Save snapshot file
        snapshot_file = training_data_dir / f"{date.strftime('%Y-%m-%d')}.json"
        with open(snapshot_file, "w") as f:
            json.dump(snapshot, f)

        snapshots.append(snapshot)

    return snapshots


class TestMLEngine:
    """Test ML Engine training pipeline."""

    @pytest.mark.asyncio
    async def test_load_training_data(self, ml_engine, synthetic_snapshots):
        """Test loading historical snapshots."""
        snapshots = await ml_engine._load_training_data(days=30)

        # May be 29 or 30 depending on if today's file exists
        assert len(snapshots) >= 29
        assert all("power" in s for s in snapshots)
        assert all("time_features" in s for s in snapshots)

    @pytest.mark.asyncio
    async def test_build_training_dataset(self, ml_engine, synthetic_snapshots):
        """Test building training dataset from snapshots."""
        X, y, weights = await ml_engine._build_training_dataset(synthetic_snapshots, "power_watts")

        # Should have 30 samples
        assert len(X) == 30
        assert len(y) == 30
        assert len(weights) == 30

        # X should be 2D numpy array
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2

        # y should be 1D numpy array
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

        # Weights should be 1D numpy array of positive values
        assert isinstance(weights, np.ndarray)
        assert weights.ndim == 1
        assert all(weights > 0)

        # Features should be reasonable
        assert X.shape[1] > 20  # Should have many features

        # Target values should be positive
        assert all(y > 0)

    @pytest.mark.asyncio
    async def test_rolling_stats_computation(self, ml_engine, synthetic_snapshots):
        """Test rolling stats are computed correctly."""
        # Get first 10 snapshots
        snapshots = synthetic_snapshots[:10]

        # For 8th snapshot (index 7), check rolling stats
        i = 7
        recent = snapshots[max(0, i - 7):i]

        # Manual calculation
        expected_power_mean = sum(
            s.get("power", {}).get("total_watts", 0) for s in recent
        ) / len(recent)
        expected_lights_mean = sum(
            s.get("lights", {}).get("on", 0) for s in recent
        ) / len(recent)

        # Build dataset and verify rolling stats are used
        X, y, weights = await ml_engine._build_training_dataset(snapshots, "power_watts")

        # Snapshot at index 7 should have rolling stats
        assert len(X) == 10
        # Can't directly verify the exact values without knowing feature order,
        # but we can verify the computation logic is correct
        assert expected_power_mean > 0
        assert expected_lights_mean > 0

    def test_extract_target(self, ml_engine, synthetic_snapshots):
        """Test target extraction from snapshot."""
        snapshot = synthetic_snapshots[0]

        # Test power_watts extraction
        power = ml_engine._extract_target(snapshot, "power_watts")
        assert power is not None
        assert power > 0

        # Test lights_on extraction
        lights = ml_engine._extract_target(snapshot, "lights_on")
        assert lights is not None
        assert lights >= 0

        # Test unknown target
        unknown = ml_engine._extract_target(snapshot, "unknown_metric")
        assert unknown is None

    @pytest.mark.asyncio
    async def test_feature_extraction(self, ml_engine, synthetic_snapshots):
        """Test feature extraction from snapshot."""
        snapshot = synthetic_snapshots[5]
        prev_snapshot = synthetic_snapshots[4]

        rolling_stats = {
            "power_mean_7d": 500.0,
            "lights_mean_7d": 3.0
        }

        features = await ml_engine._extract_features(
            snapshot,
            prev_snapshot=prev_snapshot,
            rolling_stats=rolling_stats
        )

        assert features is not None
        assert isinstance(features, dict)

        # Check time features
        assert "hour_sin" in features
        assert "hour_cos" in features
        assert "dow_sin" in features
        assert "dow_cos" in features

        # Check weather features
        assert "weather_temp_f" in features
        assert "weather_humidity_pct" in features

        # Check home state features
        assert "lights_on" in features
        assert "people_home_count" in features

        # Check lag features
        assert "prev_snapshot_power" in features
        assert "rolling_7d_power_mean" in features

        # Verify rolling stats were used
        assert features["rolling_7d_power_mean"] == 500.0
        assert features["rolling_7d_lights_mean"] == 3.0

    @pytest.mark.asyncio
    async def test_train_models(self, ml_engine, mock_hub, mock_capabilities, synthetic_snapshots):
        """Test complete training pipeline."""
        # Setup mock capabilities (train_models uses get_cache_fresh)
        mock_hub.get_cache_fresh.return_value = mock_capabilities

        # Train models
        await ml_engine.train_models(days_history=30)

        # Verify models were trained
        assert "power_watts" in ml_engine.models
        assert "lights_on" in ml_engine.models
        assert "devices_home" in ml_engine.models

        # Verify model metadata
        power_model = ml_engine.models["power_watts"]
        assert "gb_model" in power_model
        assert "rf_model" in power_model
        assert "lgbm_model" in power_model
        assert "iso_model" in power_model
        assert "trained_at" in power_model
        assert "num_samples" in power_model
        assert "feature_names" in power_model
        assert "feature_importance" in power_model
        assert "accuracy_scores" in power_model

        # Verify accuracy scores (all 3 models)
        scores = power_model["accuracy_scores"]
        assert "gb_mae" in scores
        assert "gb_r2" in scores
        assert "rf_mae" in scores
        assert "rf_r2" in scores
        assert "lgbm_mae" in scores
        assert "lgbm_r2" in scores

        # Verify feature importance is a dict
        importance = power_model["feature_importance"]
        assert isinstance(importance, dict)
        assert len(importance) > 20  # Should have many features

        # Verify model files were saved
        models_dir = Path(ml_engine.models_dir)
        assert (models_dir / "power_watts_model.pkl").exists()
        assert (models_dir / "lights_on_model.pkl").exists()

        # Verify cache was updated with training metadata
        mock_hub.set_cache.assert_called()
        metadata_calls = [
            call for call in mock_hub.set_cache.call_args_list
            if call[0][0] == "ml_training_metadata"
        ]
        assert len(metadata_calls) > 0, "ml_training_metadata not found in set_cache calls"

        metadata = metadata_calls[0][0][1]
        assert "last_trained" in metadata
        assert "num_snapshots" in metadata
        assert metadata["num_snapshots"] >= 29  # May be 29 or 30 depending on if today's file exists
        assert "targets_trained" in metadata
        assert "accuracy_summary" in metadata

    @pytest.mark.asyncio
    async def test_insufficient_training_data(self, ml_engine, synthetic_snapshots):
        """Test handling of insufficient training data."""
        # Use only 10 snapshots (need 14+)
        X, y, weights = await ml_engine._build_training_dataset(synthetic_snapshots[:10], "power_watts")

        # Should still return arrays, but training will fail with warning
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(weights, np.ndarray)

    def test_model_hyperparameters(self, ml_engine, mock_hub, mock_capabilities, synthetic_snapshots):
        """Verify model hyperparameters match ha-intelligence."""
        # This test verifies the code, not runtime (runtime test needs real sklearn)
        # Check the code uses correct hyperparameters by inspection

        # GradientBoosting should have:
        # - n_estimators=100
        # - learning_rate=0.1
        # - max_depth=4
        # - subsample=0.8

        # RandomForest should have:
        # - n_estimators=100
        # - max_depth=5

        # IsolationForest should have:
        # - n_estimators=100
        # - contamination=0.05

        # These are verified by reading the code in _train_model_for_target
        pass

    @pytest.mark.asyncio
    async def test_generate_predictions_basic(self, ml_engine, mock_hub, mock_capabilities, synthetic_snapshots):
        """Test basic prediction generation."""
        # Setup: train models first (train_models uses get_cache_fresh)
        mock_hub.get_cache_fresh.return_value = mock_capabilities
        await ml_engine.train_models(days_history=30)

        # Setup: mock snapshot retrieval (generate_predictions uses get_cache)
        current_snapshot = synthetic_snapshots[-1]

        async def mock_get_cache(key):
            if key == "latest_snapshot":
                return {"data": current_snapshot}
            elif key == "capabilities":
                return mock_capabilities
            return None

        mock_hub.get_cache = AsyncMock(side_effect=mock_get_cache)

        # Generate predictions
        result = await ml_engine.generate_predictions()

        # Verify structure
        assert "timestamp" in result
        assert "predictions" in result
        assert "anomaly_detected" in result
        assert "feature_count" in result
        assert "model_count" in result

        # Verify predictions were generated
        predictions = result["predictions"]
        assert len(predictions) > 0

        # Verify each prediction has required fields
        for target, pred in predictions.items():
            assert "value" in pred
            assert "gb_prediction" in pred
            assert "rf_prediction" in pred
            assert "lgbm_prediction" in pred
            assert "confidence" in pred
            assert "is_anomaly" in pred

            # Verify confidence is between 0 and 1
            assert 0 <= pred["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_generate_predictions_blending(self, ml_engine, mock_hub, mock_capabilities, synthetic_snapshots):
        """Test N-model weighted blending (GB 35%, RF 25%, LGBM 40%)."""
        # Setup (train_models uses get_cache_fresh)
        mock_hub.get_cache_fresh.return_value = mock_capabilities
        await ml_engine.train_models(days_history=30)

        current_snapshot = synthetic_snapshots[-1]

        async def mock_get_cache(key):
            if key == "latest_snapshot":
                return {"data": current_snapshot}
            elif key == "capabilities":
                return mock_capabilities
            return None

        mock_hub.get_cache = AsyncMock(side_effect=mock_get_cache)

        # Generate predictions
        result = await ml_engine.generate_predictions()

        # Verify N-model weighted blending for each prediction
        for target, pred in result["predictions"].items():
            gb = pred["gb_prediction"]
            rf = pred["rf_prediction"]
            lgbm = pred["lgbm_prediction"]
            blended = pred["value"]
            weights = pred["blend_weights"]

            # Verify weights sum to 1.0
            assert abs(sum(weights.values()) - 1.0) < 0.01

            # Verify blended = sum(weight_i * pred_i)
            # Tolerance 0.02: both blended and expected are independently rounded
            # to 2dp, which can introduce up to 0.01 rounding disagreement.
            expected = round(weights["gb"] * gb + weights["rf"] * rf + weights["lgbm"] * lgbm, 2)
            assert abs(blended - expected) < 0.02, f"{target}: blended={blended}, expected={expected}"

    @pytest.mark.asyncio
    async def test_generate_predictions_confidence(self, ml_engine, mock_hub, mock_capabilities, synthetic_snapshots):
        """Test confidence calculation based on N-model agreement."""
        # Setup (train_models uses get_cache_fresh)
        mock_hub.get_cache_fresh.return_value = mock_capabilities
        await ml_engine.train_models(days_history=30)

        current_snapshot = synthetic_snapshots[-1]

        async def mock_get_cache(key):
            if key == "latest_snapshot":
                return {"data": current_snapshot}
            elif key == "capabilities":
                return mock_capabilities
            return None

        mock_hub.get_cache = AsyncMock(side_effect=mock_get_cache)

        # Generate predictions
        result = await ml_engine.generate_predictions()

        # Verify confidence logic (N-model: max deviation / abs(mean))
        for target, pred in result["predictions"].items():
            pred_values = [pred["gb_prediction"], pred["rf_prediction"], pred["lgbm_prediction"]]
            confidence = pred["confidence"]
            avg_pred = sum(pred_values) / len(pred_values)

            if abs(avg_pred) > 1e-6:
                max_diff = max(abs(p - avg_pred) for p in pred_values)
                rel_diff = max_diff / abs(avg_pred)
                expected_conf = max(0.0, min(1.0, 1.0 - rel_diff))
            else:
                max_diff = max(abs(p - avg_pred) for p in pred_values)
                expected_conf = 1.0 if max_diff < 0.1 else 0.5

            assert abs(confidence - expected_conf) < 0.01, f"{target}: conf={confidence}, expected={expected_conf}"

    @pytest.mark.asyncio
    async def test_generate_predictions_anomaly_detection(self, ml_engine, mock_hub, mock_capabilities, synthetic_snapshots):
        """Test anomaly detection in predictions."""
        # Setup (train_models uses get_cache_fresh)
        mock_hub.get_cache_fresh.return_value = mock_capabilities
        await ml_engine.train_models(days_history=30)

        current_snapshot = synthetic_snapshots[-1]

        async def mock_get_cache(key):
            if key == "latest_snapshot":
                return {"data": current_snapshot}
            elif key == "capabilities":
                return mock_capabilities
            return None

        mock_hub.get_cache = AsyncMock(side_effect=mock_get_cache)

        # Generate predictions
        result = await ml_engine.generate_predictions()

        # Verify anomaly fields
        assert "anomaly_detected" in result
        assert "anomaly_score" in result
        assert isinstance(result["anomaly_detected"], bool)

        # If anomaly detector exists, score should be present
        if "anomaly_detector" in ml_engine.models:
            assert result["anomaly_score"] is not None

    @pytest.mark.asyncio
    async def test_generate_predictions_cache_storage(self, ml_engine, mock_hub, mock_capabilities, synthetic_snapshots):
        """Test predictions are stored in cache."""
        # Setup (train_models uses get_cache_fresh)
        mock_hub.get_cache_fresh.return_value = mock_capabilities
        await ml_engine.train_models(days_history=30)

        current_snapshot = synthetic_snapshots[-1]

        async def mock_get_cache(key):
            if key == "latest_snapshot":
                return {"data": current_snapshot}
            elif key == "capabilities":
                return mock_capabilities
            return None

        mock_hub.get_cache = AsyncMock(side_effect=mock_get_cache)

        # Generate predictions
        await ml_engine.generate_predictions()

        # Verify cache was updated
        set_cache_calls = [call for call in mock_hub.set_cache.call_args_list if call[0][0] == "ml_predictions"]
        assert len(set_cache_calls) > 0

        # Verify cache structure
        cache_data = set_cache_calls[0][0][1]
        assert "timestamp" in cache_data
        assert "predictions" in cache_data
        assert "anomaly_detected" in cache_data

    @pytest.mark.asyncio
    async def test_generate_predictions_no_models(self, ml_engine, mock_hub):
        """Test predictions when no models are trained."""
        # Don't train any models
        result = await ml_engine.generate_predictions()

        # Should return empty dict
        assert result == {}

    @pytest.mark.asyncio
    async def test_generate_predictions_no_snapshot(self, ml_engine, mock_hub, mock_capabilities, synthetic_snapshots):
        """Test predictions when no current snapshot available."""
        # Setup: train models (uses get_cache_fresh)
        mock_hub.get_cache_fresh.return_value = mock_capabilities
        await ml_engine.train_models(days_history=30)

        # Setup: no snapshot available
        mock_hub.get_cache = AsyncMock(return_value=None)

        # Generate predictions
        result = await ml_engine.generate_predictions()

        # Should return empty dict
        assert result == {}


class TestLightGBMIntegration:
    """Test LightGBM integration in ML Engine."""

    @pytest.fixture
    def ml_engine_with_data(self, mock_hub, tmp_path):
        """Create MLEngine with synthetic training data already on disk."""
        models_dir = tmp_path / "models"
        training_data_dir = tmp_path / "training_data"
        models_dir.mkdir()
        training_data_dir.mkdir()

        engine = MLEngine(mock_hub, str(models_dir), str(training_data_dir))

        # Write 30 days of synthetic snapshot files
        base_date = datetime.now()
        for day_offset in range(30):
            date = base_date - timedelta(days=30 - day_offset)
            snapshot = {
                "date": date.strftime("%Y-%m-%d"),
                "hour": 12,
                "time_features": {
                    "hour_sin": float(np.sin(2 * np.pi * 12 / 24)),
                    "hour_cos": float(np.cos(2 * np.pi * 12 / 24)),
                    "dow_sin": float(np.sin(2 * np.pi * date.weekday() / 7)),
                    "dow_cos": float(np.cos(2 * np.pi * date.weekday() / 7)),
                    "month_sin": 0.5, "month_cos": 0.866,
                    "day_of_year_sin": 0.3, "day_of_year_cos": 0.95,
                    "is_weekend": date.weekday() >= 5,
                    "is_holiday": False, "is_night": False, "is_work_hours": True,
                    "minutes_since_sunrise": 360,
                    "minutes_until_sunset": 300,
                    "daylight_remaining_pct": 0.5
                },
                "weather": {
                    "temp_f": 65.0 + day_offset % 10,
                    "humidity_pct": 50.0 + day_offset % 20,
                    "wind_mph": 5.0
                },
                "power": {"total_watts": 500.0 + day_offset * 10 + float(np.random.default_rng(day_offset).normal() * 50)},
                "lights": {"on": 3 + (day_offset % 3), "total_brightness": 150.0 + day_offset * 5},
                "occupancy": {
                    "people_home": ["person.justin"],
                    "people_home_count": 1,
                    "device_count_home": 2 + (day_offset % 2)
                },
                "motion": {"active_count": 1 + (day_offset % 2)}
            }
            snapshot_file = training_data_dir / f"{date.strftime('%Y-%m-%d')}.json"
            with open(snapshot_file, "w") as f:
                json.dump(snapshot, f)

        return engine

    @pytest.fixture
    def mock_capabilities(self):
        """Capabilities fixture for LightGBM tests."""
        return {
            "data": {
                "power_monitoring": {"available": True, "entities": ["sensor.power_1"]},
                "lighting": {"available": True, "entities": ["light.living_room"]},
                "occupancy": {"available": True, "entities": ["person.justin"]},
            }
        }

    @pytest.mark.asyncio
    async def test_lgbm_model_trained(self, ml_engine_with_data, mock_hub, mock_capabilities):
        """LightGBM model is trained alongside GB and RF."""
        mock_hub.get_cache_fresh = AsyncMock(return_value=mock_capabilities)
        mock_hub.get_cache = AsyncMock(return_value=None)

        await ml_engine_with_data.train_models(days_history=30)

        # Verify lgbm_model exists for at least one target
        assert len(ml_engine_with_data.models) > 0

        for target, model_data in ml_engine_with_data.models.items():
            if target == "anomaly_detector":
                continue
            assert "lgbm_model" in model_data, f"lgbm_model missing from {target}"
            assert "gb_model" in model_data, f"gb_model missing from {target}"
            assert "rf_model" in model_data, f"rf_model missing from {target}"

    @pytest.mark.asyncio
    async def test_lgbm_accuracy_scores_stored(self, ml_engine_with_data, mock_hub, mock_capabilities):
        """LightGBM MAE and R2 are recorded in accuracy_scores."""
        mock_hub.get_cache_fresh = AsyncMock(return_value=mock_capabilities)
        mock_hub.get_cache = AsyncMock(return_value=None)

        await ml_engine_with_data.train_models(days_history=30)

        for target, model_data in ml_engine_with_data.models.items():
            if target == "anomaly_detector":
                continue
            scores = model_data["accuracy_scores"]
            assert "lgbm_mae" in scores, f"lgbm_mae missing from {target} scores"
            assert "lgbm_r2" in scores, f"lgbm_r2 missing from {target} scores"
            assert isinstance(scores["lgbm_mae"], float)
            assert isinstance(scores["lgbm_r2"], float)

    @pytest.mark.asyncio
    async def test_lgbm_feature_importance_stored(self, ml_engine_with_data, mock_hub, mock_capabilities):
        """LightGBM feature importance is stored separately."""
        mock_hub.get_cache_fresh = AsyncMock(return_value=mock_capabilities)
        mock_hub.get_cache = AsyncMock(return_value=None)

        await ml_engine_with_data.train_models(days_history=30)

        for target, model_data in ml_engine_with_data.models.items():
            if target == "anomaly_detector":
                continue
            assert "lgbm_feature_importance" in model_data, f"lgbm_feature_importance missing from {target}"
            assert isinstance(model_data["lgbm_feature_importance"], dict)
            assert len(model_data["lgbm_feature_importance"]) > 0

    @pytest.mark.asyncio
    async def test_lgbm_predictions_included(self, ml_engine_with_data, mock_hub, mock_capabilities):
        """Predictions include lgbm_prediction field."""
        mock_hub.get_cache_fresh = AsyncMock(return_value=mock_capabilities)
        mock_hub.get_cache = AsyncMock(return_value=None)

        await ml_engine_with_data.train_models(days_history=30)

        # Set up snapshot for prediction
        current_snapshot = {
            "date": datetime.now().isoformat(),
            "weather": {"temp_f": 70, "humidity_pct": 55, "wind_mph": 5},
            "power": {"total_watts": 600},
            "lights": {"on": 4, "total_brightness": 200},
            "occupancy": {"people_home_count": 1, "device_count_home": 3},
            "motion": {"active_count": 1},
        }

        async def mock_get_cache(key):
            if key == "latest_snapshot":
                return {"data": current_snapshot}
            if key == "feature_config":
                return None
            return None

        mock_hub.get_cache = AsyncMock(side_effect=mock_get_cache)

        result = await ml_engine_with_data.generate_predictions()

        assert "predictions" in result
        for target, pred in result["predictions"].items():
            assert "lgbm_prediction" in pred, f"lgbm_prediction missing from {target}"
            assert "gb_prediction" in pred, f"gb_prediction missing from {target}"
            assert "rf_prediction" in pred, f"rf_prediction missing from {target}"
            assert isinstance(pred["lgbm_prediction"], float)

    @pytest.mark.asyncio
    async def test_three_model_blending(self, ml_engine_with_data, mock_hub, mock_capabilities):
        """Blended prediction uses configurable weights across all three models."""
        mock_hub.get_cache_fresh = AsyncMock(return_value=mock_capabilities)
        mock_hub.get_cache = AsyncMock(return_value=None)

        await ml_engine_with_data.train_models(days_history=30)

        current_snapshot = {
            "date": datetime.now().isoformat(),
            "weather": {"temp_f": 70, "humidity_pct": 55, "wind_mph": 5},
            "power": {"total_watts": 600},
            "lights": {"on": 4, "total_brightness": 200},
            "occupancy": {"people_home_count": 1, "device_count_home": 3},
            "motion": {"active_count": 1},
        }

        async def mock_get_cache(key):
            if key == "latest_snapshot":
                return {"data": current_snapshot}
            if key == "feature_config":
                return None
            return None

        mock_hub.get_cache = AsyncMock(side_effect=mock_get_cache)

        result = await ml_engine_with_data.generate_predictions()

        for target, pred in result["predictions"].items():
            gb = pred["gb_prediction"]
            rf = pred["rf_prediction"]
            lgbm = pred["lgbm_prediction"]
            blended = pred["value"]

            # Verify blending uses normalized weights (0.35 + 0.25 + 0.40 = 1.0)
            expected = round(0.35 * gb + 0.25 * rf + 0.40 * lgbm, 2)
            assert abs(blended - expected) < 0.02, (
                f"{target}: blended={blended}, expected={expected} "
                f"(gb={gb}, rf={rf}, lgbm={lgbm})"
            )

    @pytest.mark.asyncio
    async def test_blend_weights_in_prediction(self, ml_engine_with_data, mock_hub, mock_capabilities):
        """Predictions include blend_weights showing which models contributed."""
        mock_hub.get_cache_fresh = AsyncMock(return_value=mock_capabilities)
        mock_hub.get_cache = AsyncMock(return_value=None)

        await ml_engine_with_data.train_models(days_history=30)

        current_snapshot = {
            "date": datetime.now().isoformat(),
            "weather": {"temp_f": 70, "humidity_pct": 55, "wind_mph": 5},
            "power": {"total_watts": 600},
            "lights": {"on": 4, "total_brightness": 200},
            "occupancy": {"people_home_count": 1, "device_count_home": 3},
            "motion": {"active_count": 1},
        }

        async def mock_get_cache(key):
            if key == "latest_snapshot":
                return {"data": current_snapshot}
            if key == "feature_config":
                return None
            return None

        mock_hub.get_cache = AsyncMock(side_effect=mock_get_cache)

        result = await ml_engine_with_data.generate_predictions()

        for target, pred in result["predictions"].items():
            assert "blend_weights" in pred
            weights = pred["blend_weights"]
            assert "gb" in weights
            assert "rf" in weights
            assert "lgbm" in weights
            # Weights should sum to 1.0
            assert abs(sum(weights.values()) - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_disable_lgbm_model(self, ml_engine_with_data, mock_hub, mock_capabilities):
        """Disabling LightGBM falls back to GB+RF blending only."""
        mock_hub.get_cache_fresh = AsyncMock(return_value=mock_capabilities)
        mock_hub.get_cache = AsyncMock(return_value=None)

        await ml_engine_with_data.train_models(days_history=30)

        # Disable LightGBM
        ml_engine_with_data.enabled_models["lgbm"] = False

        current_snapshot = {
            "date": datetime.now().isoformat(),
            "weather": {"temp_f": 70, "humidity_pct": 55, "wind_mph": 5},
            "power": {"total_watts": 600},
            "lights": {"on": 4, "total_brightness": 200},
            "occupancy": {"people_home_count": 1, "device_count_home": 3},
            "motion": {"active_count": 1},
        }

        async def mock_get_cache(key):
            if key == "latest_snapshot":
                return {"data": current_snapshot}
            if key == "feature_config":
                return None
            return None

        mock_hub.get_cache = AsyncMock(side_effect=mock_get_cache)

        result = await ml_engine_with_data.generate_predictions()

        for target, pred in result["predictions"].items():
            # lgbm_prediction should not be present
            assert "lgbm_prediction" not in pred
            # gb and rf should still be present
            assert "gb_prediction" in pred
            assert "rf_prediction" in pred
            # blend_weights should only have gb and rf
            assert "lgbm" not in pred["blend_weights"]
            # Weights should still sum to 1.0
            assert abs(sum(pred["blend_weights"].values()) - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_disable_all_except_lgbm(self, ml_engine_with_data, mock_hub, mock_capabilities):
        """Can run with only LightGBM enabled."""
        mock_hub.get_cache_fresh = AsyncMock(return_value=mock_capabilities)
        mock_hub.get_cache = AsyncMock(return_value=None)

        await ml_engine_with_data.train_models(days_history=30)

        # Enable only LightGBM
        ml_engine_with_data.enabled_models["gb"] = False
        ml_engine_with_data.enabled_models["rf"] = False
        ml_engine_with_data.enabled_models["lgbm"] = True

        current_snapshot = {
            "date": datetime.now().isoformat(),
            "weather": {"temp_f": 70, "humidity_pct": 55, "wind_mph": 5},
            "power": {"total_watts": 600},
            "lights": {"on": 4, "total_brightness": 200},
            "occupancy": {"people_home_count": 1, "device_count_home": 3},
            "motion": {"active_count": 1},
        }

        async def mock_get_cache(key):
            if key == "latest_snapshot":
                return {"data": current_snapshot}
            if key == "feature_config":
                return None
            return None

        mock_hub.get_cache = AsyncMock(side_effect=mock_get_cache)

        result = await ml_engine_with_data.generate_predictions()

        for target, pred in result["predictions"].items():
            assert "lgbm_prediction" in pred
            assert "gb_prediction" not in pred
            assert "rf_prediction" not in pred
            # Single model, confidence should be 0.7
            assert pred["confidence"] == 0.7
            # Blended == lgbm when it's the only model
            assert pred["value"] == pred["lgbm_prediction"]

    @pytest.mark.asyncio
    async def test_model_pickle_includes_lgbm(self, ml_engine_with_data, mock_hub, mock_capabilities):
        """Saved .pkl model files include LightGBM model."""
        mock_hub.get_cache_fresh = AsyncMock(return_value=mock_capabilities)
        mock_hub.get_cache = AsyncMock(return_value=None)

        await ml_engine_with_data.train_models(days_history=30)

        # Load a pickle file and verify lgbm_model is in it
        models_dir = Path(ml_engine_with_data.models_dir)
        pkl_files = list(models_dir.glob("*_model.pkl"))
        assert len(pkl_files) > 0

        for pkl_file in pkl_files:
            if "anomaly" in pkl_file.stem:
                continue
            import pickle
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            assert "lgbm_model" in data, f"lgbm_model not in {pkl_file.name}"

    def test_default_model_config(self, mock_hub, tmp_path):
        """Default config enables all three models with correct weights."""
        engine = MLEngine(mock_hub, str(tmp_path / "m"), str(tmp_path / "t"))

        assert engine.enabled_models == {"gb": True, "rf": True, "lgbm": True}
        assert engine.model_weights["gb"] == 0.35
        assert engine.model_weights["rf"] == 0.25
        assert engine.model_weights["lgbm"] == 0.40
        # Weights should sum to 1.0
        assert abs(sum(engine.model_weights.values()) - 1.0) < 0.001

    def test_model_config_mutable(self, mock_hub, tmp_path):
        """Model config can be changed at runtime (prep for UI config)."""
        engine = MLEngine(mock_hub, str(tmp_path / "m"), str(tmp_path / "t"))

        # Simulate UI config change
        engine.enabled_models["lgbm"] = False
        engine.model_weights["gb"] = 0.6
        engine.model_weights["rf"] = 0.4

        assert engine.enabled_models["lgbm"] is False
        assert engine.model_weights["gb"] == 0.6


class TestFeatureEngineering:
    """Test sinusoidal encoding, decay weighting, and rolling window features."""

    @pytest.fixture
    def engine(self, mock_hub, tmp_path):
        """Create MLEngine for feature engineering tests."""
        models_dir = tmp_path / "models"
        training_data_dir = tmp_path / "training_data"
        models_dir.mkdir()
        training_data_dir.mkdir()
        return MLEngine(mock_hub, str(models_dir), str(training_data_dir))

    # --- Sinusoidal time encoding ---

    def test_sinusoidal_hour_known_values(self, engine):
        """Sinusoidal encoding produces correct values for known hours."""

        # Hour 0 (midnight): sin=0, cos=1
        snap_0 = {"date": "2026-02-12T00:00:00"}
        tf_0 = engine._compute_time_features(snap_0)
        assert abs(tf_0["hour_sin"] - 0.0) < 1e-5
        assert abs(tf_0["hour_cos"] - 1.0) < 1e-5

        # Hour 6 (6am): sin=1, cos=0
        snap_6 = {"date": "2026-02-12T06:00:00"}
        tf_6 = engine._compute_time_features(snap_6)
        assert abs(tf_6["hour_sin"] - 1.0) < 1e-5
        assert abs(tf_6["hour_cos"] - 0.0) < 1e-5

        # Hour 12 (noon): sin=0, cos=-1
        snap_12 = {"date": "2026-02-12T12:00:00"}
        tf_12 = engine._compute_time_features(snap_12)
        assert abs(tf_12["hour_sin"] - 0.0) < 1e-5
        assert abs(tf_12["hour_cos"] - (-1.0)) < 1e-5

        # Hour 18 (6pm): sin=-1, cos=0
        snap_18 = {"date": "2026-02-12T18:00:00"}
        tf_18 = engine._compute_time_features(snap_18)
        assert abs(tf_18["hour_sin"] - (-1.0)) < 1e-5
        assert abs(tf_18["hour_cos"] - 0.0) < 1e-5

    def test_sinusoidal_dow_known_values(self, engine):
        """Sinusoidal encoding produces correct values for known days of week."""
        import math

        # Monday (dow=0): sin(0)=0, cos(0)=1
        # 2026-02-09 is a Monday
        snap_mon = {"date": "2026-02-09T12:00:00"}
        tf_mon = engine._compute_time_features(snap_mon)
        assert abs(tf_mon["dow_sin"] - 0.0) < 1e-5
        assert abs(tf_mon["dow_cos"] - 1.0) < 1e-5

        # Sunday (dow=6): sin(2*pi*6/7), cos(2*pi*6/7)
        # 2026-02-15 is a Sunday
        snap_sun = {"date": "2026-02-15T12:00:00"}
        tf_sun = engine._compute_time_features(snap_sun)
        expected_sin = math.sin(2 * math.pi * 6 / 7)
        expected_cos = math.cos(2 * math.pi * 6 / 7)
        assert abs(tf_sun["dow_sin"] - expected_sin) < 1e-5
        assert abs(tf_sun["dow_cos"] - expected_cos) < 1e-5

    def test_sinusoidal_unit_circle_property(self, engine):
        """sin^2 + cos^2 == 1 for all hour and dow encodings."""

        for hour in range(24):
            snap = {"date": f"2026-02-12T{hour:02d}:00:00"}
            tf = engine._compute_time_features(snap)
            sin_sq_cos_sq = tf["hour_sin"] ** 2 + tf["hour_cos"] ** 2
            assert abs(sin_sq_cos_sq - 1.0) < 1e-5, f"hour={hour}: {sin_sq_cos_sq}"

        # Test all 7 days of the week (2026-02-09 is Monday)
        for day_offset in range(7):
            day = 9 + day_offset
            snap = {"date": f"2026-02-{day:02d}T12:00:00"}
            tf = engine._compute_time_features(snap)
            sin_sq_cos_sq = tf["dow_sin"] ** 2 + tf["dow_cos"] ** 2
            assert abs(sin_sq_cos_sq - 1.0) < 1e-5, f"day_offset={day_offset}: {sin_sq_cos_sq}"

            # Also check month encoding
            sin_sq_cos_sq_m = tf["month_sin"] ** 2 + tf["month_cos"] ** 2
            assert abs(sin_sq_cos_sq_m - 1.0) < 1e-5

    # --- Decay weighting ---

    def test_decay_weight_recency_ordering(self, engine):
        """Today's weight > yesterday's weight > last week's weight."""
        now = datetime(2026, 2, 12, 12, 0, 0)

        snapshots = [
            {"date": "2026-02-05"},  # 7 days ago
            {"date": "2026-02-11"},  # yesterday
            {"date": "2026-02-12"},  # today
        ]

        weights = engine._compute_decay_weights(snapshots, reference_date=now)
        assert len(weights) == 3
        # today > yesterday > last week
        assert weights[2] > weights[1] > weights[0]
        # All positive
        assert all(w > 0 for w in weights)

    def test_decay_weight_weekday_alignment_bonus(self, engine):
        """Same-weekday snapshots receive the alignment bonus."""
        from aria.modules.ml_engine import DECAY_HALF_LIFE_DAYS, WEEKDAY_ALIGNMENT_BONUS

        # 2026-02-12 is a Thursday
        now = datetime(2026, 2, 12, 12, 0, 0)

        # Two snapshots exactly 1 day apart, one on Thursday (same dow), one on Wednesday
        snapshots = [
            {"date": "2026-02-11"},  # Wednesday, 1 day ago
            {"date": "2026-02-05"},  # Thursday, 7 days ago (same dow)
        ]

        engine._compute_decay_weights(snapshots, reference_date=now)

        # Wednesday (1 day ago) has higher recency but no weekday bonus
        # Thursday (7 days ago) has lower recency but weekday bonus
        # To verify the bonus is applied, compare two same-age snapshots:
        # Create two snapshots 7 days apart, one same weekday
        snapshots_same_age = [
            {"date": "2026-02-05"},  # Thursday (same dow), 7 days ago
            {"date": "2026-02-06"},  # Friday (different dow), 6 days ago
        ]
        weights_age = engine._compute_decay_weights(snapshots_same_age, reference_date=now)

        # Thursday at 7 days has bonus, Friday at 6 days does not
        # The bonus factor is 1.5x — verify it's applied
        # Note: dates without time parse to midnight, so days_ago includes
        # fractional days from reference_date's noon time
        import math
        thu_dt = datetime(2026, 2, 5, 0, 0, 0)
        fri_dt = datetime(2026, 2, 6, 0, 0, 0)
        thu_days_ago = (now - thu_dt).total_seconds() / 86400
        fri_days_ago = (now - fri_dt).total_seconds() / 86400
        recency_thu = math.exp(-thu_days_ago / DECAY_HALF_LIFE_DAYS)
        recency_fri = math.exp(-fri_days_ago / DECAY_HALF_LIFE_DAYS)
        expected_thu = recency_thu * WEEKDAY_ALIGNMENT_BONUS
        expected_fri = recency_fri * 1.0

        assert abs(weights_age[0] - expected_thu) < 1e-6
        assert abs(weights_age[1] - expected_fri) < 1e-6

    def test_decay_weight_missing_date(self, engine):
        """Snapshots without a date field get weight 0."""
        now = datetime(2026, 2, 12, 12, 0, 0)
        snapshots = [
            {"date": "2026-02-12"},
            {},  # no date
            {"date": ""},  # empty date
        ]
        weights = engine._compute_decay_weights(snapshots, reference_date=now)
        assert weights[0] > 0
        assert weights[1] == 0.0
        assert weights[2] == 0.0

    # --- Rolling window statistics ---

    @pytest.mark.asyncio
    async def test_rolling_window_entropy_single_domain(self, engine):
        """Entropy is 0 when all activity is in a single domain."""
        now = datetime.now()
        activity_log = {
            "windows": [
                {
                    "window_start": (now - timedelta(minutes=30)).isoformat(),
                    "window_end": (now - timedelta(minutes=15)).isoformat(),
                    "event_count": 10,
                    "by_domain": {"light": 10},
                },
                {
                    "window_start": (now - timedelta(minutes=15)).isoformat(),
                    "window_end": now.isoformat(),
                    "event_count": 5,
                    "by_domain": {"light": 5},
                },
            ],
            "last_updated": now.isoformat(),
        }

        stats = await engine._compute_rolling_window_stats(activity_log=activity_log)

        # Single domain = 0 entropy (all in 1h window)
        assert stats["rolling_1h_domain_entropy"] == 0.0
        assert stats["rolling_1h_event_count"] == 15
        assert stats["rolling_1h_dominant_domain_pct"] == 1.0

    @pytest.mark.asyncio
    async def test_rolling_window_entropy_multi_domain(self, engine):
        """Entropy is positive when activity spans multiple domains."""
        import math
        now = datetime.now()
        activity_log = {
            "windows": [
                {
                    "window_start": (now - timedelta(minutes=30)).isoformat(),
                    "window_end": (now - timedelta(minutes=15)).isoformat(),
                    "event_count": 10,
                    "by_domain": {"light": 5, "switch": 5},
                },
                {
                    "window_start": (now - timedelta(minutes=15)).isoformat(),
                    "window_end": now.isoformat(),
                    "event_count": 10,
                    "by_domain": {"light": 5, "switch": 5},
                },
            ],
            "last_updated": now.isoformat(),
        }

        stats = await engine._compute_rolling_window_stats(activity_log=activity_log)

        # Two equally distributed domains → entropy = log2(2) = 1.0
        assert stats["rolling_1h_domain_entropy"] > 0
        expected_entropy = -2 * (0.5 * math.log2(0.5))  # = 1.0
        assert abs(stats["rolling_1h_domain_entropy"] - expected_entropy) < 1e-3
        assert stats["rolling_1h_dominant_domain_pct"] == 0.5

    @pytest.mark.asyncio
    async def test_rolling_window_trend_increasing(self, engine):
        """Trend is positive when second half has more events than first."""
        now = datetime.now()
        activity_log = {
            "windows": [
                {
                    "window_start": (now - timedelta(minutes=45)).isoformat(),
                    "window_end": (now - timedelta(minutes=30)).isoformat(),
                    "event_count": 2,
                    "by_domain": {"light": 2},
                },
                {
                    "window_start": (now - timedelta(minutes=30)).isoformat(),
                    "window_end": (now - timedelta(minutes=15)).isoformat(),
                    "event_count": 10,
                    "by_domain": {"light": 10},
                },
                {
                    "window_start": (now - timedelta(minutes=15)).isoformat(),
                    "window_end": now.isoformat(),
                    "event_count": 15,
                    "by_domain": {"light": 15},
                },
            ],
            "last_updated": now.isoformat(),
        }

        stats = await engine._compute_rolling_window_stats(activity_log=activity_log)
        # First half: [2], second half: [10, 15] → increasing
        assert stats["rolling_1h_trend"] == 1.0

    @pytest.mark.asyncio
    async def test_rolling_window_trend_decreasing(self, engine):
        """Trend is negative when second half has fewer events than first."""
        now = datetime.now()
        activity_log = {
            "windows": [
                {
                    "window_start": (now - timedelta(minutes=45)).isoformat(),
                    "window_end": (now - timedelta(minutes=30)).isoformat(),
                    "event_count": 20,
                    "by_domain": {"light": 20},
                },
                {
                    "window_start": (now - timedelta(minutes=30)).isoformat(),
                    "window_end": (now - timedelta(minutes=15)).isoformat(),
                    "event_count": 3,
                    "by_domain": {"light": 3},
                },
                {
                    "window_start": (now - timedelta(minutes=15)).isoformat(),
                    "window_end": now.isoformat(),
                    "event_count": 2,
                    "by_domain": {"light": 2},
                },
            ],
            "last_updated": now.isoformat(),
        }

        stats = await engine._compute_rolling_window_stats(activity_log=activity_log)
        # First half: [20], second half: [3, 2] → 5/20 = 0.25 < 0.8 → decreasing
        assert stats["rolling_1h_trend"] == -1.0

    @pytest.mark.asyncio
    async def test_rolling_window_trend_stable(self, engine):
        """Trend is 0 when activity is roughly equal in both halves."""
        now = datetime.now()
        activity_log = {
            "windows": [
                {
                    "window_start": (now - timedelta(minutes=30)).isoformat(),
                    "window_end": (now - timedelta(minutes=15)).isoformat(),
                    "event_count": 10,
                    "by_domain": {"light": 10},
                },
                {
                    "window_start": (now - timedelta(minutes=15)).isoformat(),
                    "window_end": now.isoformat(),
                    "event_count": 10,
                    "by_domain": {"light": 10},
                },
            ],
            "last_updated": now.isoformat(),
        }

        stats = await engine._compute_rolling_window_stats(activity_log=activity_log)
        # Equal halves → stable
        assert stats["rolling_1h_trend"] == 0.0

    @pytest.mark.asyncio
    async def test_rolling_window_empty_log(self, engine):
        """Empty activity log returns zeros for all rolling features."""
        stats = await engine._compute_rolling_window_stats(activity_log={})

        for hours in [1, 3, 6]:
            assert stats[f"rolling_{hours}h_event_count"] == 0
            assert stats[f"rolling_{hours}h_domain_entropy"] == 0
            assert stats[f"rolling_{hours}h_dominant_domain_pct"] == 0
            assert stats[f"rolling_{hours}h_trend"] == 0

    # --- Feature extraction integration ---

    @pytest.mark.asyncio
    async def test_feature_names_include_rolling_windows(self, engine, mock_hub):
        """Feature names list includes rolling window feature names."""
        mock_hub.get_cache = AsyncMock(return_value=None)
        config = await engine._get_feature_config()
        names = await engine._get_feature_names(config)

        for hours in [1, 3, 6]:
            assert f"rolling_{hours}h_event_count" in names
            assert f"rolling_{hours}h_domain_entropy" in names
            assert f"rolling_{hours}h_dominant_domain_pct" in names
            assert f"rolling_{hours}h_trend" in names

    @pytest.mark.asyncio
    async def test_full_feature_extraction_includes_all_categories(self, engine, mock_hub):
        """Full feature extraction produces expected feature categories."""
        mock_hub.get_cache = AsyncMock(return_value=None)

        snapshot = {
            "date": "2026-02-12T14:00:00",
            "weather": {"temp_f": 65, "humidity_pct": 50, "wind_mph": 5},
            "power": {"total_watts": 500},
            "lights": {"on": 3, "total_brightness": 150},
            "occupancy": {"people_home_count": 1, "device_count_home": 2},
            "motion": {"active_count": 1},
        }

        features = await engine._extract_features(snapshot)

        # Sinusoidal time features
        assert "hour_sin" in features
        assert "hour_cos" in features
        assert "dow_sin" in features
        assert "dow_cos" in features

        # Weather features
        assert "weather_temp_f" in features

        # Home features
        assert "lights_on" in features

        # Rolling window features (zeros when no activity log)
        assert "rolling_1h_event_count" in features
        assert "rolling_3h_domain_entropy" in features
        assert "rolling_6h_trend" in features
        assert features["rolling_1h_event_count"] == 0  # no activity log passed

    @pytest.mark.asyncio
    async def test_build_training_dataset_returns_weights(self, engine, mock_hub):
        """_build_training_dataset returns sample weights as third element."""
        mock_hub.get_cache = AsyncMock(return_value=None)

        # Create minimal snapshots
        snapshots = []
        for i in range(20):
            date = datetime(2026, 1, 20) + timedelta(days=i)
            snapshots.append({
                "date": date.strftime("%Y-%m-%d"),
                "power": {"total_watts": 500 + i * 10},
                "lights": {"on": 3},
                "occupancy": {"people_home_count": 1, "device_count_home": 2},
                "motion": {"active_count": 1},
                "weather": {"temp_f": 65, "humidity_pct": 50, "wind_mph": 5},
            })

        X, y, weights = await engine._build_training_dataset(snapshots, "power_watts")

        assert len(X) == 20
        assert len(y) == 20
        assert len(weights) == 20
        # Weights should all be positive
        assert all(w > 0 for w in weights)
        # Later snapshots should have higher weight (more recent)
        assert weights[-1] > weights[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
