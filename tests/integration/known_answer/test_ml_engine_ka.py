"""Known-answer tests for MLEngine module.

Validates that MLEngine can be constructed with valid directories,
initializes correctly with pre-populated capabilities cache, and
produces a stable golden snapshot of post-initialization state.
"""

from unittest.mock import patch

import pytest

from aria.modules.ml_engine import MLEngine
from tests.integration.known_answer.conftest import golden_compare


# Patch hardware detection so tests are deterministic across machines.
def _fake_scan_hardware():
    from aria.engine.hardware import HardwareProfile

    return HardwareProfile(ram_gb=32.0, cpu_cores=8, gpu_available=False)


def _fake_recommend_tier(_profile):
    return 3


# Minimal capabilities data matching what discovery would produce.
# MLEngine.initialize() reads this from cache via get_cache_fresh("capabilities").
MOCK_CAPABILITIES = {
    "power_monitoring": {
        "available": True,
        "entity_count": 5,
        "entities": ["sensor.power_1", "sensor.power_2", "sensor.power_3", "sensor.power_4", "sensor.power_5"],
    },
    "lighting": {
        "available": True,
        "entity_count": 4,
        "entities": ["light.living_room", "light.bedroom", "light.kitchen", "switch.patio"],
    },
    "occupancy": {
        "available": True,
        "entity_count": 3,
        "entities": ["binary_sensor.motion_1", "binary_sensor.motion_2", "binary_sensor.occupancy_1"],
    },
}


@pytest.fixture
async def ml_engine(hub, tmp_path):
    """Create an MLEngine with pre-populated capabilities cache."""
    models_dir = tmp_path / "models"
    training_data_dir = tmp_path / "training_data"
    models_dir.mkdir()
    training_data_dir.mkdir()

    # Pre-populate capabilities cache so initialize() finds data
    await hub.set_cache("capabilities", MOCK_CAPABILITIES)

    with (
        patch("aria.modules.ml_engine.scan_hardware", _fake_scan_hardware),
        patch("aria.modules.ml_engine.recommend_tier", _fake_recommend_tier),
    ):
        engine = MLEngine(hub=hub, models_dir=str(models_dir), training_data_dir=str(training_data_dir))
    return engine


@pytest.mark.asyncio
async def test_initializes_with_data(ml_engine, hub):
    """MLEngine can be constructed and initialized with pre-populated capabilities."""
    engine = ml_engine

    # Verify construction set up expected attributes
    assert engine.module_id == "ml_engine"
    assert engine.hub is hub
    assert engine.models_dir.exists()
    assert engine.models == {}  # No models loaded yet (empty dir)

    # Verify capability_predictions mapping is populated
    assert "power_monitoring" in engine.capability_predictions
    assert "lighting" in engine.capability_predictions
    assert "occupancy" in engine.capability_predictions

    # Run initialize — should read capabilities from cache
    await engine.initialize()

    # Capabilities were found (no error logged, no early return)
    # Verify no models loaded (empty models_dir)
    assert engine.models == {}

    # Verify model configuration is set
    assert isinstance(engine.enabled_models, dict)
    assert isinstance(engine.model_weights, dict)
    assert len(engine.enabled_models) > 0
    assert len(engine.model_weights) > 0


@pytest.mark.asyncio
async def test_initialize_without_capabilities(hub, tmp_path):
    """MLEngine.initialize() handles missing capabilities gracefully."""
    models_dir = tmp_path / "models_empty"
    training_data_dir = tmp_path / "training_empty"
    models_dir.mkdir()
    training_data_dir.mkdir()

    # Do NOT pre-populate capabilities cache
    with (
        patch("aria.modules.ml_engine.scan_hardware", _fake_scan_hardware),
        patch("aria.modules.ml_engine.recommend_tier", _fake_recommend_tier),
    ):
        engine = MLEngine(hub=hub, models_dir=str(models_dir), training_data_dir=str(training_data_dir))
    await engine.initialize()

    # Should return early without error — no models loaded
    assert engine.models == {}


@pytest.mark.asyncio
async def test_golden_snapshot(ml_engine, hub, update_golden):
    """Golden snapshot of MLEngine state after initialization."""
    engine = ml_engine
    await engine.initialize()

    # Build deterministic snapshot of post-init state
    snapshot = {
        "module_name": engine.module_id,
        "models_loaded": len(engine.models),
        "capability_predictions": engine.capability_predictions,
        "enabled_models": engine.enabled_models,
        "model_weights": {k: round(v, 4) for k, v in engine.model_weights.items()},
        "current_tier": engine.current_tier,
        "online_blend_weight": engine.online_blend_weight,
        "capabilities_in_cache": list(sorted(MOCK_CAPABILITIES.keys())),
    }

    golden_compare(snapshot, "ml_engine_init", update=update_golden)

    # Structural assertions that must always hold
    assert snapshot["module_name"] == "ml_engine"
    assert snapshot["models_loaded"] == 0
    assert len(snapshot["capability_predictions"]) >= 3
    assert len(snapshot["enabled_models"]) > 0
