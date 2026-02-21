"""Tests for config defaults registry and seed logic."""

import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.hub.cache import CacheManager
from aria.hub.config_defaults import CONFIG_DEFAULTS, seed_config_defaults

# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def cache(tmp_path):
    """Create and initialize a CacheManager with a temp DB."""
    db_path = str(tmp_path / "test_hub.db")
    cm = CacheManager(db_path)
    await cm.initialize()
    yield cm
    await cm.close()


# ============================================================================
# CONFIG_DEFAULTS structure
# ============================================================================


class TestConfigDefaultsStructure:
    def test_all_entries_have_required_keys(self):
        required = {"key", "default_value", "value_type"}
        for param in CONFIG_DEFAULTS:
            missing = required - param.keys()
            assert not missing, f"Param '{param.get('key', '?')}' missing: {missing}"

    def test_all_keys_are_unique(self):
        keys = [p["key"] for p in CONFIG_DEFAULTS]
        assert len(keys) == len(set(keys)), f"Duplicate keys: {[k for k in keys if keys.count(k) > 1]}"

    def test_expected_parameter_count(self):
        assert len(CONFIG_DEFAULTS) == 148  # 113 Phase 2 + 33 Phase 3 entries

    def test_all_categories_are_set(self):
        for param in CONFIG_DEFAULTS:
            assert param.get("category"), f"Param '{param['key']}' has no category"

    def test_number_types_have_min_max(self):
        for param in CONFIG_DEFAULTS:
            if param["value_type"] == "number":
                assert "min_value" in param, f"Number param '{param['key']}' missing min_value"
                assert "max_value" in param, f"Number param '{param['key']}' missing max_value"

    def test_default_within_range(self):
        for param in CONFIG_DEFAULTS:
            if param["value_type"] == "number":
                val = float(param["default_value"])
                assert val >= param["min_value"], f"Param '{param['key']}': default {val} < min {param['min_value']}"
                assert val <= param["max_value"], f"Param '{param['key']}': default {val} > max {param['max_value']}"

    def test_all_entries_have_layman_and_technical_descriptions(self):
        for param in CONFIG_DEFAULTS:
            assert "description_layman" in param, f"{param['key']} missing description_layman"
            assert "description_technical" in param, f"{param['key']} missing description_technical"
            assert len(param["description_layman"]) > 10, f"{param['key']} layman too short"
            assert len(param["description_technical"]) > 10, f"{param['key']} technical too short"


# ============================================================================
# Presence weight/decay config entries
# ============================================================================


class TestPresenceWeightDecayEntries:
    SIGNAL_TYPES = [
        "motion",
        "door",
        "media",
        "power",
        "device_tracker",
        "camera_person",
        "camera_face",
        "light_interaction",
        "dimmer_press",
    ]

    def test_presence_weight_entries_exist(self):
        keys = {p["key"] for p in CONFIG_DEFAULTS}
        for signal in self.SIGNAL_TYPES:
            assert f"presence.weight.{signal}" in keys, f"Missing weight for {signal}"
            assert f"presence.decay.{signal}" in keys, f"Missing decay for {signal}"

    def test_presence_weight_defaults_match_sensor_config(self):
        from aria.engine.analysis.occupancy import SENSOR_CONFIG

        by_key = {p["key"]: p for p in CONFIG_DEFAULTS}
        for signal, cfg in SENSOR_CONFIG.items():
            weight_key = f"presence.weight.{signal}"
            decay_key = f"presence.decay.{signal}"
            assert float(by_key[weight_key]["default_value"]) == cfg["weight"], (
                f"{weight_key}: {by_key[weight_key]['default_value']} != {cfg['weight']}"
            )
            assert float(by_key[decay_key]["default_value"]) == cfg["decay_seconds"], (
                f"{decay_key}: {by_key[decay_key]['default_value']} != {cfg['decay_seconds']}"
            )

    def test_presence_weights_have_layman_and_technical(self):
        by_key = {p["key"]: p for p in CONFIG_DEFAULTS}
        for signal in self.SIGNAL_TYPES:
            for prefix in ("presence.weight.", "presence.decay."):
                key = f"{prefix}{signal}"
                param = by_key[key]
                assert param.get("description_layman"), f"{key} missing description_layman"
                assert param.get("description_technical"), f"{key} missing description_technical"


# ============================================================================
# seed_config_defaults
# ============================================================================


class TestSeedConfigDefaults:
    @pytest.mark.asyncio
    async def test_seed_populates_all_params(self, cache):
        seeded = await seed_config_defaults(cache)
        assert seeded == 148

        configs = await cache.get_all_config()
        assert len(configs) == 148

    @pytest.mark.asyncio
    async def test_seed_is_idempotent(self, cache):
        first = await seed_config_defaults(cache)
        assert first == 148

        second = await seed_config_defaults(cache)
        assert second == 0  # nothing new inserted

    @pytest.mark.asyncio
    async def test_seed_preserves_user_overrides(self, cache):
        await seed_config_defaults(cache)

        # User changes a value
        await cache.set_config("shadow.min_confidence", "0.5")

        # Re-seed
        await seed_config_defaults(cache)

        config = await cache.get_config("shadow.min_confidence")
        assert config["value"] == "0.5"  # preserved
        assert config["default_value"] == "0.3"  # original default

    @pytest.mark.asyncio
    async def test_get_config_value_returns_decoded_type(self, cache):
        await seed_config_defaults(cache)

        # Number
        val = await cache.get_config_value("shadow.min_confidence")
        assert isinstance(val, float)
        assert val == 0.3

        # Integer
        val = await cache.get_config_value("activity.daily_snapshot_cap")
        assert isinstance(val, int)
        assert val == 20

        # String
        val = await cache.get_config_value("curation.vehicle_patterns")
        assert isinstance(val, str)
        assert "tesla" in val
