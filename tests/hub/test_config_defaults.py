"""Tests for aria.hub.config_defaults — configuration parameter validation."""
import pytest

from aria.hub.config_defaults import CONFIG_DEFAULTS, seed_config_defaults


class TestConfigDefaults:
    """Test CONFIG_DEFAULTS structure and constraints."""

    def test_all_have_required_fields(self):
        required = {"key", "default_value", "value_type", "label", "description", "category"}
        for param in CONFIG_DEFAULTS:
            missing = required - set(param.keys())
            assert not missing, f"Parameter {param.get('key', '?')} missing: {missing}"

    def test_all_keys_are_unique(self):
        keys = [p["key"] for p in CONFIG_DEFAULTS]
        assert len(keys) == len(set(keys)), f"Duplicate keys: {[k for k in keys if keys.count(k) > 1]}"

    def test_expected_parameter_count(self):
        assert len(CONFIG_DEFAULTS) == 172  # 113 Phase 2 + 33 Phase 3 + 6 audit + 20 Phase 4 I&W entries

    def test_value_types_valid(self):
        valid_types = {"float", "int", "bool", "str", "list"}
        for param in CONFIG_DEFAULTS:
            assert param["value_type"] in valid_types, (
                f"Parameter {param['key']} has invalid type: {param['value_type']}"
            )

    def test_numeric_have_bounds(self):
        for param in CONFIG_DEFAULTS:
            if param["value_type"] in ("float", "int"):
                assert "min_value" in param, f"Numeric param {param['key']} missing min_value"
                assert "max_value" in param, f"Numeric param {param['key']} missing max_value"
                assert param["min_value"] <= param["default_value"] <= param["max_value"], (
                    f"Parameter {param['key']}: default {param['default_value']} "
                    f"not in [{param['min_value']}, {param['max_value']}]"
                )

    def test_categories_present(self):
        for param in CONFIG_DEFAULTS:
            assert param["category"], f"Parameter {param['key']} has empty category"

    def test_descriptions_present(self):
        for param in CONFIG_DEFAULTS:
            assert param["description"], f"Parameter {param['key']} has empty description"

    def test_labels_present(self):
        for param in CONFIG_DEFAULTS:
            assert param["label"], f"Parameter {param['key']} has empty label"

    def test_iw_config_entries(self):
        """Phase 4 I&W Framework config entries."""
        iw_entries = [p for p in CONFIG_DEFAULTS if p["key"].startswith("iw.")]
        assert len(iw_entries) >= 20, f"Expected >= 20 iw.* entries, got {len(iw_entries)}"

        expected_keys = {
            "iw.discovery_interval_hours", "iw.min_discovery_confidence",
            "iw.min_match_ratio", "iw.min_observations_seed",
            "iw.min_observations_emerging", "iw.min_consistency_emerging",
            "iw.min_observations_confirmed", "iw.min_consistency_confirmed",
            "iw.min_observations_mature", "iw.min_consistency_mature",
            "iw.min_density_emerging", "iw.min_density_confirmed",
            "iw.dormant_days", "iw.retired_days", "iw.max_composites",
            "iw.backtest_days", "iw.backtest_holdout_ratio",
            "iw.backtest_min_f1", "iw.detector_window_seconds",
            "iw.cold_start_replay_minutes",
        }
        actual_keys = {p["key"] for p in iw_entries}
        missing = expected_keys - actual_keys
        assert not missing, f"Missing I&W config keys: {missing}"


@pytest.fixture
async def cache():
    """Provide a mock cache for seed tests."""
    from unittest.mock import AsyncMock, MagicMock

    mock_cache = MagicMock()
    stored = {}

    async def mock_set_config(key, value, value_type="str", label="", description="",
                               category="", min_value=None, max_value=None,
                               description_layman="", description_technical="",
                               example_min="", example_max=""):
        if key not in stored:
            stored[key] = value
            return True
        return False

    async def mock_get_all_config():
        return stored

    mock_cache.set_config = AsyncMock(side_effect=mock_set_config)
    mock_cache.get_all_config = AsyncMock(side_effect=mock_get_all_config)
    return mock_cache


class TestSeedConfigDefaults:
    @pytest.mark.asyncio
    async def test_seed_populates_all_params(self, cache):
        seeded = await seed_config_defaults(cache)
        assert seeded == 172

        configs = await cache.get_all_config()
        assert len(configs) == 172

    @pytest.mark.asyncio
    async def test_seed_is_idempotent(self, cache):
        first = await seed_config_defaults(cache)
        assert first == 172

        second = await seed_config_defaults(cache)
        assert second == 0  # nothing new inserted

    @pytest.mark.asyncio
    async def test_seed_preserves_existing_values(self, cache):
        # Pre-set a config value
        await cache.set_config("ml.prediction_interval_minutes", 999)
        seeded = await seed_config_defaults(cache)
        # Should have seeded all except the pre-set one
        assert seeded == 171
