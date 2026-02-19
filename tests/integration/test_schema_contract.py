"""Contract test: engine snapshot → intelligence module schema agreement.

Verifies that IntelligenceModule._read_intelligence_data() produces all
required keys defined in aria.schemas.REQUIRED_INTELLIGENCE_KEYS.

This catches drift between engine output changes and hub expectations
before it becomes a silent data gap in production.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.modules.intelligence import IntelligenceModule
from aria.schemas import REQUIRED_INTELLIGENCE_KEYS, validate_intelligence_payload


@pytest.fixture
def intel_dir(tmp_path):
    """Create a minimal intelligence directory with enough structure for assembly."""
    intel = tmp_path / "intelligence"
    intel.mkdir()

    # Create subdirectories
    (intel / "daily").mkdir()
    (intel / "intraday").mkdir()
    (intel / "insights").mkdir()
    (intel / "models").mkdir()
    (intel / "meta-learning").mkdir()
    (intel / "insights" / "automation-suggestions").mkdir()

    # Create a minimal daily snapshot so trend extraction has something to read
    snapshot = {
        "power": {"total_watts": 500.0},
        "lights": {"on": 3},
        "occupancy": {"device_count_home": 2},
        "entities": {"unavailable": 1},
        "logbook_summary": {"useful_events": 42},
    }
    (intel / "daily" / "2025-01-01.json").write_text(json.dumps(snapshot))

    return intel


@pytest.fixture
def mock_hub():
    """Create a minimal mock hub for IntelligenceModule."""
    hub = MagicMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.is_running = MagicMock(return_value=True)
    hub.cache = MagicMock()
    hub.cache.get_config_value = AsyncMock(return_value="")
    return hub


class TestSchemaContract:
    """Verify engine→hub JSON schema contract."""

    def test_required_keys_present_in_payload(self, intel_dir, mock_hub):
        """_read_intelligence_data() must produce all REQUIRED_INTELLIGENCE_KEYS."""
        module = IntelligenceModule(mock_hub, str(intel_dir))
        data = module._read_intelligence_data()

        missing = validate_intelligence_payload(data)
        assert missing == [], f"Intelligence payload missing required keys: {missing}"

    def test_all_required_keys_are_set(self, intel_dir, mock_hub):
        """Every key in REQUIRED_INTELLIGENCE_KEYS appears in the output dict."""
        module = IntelligenceModule(mock_hub, str(intel_dir))
        data = module._read_intelligence_data()

        for key in REQUIRED_INTELLIGENCE_KEYS:
            assert key in data, f"Required key '{key}' not in intelligence payload"

    def test_data_maturity_has_structure(self, intel_dir, mock_hub):
        """data_maturity must have phase and days_of_data."""
        module = IntelligenceModule(mock_hub, str(intel_dir))
        data = module._read_intelligence_data()

        maturity = data["data_maturity"]
        assert "phase" in maturity
        assert "days_of_data" in maturity
        assert isinstance(maturity["days_of_data"], int)

    def test_ml_models_has_structure(self, intel_dir, mock_hub):
        """ml_models must have count, last_trained, scores."""
        module = IntelligenceModule(mock_hub, str(intel_dir))
        data = module._read_intelligence_data()

        ml = data["ml_models"]
        assert "count" in ml
        assert "last_trained" in ml
        assert "scores" in ml

    def test_meta_learning_has_structure(self, intel_dir, mock_hub):
        """meta_learning must have applied_count, last_applied, suggestions."""
        module = IntelligenceModule(mock_hub, str(intel_dir))
        data = module._read_intelligence_data()

        meta = data["meta_learning"]
        assert "applied_count" in meta
        assert "last_applied" in meta
        assert "suggestions" in meta

    def test_config_has_structure(self, intel_dir, mock_hub):
        """config must have anomaly_threshold and ml_weight_schedule."""
        module = IntelligenceModule(mock_hub, str(intel_dir))
        data = module._read_intelligence_data()

        config = data["config"]
        assert "anomaly_threshold" in config
        assert "ml_weight_schedule" in config

    def test_validate_intelligence_payload_catches_missing(self):
        """validate_intelligence_payload correctly identifies missing keys."""
        incomplete = {"data_maturity": {}, "predictions": None}
        missing = validate_intelligence_payload(incomplete)
        assert len(missing) > 0
        assert "baselines" in missing
        assert "ml_models" in missing

    def test_validate_intelligence_payload_passes_complete(self, intel_dir, mock_hub):
        """validate_intelligence_payload returns empty list for complete payload."""
        module = IntelligenceModule(mock_hub, str(intel_dir))
        data = module._read_intelligence_data()
        missing = validate_intelligence_payload(data)
        assert missing == []

    def test_trend_data_is_list(self, intel_dir, mock_hub):
        """trend_data must always be a list (empty OK, never None)."""
        module = IntelligenceModule(mock_hub, str(intel_dir))
        data = module._read_intelligence_data()
        assert isinstance(data["trend_data"], list)

    def test_correlations_is_list(self, intel_dir, mock_hub):
        """correlations must always be a list (empty OK, never None)."""
        module = IntelligenceModule(mock_hub, str(intel_dir))
        data = module._read_intelligence_data()
        assert isinstance(data["correlations"], list)

    def test_run_log_is_list(self, intel_dir, mock_hub):
        """run_log must always be a list."""
        module = IntelligenceModule(mock_hub, str(intel_dir))
        data = module._read_intelligence_data()
        assert isinstance(data["run_log"], list)
