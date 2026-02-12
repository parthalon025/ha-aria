"""Unit tests for IntelligenceModule.

Tests data assembly from disk files, phase detection, activity data merge,
trend extraction, digest formatting, and Telegram sending.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hub.constants import CACHE_ACTIVITY_LOG, CACHE_ACTIVITY_SUMMARY, CACHE_INTELLIGENCE
from modules.intelligence import IntelligenceModule, METRIC_PATHS


# ============================================================================
# Mock Hub
# ============================================================================


class MockHub:
    """Lightweight hub mock that provides set_cache/get_cache without SQLite."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._running = True
        self._scheduled_tasks: List[Dict[str, Any]] = []

    async def set_cache(self, category: str, data: Any, metadata: Optional[Dict] = None):
        self._cache[category] = {"data": data, "metadata": metadata}

    async def get_cache(self, category: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(category)

    def is_running(self) -> bool:
        return self._running

    async def schedule_task(self, **kwargs):
        self._scheduled_tasks.append(kwargs)

    def register_module(self, mod):
        pass

    async def publish(self, event_type: str, data: Dict[str, Any]):
        pass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hub():
    return MockHub()


@pytest.fixture
def intel_dir(tmp_path):
    """Create a structured intelligence directory with sample data."""
    d = tmp_path / "intelligence"
    (d / "daily").mkdir(parents=True)
    (d / "intraday").mkdir()
    (d / "insights").mkdir()
    (d / "models").mkdir()
    (d / "meta-learning").mkdir()
    return d


@pytest.fixture
def module(hub, intel_dir):
    """Create an IntelligenceModule pointing at the temp directory."""
    mod = IntelligenceModule(hub, str(intel_dir))
    # Override log path to avoid touching real filesystem
    mod.log_path = intel_dir / "test.log"
    return mod


def _write_daily_snapshot(intel_dir: Path, date: str, data: dict):
    """Helper: write a daily snapshot JSON file."""
    path = intel_dir / "daily" / f"{date}.json"
    path.write_text(json.dumps(data))


def _write_intraday_snapshot(intel_dir: Path, date: str, hour: int, data: dict):
    """Helper: write an intraday snapshot JSON file."""
    date_dir = intel_dir / "intraday" / date
    date_dir.mkdir(parents=True, exist_ok=True)
    path = date_dir / f"{hour}.json"
    data["hour"] = hour
    path.write_text(json.dumps(data))


def _write_insight(intel_dir: Path, date: str, report: str):
    """Helper: write an insight report JSON file."""
    path = intel_dir / "insights" / f"{date}.json"
    path.write_text(json.dumps({"date": date, "report": report}))


# ============================================================================
# Data Assembly Tests
# ============================================================================


class TestDataAssembly:
    """Test _read_intelligence_data with files on disk."""

    def test_empty_directory(self, module):
        """With no data files, returns sane defaults."""
        data = module._read_intelligence_data()
        assert data["data_maturity"]["days_of_data"] == 0
        assert data["data_maturity"]["phase"] == "collecting"
        assert data["trend_data"] == []
        assert data["predictions"] is None
        assert data["baselines"] is None

    def test_with_daily_snapshots(self, module, intel_dir):
        """Daily snapshots populate data_maturity and trend_data."""
        for i in range(5):
            date = f"2025-01-{i + 1:02d}"
            _write_daily_snapshot(intel_dir, date, {
                "power": {"total_watts": 100 + i * 10},
                "lights": {"on": i},
                "occupancy": {"device_count_home": 2},
                "entities": {"unavailable": 3},
                "logbook_summary": {"useful_events": 50 + i},
            })

        data = module._read_intelligence_data()
        assert data["data_maturity"]["days_of_data"] == 5
        assert data["data_maturity"]["first_date"] == "2025-01-01"
        assert len(data["trend_data"]) == 5
        assert data["trend_data"][0]["power_watts"] == 100
        assert data["trend_data"][4]["power_watts"] == 140

    def test_with_predictions_file(self, module, intel_dir):
        """Predictions file is read from disk."""
        predictions = {"power_watts": {"predicted": 120, "confidence": "high"}}
        (intel_dir / "predictions.json").write_text(json.dumps(predictions))

        data = module._read_intelligence_data()
        assert data["predictions"]["power_watts"]["predicted"] == 120

    def test_with_baselines_file(self, module, intel_dir):
        """Baselines file is read from disk."""
        baselines = {"Monday": {"power_watts": {"mean": 110}}}
        (intel_dir / "baselines.json").write_text(json.dumps(baselines))

        data = module._read_intelligence_data()
        assert "Monday" in data["baselines"]

    def test_intraday_count(self, module, intel_dir):
        """Intraday snapshots are counted across date directories."""
        today = datetime.now().strftime("%Y-%m-%d")
        for h in [8, 12, 16, 20]:
            _write_intraday_snapshot(intel_dir, today, h, {
                "power": {"total_watts": 200},
            })

        data = module._read_intelligence_data()
        assert data["data_maturity"]["intraday_count"] == 4

    def test_latest_insight_read(self, module, intel_dir):
        """Most recent insight report is returned."""
        _write_insight(intel_dir, "2025-01-01", "First report")
        _write_insight(intel_dir, "2025-01-02", "Second report")

        data = module._read_intelligence_data()
        assert data["daily_insight"]["date"] == "2025-01-02"
        assert data["daily_insight"]["report"] == "Second report"


# ============================================================================
# Phase Detection Tests
# ============================================================================


class TestPhaseDetection:
    """Test _determine_phase with varying maturity levels."""

    def test_collecting_phase(self, module):
        """Less than 7 days = collecting phase."""
        phase, milestone = module._determine_phase(3, False, False)
        assert phase == "collecting"
        assert "4 more days" in milestone

    def test_baselines_phase(self, module):
        """7+ days without ML = baselines phase."""
        phase, milestone = module._determine_phase(10, False, False)
        assert phase == "baselines"
        assert "14 days" in milestone

    def test_ml_training_phase(self, module):
        """ML active but no meta-learning = ml-training phase."""
        phase, milestone = module._determine_phase(15, True, False)
        assert phase == "ml-training"
        assert "Meta-learning" in milestone

    def test_ml_active_phase(self, module):
        """Both ML and meta-learning active = ml-active (fully operational)."""
        phase, milestone = module._determine_phase(60, True, True)
        assert phase == "ml-active"
        assert "fully operational" in milestone

    def test_collecting_boundary(self, module):
        """Exactly 6 days = still collecting, needs 1 more day."""
        phase, milestone = module._determine_phase(6, False, False)
        assert phase == "collecting"
        assert "1 more day" in milestone

    def test_baselines_boundary(self, module):
        """Exactly 7 days = baselines phase."""
        phase, milestone = module._determine_phase(7, False, False)
        assert phase == "baselines"


# ============================================================================
# Activity Data Merge Tests
# ============================================================================


class TestActivityDataMerge:
    """Test _read_activity_data reading from hub cache."""

    @pytest.mark.asyncio
    async def test_reads_both_caches(self, module, hub):
        """Reads both activity_log and activity_summary from cache."""
        await hub.set_cache(CACHE_ACTIVITY_LOG, {"windows": [], "events_today": 10})
        await hub.set_cache(CACHE_ACTIVITY_SUMMARY, {"occupancy": {"anyone_home": True}})

        result = await module._read_activity_data()
        assert result["activity_log"]["events_today"] == 10
        assert result["activity_summary"]["occupancy"]["anyone_home"] is True

    @pytest.mark.asyncio
    async def test_handles_missing_cache(self, module, hub):
        """Returns None entries when cache is empty."""
        result = await module._read_activity_data()
        assert result["activity_log"] is None
        assert result["activity_summary"] is None


# ============================================================================
# Trend Extraction Tests
# ============================================================================


class TestTrendExtraction:
    """Test _extract_trend_data with sample daily JSON files."""

    def test_extracts_metrics(self, module, intel_dir):
        """Metrics are correctly extracted from daily snapshot files."""
        _write_daily_snapshot(intel_dir, "2025-01-01", {
            "power": {"total_watts": 150.7},
            "lights": {"on": 3},
            "occupancy": {"device_count_home": 2},
            "entities": {"unavailable": 5},
            "logbook_summary": {"useful_events": 42},
        })

        daily_files = sorted((intel_dir / "daily").glob("*.json"))
        trends = module._extract_trend_data(daily_files)

        assert len(trends) == 1
        assert trends[0]["date"] == "2025-01-01"
        assert trends[0]["power_watts"] == 150.7
        assert trends[0]["lights_on"] == 3
        assert trends[0]["devices_home"] == 2
        assert trends[0]["unavailable"] == 5
        assert trends[0]["useful_events"] == 42

    def test_caps_at_30_days(self, module, intel_dir):
        """Only the last 30 daily files are processed."""
        for i in range(40):
            date = f"2025-01-{i + 1:02d}" if i < 31 else f"2025-02-{i - 30:02d}"
            _write_daily_snapshot(intel_dir, date, {
                "power": {"total_watts": 100},
            })

        daily_files = sorted((intel_dir / "daily").glob("*.json"))
        trends = module._extract_trend_data(daily_files)
        assert len(trends) == 30

    def test_skips_malformed_files(self, module, intel_dir):
        """Malformed JSON files are skipped without crashing."""
        (intel_dir / "daily" / "2025-01-01.json").write_text("not json")
        _write_daily_snapshot(intel_dir, "2025-01-02", {
            "power": {"total_watts": 100},
        })

        daily_files = sorted((intel_dir / "daily").glob("*.json"))
        trends = module._extract_trend_data(daily_files)
        assert len(trends) == 1
        assert trends[0]["date"] == "2025-01-02"

    def test_missing_metric_paths(self, module, intel_dir):
        """Missing metric paths result in absent keys, not errors."""
        _write_daily_snapshot(intel_dir, "2025-01-01", {
            "power": {"total_watts": 100},
            # lights, occupancy, entities, logbook_summary all missing
        })

        daily_files = sorted((intel_dir / "daily").glob("*.json"))
        trends = module._extract_trend_data(daily_files)
        assert len(trends) == 1
        assert trends[0]["power_watts"] == 100
        assert "lights_on" not in trends[0]


# ============================================================================
# Digest Formatting Tests
# ============================================================================


class TestDigestFormatting:
    """Test _format_digest produces expected Telegram markdown."""

    def test_basic_format(self, module):
        """Digest contains header and phase info."""
        data = {
            "data_maturity": {"phase": "baselines", "days_of_data": 10},
            "trend_data": [],
            "intraday_trend": [],
            "predictions": {},
            "daily_insight": {},
        }
        result = module._format_digest(data)
        assert "*HA Intelligence Daily Digest*" in result
        assert "baselines" in result
        assert "10 days" in result

    def test_includes_intraday_metrics(self, module):
        """Digest includes latest intraday metrics when available."""
        data = {
            "data_maturity": {"phase": "ml-active", "days_of_data": 30},
            "trend_data": [],
            "intraday_trend": [
                {"hour": 8, "power_watts": 200, "lights_on": 3, "devices_home": 2},
            ],
            "predictions": {},
            "daily_insight": {},
        }
        result = module._format_digest(data)
        assert "Power: 200W" in result
        assert "Lights: 3" in result
        assert "Devices: 2" in result

    def test_includes_trend_comparison(self, module):
        """Digest includes vs-yesterday comparison when 2+ days of data."""
        data = {
            "data_maturity": {"phase": "baselines", "days_of_data": 10},
            "trend_data": [
                {"date": "2025-01-01", "power_watts": 100, "lights_on": 2},
                {"date": "2025-01-02", "power_watts": 130, "lights_on": 3},
            ],
            "intraday_trend": [],
            "predictions": {},
            "daily_insight": {},
        }
        result = module._format_digest(data)
        assert "vs yesterday" in result
        assert "power_watts: +30" in result

    def test_includes_predictions(self, module):
        """Digest includes prediction summary when available."""
        data = {
            "data_maturity": {"phase": "ml-active", "days_of_data": 30},
            "trend_data": [],
            "intraday_trend": [],
            "predictions": {
                "power_watts": {"predicted": 120, "confidence": "high"},
            },
            "daily_insight": {},
        }
        result = module._format_digest(data)
        assert "*Predictions:*" in result
        assert "power_watts: 120 (high)" in result

    def test_includes_insight_excerpt(self, module):
        """Digest includes truncated insight text."""
        data = {
            "data_maturity": {"phase": "ml-active", "days_of_data": 30},
            "trend_data": [],
            "intraday_trend": [],
            "predictions": {},
            "daily_insight": {
                "date": "2025-01-02",
                "report": "### Summary\nPower usage was higher than normal today." + "x" * 400,
            },
        }
        result = module._format_digest(data)
        assert "_Insight:_" in result
        # Headers should be stripped for Telegram
        assert "###" not in result
        assert "..." in result  # truncation indicator


# ============================================================================
# Telegram Sending Tests
# ============================================================================


class TestTelegramSending:
    """Test _send_telegram with mocked aiohttp."""

    @pytest.mark.asyncio
    async def test_send_success(self, module):
        """Successful Telegram send completes without error."""
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"ok": True})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        module._telegram_token = "fake-token"
        module._telegram_chat_id = "12345"

        with patch("modules.intelligence.aiohttp.ClientSession", return_value=mock_session):
            await module._send_telegram("Test message")

        # Verify post was called with correct URL structure
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "fake-token" in call_args[0][0]
        assert call_args[1]["json"]["chat_id"] == "12345"
        assert call_args[1]["json"]["text"] == "Test message"

    @pytest.mark.asyncio
    async def test_send_api_error_raises(self, module):
        """Telegram API returning ok=false raises RuntimeError."""
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"ok": False, "description": "Bad Request"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        module._telegram_token = "fake-token"
        module._telegram_chat_id = "12345"

        with patch("modules.intelligence.aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(RuntimeError, match="Telegram API error"):
                await module._send_telegram("Test message")


# ============================================================================
# ML Models & Meta-Learning Reader Tests
# ============================================================================


class TestMLModelsReader:
    """Test _read_ml_models with various training log formats."""

    def test_no_training_log(self, module):
        """Returns defaults when no training log exists."""
        result = module._read_ml_models()
        assert result["count"] == 0
        assert result["last_trained"] is None
        assert result["scores"] == {}

    def test_list_format_training_log(self, module, intel_dir):
        """Handles training log as a list of entries."""
        log = [
            {"timestamp": "2025-01-01T10:00:00", "scores": {"r2": 0.85}},
            {"timestamp": "2025-01-02T10:00:00", "scores": {"r2": 0.90}},
        ]
        (intel_dir / "models" / "training_log.json").write_text(json.dumps(log))

        result = module._read_ml_models()
        assert result["count"] == 2
        assert result["last_trained"] == "2025-01-02T10:00:00"
        assert result["scores"]["r2"] == 0.90


class TestMetaLearningReader:
    """Test _read_meta_learning with various formats."""

    def test_no_applied_file(self, module):
        """Returns defaults when no applied.json exists."""
        result = module._read_meta_learning()
        assert result["applied_count"] == 0
        assert result["suggestions"] == []

    def test_list_format_applied(self, module, intel_dir):
        """Handles applied.json as a list of suggestions."""
        applied = [
            {"timestamp": "2025-01-01", "suggestion": "Increase weight"},
            {"timestamp": "2025-01-02", "suggestion": "Add feature"},
        ]
        (intel_dir / "meta-learning" / "applied.json").write_text(json.dumps(applied))

        result = module._read_meta_learning()
        assert result["applied_count"] == 2
        assert result["last_applied"] == "2025-01-02"
        assert len(result["suggestions"]) == 2


# ============================================================================
# Initialize & Digest Guard Tests
# ============================================================================


class TestInitializeAndDigest:
    """Test initialize() and _maybe_send_digest logic."""

    @pytest.mark.asyncio
    async def test_initialize_populates_cache(self, module, hub, intel_dir):
        """initialize() reads files and sets cache."""
        _write_daily_snapshot(intel_dir, "2025-01-01", {
            "power": {"total_watts": 100},
        })

        await module.initialize()

        cached = await hub.get_cache(CACHE_INTELLIGENCE)
        assert cached is not None
        assert cached["data"]["data_maturity"]["days_of_data"] == 1

    @pytest.mark.asyncio
    async def test_digest_not_sent_on_init(self, module, hub, intel_dir):
        """initialize() records insight date but does NOT send digest."""
        _write_insight(intel_dir, "2025-01-01", "Test report")
        _write_daily_snapshot(intel_dir, "2025-01-01", {})

        module._telegram_token = "fake"
        module._telegram_chat_id = "12345"

        with patch.object(module, "_send_telegram", new_callable=AsyncMock) as mock_send:
            await module.initialize()
            mock_send.assert_not_called()

        # But the date should be tracked so duplicate sends don't happen
        assert module._last_digest_date == "2025-01-01"

    @pytest.mark.asyncio
    async def test_digest_sent_on_new_insight(self, module, hub, intel_dir):
        """_maybe_send_digest sends when a new insight date appears."""
        module._telegram_token = "fake"
        module._telegram_chat_id = "12345"
        module._last_digest_date = "2025-01-01"

        data = {
            "data_maturity": {"phase": "baselines", "days_of_data": 10},
            "trend_data": [],
            "intraday_trend": [],
            "predictions": {},
            "daily_insight": {"date": "2025-01-02", "report": "New report"},
        }

        with patch.object(module, "_send_telegram", new_callable=AsyncMock) as mock_send:
            await module._maybe_send_digest(data)
            mock_send.assert_called_once()

        assert module._last_digest_date == "2025-01-02"

    @pytest.mark.asyncio
    async def test_digest_not_sent_without_credentials(self, module):
        """_maybe_send_digest skips when Telegram credentials are missing."""
        module._telegram_token = None
        module._telegram_chat_id = None

        data = {
            "daily_insight": {"date": "2025-01-02", "report": "New report"},
        }

        with patch.object(module, "_send_telegram", new_callable=AsyncMock) as mock_send:
            await module._maybe_send_digest(data)
            mock_send.assert_not_called()


# ============================================================================
# Config & Helpers Tests
# ============================================================================


class TestConfig:
    """Test _read_config with and without feature_config.json."""

    def test_default_config(self, module):
        """Without feature_config.json, returns defaults."""
        config = module._read_config()
        assert config["anomaly_threshold"] == 2.0
        assert config["feature_config"]["time_features"] is True

    def test_custom_config(self, module, intel_dir):
        """With feature_config.json, those values are used."""
        custom = {"time_features": False, "weather_features": True}
        (intel_dir / "feature_config.json").write_text(json.dumps(custom))

        config = module._read_config()
        assert config["feature_config"]["time_features"] is False


class TestReadJson:
    """Test _read_json helper."""

    def test_missing_file(self, module, tmp_path):
        """Returns None for missing file."""
        result = module._read_json(tmp_path / "nonexistent.json")
        assert result is None

    def test_valid_file(self, module, tmp_path):
        """Returns parsed JSON for valid file."""
        path = tmp_path / "test.json"
        path.write_text(json.dumps({"key": "value"}))
        result = module._read_json(path)
        assert result == {"key": "value"}

    def test_invalid_json(self, module, tmp_path):
        """Returns None for invalid JSON."""
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        result = module._read_json(path)
        assert result is None


# ============================================================================
# Phase 2-4 Engine Output Tests
# ============================================================================


class TestPhase2To4Outputs:
    """Test reading Phase 2-4 engine outputs (entity correlations, sequence
    anomalies, power profiles, automation suggestions)."""

    def test_entity_correlations(self, module, intel_dir):
        """entity_correlations.json is read into cache payload."""
        data = {"pairs": [{"a": "light.kitchen", "b": "light.dining", "score": 0.92}]}
        (intel_dir / "entity_correlations.json").write_text(json.dumps(data))

        result = module._read_intelligence_data()
        assert result["entity_correlations"] is not None
        assert result["entity_correlations"]["pairs"][0]["score"] == 0.92

    def test_sequence_anomalies(self, module, intel_dir):
        """sequence_anomalies.json is read into cache payload."""
        data = {"anomalies": [{"entity": "lock.front_door", "type": "unusual_time"}]}
        (intel_dir / "sequence_anomalies.json").write_text(json.dumps(data))

        result = module._read_intelligence_data()
        assert result["sequence_anomalies"] is not None
        assert result["sequence_anomalies"]["anomalies"][0]["entity"] == "lock.front_door"

    def test_power_profiles(self, module, intel_dir):
        """insights/power-profiles.json is read into cache payload."""
        data = {"profiles": [{"entity": "switch.dryer", "avg_watts": 3200}]}
        (intel_dir / "insights" / "power-profiles.json").write_text(json.dumps(data))

        result = module._read_intelligence_data()
        assert result["power_profiles"] is not None
        assert result["power_profiles"]["profiles"][0]["avg_watts"] == 3200

    def test_automation_suggestions_latest_file(self, module, intel_dir):
        """Latest file from insights/automation-suggestions/ is read."""
        suggestions_dir = intel_dir / "insights" / "automation-suggestions"
        suggestions_dir.mkdir(parents=True)

        old = {"suggestions": [{"name": "Old suggestion"}]}
        new = {"suggestions": [{"name": "Turn off lights at midnight"}]}
        (suggestions_dir / "2026-02-10.json").write_text(json.dumps(old))
        (suggestions_dir / "2026-02-12.json").write_text(json.dumps(new))

        result = module._read_intelligence_data()
        assert result["automation_suggestions"] is not None
        assert result["automation_suggestions"]["suggestions"][0]["name"] == "Turn off lights at midnight"

    def test_missing_files_return_none(self, module):
        """When Phase 2-4 files don't exist, their keys are None (no errors)."""
        result = module._read_intelligence_data()
        assert result["entity_correlations"] is None
        assert result["sequence_anomalies"] is None
        assert result["power_profiles"] is None
        assert result["automation_suggestions"] is None

    def test_automation_suggestions_empty_dir(self, module, intel_dir):
        """Empty automation-suggestions directory returns None."""
        suggestions_dir = intel_dir / "insights" / "automation-suggestions"
        suggestions_dir.mkdir(parents=True)

        result = module._read_latest_automation_suggestion()
        assert result is None

    def test_automation_suggestions_malformed_json(self, module, intel_dir):
        """Malformed automation suggestion file returns None."""
        suggestions_dir = intel_dir / "insights" / "automation-suggestions"
        suggestions_dir.mkdir(parents=True)
        (suggestions_dir / "2026-02-12.json").write_text("not valid json {{")

        result = module._read_latest_automation_suggestion()
        assert result is None
