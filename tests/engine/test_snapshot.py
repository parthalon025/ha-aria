"""Tests for snapshot construction — race condition (#22) and presence validation (#23)."""

import json
from unittest.mock import patch

import pytest

from aria.engine.collectors.snapshot import (
    _validate_presence,
    build_intraday_snapshot,
)
from aria.engine.config import (
    AppConfig,
    HAConfig,
    HolidayConfig,
    ModelConfig,
    OllamaConfig,
    PathConfig,
    SafetyConfig,
    WeatherConfig,
)
from aria.engine.storage.data_store import DataStore

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_paths(tmp_path):
    """Create a PathConfig pointed at a temp directory."""
    paths = PathConfig(
        data_dir=tmp_path / "intelligence",
        logbook_path=tmp_path / "current.json",
    )
    paths.ensure_dirs()
    return paths


@pytest.fixture
def store(tmp_paths):
    """Create a DataStore backed by temp directories."""
    return DataStore(tmp_paths)


@pytest.fixture
def app_config(tmp_paths):
    """Create an AppConfig with temp paths."""
    return AppConfig(
        ha=HAConfig(),
        paths=tmp_paths,
        model=ModelConfig(),
        ollama=OllamaConfig(),
        weather=WeatherConfig(),
        safety=SafetyConfig(),
        holidays=HolidayConfig(),
    )


# ============================================================================
# #23: Presence cold-start validation
# ============================================================================


class TestValidatePresence:
    """_validate_presence detects all-zero presence features."""

    def test_all_zero_presence_marks_invalid(self):
        """Snapshot with all-zero presence fields → presence_valid=False."""
        snapshot = {
            "presence": {
                "overall_probability": 0,
                "occupied_room_count": 0,
                "identified_person_count": 0,
                "camera_signal_count": 0,
                "rooms": {},
            }
        }
        _validate_presence(snapshot)
        assert snapshot["presence_valid"] is False

    def test_nonzero_presence_marks_valid(self):
        """Snapshot with at least one nonzero field → presence_valid=True."""
        snapshot = {
            "presence": {
                "overall_probability": 0.85,
                "occupied_room_count": 2,
                "identified_person_count": 1,
                "camera_signal_count": 3,
                "rooms": {"living_room": {"probability": 0.85}},
            }
        }
        _validate_presence(snapshot)
        assert snapshot["presence_valid"] is True

    def test_partial_nonzero_marks_valid(self):
        """Even one nonzero field is enough to mark valid."""
        snapshot = {
            "presence": {
                "overall_probability": 0.5,
                "occupied_room_count": 0,
                "identified_person_count": 0,
                "camera_signal_count": 0,
                "rooms": {},
            }
        }
        _validate_presence(snapshot)
        assert snapshot["presence_valid"] is True

    def test_missing_presence_key_marks_invalid(self):
        """Snapshot with no 'presence' key → presence_valid=False."""
        snapshot = {}
        _validate_presence(snapshot)
        assert snapshot["presence_valid"] is False

    def test_empty_presence_dict_marks_invalid(self):
        """Snapshot with empty presence dict → presence_valid=False."""
        snapshot = {"presence": {}}
        _validate_presence(snapshot)
        assert snapshot["presence_valid"] is False


# ============================================================================
# #22: Snapshot write race — file lock and deduplication
# ============================================================================


class TestIntradaySnapshotDedup:
    """build_intraday_snapshot returns existing snapshot if one exists for the hour."""

    @patch("aria.engine.collectors.snapshot.fetch_ha_states", return_value=[])
    @patch("aria.engine.collectors.snapshot._fetch_presence_cache", return_value=None)
    @patch("aria.engine.collectors.snapshot.fetch_weather", return_value=None)
    @patch("aria.engine.collectors.snapshot.parse_weather", return_value={})
    @patch("aria.engine.collectors.snapshot.summarize_logbook", return_value={})
    def test_existing_snapshot_returns_cached(
        self, _mock_log, _mock_pw, _mock_fw, _mock_pc, _mock_ha, store, app_config
    ):
        """If HH.json already exists, build_intraday_snapshot returns it without rebuilding."""
        date_str = "2026-02-19"
        hour = 14

        # Pre-create the snapshot file
        day_dir = store.paths.intraday_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        existing = {"date": date_str, "hour": hour, "preexisting": True}
        with open(day_dir / f"{hour:02d}.json", "w") as f:
            json.dump(existing, f)

        result = build_intraday_snapshot(hour=hour, date_str=date_str, config=app_config, store=store)

        # Should return the pre-existing snapshot, not build a new one
        assert result["preexisting"] is True
        # fetch_ha_states should NOT have been called since we short-circuited
        _mock_ha.assert_not_called()

    @patch("aria.engine.collectors.snapshot.fetch_ha_states", return_value=[])
    @patch("aria.engine.collectors.snapshot._fetch_presence_cache", return_value=None)
    @patch("aria.engine.collectors.snapshot.fetch_weather", return_value=None)
    @patch("aria.engine.collectors.snapshot.parse_weather", return_value={})
    @patch("aria.engine.collectors.snapshot.summarize_logbook", return_value={})
    def test_no_existing_snapshot_builds_new(
        self, _mock_log, _mock_pw, _mock_fw, _mock_pc, _mock_ha, store, app_config
    ):
        """When no snapshot exists for the hour, it builds a new one normally."""
        result = build_intraday_snapshot(hour=10, date_str="2026-02-19", config=app_config, store=store)

        assert result["hour"] == 10
        assert result["date"] == "2026-02-19"
        # presence_valid should be set (from #23 validation)
        assert "presence_valid" in result

    @patch("aria.engine.collectors.snapshot.fetch_ha_states", return_value=[])
    @patch("aria.engine.collectors.snapshot._fetch_presence_cache", return_value=None)
    @patch("aria.engine.collectors.snapshot.fetch_weather", return_value=None)
    @patch("aria.engine.collectors.snapshot.parse_weather", return_value={})
    @patch("aria.engine.collectors.snapshot.summarize_logbook", return_value={})
    def test_lock_file_created(self, _mock_log, _mock_pw, _mock_fw, _mock_pc, _mock_ha, store, app_config):
        """Building a snapshot creates a .lock file in the intraday directory."""
        date_str = "2026-02-19"
        hour = 8
        build_intraday_snapshot(hour=hour, date_str=date_str, config=app_config, store=store)

        lock_path = store.paths.intraday_dir / date_str / f"{hour:02d}.lock"
        assert lock_path.exists()


# ============================================================================
# #23: Presence validation in intraday snapshot pipeline
# ============================================================================


class TestIntradayPresenceValidation:
    """build_intraday_snapshot includes presence_valid flag."""

    @patch("aria.engine.collectors.snapshot.fetch_ha_states", return_value=[])
    @patch("aria.engine.collectors.snapshot._fetch_presence_cache", return_value=None)
    @patch("aria.engine.collectors.snapshot.fetch_weather", return_value=None)
    @patch("aria.engine.collectors.snapshot.parse_weather", return_value={})
    @patch("aria.engine.collectors.snapshot.summarize_logbook", return_value={})
    def test_cold_start_presence_invalid(self, _mock_log, _mock_pw, _mock_fw, _mock_pc, _mock_ha, store, app_config):
        """When presence cache is unavailable (cold-start), presence_valid is False."""
        result = build_intraday_snapshot(hour=12, date_str="2026-02-19", config=app_config, store=store)

        assert result["presence_valid"] is False
        assert result["presence"]["overall_probability"] == 0

    @patch("aria.engine.collectors.snapshot.fetch_ha_states", return_value=[])
    @patch(
        "aria.engine.collectors.snapshot._fetch_presence_cache",
        return_value={
            "rooms": {"living_room": {"probability": 0.9, "signals": [], "persons": ["alice"]}},
            "identified_persons": {"alice": {}},
        },
    )
    @patch("aria.engine.collectors.snapshot.fetch_weather", return_value=None)
    @patch("aria.engine.collectors.snapshot.parse_weather", return_value={})
    @patch("aria.engine.collectors.snapshot.summarize_logbook", return_value={})
    def test_valid_presence_marks_true(self, _mock_log, _mock_pw, _mock_fw, _mock_pc, _mock_ha, store, app_config):
        """When presence cache has real data, presence_valid is True."""
        result = build_intraday_snapshot(hour=12, date_str="2026-02-19", config=app_config, store=store)

        assert result["presence_valid"] is True
        assert result["presence"]["overall_probability"] > 0
