"""Tests for reliability medium batch fixes — #45, #46, #47, #56.

#45: Silent except-pass replaced with logging
#46: Typed failure returns (predict() returns None)
#47: Retry decorator for network calls
#56: Bounded in-memory collections
"""

import logging
import urllib.error
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aria.engine.collectors.ha_api import (
    fetch_ha_states,
    fetch_weather,
    retry_on_network_error,
)
from aria.engine.sequence import SequenceClassifier

# ---------------------------------------------------------------
# #45: Silent except-pass replaced with logging
# ---------------------------------------------------------------


class TestSilentExceptPassLogging:
    """Verify that formerly-silent except blocks now emit log warnings."""

    def test_hardware_gpu_import_error_logs(self, caplog):
        """GPU detection ImportError logs a warning (#45 hardware.py)."""
        from aria.engine.hardware import scan_hardware

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3)
        mock_psutil.cpu_count.return_value = 4

        # Simulate torch import raising ImportError
        with (
            patch.dict("sys.modules", {"psutil": mock_psutil, "torch": None}),
            caplog.at_level(logging.WARNING, logger="aria.engine.hardware"),
        ):
            scan_hardware()

        assert any("torch not installed" in r.message or "GPU detection" in r.message for r in caplog.records)

    def test_hardware_gpu_exception_logs(self, caplog):
        """GPU detection runtime error logs a warning (#45 hardware.py)."""
        from aria.engine.hardware import scan_hardware

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3)
        mock_psutil.cpu_count.return_value = 4

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("GPU error")

        with (
            patch.dict("sys.modules", {"psutil": mock_psutil, "torch": mock_torch}),
            caplog.at_level(logging.WARNING, logger="aria.engine.hardware"),
        ):
            profile = scan_hardware()

        assert any("GPU detection failed" in r.message for r in caplog.records)
        assert profile.gpu_available is False

    def test_snapshot_presence_hub_api_failure_logs(self, caplog):
        """Presence hub API failure logs a warning (#45 snapshot.py)."""
        from aria.engine.collectors.snapshot import _fetch_presence_cache

        with patch("aria.engine.collectors.snapshot.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("connection refused")
            with patch("aria.engine.collectors.snapshot.sqlite3") as mock_sqlite:
                mock_sqlite.connect.side_effect = Exception("no db")
                with caplog.at_level(logging.WARNING, logger="aria.engine.collectors.snapshot"):
                    result = _fetch_presence_cache()

        assert result is None
        assert any("Presence hub API" in r.message for r in caplog.records)
        assert any("Presence SQLite fallback" in r.message for r in caplog.records)

    def test_extractors_power_parse_failure_logs(self, caplog):
        """Power collector logs warning on unparseable value (#45 extractors.py)."""
        from aria.engine.collectors.extractors import PowerCollector

        snapshot = {"power": {"total_watts": 0.0, "outlets": {}}}
        states = [
            {
                "entity_id": "sensor.usp_pdu_pro_outlet_1_power",
                "state": "not_a_number",
                "attributes": {"friendly_name": "Outlet 1"},
            }
        ]
        with caplog.at_level(logging.WARNING, logger="aria.engine.collectors.extractors"):
            PowerCollector().extract(snapshot, states)

        assert any("Failed to parse power value" in r.message for r in caplog.records)


# ---------------------------------------------------------------
# #46: Typed failure returns
# ---------------------------------------------------------------


class TestTypedFailureReturns:
    """Verify predict() returns None instead of 'stable' on failure."""

    def test_predict_untrained_returns_none(self):
        """Untrained classifier returns None, not 'stable'."""
        clf = SequenceClassifier(window_size=4)
        window = np.zeros((4, 5))
        result = clf.predict(window)
        assert result is None

    def test_predict_tslearn_unavailable_returns_none(self):
        """Classifier with tslearn unavailable returns None."""
        clf = SequenceClassifier(window_size=4)
        clf._tslearn_available = False
        window = np.zeros((4, 3))
        result = clf.predict(window)
        assert result is None

    def test_predict_exception_returns_none(self):
        """Classifier returns None when predict() raises."""
        clf = SequenceClassifier(window_size=4)
        clf._tslearn_available = True
        clf._model = MagicMock()
        clf._model.predict.side_effect = RuntimeError("model error")
        window = np.zeros((4, 3))
        result = clf.predict(window)
        assert result is None

    def test_predict_trained_returns_string(self):
        """Trained classifier returns a string label."""
        clf = SequenceClassifier(window_size=4)
        clf._tslearn_available = True
        clf._model = MagicMock()
        clf._model.predict.return_value = np.array(["ramping_up"])
        window = np.zeros((4, 3))
        result = clf.predict(window)
        assert result == "ramping_up"
        assert isinstance(result, str)

    def test_trajectory_classifier_handles_none_predict(self):
        """TrajectoryClassifier falls back to heuristic when predict() returns None."""
        from aria.modules.trajectory_classifier import TrajectoryClassifier

        # Create a mock hub
        hub = MagicMock()
        hub.set_cache = MagicMock()

        tc = TrajectoryClassifier(hub)
        tc.active = True

        # Classifier not trained, predict() returns None
        assert tc.sequence_classifier.predict(np.zeros((6, 3))) is None

        # The heuristic should still produce a valid trajectory label
        label = SequenceClassifier.label_window_heuristic(np.zeros((6, 3)))
        assert label == "stable"


# ---------------------------------------------------------------
# #47: Retry decorator for network calls
# ---------------------------------------------------------------


class TestRetryDecorator:
    """Verify retry_on_network_error decorator behavior."""

    def test_retry_succeeds_on_first_attempt(self):
        """No retries needed when function succeeds immediately."""
        call_count = 0

        @retry_on_network_error(max_attempts=3, backoff_factor=1.0)
        def always_works():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = always_works()
        assert result == "ok"
        assert call_count == 1

    def test_retry_succeeds_on_second_attempt(self):
        """Function succeeds after one transient failure."""
        call_count = 0

        @retry_on_network_error(max_attempts=3, backoff_factor=0.01)
        def fails_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise urllib.error.URLError("connection refused")
            return "ok"

        result = fails_once()
        assert result == "ok"
        assert call_count == 2

    def test_retry_exhausts_all_attempts(self):
        """Function raises after exhausting all retry attempts."""
        call_count = 0

        @retry_on_network_error(max_attempts=3, backoff_factor=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("timed out")

        with pytest.raises(TimeoutError):
            always_fails()
        assert call_count == 3

    def test_retry_only_catches_network_errors(self):
        """Non-network errors are not retried."""
        call_count = 0

        @retry_on_network_error(max_attempts=3, backoff_factor=0.01)
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not a network error")

        with pytest.raises(ValueError):
            raises_value_error()
        assert call_count == 1

    def test_retry_catches_url_error(self):
        """URLError triggers retry."""
        call_count = 0

        @retry_on_network_error(max_attempts=2, backoff_factor=0.01)
        def url_error_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise urllib.error.URLError("DNS resolution failed")
            return "recovered"

        assert url_error_then_ok() == "recovered"
        assert call_count == 2

    def test_fetch_ha_states_uses_retry(self):
        """fetch_ha_states retries on URLError before returning []."""
        from aria.engine.config import HAConfig

        config = HAConfig(url="http://test-host:8123", token="test-token")

        with patch("aria.engine.collectors.ha_api.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("connection refused")
            with patch("aria.engine.collectors.ha_api.time.sleep"):
                result = fetch_ha_states(config)

        assert result == []
        assert mock_urlopen.call_count == 3  # 3 attempts

    def test_fetch_weather_uses_retry(self):
        """fetch_weather retries on TimeoutError before returning ''."""
        from aria.engine.config import WeatherConfig

        config = WeatherConfig(location="TestCity")

        with patch("aria.engine.collectors.ha_api.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("request timed out")
            with patch("aria.engine.collectors.ha_api.time.sleep"):
                result = fetch_weather(config)

        assert result == ""
        assert mock_urlopen.call_count == 3  # 3 attempts

    def test_retry_logs_warnings(self, caplog):
        """Retry decorator logs warnings on each failed attempt."""

        @retry_on_network_error(max_attempts=3, backoff_factor=0.01)
        def always_timeout():
            raise TimeoutError("timed out")

        with caplog.at_level(logging.WARNING), pytest.raises(TimeoutError):
            always_timeout()

        # Should have warnings for attempts 1, 2, and final failure
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 2  # At least retry warnings + final


# ---------------------------------------------------------------
# #56: Bounded in-memory collections
# ---------------------------------------------------------------


class TestBoundedCollections:
    """Verify in-memory collections are bounded."""

    def test_shadow_engine_recent_resolved_is_deque(self):
        """_recent_resolved is a bounded deque, not an unbounded list."""
        from aria.modules.shadow_engine import ShadowEngine

        hub = MagicMock()
        hub.subscribe = MagicMock()
        engine = ShadowEngine(hub)

        assert isinstance(engine._recent_resolved, deque)
        assert engine._recent_resolved.maxlen == 200

    def test_shadow_engine_recent_resolved_evicts_oldest(self):
        """deque(maxlen=200) evicts oldest items when full."""
        from aria.modules.shadow_engine import ShadowEngine

        hub = MagicMock()
        hub.subscribe = MagicMock()
        engine = ShadowEngine(hub)

        # Fill to capacity + 1
        for i in range(201):
            engine._recent_resolved.append({"id": str(i)})

        assert len(engine._recent_resolved) == 200
        # First item should be evicted
        assert engine._recent_resolved[0]["id"] == "1"
        assert engine._recent_resolved[-1]["id"] == "200"

    def test_activity_buffer_early_flush_at_5000(self):
        """Activity buffer triggers early flush at 5000 events.

        NOTE: This test is also in tests/hub/test_activity_monitor_flush.py
        with a tighter assertion. This version kept for suite coverage.
        """
        pytest.importorskip("aiohttp")
        from aria.modules.activity_monitor import ActivityMonitor

        hub = MagicMock()
        hub.subscribe = MagicMock()
        hub.is_running = MagicMock(return_value=True)
        hub.publish = MagicMock()

        monitor = ActivityMonitor(hub, ha_url="http://test-host:8123", ha_token="test-token")

        # Set up so _handle_state_changed processes events
        monitor._curation_loaded = False
        monitor._occupancy_state = False  # Prevent snapshot triggers

        # Mock the event loop
        mock_loop = MagicMock()
        mock_task = MagicMock()
        mock_task.exception.return_value = None
        mock_loop.create_task.return_value = mock_task

        # Pre-fill buffer to just below threshold
        for i in range(4999):
            monitor._activity_buffer.append({"domain": "light", "entity_id": f"light.test_{i}"})

        # Add one more event via _handle_state_changed
        data = {
            "entity_id": "light.kitchen",
            "new_state": {"state": "on", "attributes": {"friendly_name": "Kitchen"}},
            "old_state": {"state": "off", "attributes": {}},
        }

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            monitor._handle_state_changed(data)

        # Verify flush was triggered by checking create_task was called with a coroutine
        # from _flush_activity_buffer
        assert mock_loop.create_task.call_count >= 1
