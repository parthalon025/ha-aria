"""Test activity monitor early flush at buffer threshold (#56)."""

from unittest.mock import MagicMock, patch

from aria.modules.activity_monitor import ActivityMonitor


class TestActivityBufferEarlyFlush:
    """Verify buffer triggers early flush at 5000 events."""

    def test_early_flush_triggered_at_5000(self):
        """Buffer reaching 5000 events fires _flush_activity_buffer via create_task."""
        hub = MagicMock()
        hub.subscribe = MagicMock()
        hub.is_running = MagicMock(return_value=True)
        hub.publish = MagicMock()

        monitor = ActivityMonitor(hub, ha_url="http://test-host:8123", ha_token="test-token")
        monitor._curation_loaded = False
        monitor._occupancy_state = False

        mock_loop = MagicMock()
        mock_task = MagicMock()
        mock_task.exception.return_value = None
        mock_loop.create_task.return_value = mock_task

        # Pre-fill buffer to just below threshold
        for i in range(4999):
            monitor._activity_buffer.append({"domain": "light", "entity_id": f"light.test_{i}"})

        data = {
            "entity_id": "light.kitchen",
            "new_state": {"state": "on", "attributes": {"friendly_name": "Kitchen"}},
            "old_state": {"state": "off", "attributes": {}},
        }

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            monitor._handle_state_changed(data)

        # Verify _flush_activity_buffer coroutine was passed to create_task
        flush_calls = [call for call in mock_loop.create_task.call_args_list if "_flush_activity_buffer" in str(call)]
        assert len(flush_calls) >= 1, (
            f"Expected _flush_activity_buffer in create_task calls, got: {mock_loop.create_task.call_args_list}"
        )

    def test_no_flush_below_threshold(self):
        """Buffer below 5000 does not trigger early flush."""
        hub = MagicMock()
        hub.subscribe = MagicMock()
        hub.is_running = MagicMock(return_value=True)
        hub.publish = MagicMock()

        monitor = ActivityMonitor(hub, ha_url="http://test-host:8123", ha_token="test-token")
        monitor._curation_loaded = False
        monitor._occupancy_state = False

        mock_loop = MagicMock()
        mock_task = MagicMock()
        mock_task.exception.return_value = None
        mock_loop.create_task.return_value = mock_task

        # Pre-fill to 100 events (well below threshold)
        for i in range(100):
            monitor._activity_buffer.append({"domain": "light", "entity_id": f"light.test_{i}"})

        data = {
            "entity_id": "light.kitchen",
            "new_state": {"state": "on", "attributes": {"friendly_name": "Kitchen"}},
            "old_state": {"state": "off", "attributes": {}},
        }

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            monitor._handle_state_changed(data)

        # Only the event bus publish should have called create_task, not flush
        flush_calls = [call for call in mock_loop.create_task.call_args_list if "_flush_activity_buffer" in str(call)]
        assert len(flush_calls) == 0, f"Unexpected flush calls: {flush_calls}"
