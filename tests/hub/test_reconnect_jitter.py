"""Tests for per-module reconnect stagger to prevent thundering herd.

When HA restarts, all WebSocket/MQTT connections reconnect simultaneously.
Each module should have a different initial reconnect delay (stagger) to
spread reconnection attempts across time.
"""

from aria.hub.constants import RECONNECT_STAGGER


class TestReconnectStagger:
    """Verify per-module reconnect stagger configuration."""

    def test_all_modules_have_stagger(self):
        """All connection-holding modules must be in RECONNECT_STAGGER."""
        expected_modules = {"discovery", "activity_monitor", "presence_mqtt", "presence_ws"}
        assert set(RECONNECT_STAGGER.keys()) == expected_modules

    def test_staggers_are_at_least_1s_apart(self):
        """Each module must have a stagger at least 1s different from all others."""
        values = sorted(RECONNECT_STAGGER.values())
        for i in range(1, len(values)):
            gap = values[i] - values[i - 1]
            assert gap >= 1, (
                f"Stagger gap between sorted values {values[i - 1]} and {values[i]} is only {gap}s (need >= 1s)"
            )

    def test_discovery_has_lowest_stagger(self):
        """Discovery should reconnect first (0s stagger) since other modules depend on it."""
        assert RECONNECT_STAGGER["discovery"] == 0

    def test_presence_ws_has_highest_stagger(self):
        """Presence WS has highest stagger (last to reconnect)."""
        max_stagger = max(RECONNECT_STAGGER.values())
        assert RECONNECT_STAGGER["presence_ws"] == max_stagger

    def test_staggers_are_non_negative(self):
        """All stagger values must be >= 0."""
        for module, stagger in RECONNECT_STAGGER.items():
            assert stagger >= 0, f"{module} has negative stagger: {stagger}"

    def test_activity_monitor_after_discovery(self):
        """Activity monitor reconnects after discovery."""
        assert RECONNECT_STAGGER["activity_monitor"] > RECONNECT_STAGGER["discovery"]

    def test_presence_mqtt_after_activity_monitor(self):
        """Presence MQTT reconnects after activity monitor."""
        assert RECONNECT_STAGGER["presence_mqtt"] > RECONNECT_STAGGER["activity_monitor"]

    def test_presence_ws_after_presence_mqtt(self):
        """Presence WS reconnects after presence MQTT."""
        assert RECONNECT_STAGGER["presence_ws"] > RECONNECT_STAGGER["presence_mqtt"]

    def test_stagger_values_match_spec(self):
        """Verify exact stagger values match the design spec."""
        assert RECONNECT_STAGGER == {
            "discovery": 0,
            "activity_monitor": 2,
            "presence_mqtt": 4,
            "presence_ws": 6,
        }
