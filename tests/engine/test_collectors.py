"""Tests for collectors: snapshot building, entity extraction, weather, logbook."""

import unittest

from conftest import EXTENDED_STATES, SAMPLE_STATES

from aria.engine.collectors.extractors import (
    BatteriesCollector,
    ClimateCollector,
    DoorsWindowsCollector,
    EVCollector,
    LightsCollector,
    MediaCollector,
    NetworkCollector,
    OccupancyCollector,
    PowerCollector,
    SunCollector,
    VacuumCollector,
)
from aria.engine.collectors.ha_api import parse_weather
from aria.engine.collectors.logbook import summarize_logbook
from aria.engine.collectors.snapshot import build_empty_snapshot
from aria.engine.config import HolidayConfig


class TestDailySnapshot(unittest.TestCase):
    def test_snapshot_schema_has_required_keys(self):
        snapshot = build_empty_snapshot("2026-02-10", HolidayConfig())
        required_keys = [
            "date",
            "day_of_week",
            "is_weekend",
            "is_holiday",
            "weather",
            "calendar_events",
            "entities",
            "power",
            "occupancy",
            "climate",
            "locks",
            "lights",
            "motion",
            "automations",
            "ev",
            "logbook_summary",
        ]
        for key in required_keys:
            self.assertIn(key, snapshot, f"Missing key: {key}")

    def test_snapshot_date_metadata(self):
        snapshot = build_empty_snapshot("2026-02-10", HolidayConfig())
        self.assertEqual(snapshot["date"], "2026-02-10")
        self.assertEqual(snapshot["day_of_week"], "Tuesday")
        self.assertFalse(snapshot["is_weekend"])


class TestExternalData(unittest.TestCase):
    def test_parse_weather(self):
        raw = "Partly cloudy +62°F 70% →11mph"
        result = parse_weather(raw)
        self.assertEqual(result["temp_f"], 62)
        self.assertEqual(result["humidity_pct"], 70)
        self.assertIn("cloudy", result["condition"].lower())

    def test_parse_logbook_summary(self):
        entries = [
            {"entity_id": "sensor.time", "when": "2026-02-10T01:02:00+00:00"},
            {"entity_id": "sensor.time", "when": "2026-02-10T01:03:00+00:00"},
            {"entity_id": "light.atrium", "when": "2026-02-10T18:30:00+00:00"},
            {"entity_id": "lock.front_door", "when": "2026-02-10T18:35:00+00:00"},
        ]
        result = summarize_logbook(entries)
        self.assertEqual(result["total_events"], 4)
        self.assertEqual(result["useful_events"], 2)
        self.assertEqual(result["by_domain"]["sensor"], 2)
        self.assertEqual(result["by_domain"]["light"], 1)
        self.assertIn("18", result["hourly"])


class TestEntityExtraction(unittest.TestCase):
    def _snap(self):
        return build_empty_snapshot("2026-02-10", HolidayConfig())

    def test_extract_power(self):
        snapshot = self._snap()
        PowerCollector().extract(snapshot, SAMPLE_STATES)
        self.assertAlmostEqual(snapshot["power"]["total_watts"], 156.5, places=1)

    def test_extract_occupancy(self):
        snapshot = self._snap()
        OccupancyCollector().extract(snapshot, SAMPLE_STATES)
        self.assertIn("Justin", snapshot["occupancy"]["people_home"])
        self.assertIn("Lisa", snapshot["occupancy"]["people_away"])
        self.assertGreater(snapshot["occupancy"]["device_count_home"], 0)

    def test_extract_climate(self):
        snapshot = self._snap()
        ClimateCollector().extract(snapshot, SAMPLE_STATES)
        self.assertEqual(len(snapshot["climate"]), 1)
        self.assertEqual(snapshot["climate"][0]["name"], "Bedroom")
        self.assertEqual(snapshot["climate"][0]["current_temp"], 72)

    def test_extract_lights(self):
        snapshot = self._snap()
        LightsCollector().extract(snapshot, SAMPLE_STATES)
        self.assertEqual(snapshot["lights"]["on"], 1)
        self.assertEqual(snapshot["lights"]["off"], 1)

    def test_extract_ev(self):
        snapshot = self._snap()
        EVCollector().extract(snapshot, SAMPLE_STATES)
        self.assertIn("TARS", snapshot["ev"])
        self.assertEqual(snapshot["ev"]["TARS"]["battery_pct"], 71)
        self.assertAlmostEqual(snapshot["ev"]["TARS"]["charger_power_kw"], 4.0)


class TestNewExtraction(unittest.TestCase):
    def _snap(self):
        return build_empty_snapshot("2026-02-10", HolidayConfig())

    def test_extract_doors_windows(self):
        snapshot = self._snap()
        DoorsWindowsCollector().extract(snapshot, EXTENDED_STATES)
        self.assertIn("Front Door", snapshot["doors_windows"])
        self.assertIn("Garage Door", snapshot["doors_windows"])
        self.assertIn("Kitchen Window", snapshot["doors_windows"])
        self.assertNotIn("Motion 1", snapshot["doors_windows"])
        self.assertEqual(snapshot["doors_windows"]["Garage Door"]["state"], "on")

    def test_extract_batteries(self):
        snapshot = self._snap()
        BatteriesCollector().extract(snapshot, EXTENDED_STATES)
        self.assertIn("lock.back_door", snapshot["batteries"])
        self.assertEqual(snapshot["batteries"]["lock.back_door"]["level"], 58)
        self.assertIn("sensor.hue_motion_battery", snapshot["batteries"])
        self.assertEqual(snapshot["batteries"]["sensor.hue_motion_battery"]["level"], 82.0)

    def test_extract_network(self):
        snapshot = self._snap()
        NetworkCollector().extract(snapshot, EXTENDED_STATES)
        self.assertEqual(snapshot["network"]["devices_home"], 2)
        self.assertEqual(snapshot["network"]["devices_away"], 1)
        self.assertEqual(snapshot["network"]["devices_unavailable"], 1)

    def test_extract_media(self):
        snapshot = self._snap()
        MediaCollector().extract(snapshot, EXTENDED_STATES)
        self.assertEqual(snapshot["media"]["total_active"], 1)
        self.assertIn("Living Room", snapshot["media"]["active_players"])

    def test_extract_sun(self):
        snapshot = self._snap()
        SunCollector().extract(snapshot, EXTENDED_STATES)
        self.assertEqual(snapshot["sun"]["sunrise"], "06:42")
        self.assertEqual(snapshot["sun"]["sunset"], "17:58")
        self.assertAlmostEqual(snapshot["sun"]["daylight_hours"], 11.27, places=1)
        self.assertEqual(snapshot["sun"]["solar_elevation"], 32.5)

    def test_extract_vacuum(self):
        snapshot = self._snap()
        VacuumCollector().extract(snapshot, EXTENDED_STATES)
        self.assertIn("Roborock", snapshot["vacuum"])
        self.assertEqual(snapshot["vacuum"]["Roborock"]["status"], "docked")
        self.assertEqual(snapshot["vacuum"]["Roborock"]["battery"], 100)


class TestSnapshotAssembly(unittest.TestCase):
    def test_build_snapshot_assembles_all_sections(self):
        import shutil
        import tempfile
        from unittest.mock import patch

        from aria.engine.collectors.snapshot import build_snapshot
        from aria.engine.config import AppConfig
        from aria.engine.storage.data_store import DataStore

        tmpdir = tempfile.mkdtemp()
        try:
            config = AppConfig()
            config.paths.data_dir = __import__("pathlib").Path(tmpdir) / "data"
            config.paths.logbook_path = __import__("pathlib").Path(tmpdir) / "current.json"
            store = DataStore(config.paths)
            store.ensure_dirs()

            with (
                patch("aria.engine.collectors.snapshot.fetch_ha_states", return_value=SAMPLE_STATES),
                patch("aria.engine.collectors.snapshot.fetch_weather", return_value="Clear +75°F 50% →5mph"),
                patch(
                    "aria.engine.collectors.snapshot.fetch_calendar_events",
                    return_value=[{"start": "2026-02-10T09:00:00", "end": "2026-02-10T10:00:00", "summary": "Meeting"}],
                ),
            ):
                snapshot = build_snapshot("2026-02-10", config=config, store=store)

            self.assertEqual(snapshot["weather"]["temp_f"], 75)
            self.assertEqual(len(snapshot["calendar_events"]), 1)
            self.assertGreater(snapshot["entities"]["total"], 0)
            self.assertGreater(snapshot["power"]["total_watts"], 0)
            self.assertGreater(len(snapshot["occupancy"]["people_home"]), 0)
        finally:
            shutil.rmtree(tmpdir)


class TestPresenceCollector(unittest.TestCase):
    def test_presence_collector_with_data(self):
        """PresenceCollector extracts summary from presence cache."""
        from aria.engine.collectors.registry import CollectorRegistry

        collector = CollectorRegistry.get("presence")()
        snapshot = {"date": "2026-02-16"}
        presence_cache = {
            "rooms": {
                "bedroom": {
                    "probability": 0.9,
                    "persons": [{"name": "Justin"}],
                    "signals": [{"type": "camera_person"}],
                },
                "kitchen": {"probability": 0.7, "persons": [], "signals": []},
                "driveway": {
                    "probability": 0.3,
                    "persons": [],
                    "signals": [{"type": "camera_person"}],
                },
            },
            "identified_persons": {"Justin": {"room": "bedroom"}},
        }
        collector.inject_presence(snapshot, presence_cache)
        assert "presence" in snapshot
        assert snapshot["presence"]["overall_probability"] == 0.9
        assert snapshot["presence"]["occupied_room_count"] == 2  # bedroom + kitchen > 0.5
        assert snapshot["presence"]["identified_person_count"] == 1
        assert snapshot["presence"]["camera_signal_count"] == 2

    def test_presence_collector_no_data(self):
        """PresenceCollector defaults to zeros when no cache available."""
        from aria.engine.collectors.registry import CollectorRegistry

        collector = CollectorRegistry.get("presence")()
        snapshot = {"date": "2026-02-16"}
        collector.inject_presence(snapshot, None)
        assert snapshot["presence"]["overall_probability"] == 0
        assert snapshot["presence"]["occupied_room_count"] == 0

    def test_presence_collector_via_extract_interface(self):
        """PresenceCollector works through standard extract() interface with kwargs."""
        from aria.engine.collectors.registry import CollectorRegistry

        collector = CollectorRegistry.get("presence")()
        snapshot = {"date": "2026-02-16"}
        presence_cache = {
            "rooms": {"living_room": {"probability": 0.8, "persons": [], "signals": []}},
            "identified_persons": {},
        }
        collector.extract(snapshot, [], presence_cache=presence_cache)
        assert snapshot["presence"]["overall_probability"] == 0.8
        assert snapshot["presence"]["occupied_room_count"] == 1

    def test_presence_collector_empty_rooms(self):
        """PresenceCollector handles empty rooms dict."""
        from aria.engine.collectors.registry import CollectorRegistry

        collector = CollectorRegistry.get("presence")()
        snapshot = {"date": "2026-02-16"}
        collector.inject_presence(snapshot, {"rooms": {}, "identified_persons": {}})
        assert snapshot["presence"]["overall_probability"] == 0
        assert snapshot["presence"]["occupied_room_count"] == 0


if __name__ == "__main__":
    unittest.main()
