"""Tests for analysis: baselines, correlations, anomalies, reliability."""

import unittest

from ha_intelligence.config import HolidayConfig
from ha_intelligence.collectors.snapshot import build_empty_snapshot
from ha_intelligence.analysis.baselines import compute_baselines
from ha_intelligence.analysis.correlations import pearson_r, cross_correlate
from ha_intelligence.analysis.reliability import compute_device_reliability
from ha_intelligence.analysis.anomalies import detect_anomalies

from conftest import make_snapshot


class TestBaselines(unittest.TestCase):
    def test_compute_baselines_groups_by_day_of_week(self):
        snapshots = [
            make_snapshot("2026-02-03", power=100),  # Tuesday
            make_snapshot("2026-02-10", power=200),  # Tuesday
            make_snapshot("2026-02-04", power=150),  # Wednesday
        ]
        baselines = compute_baselines(snapshots)
        self.assertIn("Tuesday", baselines)
        self.assertAlmostEqual(baselines["Tuesday"]["power_watts"]["mean"], 150.0)
        self.assertIn("Wednesday", baselines)

    def test_baseline_includes_stddev(self):
        snapshots = [
            make_snapshot("2026-02-03", power=100),
            make_snapshot("2026-02-10", power=200),
        ]
        baselines = compute_baselines(snapshots)
        self.assertGreater(baselines["Tuesday"]["power_watts"]["stddev"], 0)


class TestCorrelation(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r = pearson_r(x, y)
        self.assertAlmostEqual(r, 1.0, places=5)

    def test_no_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 1, 4, 2, 3]
        r = pearson_r(x, y)
        self.assertLess(abs(r), 0.5)

    def test_cross_correlate_finds_weather_power_link(self):
        snapshots = []
        for i, temp in enumerate([60, 70, 80, 90, 95, 65, 75, 85, 92, 88]):
            snap = build_empty_snapshot(f"2026-02-{i+1:02d}", HolidayConfig())
            snap["weather"]["temp_f"] = temp
            snap["power"]["total_watts"] = 100 + (temp - 60) * 3
            snap["lights"]["on"] = 30
            snap["occupancy"]["device_count_home"] = 50
            snap["entities"]["unavailable"] = 900
            snap["logbook_summary"]["useful_events"] = 2500
            snapshots.append(snap)

        corrs = cross_correlate(snapshots)
        temp_power = [c for c in corrs if c["x"] == "weather_temp" and c["y"] == "power_watts"]
        self.assertTrue(len(temp_power) > 0)
        self.assertGreater(temp_power[0]["r"], 0.9)


class TestDeviceReliability(unittest.TestCase):
    def test_reliability_score_decreases_with_more_outages(self):
        snapshots = []
        unavail_days = {"2026-02-04", "2026-02-06", "2026-02-09"}
        for i in range(7):
            date = f"2026-02-{4+i:02d}"
            snap = build_empty_snapshot(date, HolidayConfig())
            if date in unavail_days:
                snap["entities"]["unavailable_list"] = ["sensor.flaky_device"]
            else:
                snap["entities"]["unavailable_list"] = []
            snapshots.append(snap)

        scores = compute_device_reliability(snapshots)
        self.assertIn("sensor.flaky_device", scores)
        self.assertLess(scores["sensor.flaky_device"]["score"], 100)
        self.assertEqual(scores["sensor.flaky_device"]["outage_days"], 3)

    def test_healthy_device_gets_100_score(self):
        snapshots = []
        for i in range(7):
            snap = build_empty_snapshot(f"2026-02-{4+i:02d}", HolidayConfig())
            snap["entities"]["unavailable_list"] = []
            snapshots.append(snap)

        scores = compute_device_reliability(snapshots)
        for eid, data in scores.items():
            self.assertEqual(data["score"], 100)


class TestAnomalyDetection(unittest.TestCase):
    def test_detects_high_power_anomaly(self):
        baselines = {
            "Tuesday": {
                "power_watts": {"mean": 150, "stddev": 10},
                "lights_on": {"mean": 30, "stddev": 5},
                "devices_home": {"mean": 50, "stddev": 10},
                "unavailable": {"mean": 900, "stddev": 20},
                "useful_events": {"mean": 2500, "stddev": 300},
            }
        }
        snapshot = make_snapshot("2026-02-10", power=300)  # 15sigma above
        anomalies = detect_anomalies(snapshot, baselines)
        power_anomalies = [a for a in anomalies if a["metric"] == "power_watts"]
        self.assertTrue(len(power_anomalies) > 0)
        self.assertGreater(power_anomalies[0]["z_score"], 2.0)

    def test_no_anomaly_within_normal_range(self):
        baselines = {
            "Tuesday": {
                "power_watts": {"mean": 150, "stddev": 10},
                "lights_on": {"mean": 30, "stddev": 5},
                "devices_home": {"mean": 50, "stddev": 10},
                "unavailable": {"mean": 900, "stddev": 20},
                "useful_events": {"mean": 2500, "stddev": 300},
            }
        }
        snapshot = make_snapshot("2026-02-10", power=155)
        anomalies = detect_anomalies(snapshot, baselines)
        power_anomalies = [a for a in anomalies if a["metric"] == "power_watts"]
        self.assertEqual(len(power_anomalies), 0)


if __name__ == "__main__":
    unittest.main()
