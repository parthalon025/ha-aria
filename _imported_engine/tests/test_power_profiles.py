"""Tests for appliance power profiling and cycle detection."""

import unittest

from ha_intelligence.analysis.power_profiles import (
    ApplianceProfiler, ApplianceProfile, profile_correlation,
)


def _make_cycle_series(n_cycles=3, on_watts=100, off_watts=1,
                       cycle_points=5, gap_points=3):
    """Build a synthetic power time series with N on/off cycles."""
    series = []
    t = 0
    for i in range(n_cycles):
        # On phase
        for j in range(cycle_points):
            ts = f"2026-02-10T{10 + t // 60:02d}:{t % 60:02d}:00"
            series.append((ts, on_watts + (j * 5)))
            t += 5
        # Off phase
        for j in range(gap_points):
            ts = f"2026-02-10T{10 + t // 60:02d}:{t % 60:02d}:00"
            series.append((ts, off_watts))
            t += 5
    return series


class TestCycleDetection(unittest.TestCase):
    def test_detects_simple_cycles(self):
        profiler = ApplianceProfiler(on_threshold=5, off_threshold=2)
        series = _make_cycle_series(n_cycles=3, on_watts=100, off_watts=1)
        cycles = profiler.detect_cycles(series)
        self.assertEqual(len(cycles), 3)

    def test_cycle_has_expected_fields(self):
        profiler = ApplianceProfiler(on_threshold=5, off_threshold=2)
        series = _make_cycle_series(n_cycles=1, on_watts=80)
        cycles = profiler.detect_cycles(series)
        self.assertGreater(len(cycles), 0)
        cycle = cycles[0]
        self.assertIn("start", cycle)
        self.assertIn("end", cycle)
        self.assertIn("duration_minutes", cycle)
        self.assertIn("peak_watts", cycle)
        self.assertIn("avg_watts", cycle)
        self.assertIn("energy_wh", cycle)

    def test_no_cycles_below_threshold(self):
        profiler = ApplianceProfiler(on_threshold=50)
        # All readings below threshold
        series = [(f"2026-02-10T10:{i:02d}:00", 3.0) for i in range(10)]
        cycles = profiler.detect_cycles(series)
        self.assertEqual(len(cycles), 0)

    def test_empty_series(self):
        profiler = ApplianceProfiler()
        cycles = profiler.detect_cycles([])
        self.assertEqual(len(cycles), 0)

    def test_peak_watts_correct(self):
        profiler = ApplianceProfiler(on_threshold=5, off_threshold=2)
        series = [
            ("2026-02-10T10:00:00", 50),
            ("2026-02-10T10:05:00", 100),
            ("2026-02-10T10:10:00", 80),
            ("2026-02-10T10:15:00", 1),
        ]
        cycles = profiler.detect_cycles(series)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles[0]["peak_watts"], 100)


class TestProfileLearning(unittest.TestCase):
    def test_learn_profile_from_cycles(self):
        profiler = ApplianceProfiler(on_threshold=5, off_threshold=2)
        series = _make_cycle_series(n_cycles=4, on_watts=80)
        cycles = profiler.detect_cycles(series)
        profile = profiler.learn_profile("outlet_1", cycles)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "outlet_1")
        self.assertGreater(profile.peak_watts, 0)
        self.assertGreater(profile.typical_duration_minutes, 0)

    def test_insufficient_cycles_returns_none(self):
        profiler = ApplianceProfiler(on_threshold=5, off_threshold=2)
        series = _make_cycle_series(n_cycles=1, on_watts=80)
        cycles = profiler.detect_cycles(series)
        profile = profiler.learn_profile("outlet_1", cycles)
        self.assertIsNone(profile)

    def test_profile_serialization(self):
        profile = ApplianceProfile(
            name="test_outlet",
            reference_watts=[10.0, 50.0, 80.0, 50.0, 10.0],
            typical_duration_minutes=25.0,
            peak_watts=80.0,
            idle_watts=1.0,
        )
        data = profile.to_dict()
        restored = ApplianceProfile.from_dict(data)
        self.assertEqual(restored.name, "test_outlet")
        self.assertEqual(restored.peak_watts, 80.0)
        self.assertEqual(len(restored.reference_watts), 5)


class TestHealthAssessment(unittest.TestCase):
    def test_healthy_appliance_scores_high(self):
        profiler = ApplianceProfiler(on_threshold=5, off_threshold=2)
        # Learn a consistent profile
        series = _make_cycle_series(n_cycles=5, on_watts=100, off_watts=1)
        cycles = profiler.detect_cycles(series)
        profiler.learn_profile("outlet_1", cycles)

        # Assess with similar cycles
        health = profiler.assess_health("outlet_1", cycles)
        self.assertIsNotNone(health["score"])
        self.assertGreaterEqual(health["score"], 70)

    def test_no_profile_returns_insufficient(self):
        profiler = ApplianceProfiler()
        health = profiler.assess_health("unknown_outlet", [])
        self.assertIsNone(health["score"])
        self.assertIn("insufficient", health["reason"])

    def test_degraded_duration_triggers_alert(self):
        profiler = ApplianceProfiler(on_threshold=5, off_threshold=2)
        # Learn profile from consistent cycles
        series = _make_cycle_series(n_cycles=5, on_watts=100)
        cycles = profiler.detect_cycles(series)
        profiler.learn_profile("outlet_1", cycles)

        # Create cycles with much longer duration (simulate degradation)
        long_series = _make_cycle_series(n_cycles=3, on_watts=100,
                                          cycle_points=15)  # 3x longer cycles
        long_cycles = profiler.detect_cycles(long_series)

        health = profiler.assess_health("outlet_1", long_cycles)
        # Should have lower score or alert about duration
        self.assertTrue(
            health["score"] < 90 or len(health["alerts"]) > 0,
            f"Expected degradation signal: score={health['score']}, alerts={health['alerts']}"
        )


class TestSnapshotAnalysis(unittest.TestCase):
    def test_analyze_snapshot_outlets(self):
        profiler = ApplianceProfiler(on_threshold=5, off_threshold=2)
        snapshots = []
        for i in range(10):
            snap = {
                "power": {
                    "outlets": {
                        "NVR Outlet": 35.0 + (i % 3),
                        "Empty Outlet": 0.0,
                    }
                }
            }
            snapshots.append((f"2026-02-{i+1:02d}", snap))

        result = profiler.analyze_snapshot_outlets(snapshots)
        self.assertIn("NVR Outlet", result["outlets"])
        self.assertIn("Empty Outlet", result["outlets"])
        self.assertTrue(result["outlets"]["NVR Outlet"]["is_active"])
        self.assertFalse(result["outlets"]["Empty Outlet"]["is_active"])
        self.assertGreater(result["active_count"], 0)


class TestProfileCorrelation(unittest.TestCase):
    def test_identical_profiles_correlation_1(self):
        ref = [10.0, 50.0, 80.0, 50.0, 10.0]
        a = ApplianceProfile("a", reference_watts=ref)
        b = ApplianceProfile("b", reference_watts=ref)
        corr = profile_correlation(a, b)
        self.assertAlmostEqual(corr, 1.0, places=2)

    def test_inverse_profiles_negative_correlation(self):
        a = ApplianceProfile("a", reference_watts=[10, 20, 30, 40, 50])
        b = ApplianceProfile("b", reference_watts=[50, 40, 30, 20, 10])
        corr = profile_correlation(a, b)
        self.assertLess(corr, 0)

    def test_empty_profiles_zero(self):
        a = ApplianceProfile("a", reference_watts=[])
        b = ApplianceProfile("b", reference_watts=[])
        corr = profile_correlation(a, b)
        self.assertEqual(corr, 0.0)


if __name__ == "__main__":
    unittest.main()
