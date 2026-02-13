"""Tests for Markov chain sequence anomaly detection."""

import unittest

from ha_intelligence.analysis.sequence_anomalies import MarkovChainDetector


def _make_entries(triples):
    """Build logbook entries from (entity_id, timestamp_str, state) triples."""
    return [{"entity_id": eid, "when": ts, "state": st} for eid, ts, st in triples]


class TestMarkovChainTraining(unittest.TestCase):

    def test_builds_transitions_from_consecutive_events(self):
        entries = _make_entries([
            ("light.kitchen", "2026-02-10T18:00:00+00:00", "on"),
            ("light.living_room", "2026-02-10T18:00:30+00:00", "on"),
            ("light.kitchen", "2026-02-10T18:01:00+00:00", "off"),
        ])
        detector = MarkovChainDetector(window_seconds=300)
        result = detector.train(entries)
        self.assertEqual(result["transitions"], 2)
        self.assertEqual(result["unique_entities"], 2)
        self.assertIn("light.living_room", detector.transition_counts["light.kitchen"])

    def test_ignores_events_outside_window(self):
        entries = _make_entries([
            ("light.kitchen", "2026-02-10T18:00:00+00:00", "on"),
            ("light.living_room", "2026-02-10T18:10:00+00:00", "on"),
        ])
        detector = MarkovChainDetector(window_seconds=300)
        result = detector.train(entries)
        self.assertEqual(result["transitions"], 0)

    def test_filters_non_trackable_entities(self):
        entries = _make_entries([
            ("sensor.temperature", "2026-02-10T18:00:00+00:00", "72"),
            ("sensor.humidity", "2026-02-10T18:00:30+00:00", "45"),
        ])
        detector = MarkovChainDetector(window_seconds=300)
        result = detector.train(entries)
        self.assertEqual(result["transitions"], 0)

    def test_insufficient_data_returns_status(self):
        entries = _make_entries([
            ("light.kitchen", "2026-02-10T18:00:00+00:00", "on"),
        ])
        detector = MarkovChainDetector(window_seconds=300)
        result = detector.train(entries)
        self.assertEqual(result["status"], "insufficient_data")

    def test_serialization_roundtrip(self):
        entries = _make_entries([
            ("light.kitchen", "2026-02-10T18:00:00+00:00", "on"),
            ("light.living_room", "2026-02-10T18:00:30+00:00", "on"),
        ] * 30)
        detector = MarkovChainDetector(window_seconds=300, min_transitions=5)
        detector.train(entries)
        data = detector.to_dict()
        restored = MarkovChainDetector.from_dict(data)
        self.assertEqual(restored.total_transitions, detector.total_transitions)
        self.assertEqual(restored.threshold, detector.threshold)
        self.assertEqual(
            dict(restored.transition_counts["light.kitchen"]),
            dict(detector.transition_counts["light.kitchen"]),
        )

    def test_trained_status_with_enough_data(self):
        entries = []
        for i in range(60):
            sec = i * 30
            eid = "light.kitchen" if i % 2 == 0 else "light.living_room"
            entries.append({
                "entity_id": eid,
                "when": f"2026-02-10T18:{sec // 60:02d}:{sec % 60:02d}+00:00",
                "state": "on",
            })
        detector = MarkovChainDetector(window_seconds=300, min_transitions=5)
        result = detector.train(entries)
        self.assertEqual(result["status"], "trained")
        self.assertIsNotNone(result["threshold"])


class TestMarkovChainDetection(unittest.TestCase):

    def _trained_detector(self):
        """Build a detector trained on a regular kitchen<->living_room pattern."""
        entries = []
        for i in range(200):
            sec = i * 30
            minute = sec // 60
            second = sec % 60
            if minute >= 60:
                hour = 18 + minute // 60
                minute = minute % 60
            else:
                hour = 18
            eid = "light.kitchen" if i % 2 == 0 else "light.living_room"
            entries.append({
                "entity_id": eid,
                "when": f"2026-02-10T{hour:02d}:{minute:02d}:{second:02d}+00:00",
                "state": "on",
            })
        detector = MarkovChainDetector(window_seconds=300, min_transitions=5)
        detector.train(entries)
        return detector

    def test_detect_returns_empty_when_untrained(self):
        detector = MarkovChainDetector()
        result = detector.detect([])
        self.assertEqual(result, [])

    def test_normal_sequence_not_flagged(self):
        detector = self._trained_detector()
        normal_entries = _make_entries([
            ("light.kitchen", "2026-02-11T18:00:00+00:00", "on"),
            ("light.living_room", "2026-02-11T18:00:30+00:00", "on"),
            ("light.kitchen", "2026-02-11T18:01:00+00:00", "off"),
            ("light.living_room", "2026-02-11T18:01:30+00:00", "off"),
            ("light.kitchen", "2026-02-11T18:02:00+00:00", "on"),
            ("light.living_room", "2026-02-11T18:02:30+00:00", "on"),
            ("light.kitchen", "2026-02-11T18:03:00+00:00", "off"),
            ("light.living_room", "2026-02-11T18:03:30+00:00", "off"),
            ("light.kitchen", "2026-02-11T18:04:00+00:00", "on"),
            ("light.living_room", "2026-02-11T18:04:30+00:00", "on"),
        ])
        anomalies = detector.detect(normal_entries)
        self.assertEqual(len(anomalies), 0)

    def test_novel_entity_sequence_flagged(self):
        detector = self._trained_detector()
        novel_entries = _make_entries([
            ("lock.back_door", "2026-02-11T03:00:00+00:00", "unlocked"),
            ("binary_sensor.motion_office", "2026-02-11T03:00:10+00:00", "on"),
            ("lock.back_door", "2026-02-11T03:00:20+00:00", "unlocked"),
            ("binary_sensor.motion_office", "2026-02-11T03:00:30+00:00", "on"),
            ("lock.back_door", "2026-02-11T03:00:40+00:00", "unlocked"),
            ("binary_sensor.motion_office", "2026-02-11T03:00:50+00:00", "on"),
            ("lock.back_door", "2026-02-11T03:01:00+00:00", "unlocked"),
            ("binary_sensor.motion_office", "2026-02-11T03:01:10+00:00", "on"),
            ("lock.back_door", "2026-02-11T03:01:20+00:00", "unlocked"),
            ("binary_sensor.motion_office", "2026-02-11T03:01:30+00:00", "on"),
        ])
        anomalies = detector.detect(novel_entries)
        self.assertGreater(len(anomalies), 0)
        self.assertIn("score", anomalies[0])
        self.assertIn("entities", anomalies[0])

    def test_detect_returns_time_range(self):
        detector = self._trained_detector()
        novel_entries = _make_entries([
            ("lock.back_door", "2026-02-11T03:00:00+00:00", "unlocked"),
            ("binary_sensor.motion_office", "2026-02-11T03:00:10+00:00", "on"),
            ("lock.back_door", "2026-02-11T03:00:20+00:00", "unlocked"),
            ("binary_sensor.motion_office", "2026-02-11T03:00:30+00:00", "on"),
            ("lock.back_door", "2026-02-11T03:00:40+00:00", "unlocked"),
            ("binary_sensor.motion_office", "2026-02-11T03:00:50+00:00", "on"),
            ("lock.back_door", "2026-02-11T03:01:00+00:00", "unlocked"),
            ("binary_sensor.motion_office", "2026-02-11T03:01:10+00:00", "on"),
            ("lock.back_door", "2026-02-11T03:01:20+00:00", "unlocked"),
            ("binary_sensor.motion_office", "2026-02-11T03:01:30+00:00", "on"),
        ])
        anomalies = detector.detect(novel_entries)
        if anomalies:
            self.assertIn("time_start", anomalies[0])
            self.assertIn("time_end", anomalies[0])
            self.assertIn("severity", anomalies[0])


class TestSummarizeSequenceAnomalies(unittest.TestCase):

    def test_summarize_returns_overview(self):
        from ha_intelligence.analysis.sequence_anomalies import summarize_sequence_anomalies
        anomalies = [
            {"time_start": "2026-02-11T03:00:00", "time_end": "2026-02-11T03:01:30",
             "score": -5.2, "threshold": -3.0, "severity": "high",
             "entities": ["lock.back_door", "binary_sensor.motion_office"]},
        ]
        summary = summarize_sequence_anomalies(anomalies, total_windows_checked=20)
        self.assertEqual(summary["anomalies_found"], 1)
        self.assertEqual(summary["total_windows_checked"], 20)
        self.assertIn("lock.back_door", summary["involved_entities"])


if __name__ == "__main__":
    unittest.main()
