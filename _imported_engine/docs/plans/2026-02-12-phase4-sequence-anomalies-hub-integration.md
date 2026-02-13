# Phase 4: Sequence Anomaly Detection + Hub Integration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Markov chain-based sequence anomaly detection to the HA Intelligence engine, then integrate all Phase 2-4 outputs into the Intelligence Hub dashboard.

**Architecture:** A `MarkovChainDetector` class learns entity event transition probabilities from logbook data, then flags unusual event sequences (e.g., front door opens at 3am → motion in office → no lights). Hub integration adds reads for entity correlations, automation suggestions, power profiles, and sequence anomalies to the existing `intelligence.py` module.

**Tech Stack:** Python stdlib (collections, math), existing DataStore/PathConfig patterns, existing entity_correlations filtering utilities. No new dependencies.

---

## Context for Implementer

### Project locations
- **Engine:** `~/Documents/projects/ha-intelligence/`
- **Hub:** `~/Documents/projects/ha-intelligence-hub/`

### Key patterns to follow
- **Lazy imports in CLI:** Each `cmd_*()` function imports at top of function body, not module level
- **Mock at import site:** `@patch("ha_intelligence.analysis.sequence_anomalies.func")`, not definition site
- **DataStore for all I/O:** No direct file reads/writes in business logic
- **PathConfig for all paths:** Add new path properties, reference via `self.paths.xxx`
- **Entity filtering:** Reuse `TRACKABLE_DOMAINS`, `EXCLUDED_PATTERNS`, `_is_trackable()`, `_parse_timestamp()` from `entity_correlations.py` — import, don't duplicate
- **Test fixtures:** Use `tmp_paths`, `store`, `app_config` from `conftest.py`
- **Tests run with:** `python3 -m pytest tests/ -v` (uses Homebrew Python 3.14)

### Data shape: Logbook entries
Loaded via `store.load_logbook()` — returns list of dicts:
```python
{"entity_id": "light.kitchen", "when": "2026-02-10T18:00:30+00:00", "state": "on"}
```

---

## Task 1: MarkovChainDetector — Core Class + Training

**Files:**
- Create: `ha_intelligence/analysis/sequence_anomalies.py`
- Create: `tests/test_sequence_anomalies.py`

**Step 1: Write the failing tests**

```python
# tests/test_sequence_anomalies.py
"""Tests for Markov chain sequence anomaly detection."""

import unittest

from ha_intelligence.analysis.sequence_anomalies import MarkovChainDetector


def _make_entries(triples):
    """Build logbook entries from (entity_id, timestamp_str, state) triples."""
    return [{"entity_id": eid, "when": ts, "state": st} for eid, ts, st in triples]


class TestMarkovChainTraining(unittest.TestCase):
    """Tests for transition matrix building."""

    def test_builds_transitions_from_consecutive_events(self):
        """Consecutive events within window should create transitions."""
        entries = _make_entries([
            ("light.kitchen", "2026-02-10T18:00:00+00:00", "on"),
            ("light.living_room", "2026-02-10T18:00:30+00:00", "on"),
            ("light.kitchen", "2026-02-10T18:01:00+00:00", "off"),
        ])
        detector = MarkovChainDetector(window_seconds=300)
        result = detector.train(entries)

        self.assertEqual(result["transitions"], 2)
        self.assertEqual(result["unique_entities"], 2)
        # kitchen → living_room transition should exist
        self.assertIn("light.living_room", detector.transition_counts["light.kitchen"])

    def test_ignores_events_outside_window(self):
        """Events separated by more than window_seconds should not create transitions."""
        entries = _make_entries([
            ("light.kitchen", "2026-02-10T18:00:00+00:00", "on"),
            ("light.living_room", "2026-02-10T18:10:00+00:00", "on"),  # 10 min later
        ])
        detector = MarkovChainDetector(window_seconds=300)  # 5 min window
        result = detector.train(entries)

        self.assertEqual(result["transitions"], 0)

    def test_filters_non_trackable_entities(self):
        """Sensor entities (not in TRACKABLE_DOMAINS) should be filtered out."""
        entries = _make_entries([
            ("sensor.temperature", "2026-02-10T18:00:00+00:00", "72"),
            ("sensor.humidity", "2026-02-10T18:00:30+00:00", "45"),
        ])
        detector = MarkovChainDetector(window_seconds=300)
        result = detector.train(entries)

        self.assertEqual(result["transitions"], 0)

    def test_insufficient_data_returns_status(self):
        """Too few events should report insufficient_data status."""
        entries = _make_entries([
            ("light.kitchen", "2026-02-10T18:00:00+00:00", "on"),
        ])
        detector = MarkovChainDetector(window_seconds=300)
        result = detector.train(entries)

        self.assertEqual(result["status"], "insufficient_data")

    def test_serialization_roundtrip(self):
        """to_dict() → from_dict() should preserve detector state."""
        entries = _make_entries([
            ("light.kitchen", "2026-02-10T18:00:00+00:00", "on"),
            ("light.living_room", "2026-02-10T18:00:30+00:00", "on"),
        ] * 30)  # repeat for enough transitions
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
        """Should report 'trained' when enough transitions exist."""
        # Generate enough transitions (>= min_transitions)
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


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_sequence_anomalies.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ha_intelligence.analysis.sequence_anomalies'`

**Step 3: Write the implementation**

```python
# ha_intelligence/analysis/sequence_anomalies.py
"""Markov chain-based sequence anomaly detection.

Learns entity state-change transition probabilities from logbook events.
Flags sequences with unusually low transition probability as anomalous.
Lightweight alternative to LSTM-Autoencoder — zero new dependencies.
"""

import math
from collections import defaultdict

from ha_intelligence.analysis.entity_correlations import (
    _is_trackable,
    _parse_timestamp,
)


class MarkovChainDetector:
    """First-order Markov chain for entity event sequence anomaly detection.

    Learns P(entity_B fires | entity_A just fired) from logbook history,
    then scores new event windows by average log-probability of transitions.
    Windows scoring below the 5th percentile of training data are anomalous.
    """

    def __init__(self, window_seconds=300, min_transitions=50):
        self.window_seconds = window_seconds
        self.min_transitions = min_transitions
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.entity_counts = defaultdict(int)
        self.total_transitions = 0
        self.threshold = None

    def train(self, logbook_entries):
        """Build transition matrix from historical logbook events.

        Args:
            logbook_entries: List of dicts with entity_id, when, state keys.

        Returns:
            Dict with transitions, unique_entities, threshold, status.
        """
        events = self._filter_and_sort(logbook_entries)

        if len(events) < 2:
            return {"transitions": 0, "unique_entities": 0,
                    "threshold": None, "status": "insufficient_data"}

        # Build transition counts from consecutive events within window
        for i in range(len(events) - 1):
            ts_a, eid_a = events[i]
            ts_b, eid_b = events[i + 1]

            gap = (ts_b - ts_a).total_seconds()
            if gap > self.window_seconds:
                continue

            self.transition_counts[eid_a][eid_b] += 1
            self.entity_counts[eid_a] += 1
            self.total_transitions += 1

        # Compute anomaly threshold from training data
        if self.total_transitions >= self.min_transitions:
            scores = self._score_training_windows(events)
            if scores:
                scores.sort()
                idx = max(0, int(len(scores) * 0.05))
                self.threshold = scores[idx]

        return {
            "transitions": self.total_transitions,
            "unique_entities": len(self.entity_counts),
            "threshold": self.threshold,
            "status": "trained" if self.total_transitions >= self.min_transitions
                      else "insufficient_data",
        }

    def _filter_and_sort(self, logbook_entries):
        """Filter to trackable entities, parse timestamps, sort by time."""
        events = []
        for entry in logbook_entries:
            eid = entry.get("entity_id", "")
            if not _is_trackable(eid):
                continue
            ts = _parse_timestamp(entry.get("when", ""))
            if ts is None:
                continue
            events.append((ts, eid))
        events.sort(key=lambda x: x[0])
        return events

    def _score_training_windows(self, events, window_size=10):
        """Score sliding windows of training data to establish baseline."""
        scores = []
        step = max(1, window_size // 2)
        for i in range(0, len(events) - window_size + 1, step):
            window = events[i:i + window_size]
            score = self._score_window(window)
            if score is not None:
                scores.append(score)
        return scores

    def _score_window(self, events):
        """Score a window of events by average log transition probability."""
        if len(events) < 2:
            return None

        log_probs = []
        for i in range(len(events) - 1):
            ts_a, eid_a = events[i]
            ts_b, eid_b = events[i + 1]

            gap = (ts_b - ts_a).total_seconds()
            if gap > self.window_seconds:
                continue

            total = self.entity_counts.get(eid_a, 0)
            if total == 0:
                continue

            count = self.transition_counts.get(eid_a, {}).get(eid_b, 0)
            if count == 0:
                # Unseen transition — Laplace smoothing
                prob = 1.0 / (total + len(self.entity_counts))
            else:
                prob = count / total

            log_probs.append(math.log(prob))

        if not log_probs:
            return None
        return sum(log_probs) / len(log_probs)

    def to_dict(self):
        """Serialize detector state for JSON storage."""
        return {
            "transition_counts": {k: dict(v) for k, v in self.transition_counts.items()},
            "entity_counts": dict(self.entity_counts),
            "total_transitions": self.total_transitions,
            "threshold": self.threshold,
            "window_seconds": self.window_seconds,
            "min_transitions": self.min_transitions,
        }

    @classmethod
    def from_dict(cls, data):
        """Restore detector from serialized dict."""
        detector = cls(
            window_seconds=data.get("window_seconds", 300),
            min_transitions=data.get("min_transitions", 50),
        )
        for k, v in data.get("transition_counts", {}).items():
            for k2, count in v.items():
                detector.transition_counts[k][k2] = count
        detector.entity_counts = defaultdict(int, data.get("entity_counts", {}))
        detector.total_transitions = data.get("total_transitions", 0)
        detector.threshold = data.get("threshold")
        return detector
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_sequence_anomalies.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add ha_intelligence/analysis/sequence_anomalies.py tests/test_sequence_anomalies.py
git commit -m "feat: add MarkovChainDetector core class + training tests"
```

---

## Task 2: Sequence Anomaly Detection + Scoring

**Files:**
- Modify: `ha_intelligence/analysis/sequence_anomalies.py` (add `detect()` method)
- Modify: `tests/test_sequence_anomalies.py` (add detection tests)

**Step 1: Write the failing tests**

Add to `tests/test_sequence_anomalies.py`:

```python
class TestMarkovChainDetection(unittest.TestCase):
    """Tests for anomaly detection on new event sequences."""

    def _trained_detector(self):
        """Build a detector trained on a regular kitchen↔living_room pattern."""
        entries = []
        for i in range(200):
            sec = i * 30
            minute = sec // 60
            second = sec % 60
            if minute >= 60:
                # wrap to next hour
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
        """Untrained detector should return no anomalies."""
        detector = MarkovChainDetector()
        result = detector.detect([])
        self.assertEqual(result, [])

    def test_normal_sequence_not_flagged(self):
        """Known normal pattern should not be flagged as anomalous."""
        detector = self._trained_detector()
        # Normal: alternating kitchen and living_room
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
        """Sequence with unseen entities should score low and be flagged."""
        detector = self._trained_detector()
        # Introduce entities never seen in training
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
        """Anomaly results should include time_start and time_end."""
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

    def test_summarize_returns_overview(self):
        """summarize_sequence_anomalies should produce a structured summary."""
        anomalies = [
            {"time_start": "2026-02-11T03:00:00", "time_end": "2026-02-11T03:01:30",
             "score": -5.2, "threshold": -3.0, "severity": "high",
             "entities": ["lock.back_door", "binary_sensor.motion_office"]},
        ]
        from ha_intelligence.analysis.sequence_anomalies import summarize_sequence_anomalies
        summary = summarize_sequence_anomalies(anomalies, total_windows_checked=20)

        self.assertEqual(summary["anomalies_found"], 1)
        self.assertEqual(summary["total_windows_checked"], 20)
        self.assertIn("lock.back_door", summary["involved_entities"])
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_sequence_anomalies.py::TestMarkovChainDetection -v`
Expected: FAIL (methods not yet implemented)

**Step 3: Write the implementation**

Add to `ha_intelligence/analysis/sequence_anomalies.py`:

```python
    # Add these methods to MarkovChainDetector class:

    def detect(self, logbook_entries, window_size=10, step=5):
        """Detect anomalous event sequences in recent logbook data.

        Args:
            logbook_entries: Recent logbook events.
            window_size: Events per sliding window.
            step: Window slide step.

        Returns:
            List of anomaly dicts: time_start, time_end, score, threshold,
            severity, entities.
        """
        if self.threshold is None:
            return []

        events = self._filter_and_sort(logbook_entries)
        if len(events) < 3:
            return []

        anomalies = []
        for i in range(0, max(1, len(events) - window_size + 1), step):
            window = events[i:i + window_size]
            if len(window) < 3:
                continue

            score = self._score_window(window)
            if score is not None and score < self.threshold:
                entities = list(set(eid for _, eid in window))
                anomalies.append({
                    "time_start": window[0][0].isoformat(),
                    "time_end": window[-1][0].isoformat(),
                    "score": round(score, 4),
                    "threshold": round(self.threshold, 4),
                    "severity": "high" if score < self.threshold * 1.5 else "medium",
                    "entities": entities,
                })

        return anomalies


# Module-level function:

def summarize_sequence_anomalies(anomalies, total_windows_checked=0):
    """Produce a summary dict suitable for storage and LLM context.

    Args:
        anomalies: List of anomaly dicts from detect().
        total_windows_checked: How many windows were scored.

    Returns:
        Summary dict with counts, entities, and severity breakdown.
    """
    all_entities = set()
    severity_counts = {"high": 0, "medium": 0}
    for a in anomalies:
        all_entities.update(a.get("entities", []))
        sev = a.get("severity", "medium")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    return {
        "anomalies_found": len(anomalies),
        "total_windows_checked": total_windows_checked,
        "involved_entities": sorted(all_entities),
        "severity_breakdown": severity_counts,
        "anomalies": anomalies[:10],  # Keep top 10 for storage
    }
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_sequence_anomalies.py -v`
Expected: 11 passed (6 from Task 1 + 5 new)

**Step 5: Run full test suite to verify no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: All 164 passed (153 existing + 11 new)

**Step 6: Commit**

```bash
git add ha_intelligence/analysis/sequence_anomalies.py tests/test_sequence_anomalies.py
git commit -m "feat: add Markov chain sequence anomaly detection + scoring"
```

---

## Task 3: DataStore + PathConfig Integration

**Files:**
- Modify: `ha_intelligence/config.py:74` (add `sequence_model_path` property)
- Modify: `ha_intelligence/storage/data_store.py:216` (add sequence methods)
- Modify: `tests/test_storage.py` (add round-trip tests)

**Step 1: Write the failing tests**

Add to `tests/test_storage.py`:

```python
class TestSequenceStorage(unittest.TestCase):
    """DataStore methods for sequence anomaly model and results."""

    def test_save_load_sequence_model_roundtrip(self):
        """Saving and loading sequence model should preserve data."""
        store = self._make_store()
        model_data = {
            "transition_counts": {"light.kitchen": {"light.living_room": 50}},
            "entity_counts": {"light.kitchen": 50},
            "total_transitions": 50,
            "threshold": -2.5,
            "window_seconds": 300,
            "min_transitions": 50,
        }
        store.save_sequence_model(model_data)
        loaded = store.load_sequence_model()
        self.assertEqual(loaded["total_transitions"], 50)
        self.assertEqual(loaded["threshold"], -2.5)

    def test_load_sequence_model_returns_none_when_missing(self):
        """Should return None when no model has been saved."""
        store = self._make_store()
        self.assertIsNone(store.load_sequence_model())

    def test_save_load_sequence_anomalies_roundtrip(self):
        """Saving and loading anomaly results should preserve data."""
        store = self._make_store()
        anomalies = {
            "anomalies_found": 2,
            "total_windows_checked": 100,
            "involved_entities": ["lock.back_door"],
            "anomalies": [{"score": -5.2, "severity": "high"}],
        }
        store.save_sequence_anomalies(anomalies)
        loaded = store.load_sequence_anomalies()
        self.assertEqual(loaded["anomalies_found"], 2)

    def _make_store(self):
        import tempfile
        from ha_intelligence.config import PathConfig
        from ha_intelligence.storage.data_store import DataStore
        tmp = tempfile.mkdtemp()
        paths = PathConfig(data_dir=Path(tmp) / "intelligence")
        paths.ensure_dirs()
        return DataStore(paths)
```

Note: The implementer should integrate this into the existing test_storage.py file structure, using whatever pattern exists there (pytest fixtures or unittest setUp).

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_storage.py -v -k sequence`
Expected: FAIL (methods don't exist on DataStore)

**Step 3: Add PathConfig property**

In `ha_intelligence/config.py`, add after the `capabilities_path` property (line ~77):

```python
    @property
    def sequence_model_path(self) -> Path:
        return self.models_dir / "sequence_model.json"
```

**Step 4: Add DataStore methods**

In `ha_intelligence/storage/data_store.py`, add after `load_logbook()` (line ~216):

```python
    # --- Sequence Anomaly Detection ---

    def save_sequence_model(self, model_data: dict):
        """Save trained Markov chain model."""
        self.ensure_dirs()
        with open(self.paths.sequence_model_path, "w") as f:
            json.dump(model_data, f, indent=2)

    def load_sequence_model(self) -> dict | None:
        """Load trained Markov chain model. Returns None if not yet trained."""
        if not self.paths.sequence_model_path.is_file():
            return None
        with open(self.paths.sequence_model_path) as f:
            return json.load(f)

    def save_sequence_anomalies(self, summary: dict):
        """Save sequence anomaly detection results."""
        self.ensure_dirs()
        path = self.paths.data_dir / "sequence_anomalies.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def load_sequence_anomalies(self) -> dict | None:
        """Load sequence anomaly results. Returns None if none exist."""
        path = self.paths.data_dir / "sequence_anomalies.json"
        if not path.is_file():
            return None
        with open(path) as f:
            return json.load(f)
```

**Step 5: Run tests**

Run: `python3 -m pytest tests/test_storage.py -v`
Expected: All storage tests pass

**Step 6: Run full suite**

Run: `python3 -m pytest tests/ -q`
Expected: All passing

**Step 7: Commit**

```bash
git add ha_intelligence/config.py ha_intelligence/storage/data_store.py tests/test_storage.py
git commit -m "feat: add DataStore methods for sequence model + anomaly persistence"
```

---

## Task 4: CLI Integration

**Files:**
- Modify: `ha_intelligence/cli.py:24` (add to docstring)
- Modify: `ha_intelligence/cli.py:545` (add cmd functions before `cmd_full`)
- Modify: `ha_intelligence/cli.py:584` (add to dispatcher)
- Modify: `tests/test_cli.py` (add CLI dispatch tests)

**Step 1: Write the failing tests**

Add to `tests/test_cli.py` (match existing test patterns in that file):

```python
class TestSequenceAnomalyCLI(unittest.TestCase):
    """CLI commands for sequence anomaly detection."""

    @patch("ha_intelligence.cli._init")
    def test_train_sequences_loads_logbook(self, mock_init):
        """--train-sequences should load logbook and train detector."""
        config, store = _mock_config_store(mock_init)
        store.load_logbook.return_value = [
            {"entity_id": "light.kitchen", "when": f"2026-02-10T18:{i:02d}:00+00:00", "state": "on"}
            for i in range(60)
        ]
        from ha_intelligence.cli import cmd_train_sequences
        result = cmd_train_sequences()
        store.load_logbook.assert_called_once()
        store.save_sequence_model.assert_called_once()

    @patch("ha_intelligence.cli._init")
    def test_sequence_anomalies_loads_model_and_logbook(self, mock_init):
        """--sequence-anomalies should load model and detect on recent logbook."""
        config, store = _mock_config_store(mock_init)
        store.load_sequence_model.return_value = {
            "transition_counts": {"light.kitchen": {"light.living_room": 50}},
            "entity_counts": {"light.kitchen": 50},
            "total_transitions": 50,
            "threshold": -2.5,
            "window_seconds": 300,
            "min_transitions": 50,
        }
        store.load_logbook.return_value = [
            {"entity_id": "light.kitchen", "when": "2026-02-10T18:00:00+00:00", "state": "on"},
        ]
        from ha_intelligence.cli import cmd_sequence_anomalies
        result = cmd_sequence_anomalies()
        store.load_sequence_model.assert_called_once()
```

Note: The implementer should adapt `_mock_config_store` to match whatever mock pattern exists in test_cli.py. If test_cli.py doesn't have CLI tests for other commands, model after the pattern in test_llm.py or test_drift.py.

**Step 2: Implement CLI commands**

Add to `ha_intelligence/cli.py` docstring (line 23):
```
  ha-intelligence --train-sequences     # Train Markov chain from logbook events
  ha-intelligence --sequence-anomalies  # Detect anomalous event sequences
```

Add two command functions before `cmd_full()` (around line 545):

```python
def cmd_train_sequences():
    """Train Markov chain sequence model from logbook data."""
    config, store = _init()

    from ha_intelligence.analysis.sequence_anomalies import MarkovChainDetector

    entries = store.load_logbook()
    if not entries:
        print("No logbook data available.")
        return None

    detector = MarkovChainDetector(window_seconds=300, min_transitions=50)
    result = detector.train(entries)

    store.save_sequence_model(detector.to_dict())
    print(f"Sequence model: {result['status']} "
          f"({result['transitions']} transitions, "
          f"{result['unique_entities']} entities)")
    if result["threshold"] is not None:
        print(f"  Anomaly threshold: {result['threshold']:.4f}")
    return result


def cmd_sequence_anomalies():
    """Detect anomalous event sequences using trained Markov chain."""
    config, store = _init()

    from ha_intelligence.analysis.sequence_anomalies import (
        MarkovChainDetector,
        summarize_sequence_anomalies,
    )

    model_data = store.load_sequence_model()
    if not model_data:
        print("No sequence model found. Run --train-sequences first.")
        return None

    detector = MarkovChainDetector.from_dict(model_data)
    if detector.threshold is None:
        print("Sequence model not fully trained (no threshold). Need more data.")
        return None

    entries = store.load_logbook()
    if not entries:
        print("No logbook data available.")
        return None

    anomalies = detector.detect(entries)
    summary = summarize_sequence_anomalies(
        anomalies, total_windows_checked=max(1, len(entries) // 5))
    store.save_sequence_anomalies(summary)

    print(f"Sequence anomalies: {summary['anomalies_found']} found "
          f"({summary['total_windows_checked']} windows checked)")
    for a in anomalies[:5]:
        print(f"  {a['time_start']} → {a['time_end']}: "
              f"score={a['score']} ({a['severity']})")
        print(f"    entities: {', '.join(a['entities'][:5])}")
    return summary
```

Add to dispatcher in `main()` (after `--power-profiles` block, before `--retrain`):

```python
    elif "--train-sequences" in args:
        cmd_train_sequences()
    elif "--sequence-anomalies" in args:
        cmd_sequence_anomalies()
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/ -v`
Expected: All passing

**Step 4: Commit**

```bash
git add ha_intelligence/cli.py tests/test_cli.py
git commit -m "feat: add --train-sequences and --sequence-anomalies CLI commands"
```

---

## Task 5: Hub Integration — Read Engine Phase 2-4 Outputs

**Project:** `~/Documents/projects/ha-intelligence-hub/` (separate repo)

**Files:**
- Modify: `modules/intelligence.py:153-175` (add new fields to cache payload)
- Modify: `tests/test_intelligence.py` (add tests for new data reads)

**Step 1: Write the failing tests**

Add to `tests/test_intelligence.py` (match existing test patterns):

```python
class TestPhase2To4DataReads(unittest.TestCase):
    """Hub should read Phase 2-4 engine outputs."""

    def test_reads_entity_correlations(self):
        """entity_correlations.json should be included in cache payload."""
        # Write entity_correlations.json to test intel_dir
        intel_dir = self._setup_intel_dir()
        corr = {"top_co_occurrences": [], "total_pairs_found": 5}
        (intel_dir / "entity_correlations.json").write_text(json.dumps(corr))

        module = self._create_module(intel_dir)
        data = module._read_intelligence_data()
        self.assertEqual(data["entity_correlations"]["total_pairs_found"], 5)

    def test_reads_sequence_anomalies(self):
        """sequence_anomalies.json should be included in cache payload."""
        intel_dir = self._setup_intel_dir()
        anomalies = {"anomalies_found": 2, "involved_entities": ["lock.back_door"]}
        (intel_dir / "sequence_anomalies.json").write_text(json.dumps(anomalies))

        module = self._create_module(intel_dir)
        data = module._read_intelligence_data()
        self.assertEqual(data["sequence_anomalies"]["anomalies_found"], 2)

    def test_reads_power_profiles(self):
        """insights/power-profiles.json should be included in cache payload."""
        intel_dir = self._setup_intel_dir()
        insights_dir = intel_dir / "insights"
        insights_dir.mkdir(exist_ok=True)
        profiles = {"active_count": 3, "profiles_learned": 2}
        (insights_dir / "power-profiles.json").write_text(json.dumps(profiles))

        module = self._create_module(intel_dir)
        data = module._read_intelligence_data()
        self.assertEqual(data["power_profiles"]["active_count"], 3)

    def test_reads_automation_suggestions(self):
        """Latest automation suggestion file should be included."""
        intel_dir = self._setup_intel_dir()
        suggestions_dir = intel_dir / "insights" / "automation-suggestions"
        suggestions_dir.mkdir(parents=True, exist_ok=True)
        suggestion = {"suggestions": [{"description": "Turn on lights with motion"}]}
        (suggestions_dir / "2026-02-12.json").write_text(json.dumps(suggestion))

        module = self._create_module(intel_dir)
        data = module._read_intelligence_data()
        self.assertEqual(len(data["automation_suggestions"]["suggestions"]), 1)

    def test_missing_files_return_none(self):
        """Missing Phase 2-4 files should return None, not error."""
        intel_dir = self._setup_intel_dir()
        module = self._create_module(intel_dir)
        data = module._read_intelligence_data()
        self.assertIsNone(data.get("entity_correlations"))
        self.assertIsNone(data.get("sequence_anomalies"))
        self.assertIsNone(data.get("power_profiles"))
        self.assertIsNone(data.get("automation_suggestions"))
```

Note: The implementer must adapt `_setup_intel_dir` and `_create_module` helpers to match whatever test setup exists in `test_intelligence.py`.

**Step 2: Implement in intelligence.py**

Add four new reads to `_read_intelligence_data()` in `modules/intelligence.py` (inside the return dict, after line 174):

```python
            "entity_correlations": self._read_json(self.intel_dir / "entity_correlations.json"),
            "sequence_anomalies": self._read_json(self.intel_dir / "sequence_anomalies.json"),
            "power_profiles": self._read_json(self.intel_dir / "insights" / "power-profiles.json"),
            "automation_suggestions": self._read_latest_automation_suggestion(),
```

Add new method:

```python
    def _read_latest_automation_suggestion(self) -> Optional[Dict[str, Any]]:
        """Read the most recent automation suggestion file."""
        suggestions_dir = self.intel_dir / "insights" / "automation-suggestions"
        if not suggestions_dir.exists():
            return None
        files = sorted(suggestions_dir.glob("*.json"))
        if not files:
            # Also check for .yaml files
            files = sorted(suggestions_dir.glob("*.yaml"))
        if not files:
            return None
        try:
            return json.loads(files[-1].read_text())
        except Exception as e:
            self.logger.debug(f"Failed to read automation suggestion: {e}")
            return None
```

**Step 3: Run hub tests**

Run: `cd ~/Documents/projects/ha-intelligence-hub && .venv/bin/python -m pytest tests/test_intelligence.py -v`
Expected: All passing

**Step 4: Commit**

```bash
cd ~/Documents/projects/ha-intelligence-hub
git add modules/intelligence.py tests/test_intelligence.py
git commit -m "feat: read Phase 2-4 engine outputs (correlations, anomalies, power, automations)"
```

---

## Task 6: Update CLAUDE.md + Cron Entry

**Files:**
- Modify: `~/Documents/projects/ha-intelligence/CLAUDE.md` (add new CLI commands, update structure)
- Modify: crontab (add weekly --train-sequences + --sequence-anomalies)

**Step 1: Update CLAUDE.md**

Add to the CLI Commands table:
```
| `--train-sequences` | Train Markov chain sequence model from logbook |
| `--sequence-anomalies` | Detect anomalous event sequences |
```

Add to Structure section under `analysis/`:
```
    sequence_anomalies.py   # Markov chain sequence anomaly detection
```

Add to test file listing:
```
  test_sequence_anomalies.py  # N tests (sequence training, detection, scoring)
```

Add to Gotchas:
```
- Sequence model needs enough logbook data to build meaningful transitions (>= 50 transitions)
- `--sequence-anomalies` requires `--train-sequences` to have run first (needs saved model)
```

**Step 2: Add cron entry**

Add weekly sequence training + detection (Sunday 3:45am — 30 min after entity correlations, no Ollama):
```
# HA Intelligence Engine — weekly sequence anomaly training + detection (Sunday 3:45am, no Ollama)
45 3 * * 0 . /home/justin/.env && /home/justin/.local/bin/ha-intelligence --train-sequences && /home/justin/.local/bin/ha-intelligence --sequence-anomalies >> /home/justin/.local/log/ha-intelligence.log 2>&1
```

**Step 3: Update Ollama schedule docs**

This task uses NO Ollama (pure Python math), so just add to the non-Ollama task list in `~/Documents/CLAUDE.md`.

**Step 4: Run full test suite one final time**

Run: `python3 -m pytest tests/ -v`
Expected: All passing

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with sequence anomaly detection commands"
```

---

## Verification Checklist

After all tasks complete:

- [ ] `python3 -m pytest tests/ -v` — all tests pass in ha-intelligence
- [ ] `.venv/bin/python -m pytest tests/ -v` — all tests pass in ha-intelligence-hub
- [ ] `ha-intelligence --train-sequences` runs without error (may report "insufficient_data" — OK, needs more logbook history)
- [ ] `ha-intelligence --sequence-anomalies` runs without error
- [ ] Hub cache (`curl http://127.0.0.1:8001/api/cache/intelligence`) includes `entity_correlations`, `sequence_anomalies`, `power_profiles`, `automation_suggestions` keys
- [ ] Crontab has new sequence entry at Sunday 3:45am
- [ ] No Ollama contention (sequence tasks use no LLM)
