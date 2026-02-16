"""Markov chain-based sequence anomaly detection.

Learns entity state-change transition probabilities from logbook events.
Flags sequences with unusually low transition probability as anomalous.
Lightweight alternative to LSTM-Autoencoder — zero new dependencies.
"""

import math
from collections import defaultdict

from aria.engine.analysis.entity_correlations import (
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
            return {"transitions": 0, "unique_entities": 0, "threshold": None, "status": "insufficient_data"}

        for i in range(len(events) - 1):
            ts_a, eid_a = events[i]
            ts_b, eid_b = events[i + 1]

            gap = (ts_b - ts_a).total_seconds()
            if gap > self.window_seconds:
                continue

            self.transition_counts[eid_a][eid_b] += 1
            self.entity_counts[eid_a] += 1
            self.total_transitions += 1

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
            "status": "trained" if self.total_transitions >= self.min_transitions else "insufficient_data",
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
            window = events[i : i + window_size]
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
                # Completely unknown entity — assign minimum probability
                prob = 1.0 / max(self.total_transitions, len(self.entity_counts) + 1)
            else:
                count = self.transition_counts.get(eid_a, {}).get(eid_b, 0)
                prob = 1.0 / (total + len(self.entity_counts)) if count == 0 else count / total

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
            window = events[i : i + window_size]
            if len(window) < 3:
                continue

            score = self._score_window(window)
            if score is not None and score < self.threshold:
                entities = list(set(eid for _, eid in window))
                anomalies.append(
                    {
                        "time_start": window[0][0].isoformat(),
                        "time_end": window[-1][0].isoformat(),
                        "score": round(score, 4),
                        "threshold": round(self.threshold, 4),
                        "severity": "high" if score < self.threshold * 1.5 else "medium",
                        "entities": entities,
                    }
                )

        return anomalies

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


def summarize_sequence_anomalies(anomalies, total_windows_checked=0):
    """Produce a summary dict suitable for storage and LLM context."""
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
        "anomalies": anomalies[:10],
    }
