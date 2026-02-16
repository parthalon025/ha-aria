"""Tests for CLI command functions (train-sequences, sequence-anomalies)."""

import json
import unittest
from unittest.mock import MagicMock, patch

from aria.engine.cli import cmd_sequence_anomalies, cmd_train_sequences


def _make_logbook_entries(n=200):
    """Generate enough logbook entries to train a Markov chain."""
    entities = [
        "light.living_room",
        "light.kitchen",
        "light.bedroom",
        "switch.fan",
        "binary_sensor.motion_hallway",
        "lock.front_door",
        "light.hallway",
    ]
    entries = []
    for i in range(n):
        eid = entities[i % len(entities)]
        minute = i * 2
        hour = minute // 60
        ts = f"2026-02-10T{hour % 24:02d}:{minute % 60:02d}:00+00:00"
        entries.append({"entity_id": eid, "when": ts})
    return entries


class TestCmdTrainSequences(unittest.TestCase):
    @patch("aria.engine.cli._init")
    def test_trains_and_saves_model(self, mock_init):
        """Train sequences loads logbook, trains detector, saves model."""
        mock_config = MagicMock()
        mock_store = MagicMock()
        mock_store.load_logbook.return_value = _make_logbook_entries(200)
        mock_init.return_value = (mock_config, mock_store)

        result = cmd_train_sequences()

        mock_store.load_logbook.assert_called_once()
        mock_store.save_sequence_model.assert_called_once()
        self.assertIsNotNone(result)
        self.assertIn("transitions", result)
        self.assertIn("unique_entities", result)
        self.assertIn("status", result)

    @patch("aria.engine.cli._init")
    def test_no_logbook_data(self, mock_init):
        """Train sequences with no logbook prints message and returns None."""
        mock_config = MagicMock()
        mock_store = MagicMock()
        mock_store.load_logbook.return_value = []
        mock_init.return_value = (mock_config, mock_store)

        result = cmd_train_sequences()

        self.assertIsNone(result)
        mock_store.save_sequence_model.assert_not_called()

    @patch("aria.engine.cli._init")
    def test_saved_model_is_dict(self, mock_init):
        """Saved model data should be a serializable dict."""
        mock_config = MagicMock()
        mock_store = MagicMock()
        mock_store.load_logbook.return_value = _make_logbook_entries(200)
        mock_init.return_value = (mock_config, mock_store)

        cmd_train_sequences()

        saved = mock_store.save_sequence_model.call_args[0][0]
        self.assertIsInstance(saved, dict)
        self.assertIn("transition_counts", saved)
        self.assertIn("threshold", saved)
        # Verify it's JSON-serializable
        json.dumps(saved)


class TestCmdSequenceAnomalies(unittest.TestCase):
    @patch("aria.engine.cli._init")
    def test_detects_and_saves_anomalies(self, mock_init):
        """Sequence anomalies loads model and logbook, detects, saves results."""
        mock_config = MagicMock()
        mock_store = MagicMock()

        # Train a real model to get valid model_data
        from aria.engine.analysis.sequence_anomalies import MarkovChainDetector

        entries = _make_logbook_entries(200)
        detector = MarkovChainDetector(window_seconds=300, min_transitions=50)
        detector.train(entries)
        model_data = detector.to_dict()

        mock_store.load_sequence_model.return_value = model_data
        mock_store.load_logbook.return_value = entries
        mock_init.return_value = (mock_config, mock_store)

        result = cmd_sequence_anomalies()

        mock_store.load_sequence_model.assert_called_once()
        mock_store.load_logbook.assert_called_once()
        mock_store.save_sequence_anomalies.assert_called_once()
        self.assertIsNotNone(result)
        self.assertIn("anomalies_found", result)
        self.assertIn("total_windows_checked", result)

    @patch("aria.engine.cli._init")
    def test_no_model_returns_none(self, mock_init):
        """Sequence anomalies with no model prints error and returns None."""
        mock_config = MagicMock()
        mock_store = MagicMock()
        mock_store.load_sequence_model.return_value = None
        mock_init.return_value = (mock_config, mock_store)

        result = cmd_sequence_anomalies()

        self.assertIsNone(result)
        mock_store.save_sequence_anomalies.assert_not_called()

    @patch("aria.engine.cli._init")
    def test_no_threshold_returns_none(self, mock_init):
        """Sequence anomalies with untrained model (no threshold) returns None."""
        mock_config = MagicMock()
        mock_store = MagicMock()
        # Model with no threshold (insufficient training data)
        mock_store.load_sequence_model.return_value = {
            "transition_counts": {},
            "entity_counts": {},
            "total_transitions": 5,
            "threshold": None,
            "window_seconds": 300,
            "min_transitions": 50,
        }
        mock_init.return_value = (mock_config, mock_store)

        result = cmd_sequence_anomalies()

        self.assertIsNone(result)
        mock_store.save_sequence_anomalies.assert_not_called()

    @patch("aria.engine.cli._init")
    def test_no_logbook_for_detection(self, mock_init):
        """Sequence anomalies with model but no logbook returns None."""
        mock_config = MagicMock()
        mock_store = MagicMock()

        from aria.engine.analysis.sequence_anomalies import MarkovChainDetector

        entries = _make_logbook_entries(200)
        detector = MarkovChainDetector(window_seconds=300, min_transitions=50)
        detector.train(entries)
        model_data = detector.to_dict()

        mock_store.load_sequence_model.return_value = model_data
        mock_store.load_logbook.return_value = []
        mock_init.return_value = (mock_config, mock_store)

        result = cmd_sequence_anomalies()

        self.assertIsNone(result)
        mock_store.save_sequence_anomalies.assert_not_called()


class TestDispatch(unittest.TestCase):
    @patch("aria.engine.cli.cmd_train_sequences")
    def test_dispatch_train_sequences(self, mock_cmd):
        """--train-sequences dispatches to cmd_train_sequences."""
        from aria.engine.cli import main

        with patch("sys.argv", ["aria", "--train-sequences"]):
            main()
        mock_cmd.assert_called_once()

    @patch("aria.engine.cli.cmd_sequence_anomalies")
    def test_dispatch_sequence_anomalies(self, mock_cmd):
        """--sequence-anomalies dispatches to cmd_sequence_anomalies."""
        from aria.engine.cli import main

        with patch("sys.argv", ["aria", "--sequence-anomalies"]):
            main()
        mock_cmd.assert_called_once()


if __name__ == "__main__":
    unittest.main()
