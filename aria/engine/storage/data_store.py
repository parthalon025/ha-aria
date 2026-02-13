"""Unified data persistence layer for all JSON I/O.

Single class that handles all file-based storage, making it trivial to
mock in tests or swap backends (e.g., SQLite) in the future.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from aria.engine.config import PathConfig


def _atomic_write_json(path, data, **kwargs):
    """Write JSON atomically using temp file + rename."""
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, **kwargs)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class DataStore:
    """Centralized JSON I/O for snapshots, baselines, predictions, and config."""

    def __init__(self, paths: PathConfig):
        self.paths = paths

    def ensure_dirs(self):
        """Create all required directories."""
        self.paths.ensure_dirs()

    # --- Daily Snapshots ---

    def save_snapshot(self, snapshot: dict) -> Path:
        """Save snapshot to daily directory."""
        self.ensure_dirs()
        path = self.paths.daily_dir / f"{snapshot['date']}.json"
        _atomic_write_json(path, snapshot, indent=2)
        return path

    def load_snapshot(self, date_str: str) -> dict | None:
        """Load a previously saved daily snapshot."""
        path = self.paths.daily_dir / f"{date_str}.json"
        if not path.is_file():
            return None
        with open(path) as f:
            return json.load(f)

    def load_recent_snapshots(self, days: int = 30) -> list[dict]:
        """Load up to N days of recent daily snapshots."""
        snapshots = []
        today = datetime.now()
        for i in range(days):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            snap = self.load_snapshot(date_str)
            if snap:
                snapshots.append(snap)
        return snapshots

    # --- Intraday Snapshots ---

    def save_intraday_snapshot(self, snapshot: dict) -> Path:
        """Save intra-day snapshot to intraday/YYYY-MM-DD/HH.json."""
        self.ensure_dirs()
        day_dir = self.paths.intraday_dir / snapshot["date"]
        day_dir.mkdir(parents=True, exist_ok=True)
        hour = snapshot.get("hour", 0)
        path = day_dir / f"{hour:02d}.json"
        _atomic_write_json(path, snapshot, indent=2)
        return path

    def load_intraday_snapshots(self, date_str: str) -> list[dict]:
        """Load all intra-day snapshots for a given date."""
        day_dir = self.paths.intraday_dir / date_str
        if not day_dir.is_dir():
            return []
        snapshots = []
        for fname in sorted(os.listdir(day_dir)):
            if fname.endswith(".json"):
                try:
                    with open(day_dir / fname) as f:
                        snapshots.append(json.load(f))
                except Exception:
                    pass
        return snapshots

    def load_all_intraday_snapshots(self, days: int = 30) -> list[dict]:
        """Load all intra-day snapshots for the last N days."""
        all_snapshots = []
        today = datetime.now()
        for i in range(days):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            all_snapshots.extend(self.load_intraday_snapshots(date_str))
        return all_snapshots

    # --- Baselines ---

    def save_baselines(self, baselines: dict):
        """Save computed baselines."""
        self.ensure_dirs()
        _atomic_write_json(self.paths.baselines_path, baselines, indent=2)

    def load_baselines(self) -> dict:
        """Load baselines. Returns empty dict if not yet computed."""
        if not self.paths.baselines_path.is_file():
            return {}
        with open(self.paths.baselines_path) as f:
            return json.load(f)

    # --- Predictions ---

    def save_predictions(self, predictions: dict):
        """Save generated predictions."""
        self.ensure_dirs()
        _atomic_write_json(self.paths.predictions_path, predictions, indent=2)

    def load_predictions(self) -> dict:
        """Load predictions. Returns empty dict if none exist."""
        if not self.paths.predictions_path.is_file():
            return {}
        with open(self.paths.predictions_path) as f:
            return json.load(f)

    # --- Correlations ---

    def save_correlations(self, correlations: dict):
        """Save cross-correlation results."""
        self.ensure_dirs()
        _atomic_write_json(self.paths.correlations_path, correlations, indent=2)

    def load_correlations(self) -> dict:
        """Load correlations. Returns empty dict if not yet computed."""
        if not self.paths.correlations_path.is_file():
            return {}
        with open(self.paths.correlations_path) as f:
            return json.load(f)

    # --- Entity Correlations ---

    def save_entity_correlations(self, summary: dict):
        """Save entity co-occurrence correlation summary."""
        self.ensure_dirs()
        path = self.paths.data_dir / "entity_correlations.json"
        _atomic_write_json(path, summary, indent=2)

    def load_entity_correlations(self) -> dict:
        """Load entity co-occurrence correlations. Returns empty dict if none."""
        path = self.paths.data_dir / "entity_correlations.json"
        if not path.is_file():
            return {}
        with open(path) as f:
            return json.load(f)

    # --- Accuracy History ---

    def load_accuracy_history(self) -> dict:
        """Load prediction accuracy history."""
        if not self.paths.accuracy_path.is_file():
            return {"scores": []}
        with open(self.paths.accuracy_path) as f:
            return json.load(f)

    def update_accuracy_history(self, new_score: dict) -> dict:
        """Append score to accuracy history, keep last 90 entries."""
        self.ensure_dirs()
        history = self.load_accuracy_history()
        history["scores"].append(new_score)
        history["scores"] = history["scores"][-90:]
        # Trend computation imported from analysis when needed
        _atomic_write_json(self.paths.accuracy_path, history, indent=2)
        return history

    # --- Feature Config ---

    def load_feature_config(self) -> dict | None:
        """Load feature config. Returns None if not yet created."""
        if not self.paths.feature_config_path.is_file():
            return None
        with open(self.paths.feature_config_path) as f:
            return json.load(f)

    def save_feature_config(self, config: dict):
        """Save feature config."""
        self.ensure_dirs()
        _atomic_write_json(self.paths.feature_config_path, config, indent=2)

    # --- Meta-Learning Suggestions ---

    def load_applied_suggestions(self) -> dict:
        """Load history of applied meta-learning suggestions."""
        path = self.paths.meta_dir / "applied.json"
        if path.is_file():
            with open(path) as f:
                return json.load(f)
        return {"applied": [], "total_applied": 0}

    def save_applied_suggestions(self, history: dict):
        """Save meta-learning applied suggestions history."""
        self.paths.meta_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.meta_dir / "applied.json"
        _atomic_write_json(path, history, indent=2)

    # --- Sequence Anomaly Detection ---

    def save_sequence_model(self, model_data: dict):
        """Save trained Markov chain model."""
        self.ensure_dirs()
        _atomic_write_json(self.paths.sequence_model_path, model_data, indent=2)

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
        _atomic_write_json(path, summary, indent=2)

    def load_sequence_anomalies(self) -> dict | None:
        """Load sequence anomaly results. Returns None if none exist."""
        path = self.paths.data_dir / "sequence_anomalies.json"
        if not path.is_file():
            return None
        with open(path) as f:
            return json.load(f)

    # --- Logbook ---

    def load_logbook(self) -> list:
        """Load current logbook from synced JSON file."""
        if not self.paths.logbook_path.is_file():
            return []
        try:
            with open(self.paths.logbook_path) as f:
                return json.load(f)
        except Exception:
            return []
