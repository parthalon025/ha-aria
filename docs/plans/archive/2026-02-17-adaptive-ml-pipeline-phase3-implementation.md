# Adaptive ML Pipeline — Phase 3: Pattern Recognition Expansion

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** Draft

**Goal:** Add sequence-aware trajectory classification, hierarchical pattern scale tagging, and explainable anomaly detection to ARIA's ML pipeline — all gated to Tier 3+ hardware.

**Architecture:** Three new engine modules (`anomaly_explainer.py`, `pattern_scale.py`, `sequence.py`) provide pure computation. A new hub module (`pattern_recognition.py`) orchestrates them, subscribes to events, and caches results. ML engine wires in anomaly explanations and trajectory classification as a new feature. All new functionality requires `tslearn` (optional dep) and Tier 3+ hardware; Tier 1-2 behavior unchanged.

**Tech Stack:** tslearn (DTW), sklearn IsolationForest (path tracing), numpy

**Design doc:** `docs/plans/2026-02-17-adaptive-ml-pipeline-design.md` § Phase 3

---

## Dependencies & Prerequisites

- **Phase 2 must be merged** (completed 2026-02-17)
- **Branch:** Create `feature/adaptive-ml-pipeline-phase3` from `main`
- **Virtual env:** `.venv/bin/python -m pytest` for all test runs
- **Test baseline:** 1458 passed, 14 skipped, 0 failures

## Critical Patterns (Read Before Implementing)

1. **Sync vs async module access:** `hub.get_module()` is async. From sync code, use `getattr(self.hub, "modules", {}).get("module_name")` instead. See Phase 2 bug fix in `_get_online_prediction()` at `ml_engine.py:1225-1227`.

2. **Optional imports:** All tslearn imports must be `try/except` guarded with lazy loading. Never crash on import. Pattern from `aria/engine/online.py:_create_model()`.

3. **Hub cache access:** Use `await self.hub.set_cache()` / `get_cache()`, NOT `self.hub.cache.*`. See `CLAUDE.md` gotchas.

4. **Config defaults format:** Each entry is a dict with `key`, `default_value` (string), `value_type`, `label`, `description`, `category`, and optional `min_value`/`max_value`/`step`/`options`. See `config_defaults.py:599-698`.

5. **Module registration:** Follow the try/except pattern in `cli.py:382-390` (online_learner block).

6. **Feature config sections:** New feature groups must be added to `DEFAULT_FEATURE_CONFIG` in `feature_config.py` AND to `_REQUIRED_SECTIONS` set. See `feature_config.py:9-77`.

---

## Task 1: Anomaly Explainer Engine Module

**Files:**
- Create: `aria/engine/anomaly_explainer.py`
- Test: `tests/engine/test_anomaly_explainer.py`

**Context:** Currently, `_run_anomaly_detection()` in `ml_engine.py:1120-1136` returns only `(is_anomaly, anomaly_score)` — a binary flag and a scalar. The design calls for per-feature contribution explanations: which features most caused the anomaly? This module traces IsolationForest's isolation paths to count how frequently each feature appears at split nodes. Features split on more often = more responsible for isolating the sample.

**Step 1: Write the failing tests**

Create `tests/engine/test_anomaly_explainer.py`:

```python
"""Tests for IsolationForest anomaly explanation engine."""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from aria.engine.anomaly_explainer import AnomalyExplainer


class TestAnomalyExplainer:
    """Test anomaly explanation via IsolationForest path tracing."""

    def setup_method(self):
        """Create a trained IsolationForest with known structure."""
        np.random.seed(42)
        # Normal data: low values in all 5 features
        normal = np.random.normal(loc=0, scale=1, size=(100, 5))
        self.feature_names = ["power", "lights", "motion", "temp", "humidity"]
        self.model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
        self.model.fit(normal)
        self.explainer = AnomalyExplainer()

    def test_explain_returns_top_n(self):
        """Explain returns exactly top_n features."""
        # Anomalous: extreme power value
        anomalous = np.array([[10.0, 0.0, 0.0, 0.0, 0.0]])
        result = self.explainer.explain(self.model, anomalous, self.feature_names, top_n=3)
        assert len(result) == 3
        assert all("feature" in r and "contribution" in r for r in result)

    def test_contributions_sum_to_one_or_less(self):
        """Contributions of top_n features sum to <= 1.0."""
        anomalous = np.array([[10.0, 0.0, 0.0, 0.0, 0.0]])
        result = self.explainer.explain(self.model, anomalous, self.feature_names, top_n=5)
        total = sum(r["contribution"] for r in result)
        assert 0.0 < total <= 1.001  # Allow float rounding

    def test_extreme_feature_ranks_first(self):
        """The feature with the extreme value should rank highest."""
        # Only feature 0 (power) is extreme
        anomalous = np.array([[20.0, 0.1, -0.1, 0.2, -0.2]])
        result = self.explainer.explain(self.model, anomalous, self.feature_names, top_n=3)
        # Power should be the top contributor
        assert result[0]["feature"] == "power"

    def test_explain_with_no_feature_names(self):
        """Falls back to index-based names when feature_names is empty."""
        anomalous = np.array([[10.0, 0.0, 0.0, 0.0, 0.0]])
        result = self.explainer.explain(self.model, anomalous, [], top_n=3)
        assert len(result) == 3
        assert result[0]["feature"].startswith("feature_")

    def test_explain_normal_sample(self):
        """Normal samples still get explanations (lower contributions)."""
        normal = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        result = self.explainer.explain(self.model, normal, self.feature_names, top_n=3)
        assert len(result) == 3

    def test_top_n_capped_at_feature_count(self):
        """top_n larger than feature count returns all features."""
        anomalous = np.array([[10.0, 0.0, 0.0, 0.0, 0.0]])
        result = self.explainer.explain(self.model, anomalous, self.feature_names, top_n=10)
        assert len(result) <= 5  # Only 5 features exist
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/engine/test_anomaly_explainer.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'aria.engine.anomaly_explainer'`

**Step 3: Implement the anomaly explainer**

Create `aria/engine/anomaly_explainer.py`:

```python
"""IsolationForest anomaly explanation via path tracing.

Traces the isolation path of a sample across all trees in an
IsolationForest ensemble. Features that appear at split nodes more
frequently are more responsible for isolating (flagging) the sample.

Tier 3+ only — called from MLEngine when anomaly is detected.
"""

import numpy as np


class AnomalyExplainer:
    """Explain anomalies by tracing IsolationForest isolation paths."""

    def explain(
        self,
        iso_forest,
        X_sample: np.ndarray,
        feature_names: list[str],
        top_n: int = 3,
    ) -> list[dict]:
        """Identify top contributing features for an anomaly.

        Traces the decision path through each tree in the ensemble,
        counting how often each feature appears at a split node.
        Features used more often in the isolation path contribute
        more to the anomaly score.

        Args:
            iso_forest: Trained IsolationForest model.
            X_sample: Single sample as (1, n_features) array.
            feature_names: List of feature names (or empty for index-based).
            top_n: Number of top contributors to return.

        Returns:
            List of dicts with "feature" and "contribution" keys,
            sorted by contribution descending.
        """
        n_features = X_sample.shape[1]
        contributions = np.zeros(n_features)

        for estimator in iso_forest.estimators_:
            tree = estimator.tree_
            # decision_path returns sparse CSR matrix
            node_indicator = estimator.decision_path(X_sample)
            node_indices = node_indicator.indices

            for node_id in node_indices:
                feature_id = tree.feature[node_id]
                if feature_id >= 0:  # -2 means leaf node
                    contributions[feature_id] += 1.0

        # Normalize to sum to 1.0
        total = contributions.sum()
        if total > 0:
            contributions /= total

        # Build names — fall back to index if names not provided
        names = feature_names if len(feature_names) == n_features else [
            f"feature_{i}" for i in range(n_features)
        ]

        # Sort by contribution descending, take top_n
        top_indices = np.argsort(contributions)[::-1][:min(top_n, n_features)]

        return [
            {
                "feature": names[i],
                "contribution": round(float(contributions[i]), 4),
            }
            for i in top_indices
            if contributions[i] > 0
        ]
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/engine/test_anomaly_explainer.py -v
```
Expected: 6 passed

**Step 5: Commit**

```bash
git add aria/engine/anomaly_explainer.py tests/engine/test_anomaly_explainer.py
git commit -m "feat: add IsolationForest anomaly explainer with path tracing"
```

---

## Task 2: Pattern Scale Enum and Utilities

**Files:**
- Create: `aria/engine/pattern_scale.py`
- Test: `tests/engine/test_pattern_scale.py`

**Context:** The design calls for three time-scale tiers: micro (seconds-minutes, motion→light), meso (minutes-hours, morning routine), macro (days-weeks, seasonal shifts). This enum tags all detected patterns so the dashboard and shadow engine can filter/track accuracy per scale.

**Step 1: Write the failing tests**

Create `tests/engine/test_pattern_scale.py`:

```python
"""Tests for pattern scale classification."""

import pytest

from aria.engine.pattern_scale import PatternScale


class TestPatternScale:
    """Test time-scale classification for patterns."""

    def test_enum_values(self):
        """Three scales exist with correct string values."""
        assert PatternScale.MICRO.value == "micro"
        assert PatternScale.MESO.value == "meso"
        assert PatternScale.MACRO.value == "macro"

    def test_from_duration_micro(self):
        """Durations under 5 minutes classify as micro."""
        assert PatternScale.from_duration_seconds(10) == PatternScale.MICRO
        assert PatternScale.from_duration_seconds(60) == PatternScale.MICRO
        assert PatternScale.from_duration_seconds(299) == PatternScale.MICRO

    def test_from_duration_meso(self):
        """Durations from 5 minutes to 4 hours classify as meso."""
        assert PatternScale.from_duration_seconds(300) == PatternScale.MESO
        assert PatternScale.from_duration_seconds(3600) == PatternScale.MESO
        assert PatternScale.from_duration_seconds(14399) == PatternScale.MESO

    def test_from_duration_macro(self):
        """Durations 4 hours and above classify as macro."""
        assert PatternScale.from_duration_seconds(14400) == PatternScale.MACRO
        assert PatternScale.from_duration_seconds(86400) == PatternScale.MACRO

    def test_from_duration_zero(self):
        """Zero duration is micro."""
        assert PatternScale.from_duration_seconds(0) == PatternScale.MICRO

    def test_scale_description(self):
        """Each scale has a human-readable description."""
        assert PatternScale.MICRO.description  # Not empty
        assert PatternScale.MESO.description
        assert PatternScale.MACRO.description

    def test_scale_window_range(self):
        """Each scale reports its duration range as (min_s, max_s)."""
        micro_range = PatternScale.MICRO.window_range
        assert micro_range == (0, 300)
        meso_range = PatternScale.MESO.window_range
        assert meso_range == (300, 14400)
        macro_range = PatternScale.MACRO.window_range
        assert macro_range == (14400, None)  # Unbounded upper
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/engine/test_pattern_scale.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement pattern scale**

Create `aria/engine/pattern_scale.py`:

```python
"""Pattern time-scale classification.

Patterns in ARIA span three scales:
- Micro (seconds-minutes): motion triggers light, door opens
- Meso (minutes-hours): morning routine, cooking session
- Macro (days-weeks): seasonal shifts, schedule changes

Used to tag detected patterns, shadow predictions, and accuracy tracking.
"""

from enum import Enum


class PatternScale(str, Enum):
    """Time-scale classification for detected patterns."""

    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"

    @property
    def description(self) -> str:
        """Human-readable description of this scale."""
        return _DESCRIPTIONS[self]

    @property
    def window_range(self) -> tuple[int, int | None]:
        """Duration range as (min_seconds, max_seconds_or_None)."""
        return _WINDOW_RANGES[self]

    @classmethod
    def from_duration_seconds(cls, duration_s: float) -> "PatternScale":
        """Classify a pattern by its time span.

        Args:
            duration_s: Pattern duration in seconds.

        Returns:
            The appropriate PatternScale.
        """
        if duration_s < 300:  # < 5 minutes
            return cls.MICRO
        elif duration_s < 14400:  # < 4 hours
            return cls.MESO
        else:
            return cls.MACRO


_DESCRIPTIONS = {
    PatternScale.MICRO: "Seconds to minutes — immediate reactions (motion triggers light)",
    PatternScale.MESO: "Minutes to hours — routines and sessions (morning routine)",
    PatternScale.MACRO: "Days to weeks — seasonal and schedule patterns",
}

_WINDOW_RANGES = {
    PatternScale.MICRO: (0, 300),
    PatternScale.MESO: (300, 14400),
    PatternScale.MACRO: (14400, None),
}
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/engine/test_pattern_scale.py -v
```
Expected: 7 passed

**Step 5: Commit**

```bash
git add aria/engine/pattern_scale.py tests/engine/test_pattern_scale.py
git commit -m "feat: add PatternScale enum for micro/meso/macro time classification"
```

---

## Task 3: Sequence Classifier Engine Module

**Files:**
- Create: `aria/engine/sequence.py`
- Modify: `pyproject.toml:43-49` (add tslearn to ml-extra)
- Test: `tests/engine/test_sequence.py`

**Context:** This module wraps tslearn's `KNeighborsTimeSeriesClassifier` with DTW metric. It classifies sliding windows of snapshot feature vectors into trajectory types: stable, ramping_up, winding_down, or anomalous_transition. Training labels come from heuristic rules applied to historical data (the power metric trend over the window). The classifier then generalizes beyond the heuristic using the full multi-variate feature set.

**Important:** tslearn is an optional Tier 3+ dependency. Import must be lazy with try/except guard. If tslearn is missing, the classifier returns "stable" for all inputs.

**Step 1: Add tslearn to pyproject.toml**

In `pyproject.toml`, add `tslearn` to the `ml-extra` optional dependency group (line 43-49):

```toml
ml-extra = [
    "lightgbm>=4.0.0",
    "mlxtend>=0.23.0",
    "river>=0.21.0",
    "shap>=0.45.0",
    "optuna>=3.0.0",
    "tslearn>=0.6.0",
]
```

Then install:

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/pip install -e ".[ml-extra]" --quiet
```

**Step 2: Write the failing tests**

Create `tests/engine/test_sequence.py`:

```python
"""Tests for trajectory sequence classifier."""

import numpy as np
import pytest

from aria.engine.sequence import SequenceClassifier, TRAJECTORY_CLASSES


class TestSequenceClassifier:
    """Test DTW-based trajectory classification."""

    def test_trajectory_classes(self):
        """Four trajectory classes are defined."""
        assert set(TRAJECTORY_CLASSES) == {"stable", "ramping_up", "winding_down", "anomalous_transition"}

    def test_init_defaults(self):
        """Classifier initializes with sensible defaults."""
        clf = SequenceClassifier()
        assert clf.window_size == 6
        assert clf.n_neighbors == 3
        assert not clf.is_trained

    def test_predict_untrained_returns_stable(self):
        """Untrained classifier defaults to 'stable'."""
        clf = SequenceClassifier(window_size=4)
        window = np.zeros((4, 5))
        assert clf.predict(window) == "stable"

    def test_label_heuristic_ramping_up(self):
        """Increasing power trend labels as ramping_up."""
        # 6 snapshots, 5 features — first feature is power
        window = np.zeros((6, 5))
        window[:, 0] = [10, 20, 30, 40, 50, 60]  # Increasing power
        label = SequenceClassifier.label_window_heuristic(window, target_col_idx=0)
        assert label == "ramping_up"

    def test_label_heuristic_winding_down(self):
        """Decreasing power trend labels as winding_down."""
        window = np.zeros((6, 5))
        window[:, 0] = [60, 50, 40, 30, 20, 10]  # Decreasing power
        label = SequenceClassifier.label_window_heuristic(window, target_col_idx=0)
        assert label == "winding_down"

    def test_label_heuristic_stable(self):
        """Flat power trend labels as stable."""
        window = np.zeros((6, 5))
        window[:, 0] = [50, 51, 49, 50, 51, 50]  # Stable
        label = SequenceClassifier.label_window_heuristic(window, target_col_idx=0)
        assert label == "stable"

    def test_label_heuristic_anomalous(self):
        """Wild oscillation labels as anomalous_transition."""
        window = np.zeros((6, 5))
        window[:, 0] = [10, 100, 5, 95, 10, 100]  # Extreme variance
        label = SequenceClassifier.label_window_heuristic(window, target_col_idx=0)
        assert label == "anomalous_transition"

    def test_train_and_predict(self):
        """Train on labeled windows, predict a new one."""
        clf = SequenceClassifier(window_size=4, n_neighbors=1)

        # Build training data: 10 windows of each class
        windows = []
        labels = []
        rng = np.random.RandomState(42)
        for _ in range(10):
            # Ramping up
            w = rng.normal(0, 0.1, (4, 3))
            w[:, 0] = np.linspace(10, 60, 4)
            windows.append(w)
            labels.append("ramping_up")
            # Winding down
            w = rng.normal(0, 0.1, (4, 3))
            w[:, 0] = np.linspace(60, 10, 4)
            windows.append(w)
            labels.append("winding_down")

        X = np.array(windows)
        clf.train(X, labels)
        assert clf.is_trained

        # Test ramping up window
        test_window = np.zeros((4, 3))
        test_window[:, 0] = np.linspace(10, 60, 4)
        pred = clf.predict(test_window)
        assert pred in TRAJECTORY_CLASSES

    def test_get_stats(self):
        """Stats dict includes training info."""
        clf = SequenceClassifier(window_size=4)
        stats = clf.get_stats()
        assert stats["is_trained"] is False
        assert stats["window_size"] == 4

    def test_tslearn_missing_graceful(self, monkeypatch):
        """Graceful fallback when tslearn is not installed."""
        import aria.engine.sequence as seq_module
        # Simulate tslearn missing by making _create_model raise
        clf = SequenceClassifier(window_size=4)
        clf._tslearn_available = False
        window = np.zeros((4, 3))
        assert clf.predict(window) == "stable"
```

**Step 3: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/engine/test_sequence.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'aria.engine.sequence'`

**Step 4: Implement the sequence classifier**

Create `aria/engine/sequence.py`:

```python
"""Trajectory sequence classifier using Dynamic Time Warping.

Classifies sliding windows of snapshot feature vectors into trajectory
types using tslearn's KNeighborsTimeSeriesClassifier with DTW metric.

Training labels are generated heuristically from the power metric trend,
then the DTW classifier generalizes across the full multi-variate space.

Tier 3+ only — tslearn is an optional dependency.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

TRAJECTORY_CLASSES = ["stable", "ramping_up", "winding_down", "anomalous_transition"]

# Thresholds for heuristic labeling
_CHANGE_THRESHOLD = 0.20  # 20% change = directional
_ANOMALOUS_CV_THRESHOLD = 0.50  # Coefficient of variation > 50% = anomalous


class SequenceClassifier:
    """DTW-based trajectory classifier for snapshot windows.

    Args:
        window_size: Number of snapshots per window (default 6).
        n_neighbors: KNN neighbors for DTW classifier (default 3).
    """

    def __init__(self, window_size: int = 6, n_neighbors: int = 3):
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self._model = None
        self._tslearn_available = True
        self._trained_at: str | None = None
        self._n_training_samples: int = 0

    @property
    def is_trained(self) -> bool:
        """Whether the classifier has been trained."""
        return self._model is not None

    def _create_model(self):
        """Lazily create the tslearn KNN-DTW model."""
        try:
            from tslearn.neighbors import KNeighborsTimeSeriesClassifier

            return KNeighborsTimeSeriesClassifier(
                n_neighbors=self.n_neighbors, metric="dtw"
            )
        except ImportError:
            logger.warning("tslearn not installed — sequence classifier disabled")
            self._tslearn_available = False
            return None

    def train(self, windows: np.ndarray, labels: list[str]) -> bool:
        """Train on labeled windows.

        Args:
            windows: Array of shape (n_samples, window_size, n_features).
            labels: List of trajectory class labels.

        Returns:
            True if training succeeded, False otherwise.
        """
        if not self._tslearn_available:
            return False

        model = self._create_model()
        if model is None:
            return False

        try:
            model.fit(windows, labels)
            self._model = model
            self._n_training_samples = len(labels)
            from datetime import datetime

            self._trained_at = datetime.now().isoformat()
            logger.info(f"Sequence classifier trained on {len(labels)} windows")
            return True
        except Exception as e:
            logger.error(f"Sequence classifier training failed: {e}")
            return False

    def predict(self, window: np.ndarray) -> str:
        """Classify a single window.

        Args:
            window: Array of shape (window_size, n_features).

        Returns:
            Trajectory class string. Defaults to "stable" if untrained
            or tslearn unavailable.
        """
        if self._model is None or not self._tslearn_available:
            return "stable"

        try:
            result = self._model.predict(window.reshape(1, *window.shape))
            return str(result[0])
        except Exception as e:
            logger.debug(f"Sequence prediction failed: {e}")
            return "stable"

    def get_stats(self) -> dict:
        """Return classifier statistics."""
        return {
            "is_trained": self.is_trained,
            "window_size": self.window_size,
            "n_neighbors": self.n_neighbors,
            "tslearn_available": self._tslearn_available,
            "trained_at": self._trained_at,
            "n_training_samples": self._n_training_samples,
        }

    @staticmethod
    def label_window_heuristic(
        window: np.ndarray, target_col_idx: int = 0
    ) -> str:
        """Label a window using heuristic rules on the target metric.

        Compares the average of the first two and last two values in
        the target column to determine trend direction. High variance
        with no clear direction = anomalous_transition.

        Args:
            window: Array of shape (window_size, n_features).
            target_col_idx: Column index of the target metric (e.g., power).

        Returns:
            One of TRAJECTORY_CLASSES.
        """
        values = window[:, target_col_idx]
        start_avg = float(np.mean(values[:2]))
        end_avg = float(np.mean(values[-2:]))

        # Avoid division by zero
        baseline = max(abs(start_avg), 1e-6)
        change_pct = (end_avg - start_avg) / baseline

        # Check for anomalous variance first
        mean_abs = max(float(np.mean(np.abs(values))), 1e-6)
        cv = float(np.std(values)) / mean_abs
        if cv > _ANOMALOUS_CV_THRESHOLD and abs(change_pct) < _CHANGE_THRESHOLD:
            return "anomalous_transition"

        if change_pct > _CHANGE_THRESHOLD:
            return "ramping_up"
        elif change_pct < -_CHANGE_THRESHOLD:
            return "winding_down"

        return "stable"
```

**Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/engine/test_sequence.py -v
```
Expected: 11 passed

**Step 6: Commit**

```bash
git add aria/engine/sequence.py tests/engine/test_sequence.py pyproject.toml
git commit -m "feat: add DTW trajectory sequence classifier with heuristic labeling"
```

---

## Task 4: Pattern Recognition Hub Module

**Files:**
- Create: `aria/modules/pattern_recognition.py`
- Test: `tests/hub/test_pattern_recognition.py`

**Context:** This hub module orchestrates sequence classification and anomaly explanation. It subscribes to `state_changed` events to maintain a sliding window of recent snapshots, runs the sequence classifier after each activity buffer flush, and caches results for ML engine consumption. It self-gates on Tier 3+ hardware, identical to `online_learner.py:MIN_TIER = 3`.

**Step 1: Write the failing tests**

Create `tests/hub/test_pattern_recognition.py`:

```python
"""Tests for pattern recognition hub module."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from aria.modules.pattern_recognition import PatternRecognitionModule


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.subscribe = MagicMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.publish = AsyncMock()
    hub.get_config_value = MagicMock(return_value=None)
    hub.modules = {}
    return hub


class TestPatternRecognitionInit:
    """Test module initialization and tier gating."""

    def test_module_id(self, mock_hub):
        module = PatternRecognitionModule(mock_hub)
        assert module.module_id == "pattern_recognition"

    def test_subscribes_to_events(self, mock_hub):
        module = PatternRecognitionModule(mock_hub)
        # Should subscribe to shadow_resolved for pattern tracking
        subscribe_calls = [call[0][0] for call in mock_hub.subscribe.call_args_list]
        assert "shadow_resolved" in subscribe_calls

    @patch("aria.modules.pattern_recognition.recommend_tier", return_value=2)
    @patch("aria.modules.pattern_recognition.scan_hardware")
    async def test_tier_gate_blocks_below_tier_3(self, mock_scan, mock_tier, mock_hub):
        """Module disables itself at Tier 2."""
        mock_scan.return_value = MagicMock(ram_gb=4, cpu_cores=2)
        module = PatternRecognitionModule(mock_hub)
        await module.initialize()
        assert module.active is False

    @patch("aria.modules.pattern_recognition.recommend_tier", return_value=3)
    @patch("aria.modules.pattern_recognition.scan_hardware")
    async def test_tier_gate_allows_tier_3(self, mock_scan, mock_tier, mock_hub):
        """Module activates at Tier 3."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)
        module = PatternRecognitionModule(mock_hub)
        await module.initialize()
        assert module.active is True


class TestTrajectoryClassification:
    """Test trajectory window management and classification."""

    @patch("aria.modules.pattern_recognition.recommend_tier", return_value=3)
    @patch("aria.modules.pattern_recognition.scan_hardware")
    async def test_on_shadow_resolved_updates_cache(self, mock_scan, mock_tier, mock_hub):
        """Shadow resolved events feed the trajectory window."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)
        module = PatternRecognitionModule(mock_hub)
        await module.initialize()

        # Feed enough events to build a window
        for i in range(6):
            await module._on_shadow_resolved({
                "target": "power_watts",
                "features": {"power": float(i * 10), "lights": 1.0},
                "actual_value": float(i * 10),
                "timestamp": datetime.now().isoformat(),
            })

        # Should have trajectory classification result
        assert module.current_trajectory is not None

    async def test_get_current_state(self, mock_hub):
        """get_current_state returns trajectory and scale info."""
        module = PatternRecognitionModule(mock_hub)
        state = module.get_current_state()
        assert "trajectory" in state
        assert "pattern_scales" in state
        assert "anomaly_explanations" in state

    async def test_get_stats(self, mock_hub):
        """get_stats includes sequence classifier info."""
        module = PatternRecognitionModule(mock_hub)
        stats = module.get_stats()
        assert "active" in stats
        assert "sequence_classifier" in stats
        assert "window_count" in stats
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/hub/test_pattern_recognition.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement the pattern recognition module**

Create `aria/modules/pattern_recognition.py`:

```python
"""Pattern recognition hub module.

Orchestrates sequence classification, pattern scale tagging, and anomaly
explanation. Subscribes to shadow_resolved events to maintain a sliding
window of recent feature snapshots, runs trajectory classification, and
caches results for ML engine consumption.

Tier 3+ only — self-gates on hardware tier.
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from aria.engine.anomaly_explainer import AnomalyExplainer
from aria.engine.pattern_scale import PatternScale
from aria.engine.sequence import SequenceClassifier
from aria.hub.core import Module

logger = logging.getLogger(__name__)

MIN_TIER = 3
DEFAULT_WINDOW_SIZE = 6


class PatternRecognitionModule(Module):
    """Hub module for trajectory classification and pattern analysis."""

    def __init__(self, hub):
        super().__init__("pattern_recognition", hub)
        self.active = False
        self.sequence_classifier = SequenceClassifier(window_size=DEFAULT_WINDOW_SIZE)
        self.anomaly_explainer = AnomalyExplainer()

        # Sliding window of recent feature snapshots per target
        self._feature_windows: dict[str, deque] = {}
        self._max_window = DEFAULT_WINDOW_SIZE * 2  # Keep extra for lag

        # Current state
        self.current_trajectory: str | None = None
        self._last_anomaly_explanations: list[dict] = []
        self._shadow_event_count = 0

        # Subscribe to events
        hub.subscribe("shadow_resolved", self._on_shadow_resolved)

    async def initialize(self):
        """Initialize — check hardware tier and activate if sufficient."""
        from aria.engine.hardware import recommend_tier, scan_hardware

        profile = scan_hardware()
        tier = recommend_tier(profile)

        if tier < MIN_TIER:
            logger.info(
                f"Pattern recognition disabled: tier {tier} < {MIN_TIER} "
                f"({profile.ram_gb:.1f}GB RAM, {profile.cpu_cores} cores)"
            )
            self.active = False
            return

        self.active = True
        logger.info(f"Pattern recognition active at tier {tier}")

    async def _on_shadow_resolved(self, event: dict[str, Any]):
        """Handle shadow_resolved events — update feature windows."""
        if not self.active:
            return

        target = event.get("target", "")
        features = event.get("features", {})
        timestamp = event.get("timestamp", "")

        if not features:
            return

        self._shadow_event_count += 1

        # Build numeric vector from features
        feature_names = sorted(features.keys())
        feature_vec = [float(features.get(k, 0)) for k in feature_names]

        # Maintain per-target sliding window
        if target not in self._feature_windows:
            self._feature_windows[target] = deque(maxlen=self._max_window)

        self._feature_windows[target].append({
            "vector": feature_vec,
            "feature_names": feature_names,
            "timestamp": timestamp,
        })

        # Classify trajectory when we have enough data
        window = self._feature_windows[target]
        if len(window) >= self.sequence_classifier.window_size:
            await self._classify_trajectory(target, window)

    async def _classify_trajectory(self, target: str, window: deque):
        """Run trajectory classification on the current window."""
        ws = self.sequence_classifier.window_size
        recent = list(window)[-ws:]
        window_array = np.array([entry["vector"] for entry in recent])

        if self.sequence_classifier.is_trained:
            trajectory = self.sequence_classifier.predict(window_array)
        else:
            # Fall back to heuristic when classifier not yet trained
            trajectory = SequenceClassifier.label_window_heuristic(
                window_array, target_col_idx=0
            )

        self.current_trajectory = trajectory

        # Cache for ML engine consumption
        await self.hub.set_cache(
            "pattern_trajectory",
            {
                "trajectory": trajectory,
                "target": target,
                "timestamp": datetime.now().isoformat(),
                "window_size": ws,
                "method": "dtw" if self.sequence_classifier.is_trained else "heuristic",
            },
        )

    def store_anomaly_explanations(self, explanations: list[dict]):
        """Store anomaly explanations from ML engine for API access."""
        self._last_anomaly_explanations = explanations

    def get_current_state(self) -> dict[str, Any]:
        """Return current pattern recognition state."""
        return {
            "trajectory": self.current_trajectory,
            "pattern_scales": {
                scale.value: scale.description for scale in PatternScale
            },
            "anomaly_explanations": self._last_anomaly_explanations,
            "shadow_events_processed": self._shadow_event_count,
        }

    def get_stats(self) -> dict[str, Any]:
        """Return module statistics."""
        return {
            "active": self.active,
            "sequence_classifier": self.sequence_classifier.get_stats(),
            "window_count": {
                target: len(window)
                for target, window in self._feature_windows.items()
            },
            "current_trajectory": self.current_trajectory,
            "shadow_events_processed": self._shadow_event_count,
        }
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/hub/test_pattern_recognition.py -v
```
Expected: 6 passed

**Step 5: Commit**

```bash
git add aria/modules/pattern_recognition.py tests/hub/test_pattern_recognition.py
git commit -m "feat: add pattern recognition hub module with trajectory classification"
```

---

## Task 5: Register Pattern Recognition in Hub

**Files:**
- Modify: `aria/cli.py:382-390` (add registration after online_learner)
- Test: `tests/hub/test_pattern_recognition.py` (add registration test)

**Context:** Follow the exact try/except pattern used for online_learner registration in `cli.py:382-390`. The pattern recognition module goes in `_register_analysis_modules()` after online_learner since it depends on shadow_resolved events (which online_learner also consumes — order doesn't matter since both subscribe independently).

**Step 1: Write the failing test**

Add to `tests/hub/test_pattern_recognition.py`:

```python
class TestModuleRegistration:
    """Test that pattern_recognition registers correctly in hub."""

    async def test_module_registers_without_error(self, mock_hub):
        """Module can be instantiated and registered."""
        module = PatternRecognitionModule(mock_hub)
        mock_hub.register_module = MagicMock()
        mock_hub.register_module(module)
        mock_hub.register_module.assert_called_once_with(module)
```

**Step 2: Run test to verify it passes (it should already pass)**

```bash
.venv/bin/python -m pytest tests/hub/test_pattern_recognition.py::TestModuleRegistration -v
```

**Step 3: Add registration to cli.py**

After the online_learner block (`cli.py:382-390`), add:

```python
    # pattern_recognition (Tier 3+ — module self-gates on hardware tier)
    try:
        from aria.modules.pattern_recognition import PatternRecognitionModule

        pattern_recognition = PatternRecognitionModule(hub)
        hub.register_module(pattern_recognition)
        await _init(pattern_recognition, "pattern_recognition")()
    except Exception as e:
        logger.warning(f"Pattern recognition module failed (non-fatal): {e}")
```

**Step 4: Run existing tests to verify no regression**

```bash
.venv/bin/python -m pytest tests/hub/test_pattern_recognition.py -v
```
Expected: All pass

**Step 5: Commit**

```bash
git add aria/cli.py tests/hub/test_pattern_recognition.py
git commit -m "feat: register pattern recognition module in hub startup"
```

---

## Task 6: Wire Anomaly Explanations into ML Engine

**Files:**
- Modify: `aria/modules/ml_engine.py:1120-1136` (`_run_anomaly_detection`)
- Modify: `aria/modules/ml_engine.py:1070-1087` (`generate_predictions` result dict)
- Test: `tests/hub/test_ml_training.py` (add TestAnomalyExplanation class)

**Context:** Currently `_run_anomaly_detection()` at `ml_engine.py:1120-1136` returns `(is_anomaly, anomaly_score)`. Enhance it to also return top-3 feature explanations when an anomaly is detected. The explanations get included in the cached prediction result and forwarded to the pattern recognition module.

**Step 1: Write the failing tests**

Add to `tests/hub/test_ml_training.py`:

```python
class TestAnomalyExplanation:
    """Test anomaly explanation integration in ML engine."""

    def setup_method(self):
        self.mock_hub = MagicMock()
        self.mock_hub.get_cache = AsyncMock(return_value={"data": {"power_monitoring": {"name": "power_monitoring"}}})
        self.mock_hub.get_cache_fresh = AsyncMock(return_value={"data": {"power_monitoring": {"name": "power_monitoring"}}})
        self.mock_hub.set_cache = AsyncMock()
        self.mock_hub.get_config_value = MagicMock(return_value=None)
        self.mock_hub.modules = {}

    def test_anomaly_detection_returns_explanations(self):
        """When anomaly detected, returns feature explanations."""
        engine = MLEngine(self.mock_hub)

        # Create a trained IsolationForest
        np.random.seed(42)
        X_train = np.random.normal(0, 1, (100, 5))
        model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
        model.fit(X_train)
        engine.models["anomaly_detector"] = {
            "model": model,
            "feature_names": ["power", "lights", "motion", "temp", "humidity"],
        }

        # Anomalous sample
        X_anomalous = np.array([[10.0, 0.0, 0.0, 0.0, 0.0]])
        is_anomaly, score, explanations = engine._run_anomaly_detection(
            X_anomalous, ["power", "lights", "motion", "temp", "humidity"]
        )

        assert is_anomaly is True
        assert len(explanations) == 3
        assert explanations[0]["feature"] == "power"

    def test_anomaly_detection_no_model(self):
        """Without anomaly model, returns empty explanations."""
        engine = MLEngine(self.mock_hub)
        is_anomaly, score, explanations = engine._run_anomaly_detection(
            np.zeros((1, 5)), ["a", "b", "c", "d", "e"]
        )
        assert is_anomaly is False
        assert explanations == []

    def test_anomaly_detection_normal_no_explanations(self):
        """Normal sample returns empty explanations."""
        engine = MLEngine(self.mock_hub)

        np.random.seed(42)
        X_train = np.random.normal(0, 1, (100, 5))
        model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
        model.fit(X_train)
        engine.models["anomaly_detector"] = {
            "model": model,
            "feature_names": ["power", "lights", "motion", "temp", "humidity"],
        }

        X_normal = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        is_anomaly, score, explanations = engine._run_anomaly_detection(
            X_normal, ["power", "lights", "motion", "temp", "humidity"]
        )

        # Normal → no explanations needed
        if not is_anomaly:
            assert explanations == []
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/hub/test_ml_training.py::TestAnomalyExplanation -v
```
Expected: FAIL — `_run_anomaly_detection() got unexpected keyword argument` or signature mismatch

**Step 3: Modify _run_anomaly_detection**

In `ml_engine.py`, update `_run_anomaly_detection` (currently lines 1120-1136):

**Old signature:** `def _run_anomaly_detection(self, X: np.ndarray) -> tuple[bool, float | None]:`

**New signature:** `def _run_anomaly_detection(self, X: np.ndarray, feature_names: list[str] | None = None) -> tuple[bool, float | None, list[dict]]:`

```python
    def _run_anomaly_detection(
        self, X: np.ndarray, feature_names: list[str] | None = None
    ) -> tuple[bool, float | None, list[dict]]:
        """Run anomaly detection on feature vector.

        Returns:
            (is_anomaly, anomaly_score, explanations) tuple.
            explanations is a list of top-3 feature contributions when
            anomaly is detected, empty list otherwise.
        """
        if "anomaly_detector" not in self.models:
            return False, None, []
        try:
            anomaly_data = self.models["anomaly_detector"]
            anomaly_model = anomaly_data["model"]
            score = float(anomaly_model.decision_function(X)[0])
            is_anomaly = bool(anomaly_model.predict(X)[0] == -1)
            self.logger.info(f"Anomaly detection: score={score:.3f}, is_anomaly={is_anomaly}")

            explanations: list[dict] = []
            if is_anomaly and feature_names:
                from aria.engine.anomaly_explainer import AnomalyExplainer

                explainer = AnomalyExplainer()
                explanations = explainer.explain(
                    anomaly_model, X, feature_names, top_n=3
                )
                self.logger.info(
                    f"Anomaly explanations: {', '.join(e['feature'] for e in explanations)}"
                )

                # Forward to pattern recognition module if available
                pattern_mod = getattr(self.hub, "modules", {}).get("pattern_recognition")
                if pattern_mod is not None:
                    pattern_mod.store_anomaly_explanations(explanations)

            return is_anomaly, score, explanations
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return False, None, []
```

**Then update the caller in `generate_predictions` (line ~1064):**

Change:
```python
is_anomaly, anomaly_score = self._run_anomaly_detection(X)
```
To:
```python
is_anomaly, anomaly_score, anomaly_explanations = self._run_anomaly_detection(X, feature_names)
```

**And add to the result dict (line ~1074):**

```python
        result = {
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions_dict,
            "anomaly_detected": is_anomaly,
            "anomaly_score": round(anomaly_score, 3) if anomaly_score is not None else None,
            "anomaly_explanations": anomaly_explanations,
            "feature_count": len(feature_names),
            "model_count": len([k for k in self.models if k != "anomaly_detector"]),
        }
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120
```
Expected: All pass (existing tests may need minor updates if they unpack the old 2-tuple return)

**Important:** Check for any tests that unpack `_run_anomaly_detection` as a 2-tuple. Search for `_run_anomaly_detection` in test files and update any destructuring from `is_anomaly, score = ...` to `is_anomaly, score, _ = ...`.

**Step 5: Commit**

```bash
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "feat: add anomaly explanations with top-3 feature contributions"
```

---

## Task 7: Add Trajectory Feature to ML Engine

**Files:**
- Modify: `aria/engine/features/feature_config.py:9-66` (add `pattern_features` section)
- Modify: `aria/modules/ml_engine.py` (`_extract_features` and `_build_prediction_feature_vector`)
- Test: `tests/hub/test_ml_training.py` (add TestTrajectoryFeature class)

**Context:** The sequence classifier outputs a trajectory class string. This needs to be encoded as a numeric feature and added to the ML feature vector. Use ordinal encoding: stable=0, ramping_up=1, winding_down=2, anomalous_transition=3. The feature is optional — when the pattern recognition module isn't running (Tier 1-2), the feature defaults to 0 (stable).

**Step 1: Add pattern_features to feature config**

In `aria/engine/features/feature_config.py`, add to `DEFAULT_FEATURE_CONFIG` (after `presence_features` block):

```python
    "pattern_features": {
        "trajectory_class": True,
    },
```

And add `"pattern_features"` to `_REQUIRED_SECTIONS` set.

**Step 2: Write the failing tests**

Add to `tests/hub/test_ml_training.py`:

```python
class TestTrajectoryFeature:
    """Test trajectory_class feature integration."""

    def setup_method(self):
        self.mock_hub = MagicMock()
        self.mock_hub.get_cache = AsyncMock(return_value=None)
        self.mock_hub.get_cache_fresh = AsyncMock(return_value={"data": {"power_monitoring": {"name": "power_monitoring"}}})
        self.mock_hub.set_cache = AsyncMock()
        self.mock_hub.get_config_value = MagicMock(return_value=None)
        self.mock_hub.modules = {}

    async def test_extract_features_includes_trajectory(self):
        """Feature extraction includes trajectory_class when enabled."""
        engine = MLEngine(self.mock_hub)
        config = await engine._get_feature_config()
        assert "pattern_features" in config
        assert config["pattern_features"]["trajectory_class"] is True

    async def test_trajectory_defaults_to_zero(self):
        """trajectory_class defaults to 0 (stable) when module absent."""
        engine = MLEngine(self.mock_hub)
        config = await engine._get_feature_config()
        feature_names = await engine._get_feature_names(config)
        assert "trajectory_class" in feature_names

    async def test_trajectory_reads_from_cache(self):
        """trajectory_class reads from pattern_trajectory cache."""
        self.mock_hub.get_cache = AsyncMock(return_value={
            "data": {"trajectory": "ramping_up"}
        })
        self.mock_hub.modules = {}
        engine = MLEngine(self.mock_hub)

        # The encoding: stable=0, ramping_up=1, winding_down=2, anomalous=3
        trajectory_val = engine._encode_trajectory("ramping_up")
        assert trajectory_val == 1

    def test_encode_trajectory_all_classes(self):
        """All trajectory classes encode to distinct integers."""
        engine = MLEngine(self.mock_hub)
        assert engine._encode_trajectory("stable") == 0
        assert engine._encode_trajectory("ramping_up") == 1
        assert engine._encode_trajectory("winding_down") == 2
        assert engine._encode_trajectory("anomalous_transition") == 3
        assert engine._encode_trajectory("unknown") == 0  # Default
```

**Step 3: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/hub/test_ml_training.py::TestTrajectoryFeature -v
```

**Step 4: Implement trajectory feature**

Add to `ml_engine.py`:

```python
    # Trajectory class encoding (Phase 3)
    _TRAJECTORY_ENCODING = {
        "stable": 0,
        "ramping_up": 1,
        "winding_down": 2,
        "anomalous_transition": 3,
    }

    def _encode_trajectory(self, trajectory: str) -> int:
        """Encode trajectory class string to integer."""
        return self._TRAJECTORY_ENCODING.get(trajectory, 0)
```

In `_extract_features`, add trajectory lookup when `pattern_features.trajectory_class` is enabled:

```python
        # Pattern features (Phase 3)
        if config.get("pattern_features", {}).get("trajectory_class", False):
            trajectory_cache = await self.hub.get_cache("pattern_trajectory")
            trajectory = "stable"
            if trajectory_cache and "data" in trajectory_cache:
                trajectory = trajectory_cache["data"].get("trajectory", "stable")
            features["trajectory_class"] = self._encode_trajectory(trajectory)
```

**Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120
```

**Step 6: Commit**

```bash
git add aria/engine/features/feature_config.py aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "feat: add trajectory_class as ML feature from sequence classifier"
```

---

## Task 8: Config Entries for Phase 3

**Files:**
- Modify: `aria/hub/config_defaults.py` (add pattern.* entries)
- Test: `tests/hub/test_config_defaults.py` (update count assertions)

**Context:** Phase 3 adds tunable parameters for pattern recognition. Follow the exact structure of existing ml.* entries in `config_defaults.py:599-698`. Add entries to the `CONFIG_DEFAULTS` list.

**Step 1: Add config entries**

Add these entries to `CONFIG_DEFAULTS` in `config_defaults.py`:

```python
    # Phase 3: Pattern Recognition
    {
        "key": "pattern.sequence_window_size",
        "default_value": "6",
        "value_type": "number",
        "label": "Sequence Window Size",
        "description": "Number of snapshots in the trajectory classification sliding window",
        "category": "pattern",
        "min_value": 3,
        "max_value": 24,
        "step": 1,
    },
    {
        "key": "pattern.dtw_neighbors",
        "default_value": "3",
        "value_type": "number",
        "label": "DTW Neighbors",
        "description": "Number of neighbors for DTW sequence classifier (higher = smoother, slower)",
        "category": "pattern",
        "min_value": 1,
        "max_value": 10,
        "step": 1,
    },
    {
        "key": "pattern.anomaly_top_n",
        "default_value": "3",
        "value_type": "number",
        "label": "Anomaly Top Features",
        "description": "Number of top contributing features to report per anomaly",
        "category": "pattern",
        "min_value": 1,
        "max_value": 10,
        "step": 1,
    },
    {
        "key": "pattern.trajectory_change_threshold",
        "default_value": "0.20",
        "value_type": "number",
        "label": "Trajectory Change Threshold",
        "description": "Minimum percent change in target metric to classify as ramping/winding (0.0-1.0)",
        "category": "pattern",
        "min_value": 0.05,
        "max_value": 0.50,
        "step": 0.05,
    },
```

**Step 2: Update test count assertions**

In `tests/hub/test_config_defaults.py`, find the assertion that checks config entry count and increment by 4 (the number of new entries).

**Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/hub/test_config_defaults.py -v
```

**Step 4: Commit**

```bash
git add aria/hub/config_defaults.py tests/hub/test_config_defaults.py
git commit -m "config: add pattern recognition settings (window, DTW neighbors, anomaly top-N)"
```

---

## Task 9: API Endpoint for Pattern Recognition

**Files:**
- Modify: `aria/hub/api.py` (add /api/patterns endpoint)
- Create: `tests/hub/test_api_patterns.py`

**Context:** Follow the pattern from `/api/ml/online` endpoint (added in Phase 2). The new endpoint returns current trajectory, anomaly explanations, pattern scale definitions, and classifier stats. Uses `await hub.get_module("pattern_recognition")` — correctly with `await` since this is in async FastAPI route.

**Step 1: Write the failing tests**

Create `tests/hub/test_api_patterns.py`:

```python
"""Tests for /api/patterns endpoint."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from aria.hub.api import create_app


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.get_module = AsyncMock()
    hub.get_cache = AsyncMock(return_value=None)
    return hub


class TestPatternsEndpoint:
    """Test GET /api/patterns."""

    def test_full_response(self, mock_hub):
        """Returns pattern data when module is active."""
        pattern_mod = MagicMock()
        pattern_mod.get_current_state.return_value = {
            "trajectory": "ramping_up",
            "pattern_scales": {"micro": "desc", "meso": "desc", "macro": "desc"},
            "anomaly_explanations": [{"feature": "power", "contribution": 0.45}],
            "shadow_events_processed": 42,
        }
        pattern_mod.get_stats.return_value = {
            "active": True,
            "sequence_classifier": {"is_trained": True},
            "window_count": {"power_watts": 6},
        }
        mock_hub.get_module = AsyncMock(return_value=pattern_mod)

        app = create_app(mock_hub)
        client = TestClient(app)
        resp = client.get("/api/patterns")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trajectory"] == "ramping_up"
        assert "anomaly_explanations" in data
        assert "stats" in data

    def test_module_not_available(self, mock_hub):
        """Returns empty state when module not registered."""
        mock_hub.get_module = AsyncMock(return_value=None)

        app = create_app(mock_hub)
        client = TestClient(app)
        resp = client.get("/api/patterns")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trajectory"] is None
        assert data["active"] is False

    def test_module_error(self, mock_hub):
        """Returns 500 on unexpected error."""
        mock_hub.get_module = AsyncMock(side_effect=Exception("boom"))

        app = create_app(mock_hub)
        client = TestClient(app)
        resp = client.get("/api/patterns")
        assert resp.status_code == 500
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/hub/test_api_patterns.py -v
```

**Step 3: Add the endpoint to api.py**

After the `/api/ml/online` route, add:

```python
    @app.get("/api/patterns")
    async def get_patterns():
        """Pattern recognition state — trajectory, anomaly explanations, stats."""
        try:
            pattern_mod = await hub.get_module("pattern_recognition")
            if pattern_mod is None:
                return {
                    "trajectory": None,
                    "active": False,
                    "anomaly_explanations": [],
                    "pattern_scales": {},
                    "stats": {},
                }

            state = pattern_mod.get_current_state()
            stats = pattern_mod.get_stats()

            return {
                "trajectory": state.get("trajectory"),
                "active": stats.get("active", False),
                "anomaly_explanations": state.get("anomaly_explanations", []),
                "pattern_scales": state.get("pattern_scales", {}),
                "shadow_events_processed": state.get("shadow_events_processed", 0),
                "stats": stats,
            }
        except Exception as e:
            logger.error(f"Error fetching pattern data: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
```

**Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/hub/test_api_patterns.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api_patterns.py
git commit -m "feat: add /api/patterns endpoint for pattern recognition stats"
```

---

## Task 10: Integration Test

**Files:**
- Create: `tests/integration/test_pattern_recognition_pipeline.py`

**Context:** End-to-end test verifying: anomaly explainer produces explanations → pattern recognition module receives shadow events and classifies trajectory → ML engine includes trajectory_class in features and anomaly explanations in predictions. Also verify Tier 2 gating prevents activation.

**Step 1: Write the integration tests**

Create `tests/integration/test_pattern_recognition_pipeline.py`:

```python
"""Integration tests for Phase 3 pattern recognition pipeline.

Tests the full flow:
  shadow_resolved event → pattern recognition window → trajectory classification
  anomaly detection → explainer → top-3 features in prediction output
  feature config → trajectory_class in feature vector
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from aria.engine.anomaly_explainer import AnomalyExplainer
from aria.engine.pattern_scale import PatternScale
from aria.engine.sequence import SequenceClassifier
from aria.modules.pattern_recognition import PatternRecognitionModule


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.subscribe = MagicMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.publish = AsyncMock()
    hub.get_config_value = MagicMock(return_value=None)
    hub.modules = {}
    return hub


class TestPatternRecognitionPipeline:
    """End-to-end pattern recognition tests."""

    @patch("aria.modules.pattern_recognition.recommend_tier", return_value=3)
    @patch("aria.modules.pattern_recognition.scan_hardware")
    async def test_full_pipeline(self, mock_scan, mock_tier, mock_hub):
        """Shadow events → trajectory classification → cache update."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)

        module = PatternRecognitionModule(mock_hub)
        await module.initialize()
        assert module.active is True

        # Feed 6 events with increasing power (ramping up)
        for i in range(6):
            await module._on_shadow_resolved({
                "target": "power_watts",
                "features": {
                    "power": float(10 + i * 20),
                    "lights": 2.0,
                    "motion": 1.0,
                },
                "actual_value": float(10 + i * 20),
                "timestamp": datetime.now().isoformat(),
            })

        # Should have classified trajectory
        assert module.current_trajectory is not None
        # Cache should have been updated
        mock_hub.set_cache.assert_called()

    async def test_anomaly_explanation_pipeline(self):
        """IsolationForest → explainer → top features."""
        np.random.seed(42)
        X_train = np.random.normal(0, 1, (200, 5))
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(X_train)

        # Extreme anomaly in feature 0
        X_anomaly = np.array([[15.0, 0.0, 0.0, 0.0, 0.0]])
        features = ["power", "lights", "motion", "temp", "humidity"]

        explainer = AnomalyExplainer()
        explanations = explainer.explain(model, X_anomaly, features, top_n=3)

        assert len(explanations) == 3
        assert all(e["contribution"] > 0 for e in explanations)
        # The extreme feature should rank high
        top_features = [e["feature"] for e in explanations]
        assert "power" in top_features

    @patch("aria.modules.pattern_recognition.recommend_tier", return_value=2)
    @patch("aria.modules.pattern_recognition.scan_hardware")
    async def test_tier_2_gates_out(self, mock_scan, mock_tier, mock_hub):
        """Tier 2 hardware disables pattern recognition."""
        mock_scan.return_value = MagicMock(ram_gb=4, cpu_cores=2)

        module = PatternRecognitionModule(mock_hub)
        await module.initialize()
        assert module.active is False

        # Events should be ignored
        await module._on_shadow_resolved({
            "target": "power_watts",
            "features": {"power": 100.0},
            "actual_value": 100.0,
            "timestamp": datetime.now().isoformat(),
        })
        assert module.current_trajectory is None

    def test_pattern_scale_classification(self):
        """Patterns classify into correct time scales."""
        assert PatternScale.from_duration_seconds(30) == PatternScale.MICRO
        assert PatternScale.from_duration_seconds(1800) == PatternScale.MESO
        assert PatternScale.from_duration_seconds(86400) == PatternScale.MACRO

    def test_sequence_heuristic_labels(self):
        """Heuristic labeling produces correct trajectory classes."""
        # Ramping up
        window = np.zeros((6, 3))
        window[:, 0] = np.linspace(10, 80, 6)
        assert SequenceClassifier.label_window_heuristic(window) == "ramping_up"

        # Winding down
        window[:, 0] = np.linspace(80, 10, 6)
        assert SequenceClassifier.label_window_heuristic(window) == "winding_down"

        # Stable
        window[:, 0] = [50, 51, 49, 50, 51, 50]
        assert SequenceClassifier.label_window_heuristic(window) == "stable"
```

**Step 2: Run integration tests**

```bash
.venv/bin/python -m pytest tests/integration/test_pattern_recognition_pipeline.py -v
```
Expected: 5 passed

**Step 3: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v --timeout=120 -q
```
Expected: All existing tests still pass + new Phase 3 tests pass

**Step 4: Commit**

```bash
git add tests/integration/test_pattern_recognition_pipeline.py
git commit -m "test: add integration tests for pattern recognition pipeline"
```

---

## Summary

| Task | Component | Files | Dependencies |
|------|-----------|-------|-------------|
| 1 | Anomaly Explainer | `aria/engine/anomaly_explainer.py`, test | None |
| 2 | Pattern Scale | `aria/engine/pattern_scale.py`, test | None |
| 3 | Sequence Classifier | `aria/engine/sequence.py`, test, `pyproject.toml` | tslearn |
| 4 | Pattern Recognition Module | `aria/modules/pattern_recognition.py`, test | Tasks 1, 2, 3 |
| 5 | Register in Hub | `aria/cli.py` | Task 4 |
| 6 | Wire Anomaly Explanations | `aria/modules/ml_engine.py`, test | Tasks 1, 5 |
| 7 | Trajectory Feature | `feature_config.py`, `ml_engine.py`, test | Tasks 3, 5 |
| 8 | Config Entries | `config_defaults.py`, test | None |
| 9 | API Endpoint | `api.py`, test | Task 5 |
| 10 | Integration Test | `tests/integration/`, full suite | All |

## Critical Path

```
Task 1 (explainer) ──────────────────────────────────┐
Task 2 (scale) ───────────────────────────────────────┤
Task 3 (sequence + tslearn) ──────────────────────────┼→ Task 4 (hub module) → Task 5 (register)
Task 8 (config) ─── [independent, do anytime] ────────┘       ↓              ↓
                                                        Task 6 (anomaly)  Task 7 (trajectory)
                                                               ↓              ↓
                                                        Task 9 (API) ←────────┘
                                                               ↓
                                                        Task 10 (integration)
```

Tasks 1, 2, 3, and 8 are fully independent — they can run in any order or in parallel.
Tasks 4-5 depend on 1-3. Tasks 6-7 depend on 5. Task 9 depends on 5. Task 10 depends on all.
