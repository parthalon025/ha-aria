#!/usr/bin/env python3
"""Test training dataset building with lag features and rolling stats."""

import json
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.ml_engine import MLEngine


def test_training_dataset():
    """Test building training dataset with multiple snapshots."""

    # Load recent daily snapshots
    snapshots_dir = Path.home() / "ha-logs/intelligence/daily"
    snapshot_files = sorted(snapshots_dir.glob("2026-*.json"), reverse=True)[:10]

    if len(snapshot_files) < 8:
        print(f"ERROR: Need at least 8 snapshots, found {len(snapshot_files)}")
        return False

    snapshots = []
    for path in reversed(snapshot_files):  # Chronological order
        with open(path) as f:
            snapshots.append(json.load(f))

    print(f"Loaded {len(snapshots)} snapshots")
    print(f"Date range: {snapshots[0]['date']} to {snapshots[-1]['date']}")
    print()

    # Create a minimal ML engine instance
    class MockHub:
        """Mock hub for testing."""
        pass

    engine = MLEngine(
        hub=MockHub(),
        models_dir="/tmp/test_models",
        training_data_dir=str(snapshots_dir)
    )

    # Test building training dataset for power_watts
    print("Building training dataset for target: power_watts")
    X, y = engine._build_training_dataset(snapshots, "power_watts")

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print()

    if len(X) == 0:
        print("ERROR: No training data extracted")
        return False

    # Get feature names
    config = engine._get_feature_config()
    feature_names = engine._get_feature_names(config)

    print(f"Features: {len(feature_names)}")
    print(f"Samples: {len(X)}")
    print()

    # Check that lag features are populated (should be non-zero after first snapshot)
    print("Checking lag features in sample data:")
    lag_feature_indices = [
        i for i, name in enumerate(feature_names)
        if name.startswith("prev_") or name.startswith("rolling_")
    ]

    if len(X) >= 2:
        # Second sample should have prev_snapshot features
        sample_1 = X[1]
        print(f"Sample 1 (index 1) lag features:")
        for idx in lag_feature_indices:
            print(f"  {feature_names[idx]}: {sample_1[idx]}")
        print()

    if len(X) >= 8:
        # Eighth sample should have rolling features
        sample_7 = X[7]
        print(f"Sample 7 (index 7) lag features:")
        for idx in lag_feature_indices:
            print(f"  {feature_names[idx]}: {sample_7[idx]}")
        print()

    # Verify at least some non-zero lag features exist
    non_zero_lag = sum(
        1 for i in range(1, len(X))
        for idx in lag_feature_indices
        if X[i][idx] != 0
    )

    print(f"Non-zero lag features across all samples: {non_zero_lag}")
    print()

    if non_zero_lag == 0:
        print("WARNING: No non-zero lag features found - check implementation")
        return False

    # Check target values
    print(f"Target values (power_watts):")
    print(f"  Min: {y.min():.2f}")
    print(f"  Max: {y.max():.2f}")
    print(f"  Mean: {y.mean():.2f}")
    print(f"  Std: {y.std():.2f}")
    print()

    print("✓ Training dataset build successful!")
    print(f"✓ Extracted {len(X)} samples with {len(feature_names)} features each")
    print(f"✓ Lag features populated correctly")

    return True


if __name__ == "__main__":
    success = test_training_dataset()
    sys.exit(0 if success else 1)
