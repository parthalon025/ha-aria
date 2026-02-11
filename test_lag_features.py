#!/usr/bin/env python3
"""Test lag features with synthetic snapshots."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.ml_engine import MLEngine


def create_synthetic_snapshot(date_str, power_watts=1000, lights_on=5, devices=3):
    """Create a synthetic snapshot for testing."""
    return {
        "date": date_str,
        "day_of_week": "Monday",
        "is_weekend": False,
        "is_holiday": False,
        "weather": {
            "temp_f": 70,
            "humidity_pct": 50,
            "wind_mph": 5,
        },
        "power": {
            "total_watts": power_watts,
        },
        "lights": {
            "on": lights_on,
            "total_brightness": lights_on * 100,
        },
        "occupancy": {
            "people_home_count": 2,
            "device_count_home": devices,
        },
        "motion": {
            "active_count": 1,
        },
        "media": {
            "total_active": 0,
        },
        "ev": {
            "TARS": {
                "battery_pct": 80,
                "is_charging": False,
            }
        },
        "entities": {},
        "logbook_summary": {},
    }


def test_lag_features():
    """Test that lag features are computed correctly."""

    # Create 10 synthetic snapshots with varying values
    base_date = datetime(2026, 2, 1)
    snapshots = []

    for i in range(10):
        date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        # Vary power and lights to test lag features
        power = 1000 + (i * 100)  # Increasing pattern
        lights = 5 + (i % 3)  # Cyclical pattern
        devices = 2 + (i % 2)  # Alternating pattern

        snapshots.append(create_synthetic_snapshot(date_str, power, lights, devices))

    print(f"Created {len(snapshots)} synthetic snapshots")
    print(f"Power range: {snapshots[0]['power']['total_watts']} to {snapshots[-1]['power']['total_watts']}")
    print(f"Lights range: {min(s['lights']['on'] for s in snapshots)} to {max(s['lights']['on'] for s in snapshots)}")
    print()

    # Create ML engine
    class MockHub:
        pass

    engine = MLEngine(
        hub=MockHub(),
        models_dir="/tmp/test_models",
        training_data_dir="/tmp"
    )

    # Build training dataset
    print("Building training dataset for power_watts...")
    X, y = engine._build_training_dataset(snapshots, "power_watts")

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print()

    if len(X) == 0:
        print("ERROR: No training data extracted")
        return False

    # Get feature names
    config = engine._get_feature_config()
    feature_names = engine._get_feature_names(config)

    # Find lag feature indices
    lag_indices = {}
    for name in ["prev_snapshot_power", "prev_snapshot_lights", "prev_snapshot_occupancy",
                 "rolling_7d_power_mean", "rolling_7d_lights_mean"]:
        if name in feature_names:
            lag_indices[name] = feature_names.index(name)

    print("Lag feature values by sample:")
    print(f"{'Sample':<8} {'prev_power':<12} {'prev_lights':<12} {'prev_occ':<12} {'roll_power':<12} {'roll_lights':<12} {'target':<10}")
    print("-" * 90)

    for i in range(len(X)):
        row = []
        row.append(f"{i:<8}")
        for name in ["prev_snapshot_power", "prev_snapshot_lights", "prev_snapshot_occupancy",
                     "rolling_7d_power_mean", "rolling_7d_lights_mean"]:
            if name in lag_indices:
                value = X[i][lag_indices[name]]
                row.append(f"{value:<12.1f}")
            else:
                row.append(f"{'N/A':<12}")
        row.append(f"{y[i]:<10.1f}")
        print("".join(row))

    print()

    # Verify lag features are working
    errors = []

    # Check sample 1: prev_snapshot_power should match sample 0 target
    if len(X) >= 2:
        prev_power_idx = lag_indices.get("prev_snapshot_power")
        if prev_power_idx is not None:
            expected = y[0]
            actual = X[1][prev_power_idx]
            if abs(expected - actual) > 0.1:
                errors.append(f"Sample 1 prev_power mismatch: expected {expected}, got {actual}")
            else:
                print(f"✓ Sample 1 prev_snapshot_power = {actual:.1f} (matches sample 0 target)")

    # Check sample 7: rolling_7d_power_mean should be mean of samples 0-6
    if len(X) >= 8:
        roll_power_idx = lag_indices.get("rolling_7d_power_mean")
        if roll_power_idx is not None:
            expected = sum(y[0:7]) / 7
            actual = X[7][roll_power_idx]
            if abs(expected - actual) > 0.1:
                errors.append(f"Sample 7 rolling_7d_power_mean mismatch: expected {expected:.1f}, got {actual:.1f}")
            else:
                print(f"✓ Sample 7 rolling_7d_power_mean = {actual:.1f} (mean of samples 0-6)")

    print()

    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("✓ All lag features computed correctly!")
    print(f"✓ {len(X)} samples with {len(feature_names)} features each")

    return True


if __name__ == "__main__":
    success = test_lag_features()
    sys.exit(0 if success else 1)
