#!/usr/bin/env python3
"""Test feature extraction with a real snapshot from ha-intelligence."""

import json
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.ml_engine import MLEngine


def test_feature_extraction():
    """Test feature extraction on a real daily snapshot."""

    # Load a recent daily snapshot
    snapshot_path = Path.home() / "ha-logs/intelligence/daily/2026-02-10.json"

    if not snapshot_path.exists():
        print(f"ERROR: Snapshot not found at {snapshot_path}")
        return False

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    print(f"Loaded snapshot from {snapshot_path}")
    print(f"Snapshot keys: {list(snapshot.keys())}")
    print()

    # Create a minimal ML engine instance (no hub needed for feature extraction)
    class MockHub:
        """Mock hub for testing."""
        pass

    engine = MLEngine(
        hub=MockHub(),
        models_dir="/tmp/test_models",
        training_data_dir=str(snapshot_path.parent)
    )

    # Test feature config
    config = engine._get_feature_config()
    print(f"Feature config categories:")
    for category in ["time_features", "weather_features", "home_features", "lag_features", "interaction_features"]:
        count = sum(1 for v in config.get(category, {}).values() if v)
        print(f"  {category}: {count} enabled")
    print()

    # Test feature name generation
    feature_names = engine._get_feature_names(config)
    print(f"Total feature names: {len(feature_names)}")
    print(f"Feature names: {feature_names[:10]}... (showing first 10)")
    print()

    # Test time feature computation
    time_features = engine._compute_time_features(snapshot)
    print(f"Computed time features: {len(time_features)} features")
    print(f"Sample time features:")
    for key in ["hour_sin", "hour_cos", "is_weekend", "is_work_hours", "daylight_remaining_pct"]:
        print(f"  {key}: {time_features.get(key)}")
    print()

    # Test feature extraction
    features = engine._extract_features(snapshot, config=config)

    if features is None:
        print("ERROR: Feature extraction returned None")
        return False

    print(f"Extracted features: {len(features)} features")
    print(f"Expected features: {len(feature_names)} features")
    print()

    # Verify feature count matches
    if len(features) != len(feature_names):
        print(f"WARNING: Feature count mismatch! Extracted={len(features)}, Expected={len(feature_names)}")
        missing = set(feature_names) - set(features.keys())
        extra = set(features.keys()) - set(feature_names)
        if missing:
            print(f"Missing features: {missing}")
        if extra:
            print(f"Extra features: {extra}")
        return False

    # Show sample features by category
    print("Sample features by category:")

    # Time features
    time_feats = {k: v for k, v in features.items() if k in ["hour_sin", "hour_cos", "is_weekend", "is_night"]}
    print(f"  Time: {time_feats}")

    # Weather features
    weather_feats = {k: v for k, v in features.items() if k.startswith("weather_")}
    print(f"  Weather: {weather_feats}")

    # Home features
    home_feats = {k: v for k, v in features.items() if k in ["lights_on", "people_home_count", "motion_active_count"]}
    print(f"  Home: {home_feats}")

    # Lag features (should be 0 for single snapshot)
    lag_feats = {k: v for k, v in features.items() if k.startswith("prev_") or k.startswith("rolling_")}
    print(f"  Lag: {lag_feats}")

    print()
    print("✓ Feature extraction successful!")
    print(f"✓ Generated {len(features)} features matching config")

    return True


if __name__ == "__main__":
    success = test_feature_extraction()
    sys.exit(0 if success else 1)
