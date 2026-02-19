"""Integration tests: verify engine and hub can interoperate within the aria namespace."""

import json
from unittest.mock import MagicMock

from aria.engine.schema import REQUIRED_NESTED_KEYS, validate_snapshot_schema
from aria.modules.intelligence import METRIC_PATHS

# ---------------------------------------------------------------------------
# Contract tests: engine JSON schema ↔ hub reader (RISK-01)
# ---------------------------------------------------------------------------


def _minimal_valid_snapshot() -> dict:
    """Build a minimal snapshot that satisfies every required nested key."""
    return {
        "power": {"total_watts": 450.0},
        "occupancy": {"device_count_home": 2},
        "lights": {"on": 5},
        "logbook_summary": {"useful_events": 12},
        "entities": {"unavailable": 1},
    }


def test_snapshot_schema_round_trip():
    """A minimal valid snapshot should pass validate_snapshot_schema with no errors."""
    snapshot = _minimal_valid_snapshot()
    errors = validate_snapshot_schema(snapshot)
    assert errors == [], f"Expected no validation errors, got: {errors}"


def test_required_keys_match_hub_reader():
    """Every snapshot key accessed by METRIC_PATHS must be covered by REQUIRED_NESTED_KEYS.

    METRIC_PATHS uses d.get("section", {}).get("nested_key"). Each (section, nested_key)
    pair it accesses must appear in REQUIRED_NESTED_KEYS so the schema validator enforces
    those keys are present whenever the section exists.
    """

    # Extract (section, nested_key) pairs from METRIC_PATHS lambdas by running
    # them against a probe object that records attribute access.
    class _Probe(dict):
        def __init__(self, section, results):
            super().__init__()
            self._section = section
            self._results = results

        def get(self, key, default=None):
            if self._section is not None:
                self._results.append((self._section, key))
                return None
            # First-level get — return a probe for the section
            child = _Probe(key, self._results)
            return child

    accessed_pairs = []
    probe = _Probe(None, accessed_pairs)
    for extractor in METRIC_PATHS.values():
        extractor(probe)

    # Every (section, nested_key) the hub reader touches must be covered by schema
    for section, nested_key in accessed_pairs:
        assert section in REQUIRED_NESTED_KEYS, (
            f"METRIC_PATHS accesses section '{section}' but it is not in REQUIRED_NESTED_KEYS. "
            "Add it so schema validation enforces its structure."
        )
        assert nested_key in REQUIRED_NESTED_KEYS[section], (
            f"METRIC_PATHS accesses '{section}.{nested_key}' but '{nested_key}' is not in "
            f"REQUIRED_NESTED_KEYS['{section}']. Add it to close the contract gap."
        )


def test_schema_rejects_missing_required_keys():
    """A snapshot with a present-but-incomplete section must produce validation errors."""
    # Section present but missing its required nested key
    incomplete = {
        "power": {},  # missing "total_watts"
        "occupancy": {"device_count_home": 1},
        "lights": {"on": 3},
        "logbook_summary": {"useful_events": 5},
        "entities": {"unavailable": 0},
    }
    errors = validate_snapshot_schema(incomplete)
    assert len(errors) > 0, "Expected validation errors for incomplete 'power' section"
    assert any("power" in e for e in errors), f"Expected error mentioning 'power', got: {errors}"


def test_engine_output_consumable_by_hub(tmp_path):
    """A snapshot written by the engine should be readable by the hub intelligence module.

    Creates a realistic daily snapshot on disk, instantiates IntelligenceModule with
    a mocked hub, calls _extract_trend_data(), and verifies it produces valid cache entries.
    """
    from aria.modules.intelligence import IntelligenceModule

    # Write a valid daily snapshot to a temp directory
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir(parents=True)
    snapshot = _minimal_valid_snapshot()
    snapshot_file = daily_dir / "2026-02-18.json"
    snapshot_file.write_text(json.dumps(snapshot))

    # Build a minimal mock hub (IntelligenceModule only reads hub at cache-write time)
    mock_hub = MagicMock()
    mock_hub.logger = MagicMock()

    module = IntelligenceModule(hub=mock_hub, intelligence_dir=str(tmp_path))

    # _extract_trend_data reads the daily files, validates schema, extracts metrics
    trend = module._extract_trend_data([snapshot_file])

    # Should produce exactly one trend entry (one file, valid schema)
    assert len(trend) == 1, f"Expected 1 trend entry from valid snapshot, got {len(trend)}"

    entry = trend[0]
    assert entry["date"] == "2026-02-18", f"Expected date '2026-02-18', got {entry.get('date')}"

    # All METRIC_PATHS keys that have values in the snapshot should appear in the entry
    assert entry.get("power_watts") == 450.0
    assert entry.get("lights_on") == 5
    assert entry.get("devices_home") == 2
    assert entry.get("unavailable") == 1
    assert entry.get("useful_events") == 12


def test_engine_imports_accessible_from_hub():
    """Verify hub code can import engine modules."""
    from aria.engine.analysis.entity_correlations import summarize_entity_correlations
    from aria.engine.analysis.sequence_anomalies import MarkovChainDetector
    from aria.engine.config import AppConfig
    from aria.engine.storage.data_store import DataStore

    assert AppConfig is not None
    assert DataStore is not None
    assert summarize_entity_correlations is not None
    assert MarkovChainDetector is not None


def test_hub_imports_accessible():
    """Verify hub core can be imported."""
    from aria.hub.cache import CacheManager
    from aria.hub.constants import CACHE_INTELLIGENCE
    from aria.hub.core import IntelligenceHub, Module

    assert IntelligenceHub is not None
    assert Module is not None
    assert CacheManager is not None
    assert isinstance(CACHE_INTELLIGENCE, str)


def test_module_imports_accessible():
    """Verify all hub modules can be imported."""
    from aria.modules.activity_monitor import ActivityMonitor
    from aria.modules.discovery import DiscoveryModule
    from aria.modules.intelligence import IntelligenceModule
    from aria.modules.ml_engine import MLEngine
    from aria.modules.orchestrator import OrchestratorModule
    from aria.modules.patterns import PatternRecognition
    from aria.modules.shadow_engine import ShadowEngine

    assert IntelligenceModule is not None
    assert DiscoveryModule is not None
    assert ShadowEngine is not None
    assert ActivityMonitor is not None
    assert MLEngine is not None
    assert PatternRecognition is not None
    assert OrchestratorModule is not None


def test_engine_and_hub_share_namespace():
    """Verify engine and hub live under the same aria package."""
    import aria

    assert hasattr(aria, "__version__")

    import aria.engine
    import aria.hub
    import aria.modules

    # Both are subpackages of the same top-level
    assert aria.engine.__name__.startswith("aria.")
    assert aria.hub.__name__.startswith("aria.")
    assert aria.modules.__name__.startswith("aria.")


def test_cli_entry_point_importable():
    """Verify the CLI entry point can be imported."""
    from aria.cli import main

    assert callable(main)


# ---------------------------------------------------------------------------
# Contract tests: feature vector alignment — engine vs hub (RISK-05)
# ---------------------------------------------------------------------------


def _engine_shared_feature_names() -> list[str]:
    """Get the shared (non-pattern) feature names as the engine computes them.

    The engine's get_feature_names() includes 'trajectory_class' from pattern_features,
    but that feature is only built at hub runtime (via async cache read) — the engine's
    build_training_data() zero-fills it.  The shared contract is the prefix that both
    sides actively build: time + weather + home + lag + interaction + presence.
    """
    from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
    from aria.engine.features.vector_builder import get_feature_names

    all_names = get_feature_names(DEFAULT_FEATURE_CONFIG)
    # Strip trajectory_class — it's zero-filled in engine training, built async in hub
    return [n for n in all_names if n != "trajectory_class"]


def _hub_base_feature_names_from_default() -> list[str]:
    """Get the base (non-rolling-window, non-trajectory) feature names as the hub computes them.

    The hub's _get_feature_names() extends the shared base with hub-only features
    (rolling window stats, trajectory_class) that are only available at hub runtime.
    This function extracts the shared base to allow comparison with the engine.
    """
    from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
    from aria.modules.ml_engine import MLEngine

    config = DEFAULT_FEATURE_CONFIG
    names: list[str] = []
    MLEngine._collect_time_feature_names(config.get("time_features", {}), names)
    MLEngine._collect_dict_feature_names(config.get("weather_features", {}), names, prefix="weather_")
    MLEngine._collect_dict_feature_names(config.get("home_features", {}), names)
    MLEngine._collect_dict_feature_names(config.get("lag_features", {}), names)
    MLEngine._collect_dict_feature_names(config.get("interaction_features", {}), names)
    MLEngine._collect_dict_feature_names(config.get("presence_features", {}), names)
    # Exclude hub-only rolling window and pattern features — engine has no live activity log
    return names


def test_feature_names_engine_hub_base_identical():
    """Engine and hub base feature lists must be identical (same features, same order).

    RISK-05 mitigation: if someone adds/removes/reorders a feature on one side
    but not the other, this test catches it.  The hub appends extra hub-only
    features (rolling window stats, trajectory_class) AFTER the shared base —
    this test verifies only the shared prefix matches the engine exactly.

    trajectory_class is excluded from this comparison: the engine names it but
    zero-fills it in build_training_data(); the hub builds it async from cache
    and appends it after rolling window features.  Both sides agree on the value
    (zero or encoded int) — the divergence is ordering only, and it's intentional.
    """
    engine_shared = _engine_shared_feature_names()
    hub_base_names = _hub_base_feature_names_from_default()

    assert engine_shared == hub_base_names, (
        f"Engine and hub feature name lists diverged!\n"
        f"Engine shared ({len(engine_shared)} features): {engine_shared}\n"
        f"Hub base ({len(hub_base_names)} features): {hub_base_names}\n"
        f"Engine-only: {set(engine_shared) - set(hub_base_names)}\n"
        f"Hub-only:    {set(hub_base_names) - set(engine_shared)}"
    )


def test_feature_ordering_engine_hub_base_identical():
    """Column indices must match between engine and hub for the shared base features.

    An ML model trained with column 3 = 'lights_on' will produce garbage predictions
    if the hub scores with column 3 = 'people_home_count'.  This is a separate
    assertion from name-set equality — order matters for numpy matrix slicing.
    """
    engine_shared = _engine_shared_feature_names()
    hub_base_names = _hub_base_feature_names_from_default()

    for idx, (engine_name, hub_name) in enumerate(zip(engine_shared, hub_base_names, strict=False)):
        assert engine_name == hub_name, (
            f"Feature ordering diverged at index {idx}: engine has '{engine_name}', hub has '{hub_name}'"
        )


def test_hub_rolling_window_features_extend_engine_base():
    """Hub feature list must START with the engine shared base as a strict prefix.

    The hub appends hub-only rolling window features AFTER the shared base, then
    trajectory_class last.  This verifies the extension contract:
        hub_names[:len(engine_shared)] == engine_shared
    """
    from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
    from aria.modules.ml_engine import ROLLING_WINDOWS_HOURS, MLEngine

    engine_shared = _engine_shared_feature_names()

    # Build the full hub name list (mirrors _get_feature_names logic)
    config = DEFAULT_FEATURE_CONFIG
    hub_names: list[str] = []
    MLEngine._collect_time_feature_names(config.get("time_features", {}), hub_names)
    MLEngine._collect_dict_feature_names(config.get("weather_features", {}), hub_names, prefix="weather_")
    MLEngine._collect_dict_feature_names(config.get("home_features", {}), hub_names)
    MLEngine._collect_dict_feature_names(config.get("lag_features", {}), hub_names)
    MLEngine._collect_dict_feature_names(config.get("interaction_features", {}), hub_names)
    MLEngine._collect_dict_feature_names(config.get("presence_features", {}), hub_names)
    for hours in ROLLING_WINDOWS_HOURS:
        hub_names.extend(
            [
                f"rolling_{hours}h_event_count",
                f"rolling_{hours}h_domain_entropy",
                f"rolling_{hours}h_dominant_domain_pct",
                f"rolling_{hours}h_trend",
            ]
        )
    if config.get("pattern_features", {}).get("trajectory_class", False):
        hub_names.append("trajectory_class")

    assert len(hub_names) > len(engine_shared), (
        "Hub must have MORE features than the engine shared base (rolling window extras) — "
        f"got hub={len(hub_names)}, engine_shared={len(engine_shared)}"
    )

    shared_prefix = hub_names[: len(engine_shared)]
    assert shared_prefix == engine_shared, (
        f"Hub feature list does not start with the engine shared base as a strict prefix.\n"
        f"First {len(engine_shared)} hub features: {shared_prefix}\n"
        f"Engine shared:                            {engine_shared}"
    )


def test_feature_vector_build_and_name_list_consistent():
    """build_feature_vector() must produce the keys named by get_feature_names() (minus trajectory_class).

    trajectory_class is listed in get_feature_names() but is not built by
    build_feature_vector() — it is zero-filled by build_training_data() via
    fv.get(name, 0).  This is the accepted engine behavior: the feature is
    externally provided (from shadow engine / pattern cache) rather than
    computed from the snapshot itself.

    All other named features must appear in the built vector.
    """
    from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
    from aria.engine.features.vector_builder import build_feature_vector, get_feature_names

    # Minimal snapshot with all sections populated
    snapshot = {
        "time_features": {
            "hour_sin": 0.5,
            "hour_cos": 0.5,
            "dow_sin": 0.3,
            "dow_cos": 0.3,
            "month_sin": 0.1,
            "month_cos": 0.1,
            "day_of_year_sin": 0.2,
            "day_of_year_cos": 0.2,
            "is_weekend": False,
            "is_holiday": False,
            "is_night": False,
            "is_work_hours": True,
            "minutes_since_sunrise": 120,
            "minutes_until_sunset": 300,
            "daylight_remaining_pct": 0.7,
        },
        "weather": {"temp_f": 72.0, "humidity_pct": 55.0, "wind_mph": 10.0},
        "occupancy": {"people_home_count": 2, "device_count_home": 5, "people_home": ["alice"]},
        "lights": {"on": 3, "total_brightness": 180},
        "motion": {"active_count": 1},
        "media": {"total_active": 0},
        "ev": {"TARS": {"battery_pct": 80, "is_charging": False}},
        "presence": {
            "overall_probability": 0.9,
            "occupied_room_count": 2,
            "identified_person_count": 1,
            "camera_signal_count": 3,
        },
    }

    config = DEFAULT_FEATURE_CONFIG
    feature_names = get_feature_names(config)
    feature_vector = build_feature_vector(snapshot, config)

    # trajectory_class is intentionally not built by build_feature_vector() —
    # it is zero-filled in build_training_data() and built async by the hub.
    externally_provided = {"trajectory_class"}

    missing_from_vector = [
        name for name in feature_names if name not in feature_vector and name not in externally_provided
    ]
    assert missing_from_vector == [], (
        f"get_feature_names() lists features that build_feature_vector() did not produce "
        f"(excluding known externally-provided features {externally_provided}): "
        f"{missing_from_vector}"
    )
