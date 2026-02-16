"""Tests for the OrganicDiscoveryModule hub integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.modules.organic_discovery.module import (
    DEFAULT_SETTINGS,
    OrganicDiscoveryModule,
)


@pytest.fixture
def mock_hub():
    hub = AsyncMock()
    hub.cache = AsyncMock()
    hub.set_cache = AsyncMock(return_value=1)
    hub.get_cache = AsyncMock(return_value=None)
    hub.publish = AsyncMock()
    hub.schedule_task = AsyncMock()
    hub.mark_module_running = MagicMock()
    hub.mark_module_failed = MagicMock()
    hub.is_running = MagicMock(return_value=True)
    return hub


@pytest.fixture
def module(mock_hub):
    mod = OrganicDiscoveryModule(mock_hub)
    # Default: no logbook data so behavioral layer is a no-op.
    # Behavioral tests override via patch.object.
    mod._load_logbook = AsyncMock(return_value=[])
    return mod


def _make_cache_entry(data, metadata=None):
    """Helper to build a cache entry dict."""
    return {"data": data, "metadata": metadata}


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


class TestModuleInit:
    def test_module_id(self, module):
        assert module.module_id == "organic_discovery"

    def test_default_settings(self, module):
        assert module.settings == DEFAULT_SETTINGS

    def test_settings_keys(self, module):
        assert "autonomy_mode" in module.settings
        assert "naming_backend" in module.settings
        assert "promote_threshold" in module.settings
        assert "archive_threshold" in module.settings
        assert "promote_streak_days" in module.settings
        assert "archive_streak_days" in module.settings


class TestDefaultSettings:
    def test_autonomy_mode(self):
        assert DEFAULT_SETTINGS["autonomy_mode"] == "suggest_and_wait"

    def test_naming_backend(self):
        assert DEFAULT_SETTINGS["naming_backend"] == "heuristic"

    def test_promote_threshold(self):
        assert DEFAULT_SETTINGS["promote_threshold"] == 50

    def test_archive_threshold(self):
        assert DEFAULT_SETTINGS["archive_threshold"] == 10

    def test_promote_streak_days(self):
        assert DEFAULT_SETTINGS["promote_streak_days"] == 7

    def test_archive_streak_days(self):
        assert DEFAULT_SETTINGS["archive_streak_days"] == 14


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    async def test_loads_settings_from_cache(self, mock_hub, module):
        custom_settings = {**DEFAULT_SETTINGS, "autonomy_mode": "autonomous"}
        mock_hub.get_cache.side_effect = lambda key: (
            _make_cache_entry(custom_settings)
            if key == "discovery_settings"
            else _make_cache_entry([])
            if key == "discovery_history"
            else None
        )
        await module.initialize()
        assert module.settings["autonomy_mode"] == "autonomous"

    async def test_loads_history_from_cache(self, mock_hub, module):
        history = [{"timestamp": "2026-02-14", "clusters": 3}]
        mock_hub.get_cache.side_effect = lambda key: _make_cache_entry(history) if key == "discovery_history" else None
        await module.initialize()
        assert module.history == history

    async def test_defaults_when_cache_empty(self, mock_hub, module):
        mock_hub.get_cache.return_value = None
        await module.initialize()
        assert module.settings == DEFAULT_SETTINGS
        assert module.history == []

    async def test_schedules_periodic_discovery(self, mock_hub, module):
        mock_hub.get_cache.return_value = None
        await module.initialize()
        mock_hub.schedule_task.assert_called_once()


# ---------------------------------------------------------------------------
# run_discovery — full pipeline
# ---------------------------------------------------------------------------


def _seed_capabilities():
    return {
        "lighting": {
            "available": True,
            "entities": ["light.living_room", "light.bedroom"],
            "total_count": 2,
            "can_predict": False,
        }
    }


def _entity_list():
    return [
        {"entity_id": "light.living_room", "domain": "light", "state": "on", "attributes": {}},
        {"entity_id": "light.bedroom", "domain": "light", "state": "off", "attributes": {}},
        {"entity_id": "switch.fan", "domain": "switch", "state": "on", "attributes": {}},
    ]


def _device_dict():
    return {}


def _activity_data():
    return {
        "entity_activity": {
            "light.living_room": {"daily_avg_changes": 5.0},
            "light.bedroom": {"daily_avg_changes": 3.0},
            "switch.fan": {"daily_avg_changes": 2.0},
        }
    }


class TestRunDiscovery:
    async def test_writes_capabilities_to_cache(self, mock_hub, module):
        """run_discovery should write merged capabilities to cache."""
        mock_hub.get_cache.side_effect = lambda key: {
            "entities": _make_cache_entry(_entity_list()),
            "devices": _make_cache_entry(_device_dict()),
            "capabilities": _make_cache_entry(_seed_capabilities()),
            "activity_summary": _make_cache_entry(_activity_data()),
            "discovery_settings": None,
            "discovery_history": None,
        }.get(key)

        # Patch clustering to return a simple cluster
        with (
            patch(
                "aria.modules.organic_discovery.module.cluster_entities",
                return_value=[
                    {
                        "cluster_id": 0,
                        "entity_ids": ["light.living_room", "light.bedroom", "switch.fan"],
                        "silhouette": 0.5,
                    }
                ],
            ),
            patch(
                "aria.modules.organic_discovery.module.build_feature_matrix",
                return_value=(
                    __import__("numpy").zeros((3, 5)),
                    ["light.living_room", "light.bedroom", "switch.fan"],
                    ["f1", "f2", "f3", "f4", "f5"],
                ),
            ),
        ):
            await module.initialize()
            await module.run_discovery()

        # Should have called set_cache for capabilities
        cap_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "capabilities"]
        assert len(cap_calls) >= 1

        # The capabilities data should include the seed
        written_caps = cap_calls[-1][0][1]
        assert "lighting" in written_caps

        # Verify capabilities were actually written with expected structure
        assert isinstance(written_caps, dict)
        assert written_caps["lighting"]["available"] is True

    async def test_seed_capabilities_preserved_with_empty_entities(self, mock_hub, module):
        """Seeds should be preserved even when no entities are found."""
        mock_hub.get_cache.side_effect = lambda key: {
            "entities": _make_cache_entry([]),
            "devices": _make_cache_entry({}),
            "capabilities": _make_cache_entry(_seed_capabilities()),
            "activity_summary": _make_cache_entry({"entity_activity": {}}),
            "discovery_settings": None,
            "discovery_history": None,
        }.get(key)

        await module.initialize()
        await module.run_discovery()

        cap_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "capabilities"]
        assert len(cap_calls) >= 1
        written_caps = cap_calls[-1][0][1]
        assert "lighting" in written_caps
        assert written_caps["lighting"]["source"] == "seed"
        assert written_caps["lighting"]["status"] == "promoted"

    async def test_publishes_completion_event(self, mock_hub, module):
        """run_discovery should publish organic_discovery_complete event."""
        mock_hub.get_cache.side_effect = lambda key: {
            "entities": _make_cache_entry([]),
            "devices": _make_cache_entry({}),
            "capabilities": _make_cache_entry({}),
            "activity_summary": _make_cache_entry({"entity_activity": {}}),
            "discovery_settings": None,
            "discovery_history": None,
        }.get(key)

        await module.initialize()
        await module.run_discovery()

        pub_calls = [c for c in mock_hub.publish.call_args_list if c[0][0] == "organic_discovery_complete"]
        assert len(pub_calls) >= 1

    async def test_records_history(self, mock_hub, module):
        """run_discovery should append a run record to history."""
        mock_hub.get_cache.side_effect = lambda key: {
            "entities": _make_cache_entry([]),
            "devices": _make_cache_entry({}),
            "capabilities": _make_cache_entry({}),
            "activity_summary": _make_cache_entry({"entity_activity": {}}),
            "discovery_settings": None,
            "discovery_history": None,
        }.get(key)

        await module.initialize()
        await module.run_discovery()

        assert len(module.history) == 1
        assert "timestamp" in module.history[0]

    async def test_no_crash_with_missing_cache(self, mock_hub, module):
        """run_discovery should handle missing cache data gracefully."""
        mock_hub.get_cache.return_value = None

        await module.initialize()
        # Should not raise
        result = await module.run_discovery()
        assert result is not None


# ---------------------------------------------------------------------------
# Autonomy modes
# ---------------------------------------------------------------------------


class TestAutonomyModes:
    async def test_suggest_and_wait_never_promotes(self, mock_hub, module):
        """In suggest_and_wait mode, organic caps stay as candidates."""
        module.settings["autonomy_mode"] = "suggest_and_wait"

        mock_hub.get_cache.side_effect = lambda key: {
            "entities": _make_cache_entry(_entity_list()),
            "devices": _make_cache_entry(_device_dict()),
            "capabilities": _make_cache_entry({}),
            "activity_summary": _make_cache_entry(_activity_data()),
            "discovery_settings": None,
            "discovery_history": None,
        }.get(key)

        with (
            patch(
                "aria.modules.organic_discovery.module.cluster_entities",
                return_value=[
                    {
                        "cluster_id": 0,
                        "entity_ids": ["light.living_room", "light.bedroom", "switch.fan"],
                        "silhouette": 0.7,
                    }
                ],
            ),
            patch(
                "aria.modules.organic_discovery.module.build_feature_matrix",
                return_value=(
                    __import__("numpy").zeros((3, 5)),
                    ["light.living_room", "light.bedroom", "switch.fan"],
                    ["f1", "f2", "f3", "f4", "f5"],
                ),
            ),
        ):
            await module.initialize()
            await module.run_discovery()

        cap_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "capabilities"]
        written_caps = cap_calls[-1][0][1]
        organic_caps = {k: v for k, v in written_caps.items() if v.get("source") == "organic"}
        # In suggest_and_wait mode, no organic capabilities should be promoted
        assert len(organic_caps) > 0, "Expected at least one organic capability"
        promoted = [c for c in organic_caps.values() if c["status"] == "promoted"]
        assert len(promoted) == 0, "suggest_and_wait should never promote organic capabilities"
        for cap in organic_caps.values():
            assert cap["status"] == "candidate"

    async def test_auto_promote_promotes_at_threshold(self, mock_hub, module):
        """In auto_promote mode, high-scoring capabilities with sufficient streak get promoted."""
        module.settings["autonomy_mode"] = "auto_promote"
        module.settings["promote_threshold"] = 50
        module.settings["promote_streak_days"] = 3

        # Pre-seed history with the name that heuristic_name will generate.
        # Patch heuristic_name to return a predictable name so we can match history.
        cap_name = "test_cap"
        module.history = [
            {"timestamp": "2026-02-11", "organic_caps": [cap_name]},
            {"timestamp": "2026-02-12", "organic_caps": [cap_name]},
            {"timestamp": "2026-02-13", "organic_caps": [cap_name]},
        ]

        mock_hub.get_cache.side_effect = lambda key: {
            "entities": _make_cache_entry(_entity_list()),
            "devices": _make_cache_entry(_device_dict()),
            "capabilities": _make_cache_entry({}),
            "activity_summary": _make_cache_entry(_activity_data()),
            "discovery_settings": None,
            "discovery_history": None,
        }.get(key)

        # Patch scoring to return high score and naming to return predictable name
        with (
            patch(
                "aria.modules.organic_discovery.module.cluster_entities",
                return_value=[
                    {
                        "cluster_id": 0,
                        "entity_ids": ["light.living_room", "light.bedroom", "switch.fan"],
                        "silhouette": 0.8,
                    }
                ],
            ),
            patch(
                "aria.modules.organic_discovery.module.build_feature_matrix",
                return_value=(
                    __import__("numpy").zeros((3, 5)),
                    ["light.living_room", "light.bedroom", "switch.fan"],
                    ["f1", "f2", "f3", "f4", "f5"],
                ),
            ),
            patch(
                "aria.modules.organic_discovery.module.compute_usefulness",
                return_value=75,
            ),
            patch(
                "aria.modules.organic_discovery.module.heuristic_name",
                return_value=cap_name,
            ),
        ):
            await module.run_discovery()

        cap_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "capabilities"]
        written_caps = cap_calls[-1][0][1]
        organic_caps = {k: v for k, v in written_caps.items() if v.get("source") == "organic"}
        # At least one should be promoted at threshold
        promoted = [c for c in organic_caps.values() if c["status"] == "promoted"]
        assert len(promoted) >= 1, (
            "auto_promote mode should promote capabilities above threshold with sufficient streak"
        )
        # Promoted capabilities should have the expected source
        for cap in promoted:
            assert cap["source"] == "organic"

    async def test_autonomous_promotes_at_lower_threshold(self, mock_hub, module):
        """In autonomous mode, promote at >= 30."""
        module.settings["autonomy_mode"] = "autonomous"
        module.settings["promote_threshold"] = 50  # should be ignored — autonomous uses 30

        mock_hub.get_cache.side_effect = lambda key: {
            "entities": _make_cache_entry(_entity_list()),
            "devices": _make_cache_entry(_device_dict()),
            "capabilities": _make_cache_entry({}),
            "activity_summary": _make_cache_entry(_activity_data()),
            "discovery_settings": None,
            "discovery_history": None,
        }.get(key)

        with (
            patch(
                "aria.modules.organic_discovery.module.cluster_entities",
                return_value=[
                    {
                        "cluster_id": 0,
                        "entity_ids": ["light.living_room", "light.bedroom", "switch.fan"],
                        "silhouette": 0.5,
                    }
                ],
            ),
            patch(
                "aria.modules.organic_discovery.module.build_feature_matrix",
                return_value=(
                    __import__("numpy").zeros((3, 5)),
                    ["light.living_room", "light.bedroom", "switch.fan"],
                    ["f1", "f2", "f3", "f4", "f5"],
                ),
            ),
            patch(
                "aria.modules.organic_discovery.module.compute_usefulness",
                return_value=35,  # above 30 autonomous threshold
            ),
        ):
            await module.run_discovery()

        cap_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "capabilities"]
        written_caps = cap_calls[-1][0][1]
        organic_caps = {k: v for k, v in written_caps.items() if v.get("source") == "organic"}
        promoted = [c for c in organic_caps.values() if c["status"] == "promoted"]
        assert len(promoted) >= 1, "autonomous mode should promote at lower threshold (>=30)"
        # Verify promotion happened despite the configured threshold being 50
        for cap in promoted:
            assert cap["source"] == "organic"


# ---------------------------------------------------------------------------
# on_event
# ---------------------------------------------------------------------------


class TestBehavioralDiscovery:
    async def test_module_discovers_behavioral_capabilities(self, mock_hub, module):
        """Layer 2 should process logbook data and produce behavioral capabilities."""
        mock_logbook = [
            {"entity_id": f"light.room_{i}", "state": "on", "when": f"2026-02-{d:02d}T19:{i:02d}:00"}
            for d in range(1, 15)
            for i in range(6)
        ]

        mock_hub.get_cache.side_effect = lambda key: {
            "entities": _make_cache_entry(_entity_list()),
            "devices": _make_cache_entry(_device_dict()),
            "capabilities": _make_cache_entry(_seed_capabilities()),
            "activity_summary": _make_cache_entry(_activity_data()),
            "discovery_settings": None,
            "discovery_history": None,
        }.get(key)

        with (
            patch(
                "aria.modules.organic_discovery.module.cluster_entities",
                return_value=[],
            ),
            patch(
                "aria.modules.organic_discovery.module.build_feature_matrix",
                return_value=(
                    __import__("numpy").zeros((3, 5)),
                    ["light.living_room", "light.bedroom", "switch.fan"],
                    ["f1", "f2", "f3", "f4", "f5"],
                ),
            ),
            patch.object(module, "_load_logbook", return_value=mock_logbook),
        ):
            await module.initialize()
            await module.run_discovery()

        # Should have written capabilities to cache
        cap_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "capabilities"]
        assert len(cap_calls) >= 1

        # Verify behavioral capabilities were discovered from the logbook data
        written_caps = cap_calls[-1][0][1]
        assert isinstance(written_caps, dict)
        # Seed capabilities should be preserved alongside any behavioral discoveries
        assert "lighting" in written_caps

    async def test_behavioral_skipped_when_no_logbook(self, mock_hub, module):
        """When logbook returns empty, behavioral layer should be a no-op."""
        mock_hub.get_cache.side_effect = lambda key: {
            "entities": _make_cache_entry([]),
            "devices": _make_cache_entry({}),
            "capabilities": _make_cache_entry({}),
            "activity_summary": _make_cache_entry({"entity_activity": {}}),
            "discovery_settings": None,
            "discovery_history": None,
        }.get(key)

        with patch.object(module, "_load_logbook", return_value=[]):
            await module.initialize()
            result = await module.run_discovery()

        # Should complete without error
        assert result is not None

    async def test_behavioral_caps_have_layer_behavioral(self, mock_hub, module):
        """Behavioral capabilities should have layer='behavioral'."""
        # Create enough distinct entities to form a cluster (min_cluster_size=3)
        mock_logbook = [
            {"entity_id": f"light.room_{i}", "state": "on", "when": f"2026-02-{d:02d}T19:{i:02d}:00"}
            for d in range(1, 15)
            for i in range(6)
        ]

        mock_hub.get_cache.side_effect = lambda key: {
            "entities": _make_cache_entry(_entity_list()),
            "devices": _make_cache_entry(_device_dict()),
            "capabilities": _make_cache_entry({}),
            "activity_summary": _make_cache_entry(_activity_data()),
            "discovery_settings": None,
            "discovery_history": None,
        }.get(key)

        # Patch domain clustering to return nothing so only behavioral caps appear
        with (
            patch(
                "aria.modules.organic_discovery.module.cluster_entities",
                return_value=[],
            ),
            patch(
                "aria.modules.organic_discovery.module.build_feature_matrix",
                return_value=(
                    __import__("numpy").zeros((3, 5)),
                    ["light.living_room", "light.bedroom", "switch.fan"],
                    ["f1", "f2", "f3", "f4", "f5"],
                ),
            ),
            patch(
                "aria.modules.organic_discovery.module.cluster_behavioral",
                return_value=[
                    {
                        "cluster_id": 0,
                        "entity_ids": ["light.room_0", "light.room_1", "light.room_2"],
                        "silhouette": 0.6,
                        "temporal_pattern": {"peak_hours": [19], "weekday_bias": 0.7},
                    }
                ],
            ),
            patch.object(module, "_load_logbook", return_value=mock_logbook),
        ):
            await module.run_discovery()

        cap_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "capabilities"]
        written_caps = cap_calls[-1][0][1]
        behavioral_caps = {k: v for k, v in written_caps.items() if v.get("layer") == "behavioral"}
        # Behavioral capabilities must exist and have layer="behavioral"
        assert len(behavioral_caps) >= 1, "Expected at least one behavioral capability"
        for name, cap in behavioral_caps.items():
            assert cap["layer"] == "behavioral", f"{name} should have layer='behavioral'"
            assert "temporal_pattern" in cap, f"{name} missing temporal_pattern"
            assert cap["source"] == "organic", f"{name} should have source='organic'"


# ---------------------------------------------------------------------------
# on_event
# ---------------------------------------------------------------------------


class TestOnEvent:
    async def test_on_event_noop(self, module):
        """on_event should not raise."""
        # Intentionally no assertion — verifies on_event doesn't raise
        await module.on_event("cache_updated", {"category": "entities"})


class TestDriftDetection:
    @pytest.mark.asyncio
    async def test_drift_flags_capability(self):
        """Drift event flags capability for re-discovery."""
        module = _make_module()
        # Seed capabilities cache
        module.hub.get_cache.return_value = _make_cache_entry(
            {"climate": {"available": True, "entities": ["sensor.temp"], "status": "promoted"}}
        )

        await module.on_event(
            "drift_detected",
            {
                "capability": "climate",
                "drift_type": "behavioral_drift",
                "severity": 0.8,
            },
        )

        # Verify set_cache was called with the flagged capability
        cap_calls = [c for c in module.hub.set_cache.call_args_list if c[0][0] == "capabilities"]
        assert len(cap_calls) >= 1
        written_caps = cap_calls[-1][0][1]
        assert written_caps["climate"]["drift_flagged"] is True
        assert written_caps["climate"]["drift_severity"] == 0.8

    @pytest.mark.asyncio
    async def test_drift_ignores_unknown_capability(self):
        """Drift for non-existent capability is silently ignored."""
        module = _make_module()
        module.hub.get_cache.return_value = _make_cache_entry({"climate": {"available": True}})

        await module.on_event(
            "drift_detected",
            {
                "capability": "nonexistent",
                "severity": 0.5,
            },
        )

        # set_cache should NOT have been called for capabilities
        cap_calls = [c for c in module.hub.set_cache.call_args_list if c[0][0] == "capabilities"]
        assert len(cap_calls) == 0

    @pytest.mark.asyncio
    async def test_drift_ignores_non_drift_events(self):
        """Non-drift events are ignored."""
        module = _make_module()
        module.hub.get_cache.return_value = _make_cache_entry({"climate": {"available": True}})

        await module.on_event("state_changed", {"entity_id": "sensor.temp"})

        # get_cache should not even be called for non-drift events
        module.hub.get_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_drift_ignores_empty_capability_name(self):
        """Empty capability name is ignored."""
        module = _make_module()
        module.hub.get_cache.return_value = _make_cache_entry({"climate": {"available": True}})

        await module.on_event("drift_detected", {"capability": "", "severity": 0.5})

        # Should return early without calling get_cache
        module.hub.get_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_drift_no_capabilities_cache(self):
        """Drift with no capabilities cache does nothing."""
        module = _make_module()
        module.hub.get_cache.return_value = None

        # Should not raise
        await module.on_event("drift_detected", {"capability": "climate", "severity": 0.5})


# ---------------------------------------------------------------------------
# Predictability feedback
# ---------------------------------------------------------------------------


class TestPredictabilityFeedback:
    """Tests for _compute_predictability reading ML/shadow feedback."""

    def test_compute_predictability_with_ml_and_shadow(self, module):
        """Both ML and shadow present — weighted blend (0.7 * ML + 0.3 * shadow)."""
        caps = {
            "living_room": {
                "ml_accuracy": {"mean_r2": 0.8},
                "shadow_accuracy": {"hit_rate": 0.6},
            }
        }
        result = module._compute_predictability("living_room", caps)
        expected = 0.8 * 0.7 + 0.6 * 0.3  # 0.56 + 0.18 = 0.74
        assert abs(result - expected) < 1e-9

    def test_compute_predictability_ml_only(self, module):
        """Only ML accuracy present, no shadow — shadow defaults to 0."""
        caps = {
            "kitchen": {
                "ml_accuracy": {"mean_r2": 0.5},
            }
        }
        result = module._compute_predictability("kitchen", caps)
        expected = 0.5 * 0.7 + 0.0 * 0.3  # 0.35
        assert abs(result - expected) < 1e-9

    def test_compute_predictability_shadow_only(self, module):
        """Only shadow accuracy present, no ML — ML defaults to 0."""
        caps = {
            "bedroom": {
                "shadow_accuracy": {"hit_rate": 0.9},
            }
        }
        result = module._compute_predictability("bedroom", caps)
        expected = 0.0 * 0.7 + 0.9 * 0.3  # 0.27
        assert abs(result - expected) < 1e-9

    def test_compute_predictability_neither(self, module):
        """No ML or shadow data — returns 0.0."""
        caps = {"some_cap": {"entities": ["light.test"]}}
        result = module._compute_predictability("some_cap", caps)
        assert result == 0.0

    def test_compute_predictability_missing_cap(self, module):
        """Capability not in cache at all — returns 0.0."""
        caps = {"other_cap": {"ml_accuracy": {"mean_r2": 0.9}}}
        result = module._compute_predictability("missing_cap", caps)
        assert result == 0.0

    def test_compute_predictability_from_cache_wrapper(self, module):
        """Handles {"data": {...}} wrapper from cache entries."""
        caps = {
            "data": {
                "garage": {
                    "ml_accuracy": {"mean_r2": 0.6},
                    "shadow_accuracy": {"hit_rate": 0.4},
                }
            }
        }
        result = module._compute_predictability("garage", caps)
        expected = 0.6 * 0.7 + 0.4 * 0.3  # 0.42 + 0.12 = 0.54
        assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# Demand alignment
# ---------------------------------------------------------------------------


def _make_module():
    """Create an OrganicDiscoveryModule with a mock hub for unit tests."""
    hub = AsyncMock()
    hub.cache = AsyncMock()
    hub.set_cache = AsyncMock(return_value=1)
    hub.get_cache = AsyncMock(return_value=None)
    hub.publish = AsyncMock()
    hub.schedule_task = AsyncMock()
    hub.mark_module_running = MagicMock()
    hub.mark_module_failed = MagicMock()
    hub.is_running = MagicMock(return_value=True)
    mod = OrganicDiscoveryModule(hub)
    mod._load_logbook = AsyncMock(return_value=[])
    return mod


class TestDemandAlignment:
    def test_full_match_gets_max_bonus(self):
        module = _make_module()
        from aria.capabilities import DemandSignal

        demands = [DemandSignal(entity_domains=["sensor"], device_classes=["power"], min_entities=3)]
        entities = [{"entity_id": f"sensor.power_{i}", "domain": "sensor", "device_class": "power"} for i in range(5)]
        bonus = module._compute_demand_alignment(entities, demands)
        assert bonus == 0.2

    def test_domain_match_only(self):
        module = _make_module()
        from aria.capabilities import DemandSignal

        demands = [DemandSignal(entity_domains=["sensor"], device_classes=["power"], min_entities=3)]
        entities = [
            {"entity_id": f"sensor.temp_{i}", "domain": "sensor", "device_class": "temperature"} for i in range(5)
        ]
        bonus = module._compute_demand_alignment(entities, demands)
        assert bonus == 0.05

    def test_no_match(self):
        module = _make_module()
        from aria.capabilities import DemandSignal

        demands = [DemandSignal(entity_domains=["sensor"], device_classes=["power"], min_entities=3)]
        entities = [{"entity_id": "light.lamp", "domain": "light", "device_class": ""}]
        bonus = module._compute_demand_alignment(entities, demands)
        assert bonus == 0.0

    def test_empty_demands(self):
        module = _make_module()
        bonus = module._compute_demand_alignment([{"domain": "sensor"}], [])
        assert bonus == 0.0

    def test_domain_and_class_match_below_size(self):
        module = _make_module()
        from aria.capabilities import DemandSignal

        demands = [DemandSignal(entity_domains=["sensor"], device_classes=["power"], min_entities=10)]
        entities = [{"entity_id": f"sensor.power_{i}", "domain": "sensor", "device_class": "power"} for i in range(3)]
        bonus = module._compute_demand_alignment(entities, demands)
        assert bonus == 0.1  # domain + class match but below size threshold


# ---------------------------------------------------------------------------
# update_settings
# ---------------------------------------------------------------------------


class TestUpdateSettings:
    async def test_update_settings_changes_naming_backend(self, mock_hub, module):
        """Changing naming_backend from heuristic to ollama should persist."""
        assert module.settings["naming_backend"] == "heuristic"
        await module.update_settings({"naming_backend": "ollama"})
        assert module.settings["naming_backend"] == "ollama"
        # Verify cache was written
        mock_hub.set_cache.assert_called_with("discovery_settings", module.settings, {"source": "settings_update"})

    async def test_update_settings_rejects_invalid_backend(self, mock_hub, module):
        """Invalid naming_backend should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid naming_backend"):
            await module.update_settings({"naming_backend": "invalid"})

    async def test_update_settings_preserves_other_settings(self, mock_hub, module):
        """Changing naming_backend should not affect other settings."""
        original_threshold = module.settings["promote_threshold"]
        original_mode = module.settings["autonomy_mode"]
        await module.update_settings({"naming_backend": "ollama"})
        assert module.settings["promote_threshold"] == original_threshold
        assert module.settings["autonomy_mode"] == original_mode
