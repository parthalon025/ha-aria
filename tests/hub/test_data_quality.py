"""Unit tests for DataQualityModule.

Tests entity classification pipeline: metric computation, tier assignment,
config threshold usage, human override preservation, and graceful empty-data handling.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.modules.data_quality import (
    CONFIG_NOISE_EVENT_THRESHOLD,
    DataQualityModule,
)

# ============================================================================
# Mock Hub
# ============================================================================


class MockHub:
    """Lightweight hub mock for data quality tests."""

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}
        self._running = True

        self.cache = Mock()
        self.cache.get_curation = AsyncMock(return_value=None)
        self.cache.upsert_curation = AsyncMock()
        self.cache.get_config_value = AsyncMock(side_effect=self._config_value)

        self.logger = Mock()
        self.modules = {}

        # Default config values
        self._config_overrides: dict[str, Any] = {}

    async def _config_value(self, key: str, fallback: Any = None) -> Any:
        return self._config_overrides.get(key, fallback)

    async def get_cache(self, category: str) -> dict[str, Any] | None:
        return self._cache.get(category)

    async def schedule_task(self, **kwargs):
        pass

    def register_module(self, mod):
        self.modules[mod.module_id] = mod

    def set_entities(self, entities: dict[str, dict[str, Any]]):
        self._cache["entities"] = {"data": entities}

    def set_activity(self, windows: list[dict[str, Any]]):
        self._cache["activity_log"] = {"data": {"windows": windows}}


# ============================================================================
# Helpers
# ============================================================================


def make_entity(  # noqa: PLR0913
    entity_id: str,
    domain: str = "",
    friendly_name: str = "",
    device_id: str = "",
    area_id: str = "",
    device_class: str = "",
    last_changed: str | None = None,
) -> dict[str, Any]:
    """Build an entity data dict matching the discovery cache format."""
    if not domain and "." in entity_id:
        domain = entity_id.split(".")[0]
    if not friendly_name:
        friendly_name = entity_id.replace(".", " ").title()
    if last_changed is None:
        last_changed = datetime.now(UTC).isoformat()
    return {
        "entity_id": entity_id,
        "domain": domain,
        "friendly_name": friendly_name,
        "device_id": device_id,
        "area_id": area_id,
        "device_class": device_class,
        "last_changed": last_changed,
    }


def make_window(
    by_entity: dict[str, int] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an activity window dict."""
    return {
        "window_start": datetime.now(UTC).isoformat(),
        "event_count": sum((by_entity or {}).values()),
        "by_domain": {},
        "by_entity": by_entity or {},
        "events": events or [],
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def hub():
    return MockHub()


@pytest_asyncio.fixture
async def module(hub):
    mod = DataQualityModule(hub)
    return mod


# ============================================================================
# Metric computation
# ============================================================================


@pytest.mark.asyncio
async def test_metric_computation_basic(module):
    """Verify _compute_metrics returns correct fields and values."""
    entity = make_entity("light.kitchen", domain="light", area_id="kitchen")
    windows = [
        make_window(
            by_entity={"light.kitchen": 10},
            events=[
                {"entity_id": "light.kitchen", "to": "on"},
                {"entity_id": "light.kitchen", "to": "off"},
                {"entity_id": "light.kitchen", "to": "on"},
            ],
        )
    ]

    metrics = module._compute_metrics("light.kitchen", entity, windows)

    assert metrics["domain"] == "light"
    assert metrics["area_id"] == "kitchen"
    assert metrics["unique_states"] == 2  # on, off
    assert metrics["event_rate_day"] > 0
    assert metrics["last_changed_days_ago"] is not None
    assert metrics["last_changed_days_ago"] < 1  # just created


@pytest.mark.asyncio
async def test_metric_no_activity(module):
    """Metrics with no activity windows report zero rate and states."""
    entity = make_entity("light.living_room")
    metrics = module._compute_metrics("light.living_room", entity, [])

    assert metrics["event_rate_day"] == 0
    assert metrics["unique_states"] == 0


# ============================================================================
# Tier 1: auto-excluded
# ============================================================================


@pytest.mark.asyncio
async def test_classify_auto_exclude_domain(module):
    """Domain in exclude list → tier 1, auto_excluded."""
    metrics = {"domain": "update", "event_rate_day": 5, "unique_states": 2, "last_changed_days_ago": 1}
    config = {
        "auto_exclude_domains": {"update", "automation", "script"},
        "noise_event_threshold": 1000,
        "stale_days_threshold": 30,
        "vehicle_patterns": [],
    }
    entities_data = {"update.core": make_entity("update.core", domain="update")}

    tier, status, reason, group_id = module._classify("update.core", metrics, config, entities_data, set(), set())

    assert tier == 1
    assert status == "auto_excluded"
    assert "update" in reason


@pytest.mark.asyncio
async def test_classify_stale_entity(module):
    """No changes in N days → tier 1, auto_excluded."""
    metrics = {"domain": "sensor", "event_rate_day": 0, "unique_states": 0, "last_changed_days_ago": 45}
    config = {
        "auto_exclude_domains": set(),
        "noise_event_threshold": 1000,
        "stale_days_threshold": 30,
        "vehicle_patterns": [],
    }
    entities_data = {"sensor.stale": make_entity("sensor.stale")}

    tier, status, reason, _ = module._classify("sensor.stale", metrics, config, entities_data, set(), set())

    assert tier == 1
    assert status == "auto_excluded"
    assert "45" in reason


@pytest.mark.asyncio
async def test_classify_noise_entity(module):
    """High event rate + low unique states → tier 1, auto_excluded."""
    metrics = {"domain": "sensor", "event_rate_day": 2400, "unique_states": 2, "last_changed_days_ago": 1}
    config = {
        "auto_exclude_domains": set(),
        "noise_event_threshold": 1000,
        "stale_days_threshold": 30,
        "vehicle_patterns": [],
    }
    entities_data = {"sensor.noisy": make_entity("sensor.noisy")}

    tier, status, reason, _ = module._classify("sensor.noisy", metrics, config, entities_data, set(), set())

    assert tier == 1
    assert status == "auto_excluded"
    assert "noise" in reason.lower()


@pytest.mark.asyncio
async def test_classify_vehicle_pattern(module):
    """Name matches vehicle pattern → tier 1, auto_excluded."""
    metrics = {"domain": "device_tracker", "event_rate_day": 50, "unique_states": 5, "last_changed_days_ago": 1}
    config = {
        "auto_exclude_domains": set(),
        "noise_event_threshold": 1000,
        "stale_days_threshold": 30,
        "vehicle_patterns": ["tesla", "vehicle"],
    }
    entities_data = {
        "device_tracker.tesla_location": make_entity(
            "device_tracker.tesla_location",
            friendly_name="Tesla Location",
        )
    }

    tier, status, reason, _ = module._classify(
        "device_tracker.tesla_location", metrics, config, entities_data, set(), set()
    )

    assert tier == 1
    assert status == "auto_excluded"
    assert "tesla" in reason


# ============================================================================
# Tier 2: edge cases
# ============================================================================


@pytest.mark.asyncio
async def test_classify_vehicle_group(module):
    """Shares device_id with a vehicle entity → tier 2, excluded."""
    metrics = {"domain": "sensor", "event_rate_day": 10, "unique_states": 5, "last_changed_days_ago": 1}
    config = {
        "auto_exclude_domains": set(),
        "noise_event_threshold": 1000,
        "stale_days_threshold": 30,
        "vehicle_patterns": ["tesla"],
    }
    # sensor.charger_amps shares device with device_tracker.tesla_location
    # (name does NOT match vehicle pattern — tests device group propagation only)
    entities_data = {
        "sensor.charger_amps": make_entity(
            "sensor.charger_amps",
            friendly_name="Charger Amps",
            device_id="dev_tesla_001",
        ),
        "device_tracker.tesla_location": make_entity(
            "device_tracker.tesla_location",
            friendly_name="Tesla Location",
            device_id="dev_tesla_001",
        ),
    }
    vehicle_entity_ids = {"device_tracker.tesla_location"}
    vehicle_device_ids = {"dev_tesla_001"}

    tier, status, reason, group_id = module._classify(
        "sensor.charger_amps",
        metrics,
        config,
        entities_data,
        vehicle_entity_ids,
        vehicle_device_ids,
    )

    assert tier == 2
    assert status == "excluded"
    assert "vehicle" in reason.lower()
    assert group_id == "dev_tesla_001"


@pytest.mark.asyncio
async def test_classify_high_rate_low_variety(module):
    """Moderate noise (>500/day, <5 states) → tier 2, excluded."""
    metrics = {"domain": "sensor", "event_rate_day": 600, "unique_states": 3, "last_changed_days_ago": 1}
    config = {
        "auto_exclude_domains": set(),
        "noise_event_threshold": 1000,
        "stale_days_threshold": 30,
        "vehicle_patterns": [],
    }
    entities_data = {"sensor.power": make_entity("sensor.power")}

    tier, status, reason, _ = module._classify("sensor.power", metrics, config, entities_data, set(), set())

    assert tier == 2
    assert status == "excluded"
    assert "high event rate" in reason.lower()


@pytest.mark.asyncio
async def test_classify_presence_domain(module):
    """person/device_tracker → tier 2, included (presence tracking)."""
    metrics = {"domain": "person", "event_rate_day": 20, "unique_states": 4, "last_changed_days_ago": 1}
    config = {
        "auto_exclude_domains": set(),
        "noise_event_threshold": 1000,
        "stale_days_threshold": 30,
        "vehicle_patterns": [],
    }
    entities_data = {"person.justin": make_entity("person.justin", friendly_name="Justin")}

    tier, status, reason, _ = module._classify("person.justin", metrics, config, entities_data, set(), set())

    assert tier == 2
    assert status == "included"
    assert "presence" in reason.lower()


# ============================================================================
# Tier 3: default
# ============================================================================


@pytest.mark.asyncio
async def test_classify_default(module):
    """Normal entity with no special conditions → tier 3, included."""
    metrics = {"domain": "light", "event_rate_day": 50, "unique_states": 5, "last_changed_days_ago": 1}
    config = {
        "auto_exclude_domains": set(),
        "noise_event_threshold": 1000,
        "stale_days_threshold": 30,
        "vehicle_patterns": [],
    }
    entities_data = {"light.kitchen": make_entity("light.kitchen")}

    tier, status, reason, _ = module._classify("light.kitchen", metrics, config, entities_data, set(), set())

    assert tier == 3
    assert status == "included"
    assert "general" in reason.lower()


# ============================================================================
# Human override + full pipeline
# ============================================================================


@pytest.mark.asyncio
async def test_human_override_preserved(hub, module):
    """Entity with human_override=True is skipped during classification."""
    hub.set_entities(
        {
            "light.manual": make_entity("light.manual"),
        }
    )
    hub.set_activity([])

    # Simulate existing curation with human override
    hub.cache.get_curation = AsyncMock(
        return_value={
            "entity_id": "light.manual",
            "status": "included",
            "tier": 3,
            "human_override": True,
        }
    )

    await module.run_classification()

    # upsert_curation should NOT be called for this entity
    hub.cache.upsert_curation.assert_not_called()


@pytest.mark.asyncio
async def test_run_classification_calls_upsert(hub, module):
    """Full pipeline with mock data calls upsert_curation for each entity."""
    hub.set_entities(
        {
            "light.kitchen": make_entity("light.kitchen"),
            "switch.porch": make_entity("switch.porch"),
            "update.core": make_entity("update.core", domain="update"),
        }
    )
    hub.set_activity(
        [
            make_window(
                by_entity={"light.kitchen": 5, "switch.porch": 3},
                events=[
                    {"entity_id": "light.kitchen", "to": "on"},
                    {"entity_id": "switch.porch", "to": "off"},
                ],
            )
        ]
    )

    await module.run_classification()

    assert hub.cache.upsert_curation.call_count == 3

    # Verify one of the calls was the update domain (tier 1)
    calls = hub.cache.upsert_curation.call_args_list
    update_call = [c for c in calls if c.kwargs.get("entity_id") == "update.core"]
    assert len(update_call) == 1
    assert update_call[0].kwargs["tier"] == 1
    assert update_call[0].kwargs["status"] == "auto_excluded"


@pytest.mark.asyncio
async def test_initialize_schedules_daily(hub, module):
    """initialize() calls schedule_task for daily re-classification."""
    hub.schedule_task = AsyncMock()
    hub.set_entities({})
    hub.set_activity([])

    await module.initialize()

    hub.schedule_task.assert_called_once()
    call_kwargs = hub.schedule_task.call_args.kwargs
    assert call_kwargs["task_id"] == "data_quality_reclassify"
    assert call_kwargs["run_immediately"] is False


# ============================================================================
# Edge cases: empty data
# ============================================================================


@pytest.mark.asyncio
async def test_empty_entity_cache(hub, module):
    """Handles missing/empty entity data gracefully (no crash, no upserts)."""
    # No entities set in cache
    hub.set_activity([make_window()])

    await module.run_classification()

    hub.cache.upsert_curation.assert_not_called()


@pytest.mark.asyncio
async def test_empty_activity_cache(hub, module):
    """Handles missing activity data gracefully — still classifies entities."""
    hub.set_entities(
        {
            "light.kitchen": make_entity("light.kitchen"),
        }
    )
    # No activity set

    await module.run_classification()

    assert hub.cache.upsert_curation.call_count == 1


# ============================================================================
# Config values
# ============================================================================


@pytest.mark.asyncio
async def test_config_values_used(hub, module):
    """Verify custom config thresholds are read and applied."""
    # Override config: lower noise threshold
    hub._config_overrides[CONFIG_NOISE_EVENT_THRESHOLD] = 50

    hub.set_entities(
        {
            "sensor.chatty": make_entity("sensor.chatty"),
        }
    )
    hub.set_activity(
        [
            make_window(
                by_entity={"sensor.chatty": 100},
                events=[
                    {"entity_id": "sensor.chatty", "to": "on"},
                    {"entity_id": "sensor.chatty", "to": "off"},
                ],
            )
        ]
    )

    await module.run_classification()

    # With noise threshold at 50, 100 events in one 15-min window
    # = 100/900*86400 = 9600/day → should be auto_excluded as noise
    call = hub.cache.upsert_curation.call_args
    assert call.kwargs["tier"] == 1
    assert call.kwargs["status"] == "auto_excluded"
    assert "noise" in call.kwargs["reason"].lower()
