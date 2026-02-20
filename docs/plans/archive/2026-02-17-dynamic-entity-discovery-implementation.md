# Dynamic Entity Discovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace full-replace discovery caching with merge-not-replace + staleness timer + archive for all HA entities/devices/areas, and remove all hard-coded camera/room data from the presence module.

**Architecture:** DiscoveryModule gets lifecycle management (merge, stale, archive) for entities/devices/areas caches. PresenceModule reads camera→room mappings from the enriched discovery cache instead of hard-coded defaults. Archived entities are excluded from active consumers but preserved for reference.

**Tech Stack:** Python 3.12, pytest, async/await, hub cache API (`hub.set_cache`/`hub.get_cache`)

**Design doc:** `docs/plans/2026-02-17-dynamic-entity-discovery-design.md`

---

### Task 1: Add `discovery.stale_ttl_hours` Config Key

**Files:**
- Modify: `aria/hub/config_defaults.py` (after line 507, end of "Presence Tracking" section)
- Test: `tests/hub/test_api_config.py` (existing tests verify all config keys are seeded)

**Step 1: Add config key to CONFIG_DEFAULTS**

In `aria/hub/config_defaults.py`, add a new section after the Presence Tracking entries (after line 507):

```python
    # ── Discovery Lifecycle ────────────────────────────────────────────
    {
        "key": "discovery.stale_ttl_hours",
        "default_value": "72",
        "value_type": "number",
        "label": "Entity Stale TTL (hours)",
        "description": (
            "Hours after an entity disappears from HA discovery before it is archived. "
            "While stale, entities remain usable. After archival, they are excluded from "
            "active consumers but preserved for reference. Set to 0 to archive immediately."
        ),
        "category": "Discovery",
        "min_value": 0,
        "max_value": 720,
        "step": 1,
    },
```

**Step 2: Run existing config tests to confirm no breakage**

Run: `.venv/bin/python -m pytest tests/hub/test_api_config.py -v --timeout=120`
Expected: All existing tests PASS (new key gets seeded automatically)

**Step 3: Commit**

```bash
git add aria/hub/config_defaults.py
git commit -m "feat(discovery): add discovery.stale_ttl_hours config key"
```

---

### Task 2: Add Lifecycle Merge Logic to DiscoveryModule

**Files:**
- Modify: `aria/modules/discovery.py:116-162` (`_store_discovery_results` method)
- Test: `tests/hub/test_discovery_lifecycle.py` (new file)

**Step 1: Write failing tests for merge behavior**

Create `tests/hub/test_discovery_lifecycle.py`:

```python
"""Tests for discovery entity lifecycle — merge, stale, archive."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.modules.discovery import DiscoveryModule


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.set_cache = AsyncMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.schedule_task = AsyncMock()
    hub.publish = AsyncMock()
    hub.cache = MagicMock()
    hub.cache.get_config_value = AsyncMock(return_value="72")
    return hub


@pytest.fixture
def module(mock_hub):
    return DiscoveryModule(mock_hub, "http://localhost:8123", "test-token")


# ============================================================================
# Merge Behavior
# ============================================================================


class TestMergeEntities:
    """Entity/device/area caches should merge, not replace."""

    @pytest.mark.asyncio
    async def test_new_entities_added_with_lifecycle(self, module, mock_hub):
        """First discovery should add lifecycle metadata to every entity."""
        results = {
            "entities": {"light.kitchen": {"area_id": "kitchen"}},
            "devices": {},
            "areas": {},
            "capabilities": {},
        }
        await module._store_discovery_results(results)

        # Find the set_cache call for entities
        entity_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "entities"]
        assert len(entity_calls) == 1
        stored = entity_calls[0][0][1]

        assert "light.kitchen" in stored
        lc = stored["light.kitchen"]["_lifecycle"]
        assert lc["status"] == "active"
        assert lc["first_discovered"] is not None
        assert lc["last_seen_in_discovery"] is not None
        assert lc["stale_since"] is None
        assert lc["archived_at"] is None

    @pytest.mark.asyncio
    async def test_existing_entities_updated_not_replaced(self, module, mock_hub):
        """Re-discovery should update fields but preserve lifecycle history."""
        existing = {
            "data": {
                "light.kitchen": {
                    "area_id": "kitchen",
                    "_lifecycle": {
                        "status": "active",
                        "first_discovered": "2026-02-10T10:00:00",
                        "last_seen_in_discovery": "2026-02-10T10:00:00",
                        "stale_since": None,
                        "archived_at": None,
                    },
                },
            }
        }
        mock_hub.get_cache = AsyncMock(return_value=existing)

        results = {
            "entities": {"light.kitchen": {"area_id": "living_room"}},
            "devices": {},
            "areas": {},
            "capabilities": {},
        }
        await module._store_discovery_results(results)

        entity_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "entities"]
        stored = entity_calls[0][0][1]

        # Area updated
        assert stored["light.kitchen"]["area_id"] == "living_room"
        # first_discovered preserved
        assert stored["light.kitchen"]["_lifecycle"]["first_discovered"] == "2026-02-10T10:00:00"
        # last_seen updated
        assert stored["light.kitchen"]["_lifecycle"]["last_seen_in_discovery"] != "2026-02-10T10:00:00"
        assert stored["light.kitchen"]["_lifecycle"]["status"] == "active"

    @pytest.mark.asyncio
    async def test_missing_entities_marked_stale(self, module, mock_hub):
        """Entities in cache but not in latest discovery get stale_since set."""
        existing = {
            "data": {
                "light.kitchen": {
                    "area_id": "kitchen",
                    "_lifecycle": {
                        "status": "active",
                        "first_discovered": "2026-02-10T10:00:00",
                        "last_seen_in_discovery": "2026-02-10T10:00:00",
                        "stale_since": None,
                        "archived_at": None,
                    },
                },
                "light.bedroom": {
                    "area_id": "bedroom",
                    "_lifecycle": {
                        "status": "active",
                        "first_discovered": "2026-02-10T10:00:00",
                        "last_seen_in_discovery": "2026-02-10T10:00:00",
                        "stale_since": None,
                        "archived_at": None,
                    },
                },
            }
        }
        mock_hub.get_cache = AsyncMock(return_value=existing)

        # Only kitchen in latest discovery — bedroom is missing
        results = {
            "entities": {"light.kitchen": {"area_id": "kitchen"}},
            "devices": {},
            "areas": {},
            "capabilities": {},
        }
        await module._store_discovery_results(results)

        entity_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "entities"]
        stored = entity_calls[0][0][1]

        # Kitchen still active
        assert stored["light.kitchen"]["_lifecycle"]["status"] == "active"
        # Bedroom now stale
        assert stored["light.bedroom"]["_lifecycle"]["status"] == "stale"
        assert stored["light.bedroom"]["_lifecycle"]["stale_since"] is not None

    @pytest.mark.asyncio
    async def test_stale_entity_restored_on_rediscovery(self, module, mock_hub):
        """A stale entity that reappears should be restored to active."""
        existing = {
            "data": {
                "light.kitchen": {
                    "area_id": "kitchen",
                    "_lifecycle": {
                        "status": "stale",
                        "first_discovered": "2026-02-10T10:00:00",
                        "last_seen_in_discovery": "2026-02-10T10:00:00",
                        "stale_since": "2026-02-15T10:00:00",
                        "archived_at": None,
                    },
                },
            }
        }
        mock_hub.get_cache = AsyncMock(return_value=existing)

        results = {
            "entities": {"light.kitchen": {"area_id": "kitchen"}},
            "devices": {},
            "areas": {},
            "capabilities": {},
        }
        await module._store_discovery_results(results)

        entity_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "entities"]
        stored = entity_calls[0][0][1]

        assert stored["light.kitchen"]["_lifecycle"]["status"] == "active"
        assert stored["light.kitchen"]["_lifecycle"]["stale_since"] is None

    @pytest.mark.asyncio
    async def test_archived_entity_restored_on_rediscovery(self, module, mock_hub):
        """An archived entity that reappears should be restored to active."""
        existing = {
            "data": {
                "light.kitchen": {
                    "area_id": "kitchen",
                    "_lifecycle": {
                        "status": "archived",
                        "first_discovered": "2026-02-01T10:00:00",
                        "last_seen_in_discovery": "2026-02-10T10:00:00",
                        "stale_since": "2026-02-12T10:00:00",
                        "archived_at": "2026-02-15T10:00:00",
                    },
                },
            }
        }
        mock_hub.get_cache = AsyncMock(return_value=existing)

        results = {
            "entities": {"light.kitchen": {"area_id": "kitchen"}},
            "devices": {},
            "areas": {},
            "capabilities": {},
        }
        await module._store_discovery_results(results)

        entity_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "entities"]
        stored = entity_calls[0][0][1]

        assert stored["light.kitchen"]["_lifecycle"]["status"] == "active"
        assert stored["light.kitchen"]["_lifecycle"]["stale_since"] is None
        assert stored["light.kitchen"]["_lifecycle"]["archived_at"] is None
        # first_discovered preserved from original
        assert stored["light.kitchen"]["_lifecycle"]["first_discovered"] == "2026-02-01T10:00:00"

    @pytest.mark.asyncio
    async def test_already_stale_entity_not_re_stamped(self, module, mock_hub):
        """An entity already stale should keep its original stale_since timestamp."""
        existing = {
            "data": {
                "light.gone": {
                    "area_id": "attic",
                    "_lifecycle": {
                        "status": "stale",
                        "first_discovered": "2026-02-01T10:00:00",
                        "last_seen_in_discovery": "2026-02-10T10:00:00",
                        "stale_since": "2026-02-14T10:00:00",
                        "archived_at": None,
                    },
                },
            }
        }
        mock_hub.get_cache = AsyncMock(return_value=existing)

        # Entity still missing
        results = {"entities": {}, "devices": {}, "areas": {}, "capabilities": {}}
        await module._store_discovery_results(results)

        entity_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "entities"]
        stored = entity_calls[0][0][1]

        # stale_since should be unchanged
        assert stored["light.gone"]["_lifecycle"]["stale_since"] == "2026-02-14T10:00:00"

    @pytest.mark.asyncio
    async def test_merge_applies_to_devices(self, module, mock_hub):
        """Device cache should also merge with lifecycle metadata."""
        existing = {
            "data": {
                "dev123": {
                    "name": "Hue Bridge",
                    "area_id": "living_room",
                    "_lifecycle": {
                        "status": "active",
                        "first_discovered": "2026-02-10T10:00:00",
                        "last_seen_in_discovery": "2026-02-10T10:00:00",
                        "stale_since": None,
                        "archived_at": None,
                    },
                },
            }
        }

        async def get_cache_side_effect(key):
            if key == "devices":
                return existing
            return None

        mock_hub.get_cache = AsyncMock(side_effect=get_cache_side_effect)

        results = {
            "entities": {},
            "devices": {"dev123": {"name": "Hue Bridge v2", "area_id": "living_room"}},
            "areas": {},
            "capabilities": {},
        }
        await module._store_discovery_results(results)

        device_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "devices"]
        stored = device_calls[0][0][1]

        assert stored["dev123"]["name"] == "Hue Bridge v2"
        assert stored["dev123"]["_lifecycle"]["first_discovered"] == "2026-02-10T10:00:00"
        assert stored["dev123"]["_lifecycle"]["status"] == "active"

    @pytest.mark.asyncio
    async def test_merge_applies_to_areas(self, module, mock_hub):
        """Area cache should also merge with lifecycle metadata."""
        results = {
            "entities": {},
            "devices": {},
            "areas": {"living_room": {"name": "Living Room"}},
            "capabilities": {},
        }
        await module._store_discovery_results(results)

        area_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "areas"]
        stored = area_calls[0][0][1]

        assert "living_room" in stored
        assert stored["living_room"]["_lifecycle"]["status"] == "active"


# ============================================================================
# Archive Expiry
# ============================================================================


class TestArchiveExpiry:
    """Stale entities past TTL should be archived."""

    @pytest.mark.asyncio
    async def test_archive_expired_stale_entities(self, module, mock_hub):
        """Entities stale for longer than TTL should be archived."""
        now = datetime.now()
        stale_time = (now - timedelta(hours=73)).isoformat()  # Past 72h default

        mock_hub.get_cache = AsyncMock(
            return_value={
                "data": {
                    "light.old": {
                        "area_id": "attic",
                        "_lifecycle": {
                            "status": "stale",
                            "first_discovered": "2026-02-01T10:00:00",
                            "last_seen_in_discovery": "2026-02-10T10:00:00",
                            "stale_since": stale_time,
                            "archived_at": None,
                        },
                    },
                    "light.recent_stale": {
                        "area_id": "bedroom",
                        "_lifecycle": {
                            "status": "stale",
                            "first_discovered": "2026-02-10T10:00:00",
                            "last_seen_in_discovery": "2026-02-15T10:00:00",
                            "stale_since": now.isoformat(),
                            "archived_at": None,
                        },
                    },
                }
            }
        )
        mock_hub.cache.get_config_value = AsyncMock(return_value="72")

        await module._archive_expired_entities("entities")

        entity_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "entities"]
        assert len(entity_calls) == 1
        stored = entity_calls[0][0][1]

        # Old one archived
        assert stored["light.old"]["_lifecycle"]["status"] == "archived"
        assert stored["light.old"]["_lifecycle"]["archived_at"] is not None
        # Recent stale one still stale
        assert stored["light.recent_stale"]["_lifecycle"]["status"] == "stale"

    @pytest.mark.asyncio
    async def test_archive_no_op_when_nothing_expired(self, module, mock_hub):
        """No cache write when nothing needs archiving."""
        now = datetime.now()
        mock_hub.get_cache = AsyncMock(
            return_value={
                "data": {
                    "light.fresh": {
                        "_lifecycle": {
                            "status": "active",
                            "first_discovered": now.isoformat(),
                            "last_seen_in_discovery": now.isoformat(),
                            "stale_since": None,
                            "archived_at": None,
                        },
                    },
                }
            }
        )
        mock_hub.cache.get_config_value = AsyncMock(return_value="72")

        await module._archive_expired_entities("entities")

        entity_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "entities"]
        assert len(entity_calls) == 0  # No write needed
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/hub/test_discovery_lifecycle.py -v --timeout=120`
Expected: FAIL — `_store_discovery_results` does full replace, `_archive_expired_entities` doesn't exist

**Step 3: Implement lifecycle merge in `_store_discovery_results`**

Replace the `_store_discovery_results` method in `aria/modules/discovery.py` (lines 116-162):

```python
    async def _store_discovery_results(self, capabilities: dict[str, Any]):
        """Store discovery results with lifecycle-aware merge.

        For entities/devices/areas:
        - New items: added with active lifecycle metadata
        - Existing items: fields updated, lifecycle preserved, status restored to active
        - Missing items: marked stale (never deleted)

        Capabilities still use the existing organic-preserve merge logic.
        """
        now = datetime.now().isoformat()

        # Merge entities, devices, areas with lifecycle
        for cache_key in ("entities", "devices", "areas"):
            new_items = capabilities.get(cache_key, {})
            merged = await self._merge_with_lifecycle(cache_key, new_items, now)
            if merged:
                await self.hub.set_cache(
                    cache_key, merged, {"count": len(merged), "source": "discovery"}
                )

        # Store capabilities — merge with existing to preserve organic discoveries
        caps = capabilities.get("capabilities", {})
        if caps:
            existing_entry = await self.hub.get_cache("capabilities")
            if existing_entry and existing_entry.get("data"):
                existing = existing_entry["data"]
                for name, cap_data in existing.items():
                    if cap_data.get("source") == "organic" and name not in caps:
                        caps[name] = cap_data
            await self.hub.set_cache(
                "capabilities", caps, {"count": len(caps), "source": "discovery"}
            )

        # Store metadata
        metadata = {
            "entity_count": capabilities.get("entity_count", 0),
            "device_count": capabilities.get("device_count", 0),
            "area_count": capabilities.get("area_count", 0),
            "capability_count": len(caps),
            "timestamp": capabilities.get("timestamp"),
            "ha_version": capabilities.get("ha_version"),
        }
        await self.hub.set_cache("discovery_metadata", metadata)

    async def _merge_with_lifecycle(
        self, cache_key: str, new_items: dict[str, Any], now: str
    ) -> dict[str, Any]:
        """Merge new discovery data into existing cache with lifecycle tracking.

        Args:
            cache_key: Cache category ("entities", "devices", or "areas")
            new_items: Freshly discovered items from this run
            now: ISO timestamp for this discovery run

        Returns:
            Merged dict with lifecycle metadata on every item
        """
        # Load existing cache
        existing_entry = await self.hub.get_cache(cache_key)
        existing = {}
        if existing_entry:
            existing = (
                existing_entry.get("data", existing_entry)
                if isinstance(existing_entry, dict)
                else {}
            )

        merged: dict[str, Any] = {}

        # Process items present in new discovery
        for item_id, item_data in new_items.items():
            old = existing.get(item_id, {})
            old_lc = old.get("_lifecycle", {})

            item_data["_lifecycle"] = {
                "status": "active",
                "first_discovered": old_lc.get("first_discovered", now),
                "last_seen_in_discovery": now,
                "stale_since": None,
                "archived_at": None,
            }
            merged[item_id] = item_data

        # Process items in existing cache but NOT in new discovery
        for item_id, item_data in existing.items():
            if item_id in merged:
                continue  # Already handled above

            lc = item_data.get("_lifecycle", {})
            status = lc.get("status", "active")

            if status == "active":
                # First time missing — mark stale
                item_data["_lifecycle"] = {
                    **lc,
                    "status": "stale",
                    "stale_since": now,
                }
            # If already stale or archived, keep as-is (don't re-stamp)

            merged[item_id] = item_data

        return merged

    async def _archive_expired_entities(self, cache_key: str):
        """Archive stale entities that have exceeded the TTL.

        Args:
            cache_key: Cache category to check ("entities", "devices", or "areas")
        """
        ttl_hours = float(
            await self.hub.cache.get_config_value("discovery.stale_ttl_hours", "72")
        )
        now = datetime.now()

        existing_entry = await self.hub.get_cache(cache_key)
        if not existing_entry:
            return

        items = (
            existing_entry.get("data", existing_entry)
            if isinstance(existing_entry, dict)
            else {}
        )

        changed = False
        for item_id, item_data in items.items():
            lc = item_data.get("_lifecycle", {})
            if lc.get("status") != "stale":
                continue

            stale_since = lc.get("stale_since")
            if not stale_since:
                continue

            stale_dt = datetime.fromisoformat(stale_since)
            if (now - stale_dt).total_seconds() > ttl_hours * 3600:
                lc["status"] = "archived"
                lc["archived_at"] = now.isoformat()
                changed = True

        if changed:
            await self.hub.set_cache(
                cache_key, items, {"count": len(items), "source": "discovery"}
            )
```

Also add `from datetime import datetime` to the imports at the top of `discovery.py`.

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/hub/test_discovery_lifecycle.py -v --timeout=120`
Expected: All PASS

**Step 5: Run existing discovery tests to confirm no breakage**

Run: `.venv/bin/python -m pytest tests/hub/test_discovery_preserve_organic.py tests/hub/test_discover.py -v --timeout=120`
Expected: All existing tests PASS

**Step 6: Commit**

```bash
git add aria/modules/discovery.py tests/hub/test_discovery_lifecycle.py
git commit -m "feat(discovery): lifecycle-aware merge for entities/devices/areas

Replaces full-replace caching with merge-not-replace strategy.
New entities get lifecycle metadata, missing entities marked stale,
stale entities past TTL get archived. Archived entities auto-promote
on rediscovery."
```

---

### Task 3: Schedule Archive Check in Discovery Initialize

**Files:**
- Modify: `aria/modules/discovery.py:60-69` (`initialize` method)
- Test: `tests/hub/test_discovery_lifecycle.py` (add to existing)

**Step 1: Write failing test**

Add to `tests/hub/test_discovery_lifecycle.py`:

```python
class TestArchiveScheduling:
    """Archive check should run periodically."""

    @pytest.mark.asyncio
    async def test_initialize_schedules_archive_check(self, module, mock_hub):
        """initialize() should schedule periodic archive expiry checks."""
        # Mock run_discovery so it doesn't actually run subprocess
        module.run_discovery = AsyncMock()
        await module.initialize()

        task_ids = [c.kwargs.get("task_id") or c[1].get("task_id", "")
                    for c in mock_hub.schedule_task.call_args_list]
        # Flatten — schedule_task uses keyword args
        task_ids = []
        for call in mock_hub.schedule_task.call_args_list:
            tid = call.kwargs.get("task_id", "")
            task_ids.append(tid)

        assert "discovery_archive_check" in task_ids
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_discovery_lifecycle.py::TestArchiveScheduling -v --timeout=120`
Expected: FAIL

**Step 3: Add archive scheduling to initialize()**

In `aria/modules/discovery.py`, update `initialize()`:

```python
    async def initialize(self):
        """Initialize module - run initial discovery and schedule archive checks."""
        self.logger.info("Discovery module initializing...")

        # Run initial discovery
        try:
            await self.run_discovery()
            self.logger.info("Initial discovery complete")
        except Exception as e:
            self.logger.error(f"Initial discovery failed: {e}")

        # Schedule periodic archive expiry check (every 6 hours)
        async def _check_archives():
            for cache_key in ("entities", "devices", "areas"):
                try:
                    await self._archive_expired_entities(cache_key)
                except Exception as e:
                    self.logger.warning(f"Archive check failed for {cache_key}: {e}")

        await self.hub.schedule_task(
            task_id="discovery_archive_check",
            coro=_check_archives,
            interval=timedelta(hours=6),
            run_immediately=False,
        )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_discovery_lifecycle.py -v --timeout=120`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/modules/discovery.py tests/hub/test_discovery_lifecycle.py
git commit -m "feat(discovery): schedule periodic archive expiry check every 6h"
```

---

### Task 4: Remove Hard-Coded Data from PresenceModule

**Files:**
- Modify: `aria/modules/presence.py:37-47` (delete `DEFAULT_CAMERA_ROOMS`), lines 89, 289, 340, 461-501
- Modify: `tests/hub/test_presence.py` (update affected tests)

**Step 1: Write failing tests for discovery-based camera mapping**

Add a new test class to `tests/hub/test_presence.py` (keep existing file, add at the end):

```python
# ============================================================================
# Discovery-Based Camera Mapping
# ============================================================================


class TestDiscoveryCameraMapping:
    """Camera-to-room mapping should come from discovery cache, not hard-coded defaults."""

    async def test_discover_cameras_from_entity_cache(self, hub):
        """Should build camera→room mapping from entity registry cache."""
        hub._cache_data["entities"] = {
            "camera.driveway": {
                "area_id": "driveway",
                "_lifecycle": {"status": "active"},
            },
            "camera.backyard": {
                "device_id": "dev1",
                "_lifecycle": {"status": "active"},
            },
            "light.kitchen": {
                "area_id": "kitchen",
                "_lifecycle": {"status": "active"},
            },
        }
        hub._cache_data["devices"] = {
            "dev1": {"area_id": "backyard"},
        }

        m = PresenceModule(hub, "http://x", "tok")
        mapping = await m._discover_camera_rooms()

        assert mapping["driveway"] == "driveway"
        assert mapping["backyard"] == "backyard"
        assert "kitchen" not in mapping  # light, not camera

    async def test_discover_cameras_excludes_archived(self, hub):
        """Archived cameras should not be in the active mapping."""
        hub._cache_data["entities"] = {
            "camera.old_cam": {
                "area_id": "garage",
                "_lifecycle": {"status": "archived"},
            },
            "camera.active_cam": {
                "area_id": "pool",
                "_lifecycle": {"status": "active"},
            },
        }

        m = PresenceModule(hub, "http://x", "tok")
        mapping = await m._discover_camera_rooms()

        assert "old_cam" not in mapping
        assert mapping["active_cam"] == "pool"

    async def test_discover_cameras_includes_stale(self, hub):
        """Stale cameras should still be in the mapping (not yet archived)."""
        hub._cache_data["entities"] = {
            "camera.temp_offline": {
                "area_id": "patio",
                "_lifecycle": {"status": "stale"},
            },
        }

        m = PresenceModule(hub, "http://x", "tok")
        mapping = await m._discover_camera_rooms()

        assert mapping["temp_offline"] == "patio"

    async def test_discover_cameras_fallback_to_name(self, hub):
        """Camera with no area should use camera name as room."""
        hub._cache_data["entities"] = {
            "camera.mystery_cam": {
                "_lifecycle": {"status": "active"},
            },
        }
        hub._cache_data["devices"] = {}

        m = PresenceModule(hub, "http://x", "tok")
        mapping = await m._discover_camera_rooms()

        assert mapping["mystery_cam"] == "mystery_cam"

    async def test_discover_cameras_empty_cache(self, hub):
        """Empty entity cache should return empty mapping."""
        m = PresenceModule(hub, "http://x", "tok")
        mapping = await m._discover_camera_rooms()
        assert mapping == {}

    async def test_discover_cameras_merges_not_replaces(self, hub):
        """Re-discovery should merge new cameras, not replace existing mapping."""
        m = PresenceModule(hub, "http://x", "tok")
        m.camera_rooms = {"old_cam": "garage"}

        hub._cache_data["entities"] = {
            "camera.new_cam": {
                "area_id": "pool",
                "_lifecycle": {"status": "active"},
            },
        }

        await m._refresh_camera_rooms()

        assert m.camera_rooms["old_cam"] == "garage"  # Preserved
        assert m.camera_rooms["new_cam"] == "pool"  # Added

    async def test_config_override_wins(self, hub):
        """Manual config camera_rooms should override discovery."""
        hub._cache_data["entities"] = {
            "camera.driveway": {
                "area_id": "driveway",
                "_lifecycle": {"status": "active"},
            },
        }

        m = PresenceModule(
            hub, "http://x", "tok",
            camera_rooms={"driveway": "front_yard"},
        )
        # Config override should stick
        assert m.camera_rooms["driveway"] == "front_yard"
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/hub/test_presence.py::TestDiscoveryCameraMapping -v --timeout=120`
Expected: FAIL

**Step 3: Implement changes to presence.py**

**3a. Delete `DEFAULT_CAMERA_ROOMS` (lines 37-47)**

Remove the entire dict. Replace with a comment:

```python
# Camera-to-room mapping is discovered dynamically from HA entity registry.
# Manual overrides via presence.camera_rooms config key.
```

**3b. Update `__init__` (line 89)**

Change:
```python
        self.camera_rooms = camera_rooms or DEFAULT_CAMERA_ROOMS
```
To:
```python
        self._config_camera_rooms = camera_rooms  # Explicit overrides (always win)
        self.camera_rooms: dict[str, str] = dict(camera_rooms) if camera_rooms else {}
```

**3c. Add discovery methods**

Add after `__init__`:

```python
    async def _discover_camera_rooms(self) -> dict[str, str]:
        """Build camera→room mapping from HA entity/device registry cache.

        Filters entity cache for camera.* entities with active or stale
        lifecycle status, resolves area via entity→device→area chain.

        Returns:
            Dict of camera_name → room_name
        """
        mapping: dict[str, str] = {}

        entities_entry = await self.hub.get_cache("entities")
        devices_entry = await self.hub.get_cache("devices")

        if not entities_entry:
            return mapping

        entities_data = (
            entities_entry.get("data", entities_entry)
            if isinstance(entities_entry, dict)
            else {}
        )
        devices_data = {}
        if devices_entry:
            devices_data = (
                devices_entry.get("data", devices_entry)
                if isinstance(devices_entry, dict)
                else {}
            )

        for entity_id, entity_info in entities_data.items():
            if not entity_id.startswith("camera."):
                continue

            # Skip archived cameras
            lifecycle = entity_info.get("_lifecycle", {})
            if lifecycle.get("status") == "archived":
                continue

            camera_name = entity_id.removeprefix("camera.")

            # Resolve area: entity → device → area
            area = entity_info.get("area_id")
            if not area:
                device_id = entity_info.get("device_id")
                if device_id and device_id in devices_data:
                    area = devices_data[device_id].get("area_id")

            mapping[camera_name] = area if area else camera_name

        return mapping

    async def _refresh_camera_rooms(self):
        """Refresh camera→room mapping from discovery cache (merge, not replace).

        Config overrides always take priority over discovered mappings.
        """
        discovered = await self._discover_camera_rooms()

        # Merge: discovered cameras added, existing preserved
        for cam, room in discovered.items():
            # Config overrides always win
            if self._config_camera_rooms and cam in self._config_camera_rooms:
                continue
            self.camera_rooms[cam] = room

        self.logger.info(
            f"Camera rooms refreshed: {len(self.camera_rooms)} cameras "
            f"({len(discovered)} discovered)"
        )
```

**3d. Update `initialize()` to call discovery + subscribe to events**

Add to the end of `initialize()`, before the final log line:

```python
        # Discover camera→room mapping from entity cache
        try:
            await self._refresh_camera_rooms()
        except Exception as e:
            self.logger.warning(f"Camera discovery failed (non-fatal): {e}")
```

**3e. Update `on_event` (or add it) to handle discovery_complete**

The PresenceModule needs an `on_event` method (or update it if it exists) to refresh cameras when discovery re-runs. Check if Module base class has `on_event` — it should. Add:

```python
    async def on_event(self, event_type: str, data: dict[str, Any]):
        """Handle hub events — refresh cameras on discovery completion."""
        if event_type == "discovery_complete":
            try:
                await self._refresh_camera_rooms()
            except Exception as e:
                self.logger.warning(f"Camera refresh on discovery event failed: {e}")
```

**3f. Delete hard-coded room parsing in `_resolve_room()` (lines 491-501)**

Remove:
```python
        # Fallback: parse entity_id for known patterns
        eid = entity_id.lower()
        if "bedroom" in eid:
            return "bedroom"
        if "closet" in eid:
            return "closet"
        if "front_door" in eid or "doorbell" in eid:
            return "front_door"

        self.logger.debug("Could not resolve room for %s", entity_id)
        return None
```

Replace with:
```python
        self.logger.debug("Could not resolve room for %s", entity_id)
        return None
```

**Step 4: Update existing tests that reference DEFAULT_CAMERA_ROOMS**

In `tests/hub/test_presence.py`, update:

1. Remove `DEFAULT_CAMERA_ROOMS` from the import (line 20-23):
```python
from aria.modules.presence import (
    SIGNAL_STALE_S,
    PresenceModule,
)
```

2. Update `TestInitialization`:
- Delete `test_default_camera_rooms` (line 110-111) — no longer applies
- Update `test_custom_camera_rooms` (line 113-116) — still valid, just confirm it works
- Delete `test_default_camera_rooms_coverage` (line 663-666)

3. Update `TestFrigateEvents.test_camera_room_mapping` (line 262-272):
- Pre-populate the module's `camera_rooms` with the mapping needed for the test:
```python
    async def test_camera_room_mapping(self, module):
        module.camera_rooms = {"panoramic": "backyard"}
        event = {
            "after": {
                "camera": "panoramic",
                "label": "person",
                "score": 0.9,
            }
        }
        await module._handle_frigate_event(event)
        assert "backyard" in module._room_signals
```

4. Update `TestFlushPresenceState.test_flush_includes_camera_rooms` (line 603-607):
```python
    async def test_flush_includes_camera_rooms(self, module):
        await module._flush_presence_state()
        cached = module.hub.cache._cache.get(CACHE_PRESENCE)
        assert "camera_rooms" in cached
        assert isinstance(cached["camera_rooms"], dict)
```

5. Update `TestRoomResolution` — delete fallback tests:
- Delete `test_resolve_fallback_bedroom` (line 452-454)
- Delete `test_resolve_fallback_front_door` (line 456-458)
- Delete `test_resolve_fallback_doorbell` (line 460-462)

6. Update `TestEdgeCases.test_resolve_room_cache_exception` (line 644-649):
```python
    async def test_resolve_room_cache_exception(self, module):
        """Cache errors should not propagate — returns None."""
        module.hub.get_cache = AsyncMock(side_effect=RuntimeError("broken"))
        room = await module._resolve_room("light.bedroom_lamp", {})
        assert room is None  # No hard-coded fallback
```

**Step 5: Run all presence tests**

Run: `.venv/bin/python -m pytest tests/hub/test_presence.py -v --timeout=120`
Expected: All PASS

**Step 6: Commit**

```bash
git add aria/modules/presence.py tests/hub/test_presence.py
git commit -m "feat(presence): dynamic camera discovery, remove all hard-coded data

Camera-to-room mapping now comes from HA entity registry via
discovery cache. DEFAULT_CAMERA_ROOMS deleted. Hard-coded room
name parsing in _resolve_room() deleted. Config overrides still
supported via presence.camera_rooms."
```

---

### Task 5: Publish `discovery_complete` Event from DiscoveryModule

**Files:**
- Modify: `aria/modules/discovery.py` (add publish call in `_store_discovery_results`)
- Test: `tests/hub/test_discovery_lifecycle.py` (add test)

**Step 1: Write failing test**

Add to `tests/hub/test_discovery_lifecycle.py`:

```python
class TestDiscoveryEvents:
    """Discovery should publish events for consumers."""

    @pytest.mark.asyncio
    async def test_publishes_discovery_complete(self, module, mock_hub):
        """_store_discovery_results should publish discovery_complete event."""
        results = {
            "entities": {"light.x": {"area_id": "room"}},
            "devices": {},
            "areas": {},
            "capabilities": {},
        }
        await module._store_discovery_results(results)

        publish_calls = mock_hub.publish.call_args_list
        event_types = [c[0][0] for c in publish_calls]
        assert "discovery_complete" in event_types
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_discovery_lifecycle.py::TestDiscoveryEvents -v --timeout=120`
Expected: FAIL

**Step 3: Add publish call**

At the end of `_store_discovery_results()`, add:

```python
        # Notify consumers (e.g., PresenceModule refreshes camera mapping)
        await self.hub.publish("discovery_complete", metadata)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_discovery_lifecycle.py -v --timeout=120`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/modules/discovery.py tests/hub/test_discovery_lifecycle.py
git commit -m "feat(discovery): publish discovery_complete event for consumers"
```

---

### Task 6: Full Test Suite + Integration Verification

**Files:**
- No code changes — verification only

**Step 1: Check available memory**

Run: `free -h | awk '/Mem:/{print $7}'`
If < 4G, run by suite. If >= 4G, run full.

**Step 2: Run hub tests**

Run: `.venv/bin/python -m pytest tests/hub/ -v --timeout=120 -x`
Expected: All PASS (including existing organic discovery, shadow, activity tests)

**Step 3: Run integration tests**

Run: `.venv/bin/python -m pytest tests/integration/ -v --timeout=120`
Expected: All PASS

**Step 4: If any test fails, fix and re-run**

Common issues:
- Other tests importing `DEFAULT_CAMERA_ROOMS` — grep for it: `grep -r "DEFAULT_CAMERA_ROOMS" tests/`
- Tests that assume `_resolve_room` returns a fallback room for bedroom/closet patterns
- Tests that expect the entities cache to NOT have `_lifecycle` fields

**Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: update tests for dynamic entity discovery"
```

---

### Task 7: Update CLAUDE.md Gotchas

**Files:**
- Modify: project `CLAUDE.md` — update Frigate gotcha and add discovery lifecycle note

**Step 1: Update gotchas section**

Add to the Gotchas section in the project CLAUDE.md:

```markdown
- **Entity discovery uses lifecycle merge** — Entities/devices/areas cache is never fully replaced. Missing entities are marked stale, then archived after `discovery.stale_ttl_hours` (default 72h). Archived entities auto-promote if rediscovered. See `docs/plans/2026-02-17-dynamic-entity-discovery-design.md`.
- **Camera-to-room mapping is discovery-driven** — No hard-coded camera list. Cameras are found via `camera.*` entities in HA registry, room resolved via device→area chain. Manual overrides via `presence.camera_rooms` config key.
```

Remove or update the existing Frigate gotcha if it references hard-coded camera setup.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update gotchas for dynamic entity discovery"
```

---

Plan complete and saved to `docs/plans/2026-02-17-dynamic-entity-discovery-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?
