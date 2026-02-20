# Frigate Camera Name Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Frigate MQTT camera names resolve to HA rooms by adding alias keys to camera_rooms.

**Architecture:** Extract Frigate camera names from the existing `/api/config` call, substring-match against HA camera entities, add short-name aliases to camera_rooms dict.

**Tech Stack:** Python, aiohttp (existing), aiomqtt (existing), pytest

---

### Task 1: Extract Frigate camera names from config fetch

**Files:**
- Modify: `aria/modules/presence.py` (lines 229-252, `_fetch_face_config`)
- Test: `tests/hub/test_presence.py`

**Context:** `_fetch_face_config()` already calls `GET /api/config` on Frigate. The response contains a `cameras` dict keyed by camera name. We just need to store those names.

**Step 1: Write the failing test**

Add to `TestDiscoveryCameraMapping` class in `tests/hub/test_presence.py`:

```python
@pytest.mark.asyncio
async def test_fetch_face_config_stores_frigate_cameras(self):
    """Frigate config fetch extracts camera names."""
    module = self._make_module()

    mock_config_response = MagicMock()
    mock_config_response.status = 200
    mock_config_response.json = AsyncMock(return_value={
        "face_recognition": {"enabled": True},
        "cameras": {
            "backyard": {"enabled": True},
            "driveway": {"enabled": True},
            "pool": {"enabled": True},
        },
    })

    mock_faces_response = MagicMock()
    mock_faces_response.status = 200
    mock_faces_response.json = AsyncMock(return_value={})

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(side_effect=[
            AsyncContextManager(mock_config_response),
            AsyncContextManager(mock_faces_response),
        ])
        mock_session_cls.return_value = mock_session

        await module._fetch_face_config()

    assert module._frigate_camera_names == {"backyard", "driveway", "pool"}
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_presence.py::TestDiscoveryCameraMapping::test_fetch_face_config_stores_frigate_cameras -v`
Expected: FAIL — `_frigate_camera_names` doesn't exist

**Step 3: Implement**

In `__init__`, add:
```python
self._frigate_camera_names: set[str] = set()
```

In `_fetch_face_config`, after extracting face config, add:
```python
# Extract camera names for MQTT alias resolution
cameras = config.get("cameras", {})
if cameras:
    self._frigate_camera_names = set(cameras.keys())
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_presence.py::TestDiscoveryCameraMapping::test_fetch_face_config_stores_frigate_cameras -v`
Expected: PASS

**Step 5: Commit**

```bash
git add aria/modules/presence.py tests/hub/test_presence.py
git commit -m "feat(presence): extract Frigate camera names from config fetch"
```

---

### Task 2: Add Frigate alias keys to camera_rooms during refresh

**Files:**
- Modify: `aria/modules/presence.py` (lines 150-164, `_refresh_camera_rooms`)
- Test: `tests/hub/test_presence.py`

**Context:** After `_refresh_camera_rooms()` builds the HA entity name → area mapping, we need to add Frigate short name aliases. For each Frigate camera name, find the HA camera entity that contains it as a substring, and add an alias key.

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_refresh_adds_frigate_aliases(self):
    """Frigate short names are added as alias keys to camera_rooms."""
    module = self._make_module()
    module._frigate_camera_names = {"backyard", "driveway", "pool"}

    # Simulate HA entities with long names
    self.mock_hub.get_cache = AsyncMock(side_effect=lambda key: {
        "entities": {"data": {
            "camera.backyard_high_resolution_channel": {
                "area_id": "backyard_area",
                "device_id": "dev1",
                "_lifecycle": {"status": "active"},
            },
            "camera.driveway_high_resolution_channel": {
                "area_id": "front_door",
                "device_id": "dev2",
                "_lifecycle": {"status": "active"},
            },
            "camera.pool_mainstream": {
                "area_id": "pool_area",
                "device_id": "dev3",
                "_lifecycle": {"status": "active"},
            },
            "light.kitchen": {
                "area_id": "kitchen",
                "device_id": "dev4",
                "_lifecycle": {"status": "active"},
            },
        }},
        "devices": {"data": {}},
    }.get(key))

    await module._refresh_camera_rooms()

    # HA long names should be in the map
    assert "backyard_high_resolution_channel" in module.camera_rooms
    # Frigate short names should be aliases
    assert "backyard" in module.camera_rooms
    assert module.camera_rooms["backyard"] == "backyard_area"
    assert "driveway" in module.camera_rooms
    assert module.camera_rooms["driveway"] == "front_door"
    assert "pool" in module.camera_rooms
    assert module.camera_rooms["pool"] == "pool_area"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_presence.py::TestDiscoveryCameraMapping::test_refresh_adds_frigate_aliases -v`
Expected: FAIL — aliases not added

**Step 3: Implement**

Add a method `_add_frigate_aliases()` and call it from `_refresh_camera_rooms()`:

```python
def _add_frigate_aliases(self):
    """Add Frigate short-name aliases to camera_rooms.

    For each Frigate camera name, find the HA camera entity whose name
    contains the Frigate name as a substring. Add the short name as
    an alias pointing to the same room.
    """
    if not self._frigate_camera_names:
        return

    ha_camera_names = {
        name: room for name, room in self.camera_rooms.items()
        if name not in self._frigate_camera_names  # skip already-aliased
    }

    aliases_added = 0
    for frigate_name in self._frigate_camera_names:
        if frigate_name in self.camera_rooms:
            continue  # already exists (config override or exact match)

        # Find HA entity name containing this Frigate name
        matches = [
            (ha_name, room) for ha_name, room in ha_camera_names.items()
            if frigate_name in ha_name
        ]

        if len(matches) == 1:
            self.camera_rooms[frigate_name] = matches[0][1]
            aliases_added += 1
        elif len(matches) > 1:
            # Multiple matches — use shortest (most specific)
            best = min(matches, key=lambda m: len(m[0]))
            self.camera_rooms[frigate_name] = best[1]
            aliases_added += 1
        else:
            self.logger.debug(f"No HA entity match for Frigate camera: {frigate_name}")

    if aliases_added:
        self.logger.info(f"Added {aliases_added} Frigate camera aliases")
```

In `_refresh_camera_rooms()`, add after the discovery merge loop:

```python
# Add Frigate short-name aliases
self._add_frigate_aliases()
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_presence.py::TestDiscoveryCameraMapping::test_refresh_adds_frigate_aliases -v`
Expected: PASS

**Step 5: Commit**

```bash
git add aria/modules/presence.py tests/hub/test_presence.py
git commit -m "feat(presence): add Frigate short-name aliases to camera room mapping"
```

---

### Task 3: Ensure face config runs before camera refresh on startup

**Files:**
- Modify: `aria/modules/presence.py` (lines 166-208, `initialize`)

**Context:** On startup, `_fetch_face_config` populates `_frigate_camera_names`, and `_refresh_camera_rooms` uses those names. Currently face config runs as a scheduled task (may not complete before refresh). Fix: call `_fetch_face_config()` directly before `_refresh_camera_rooms()` in `initialize()`.

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_initialize_fetches_frigate_before_refresh(self):
    """Face config (Frigate cameras) is fetched before camera room refresh."""
    module = self._make_module()
    call_order = []

    async def mock_fetch():
        call_order.append("fetch_face_config")
        module._frigate_camera_names = {"backyard"}

    async def mock_refresh():
        call_order.append("refresh_camera_rooms")

    module._fetch_face_config = mock_fetch
    module._refresh_camera_rooms = mock_refresh

    await module.initialize()

    # face config must come before camera refresh
    assert call_order.index("fetch_face_config") < call_order.index("refresh_camera_rooms")
```

**Step 2: Implement**

In `initialize()`, add before the `_refresh_camera_rooms()` call:

```python
# Fetch Frigate config first (populates _frigate_camera_names for alias resolution)
try:
    await self._fetch_face_config()
except Exception as e:
    self.logger.warning(f"Frigate config fetch failed (non-fatal): {e}")
```

**Step 3: Commit**

```bash
git add aria/modules/presence.py tests/hub/test_presence.py
git commit -m "fix(presence): fetch Frigate config before camera room refresh on startup"
```

---

### Task 4: Run full test suite and verify live

**Step 1: Run tests**

```bash
.venv/bin/python -m pytest tests/hub/test_presence.py -v --timeout=60
.venv/bin/python -m pytest tests/hub/ tests/integration/ -q --timeout=120
```

**Step 2: Restart service and verify**

```bash
systemctl --user restart aria-hub
sleep 15
# Check that Frigate short names appear as aliases
curl -s http://127.0.0.1:8001/api/cache/presence | python3 -c "
import json,sys; d=json.load(sys.stdin)['data']
for cam, room in sorted(d.get('camera_rooms',{}).items()):
    print(f'  {cam} → {room}')
"
```

Expected: Both `backyard` and `backyard_high_resolution_channel` should appear, both pointing to the same HA area.

**Step 3: Commit any fixes**

```bash
git add -u && git commit -m "fix(presence): test fixes for Frigate alias resolution"
```

**Step 4: Push**

```bash
git push origin main
```
