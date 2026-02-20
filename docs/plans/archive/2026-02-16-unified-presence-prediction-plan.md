# Unified Presence Detection + Prediction Accuracy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the presence detection system (Frigate + HA sensors → ARIA), then connect it to the prediction pipeline so real-time presence data improves ARIA's forecasting accuracy. Fix systemic data quality and timing issues along the way.

**Architecture:** Two modules — presence (real-time room occupancy via camera/sensor fusion) and prediction accuracy (unified feature extraction, snapshot validation, presence-informed priors). Presence feeds into predictions via shared cache. Frigate → MQTT → PresenceModule → BayesianOccupancy → Cache → Engine features.

**Tech Stack:** Python 3.12, asyncio, aiomqtt, aiohttp, scikit-learn, Frigate (Docker), MQTT (Mosquitto on HA Pi), pytest

---

## Current State

### What's Already Built
- `aria/modules/presence.py` — 645-line presence module (MQTT + WS listeners, BayesianOccupancy fusion, 30s flush)
- `tests/hub/test_presence.py` — 51 tests
- `aria/engine/analysis/occupancy.py` — BayesianOccupancy with 9 signal types (camera_person, camera_face, motion, door, etc.)
- `~/frigate/` — Docker running with 3 active ONVIF cameras (driveway, backyard, panoramic), 4 UniFi cameras disabled (need RTSP aliases)
- Frigate face recognition enabled (FaceNet, threshold 0.85, collecting unknowns)
- MQTT credentials configured (host/user/password via MQTT_HOST, MQTT_USER, MQTT_PASSWORD env vars)

### What's NOT Done (from design doc remaining work)
- [ ] UniFi Protect RTSP streams (4 cameras need aliases)
- [ ] Frigate HA integration (HACS)
- [ ] Hub restart to load presence module
- [ ] End-to-end verification (person → MQTT → cache → API)
- [ ] Dashboard presence card
- [ ] Face labeling workflow docs

### Systems Engineering Gaps (from previous session analysis)
- Feature extraction is duplicated: engine `vector_builder.py` and hub `activity_labeler.py` have independent implementations that can silently diverge
- No snapshot validation before training — corrupt data during HA restart can poison models
- Timer dependencies are implicit — if HA API is down, downstream timers run on stale data with no guard
- Presence data doesn't feed into engine predictions yet (only live in hub cache)

---

## Phase 1: Finish Presence Detection (make it work)

### Task 1: Verify Frigate Is Running and Publishing MQTT

**Files:**
- None (infrastructure verification only)

**Step 1: Check Frigate container is running**

Run: `docker ps --filter name=frigate --format '{{.Status}}'`
Expected: `Up X hours/days`

**Step 2: Check Frigate is detecting on active cameras**

Run: `curl -s http://127.0.0.1:5000/api/stats | python3 -m json.tool | head -40`
Expected: JSON with `driveway`, `backyard`, `panoramic` showing detection stats

**Step 3: Check MQTT messages flowing**

Run: `timeout 30 mosquitto_sub -h $MQTT_HOST -u $MQTT_USER -P $MQTT_PASSWORD -t 'frigate/#' -C 3 2>/dev/null || echo "Install: sudo apt install mosquitto-clients"`
Expected: At least 1 MQTT message (or install mosquitto-clients first)

**Step 4: Commit**

No code changes — this is a verification step only.

---

### Task 2: Verify ARIA Presence Module Loads

**Files:**
- Check: `aria/cli.py:317-324`
- Check: `aria/modules/presence.py`

**Step 1: Check if presence module is registered in running hub**

Run: `curl -s http://127.0.0.1:8001/api/cache/presence | python3 -m json.tool 2>/dev/null || echo "Presence cache empty or hub not running"`
Expected: Either JSON presence data OR empty/error (tells us if module loaded)

**Step 2: Check hub logs for presence module status**

Run: `journalctl --user -u aria-hub --since "1 hour ago" | grep -i presence | tail -10`
Expected: Either "Presence module started" or "Presence module failed (non-fatal)"

**Step 3: If module isn't loaded, restart hub**

Run: `systemctl --user restart aria-hub && sleep 5 && journalctl --user -u aria-hub --since "30 seconds ago" | grep -i presence`
Expected: "Presence module started" (or error message to debug)

**Step 4: Document findings**

If there's an error, fix it before proceeding. Common issues:
- `aiomqtt not installed` → `cd ~/Documents/projects/ha-aria && .venv/bin/pip install aiomqtt`
- MQTT connection refused → Check Frigate MQTT config matches HA Mosquitto credentials
- WS auth failed → Check HA_URL/HA_TOKEN env vars

---

### Task 3: End-to-End Presence Verification

**Files:**
- None (functional verification)

**Step 1: Trigger a person detection**

Walk past one of the active cameras (driveway, backyard, or panoramic). Or check Frigate UI for recent events:

Run: `curl -s "http://127.0.0.1:5000/api/events?limit=5&label=person" | python3 -m json.tool | head -30`
Expected: Recent person detection events

**Step 2: Check if ARIA received the MQTT event**

Run: `curl -s http://127.0.0.1:8001/api/cache/presence | python3 -m json.tool`
Expected: JSON with `rooms` containing at least one room with `probability > 0.5`

**Step 3: Check HA sensor signals are flowing**

Run: `curl -s http://127.0.0.1:8001/api/cache/presence | python3 -m json.tool | grep -A 3 "signals"`
Expected: Both camera and HA sensor signals present (motion, light_interaction, etc.)

**Step 4: Verify face recognition data**

Run: `curl -s http://127.0.0.1:8001/api/cache/presence | python3 -m json.tool | grep -A 5 "face_recognition"`
Expected: `enabled: true`, `labeled_faces` dict (may be empty if no faces labeled yet)

**Step 5: Document any gaps found**

Record which signal types are working, which aren't. This informs Phase 2 priorities.

---

### Task 4: Enable UniFi Protect RTSP Streams (Manual — User Action Required)

**Files:**
- Modify: `~/frigate/config/config.yml` (enable cameras + update RTSP URLs)

**Step 1: Document instructions for user**

The user needs to do this in the UniFi Protect UI:
1. Open Protect → Cameras → Select camera (Front Doorbell, Bedroom 1, Bedroom 2, Pool)
2. Go to Settings → Advanced → RTSP
3. Enable RTSP stream
4. Note the RTSP alias shown (e.g., `doorbell_high`, `doorbell_low`)
5. Repeat for all 4 cameras

**Step 2: After user provides aliases, update Frigate config**

For each camera, change `enabled: false` to `enabled: true` and update the RTSP path with the real alias.

Example for front_doorbell:
```yaml
  front_doorbell:
    enabled: true  # ← was false
    ffmpeg:
      inputs:
        - path: rtsps://<unifi-ip>:7441/<ACTUAL_HIGH_ALIAS>
          roles:
            - record
        - path: rtsps://<unifi-ip>:7441/<ACTUAL_LOW_ALIAS>
          roles:
            - detect
```

**Step 3: Restart Frigate**

Run: `cd ~/frigate && docker compose restart`
Expected: Frigate restarts and connects to new cameras

**Step 4: Verify new cameras active**

Run: `curl -s http://127.0.0.1:5000/api/stats | python3 -m json.tool | grep -E '"(front_doorbell|carters_room|collins_room|pool)"'`
Expected: All enabled cameras show in stats

**Step 5: Commit config change**

```bash
cd ~/frigate && git add config/config.yml && git commit -m "feat: enable UniFi Protect cameras with RTSP aliases"
```

---

## Phase 2: Systems Engineering — Data Quality (make it reliable)

### Task 5: Add Snapshot Validation Before Training

**Files:**
- Create: `aria/engine/validation.py`
- Modify: `aria/engine/models/trainer.py` (or wherever retrain is called)
- Test: `tests/engine/test_validation.py`

**Step 1: Write the failing test**

```python
# tests/engine/test_validation.py
"""Tests for snapshot validation before model training."""

import pytest
from aria.engine.validation import validate_snapshot, validate_snapshot_batch


class TestSnapshotValidation:
    """Ensure corrupt/incomplete snapshots are caught before training."""

    def test_valid_snapshot_passes(self):
        snap = {
            "date": "2026-02-15",
            "entities": {"total": 3050, "unavailable": 50},
            "power": {"total_watts": 200.0},
            "occupancy": {"people_home": ["Justin"], "device_count_home": 25},
            "motion": {"sensors": {"Closet motion": "on"}, "active_count": 1},
            "lights": {"on": 5, "off": 60},
        }
        errors = validate_snapshot(snap)
        assert errors == []

    def test_missing_date_rejected(self):
        snap = {"entities": {"total": 100}}
        errors = validate_snapshot(snap)
        assert any("date" in e for e in errors)

    def test_zero_entities_rejected(self):
        snap = {"date": "2026-02-15", "entities": {"total": 0, "unavailable": 0}}
        errors = validate_snapshot(snap)
        assert any("entities" in e.lower() for e in errors)

    def test_high_unavailable_ratio_flagged(self):
        """If >50% entities unavailable, HA was likely restarting."""
        snap = {
            "date": "2026-02-15",
            "entities": {"total": 3050, "unavailable": 2000},
            "power": {"total_watts": 0},
            "occupancy": {},
            "motion": {},
            "lights": {},
        }
        errors = validate_snapshot(snap)
        assert any("unavailable" in e.lower() for e in errors)

    def test_batch_filters_bad_snapshots(self):
        good = {"date": "2026-02-15", "entities": {"total": 3050, "unavailable": 50},
                "power": {"total_watts": 200}, "occupancy": {}, "motion": {}, "lights": {}}
        bad = {"date": "2026-02-14", "entities": {"total": 0, "unavailable": 0}}
        valid, rejected = validate_snapshot_batch([good, bad])
        assert len(valid) == 1
        assert len(rejected) == 1
        assert valid[0]["date"] == "2026-02-15"
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/engine/test_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aria.engine.validation'`

**Step 3: Write minimal implementation**

```python
# aria/engine/validation.py
"""Snapshot validation — catches corrupt/incomplete data before model training.

Prevents the scenario where HA restarts mid-snapshot, producing a snapshot with
0 entities or 90% unavailable, which poisons model training for a week.
"""

from typing import Dict, List, Tuple

# Minimum viable snapshot requirements
MIN_ENTITY_COUNT = 100  # HA has ~3050; anything below 100 means HA was down
MAX_UNAVAILABLE_RATIO = 0.5  # >50% unavailable = HA was likely restarting
REQUIRED_SECTIONS = ["date", "entities"]


def validate_snapshot(snapshot: Dict) -> List[str]:
    """Validate a single snapshot for training readiness.

    Returns list of error strings (empty = valid).
    """
    errors = []

    # Required fields
    if not snapshot.get("date"):
        errors.append("Missing required field: date")

    # Entity count sanity check
    entities = snapshot.get("entities", {})
    total = entities.get("total", 0)
    unavailable = entities.get("unavailable", 0)

    if total < MIN_ENTITY_COUNT:
        errors.append(
            f"Entity count too low: {total} (min {MIN_ENTITY_COUNT}). "
            "HA may have been down during snapshot."
        )

    if total > 0 and unavailable / total > MAX_UNAVAILABLE_RATIO:
        errors.append(
            f"High unavailable ratio: {unavailable}/{total} "
            f"({unavailable/total:.0%}). HA may have been restarting."
        )

    return errors


def validate_snapshot_batch(
    snapshots: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Validate a batch of snapshots, separating valid from rejected.

    Returns (valid_snapshots, rejected_snapshots).
    """
    valid = []
    rejected = []

    for snap in snapshots:
        errors = validate_snapshot(snap)
        if errors:
            rejected.append(snap)
        else:
            valid.append(snap)

    return valid, rejected
```

**Step 4: Run test to verify it passes**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/engine/test_validation.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/engine/validation.py tests/engine/test_validation.py
git commit -m "feat: add snapshot validation to catch corrupt data before training"
```

---

### Task 6: Wire Snapshot Validation Into Training Pipeline

**Files:**
- Modify: `aria/engine/predictions/predictor.py` (or wherever snapshots are loaded for training)
- Modify: `aria/engine/models/trainer.py` (if exists)
- Test: `tests/engine/test_validation.py` (add integration-style test)

**Step 1: Find where snapshots are loaded for training**

Run: `cd ~/Documents/projects/ha-aria && grep -rn "daily_snapshots\|load.*snapshot" aria/engine/ --include="*.py" | head -20`
Expected: File paths where snapshot lists are assembled before model training

**Step 2: Add validation call before training**

At each location where snapshots are loaded and fed to a model, add:
```python
from aria.engine.validation import validate_snapshot_batch

# Before training
valid_snapshots, rejected = validate_snapshot_batch(daily_snapshots)
if rejected:
    logger.warning(f"Rejected {len(rejected)} corrupt snapshots from training data")
daily_snapshots = valid_snapshots
```

**Step 3: Write integration test**

```python
def test_training_skips_corrupt_snapshots(self):
    """Ensure the training pipeline filters bad data."""
    # This test depends on how training is called — adapt to actual function signature
    pass  # Implement after reading the actual training code
```

**Step 4: Run existing tests to confirm no regressions**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/engine/ -v --timeout=120`
Expected: All engine tests pass

**Step 5: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add -p  # Stage only validation integration changes
git commit -m "feat: wire snapshot validation into training pipeline"
```

---

### Task 7: Add Timer Dependency Guards

**Files:**
- Create: `bin/check-ha-health.sh`
- Modify: `~/.config/systemd/user/aria-snapshot.service` (and other timer services)
- Test: Manual verification

**Step 1: Write the HA health check script**

```bash
#!/usr/bin/env bash
# bin/check-ha-health.sh — Pre-flight check before batch engine commands.
# Exit 0 if HA is healthy, exit 1 if not.
# Used as ExecStartPre in systemd timer services.

set -euo pipefail

source ~/.env

# Check HA API is reachable and returning real data
RESPONSE=$(curl -sf -m 10 \
  -H "Authorization: Bearer ${HA_TOKEN}" \
  "${HA_URL}/api/states?limit=1" 2>/dev/null) || {
    echo "ARIA guard: HA API unreachable at ${HA_URL}" >&2
    exit 1
}

# Check we got actual entity data (not an error page)
ENTITY_COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || {
    echo "ARIA guard: HA API returned invalid JSON" >&2
    exit 1
}

if [ "$ENTITY_COUNT" -lt 100 ]; then
    echo "ARIA guard: HA returned only $ENTITY_COUNT entities (expected 3000+), likely restarting" >&2
    exit 1
fi

echo "ARIA guard: HA healthy ($ENTITY_COUNT entities)"
exit 0
```

**Step 2: Make executable**

Run: `chmod +x ~/Documents/projects/ha-aria/bin/check-ha-health.sh`

**Step 3: Add ExecStartPre to key timer services**

For `aria-snapshot.service`, `aria-intraday.service`, `aria-retrain.service`:
```ini
[Service]
ExecStartPre=/home/justin/Documents/projects/ha-aria/bin/check-ha-health.sh
```

**Step 4: Test the guard manually**

Run: `~/Documents/projects/ha-aria/bin/check-ha-health.sh && echo "PASS" || echo "FAIL"`
Expected: `ARIA guard: HA healthy (3050+ entities)` + `PASS`

**Step 5: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add bin/check-ha-health.sh
git commit -m "feat: add HA health guard for batch timer services"
```

Note: Systemd service modifications are separate from the git repo. Apply them manually:
```bash
systemctl --user edit aria-snapshot.service
# Add ExecStartPre line
systemctl --user daemon-reload
```

---

## Phase 3: Connect Presence → Predictions (make it smart)

### Task 8: Add Presence Signals to Engine Feature Vectors

**Files:**
- Modify: `aria/engine/features/feature_config.py`
- Modify: `aria/engine/features/vector_builder.py`
- Test: `tests/engine/test_vector_builder.py`

**Step 1: Write the failing test**

```python
def test_presence_features_extracted(self):
    """Feature vector includes presence data when available."""
    snapshot = make_snapshot()
    snapshot["presence"] = {
        "overall_probability": 0.92,
        "occupied_room_count": 3,
        "identified_person_count": 2,
        "camera_signal_count": 5,
    }
    features = build_feature_vector(snapshot)
    assert "presence_probability" in features
    assert features["presence_probability"] == 0.92
    assert "presence_occupied_rooms" in features
    assert features["presence_occupied_rooms"] == 3

def test_presence_features_default_zero(self):
    """Feature vector defaults presence to 0 when no presence data."""
    snapshot = make_snapshot()
    features = build_feature_vector(snapshot)
    assert features.get("presence_probability", 0) == 0
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/engine/test_vector_builder.py -k "presence" -v`
Expected: FAIL

**Step 3: Add presence features to config and builder**

In `feature_config.py`, add to `DEFAULT_FEATURE_CONFIG`:
```python
"presence_features": {
    "presence_probability": True,
    "presence_occupied_rooms": True,
    "presence_identified_persons": True,
    "presence_camera_signals": True,
},
```

In `vector_builder.py`, add extraction logic:
```python
# Presence features (from real-time presence module cache)
pf = config.get("presence_features", {})
presence = snapshot.get("presence", {})
if pf.get("presence_probability"):
    features["presence_probability"] = presence.get("overall_probability", 0)
if pf.get("presence_occupied_rooms"):
    features["presence_occupied_rooms"] = presence.get("occupied_room_count", 0)
if pf.get("presence_identified_persons"):
    features["presence_identified_persons"] = presence.get("identified_person_count", 0)
if pf.get("presence_camera_signals"):
    features["presence_camera_signals"] = presence.get("camera_signal_count", 0)
```

**Step 4: Run test to verify it passes**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/engine/test_vector_builder.py -v --timeout=120`
Expected: All tests pass (existing + new)

**Step 5: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/engine/features/feature_config.py aria/engine/features/vector_builder.py tests/engine/test_vector_builder.py
git commit -m "feat: add presence data as engine feature vector inputs"
```

---

### Task 9: Inject Presence Summary Into Snapshots

**Files:**
- Modify: `aria/engine/collectors/extractors.py` (add PresenceCollector)
- Test: `tests/engine/test_collectors.py`

**Step 1: Write the failing test**

```python
def test_presence_collector_extracts_from_cache(self):
    """PresenceCollector reads presence cache and adds summary to snapshot."""
    snapshot = {"date": "2026-02-16"}
    # Mock: presence cache data as it would be read from hub cache file
    presence_cache = {
        "rooms": {
            "driveway": {"probability": 0.3, "persons": []},
            "bedroom": {"probability": 0.9, "persons": [{"name": "Justin"}]},
            "kitchen": {"probability": 0.7, "persons": []},
        },
        "occupied_rooms": ["bedroom", "kitchen"],
        "identified_persons": {"Justin": {"room": "bedroom"}},
    }
    collector = PresenceCollector()
    collector.extract(snapshot, states=[], presence_cache=presence_cache)
    assert "presence" in snapshot
    assert snapshot["presence"]["overall_probability"] > 0.5
    assert snapshot["presence"]["occupied_room_count"] == 2
    assert snapshot["presence"]["identified_person_count"] == 1
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/engine/test_collectors.py -k "presence" -v`
Expected: FAIL

**Step 3: Implement PresenceCollector**

```python
@CollectorRegistry.register("presence")
class PresenceCollector(BaseCollector):
    """Collects presence summary from hub cache for engine snapshots."""

    def extract(self, snapshot, states, presence_cache=None):
        if not presence_cache:
            snapshot["presence"] = {
                "overall_probability": 0,
                "occupied_room_count": 0,
                "identified_person_count": 0,
                "camera_signal_count": 0,
                "rooms": {},
            }
            return

        rooms = presence_cache.get("rooms", {})
        occupied = [r for r, d in rooms.items() if d.get("probability", 0) > 0.5]
        persons = presence_cache.get("identified_persons", {})
        camera_signals = sum(
            1 for r, d in rooms.items()
            for s in d.get("signals", [])
            if s.get("type", "").startswith("camera_")
        )

        # Overall probability: max of all room probabilities (someone is somewhere)
        probs = [d.get("probability", 0) for d in rooms.values()]
        overall = max(probs) if probs else 0

        snapshot["presence"] = {
            "overall_probability": round(overall, 3),
            "occupied_room_count": len(occupied),
            "identified_person_count": len(persons),
            "camera_signal_count": camera_signals,
            "rooms": {
                room: {
                    "probability": round(d.get("probability", 0), 3),
                    "person_count": len(d.get("persons", [])),
                }
                for room, d in rooms.items()
            },
        }
```

**Step 4: Run test to verify it passes**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/engine/test_collectors.py -v --timeout=120`
Expected: PASS

**Step 5: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/engine/collectors/extractors.py tests/engine/test_collectors.py
git commit -m "feat: add PresenceCollector to inject presence summary into snapshots"
```

---

### Task 10: Wire Presence Cache Into Snapshot Building

**Files:**
- Modify: `aria/engine/collectors/snapshot.py` (read presence from hub cache during snapshot)

**Step 1: Find how snapshot building currently works**

Read `aria/engine/collectors/snapshot.py` to find where collectors are called.

**Step 2: Add presence cache read**

During snapshot building, read the presence cache from the hub's SQLite DB (or via the hub API if the hub is running):

```python
# In build_snapshot(), after running collectors:
try:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get("http://127.0.0.1:8001/api/cache/presence", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                presence_cache = await resp.json()
                presence_collector = CollectorRegistry.get("presence")
                if presence_collector:
                    presence_collector().extract(snapshot, states, presence_cache=presence_cache)
except Exception:
    pass  # Hub may not be running; presence is optional
```

**Step 3: Run snapshot manually and verify**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m aria.engine.cli snapshot-intraday`
Then: `ls -la ~/ha-logs/intelligence/intraday/2026-02-16/ | tail -1`
Then check the snapshot contains presence data.

**Step 4: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/engine/collectors/snapshot.py
git commit -m "feat: inject presence cache into engine snapshots via hub API"
```

---

### Task 11: Update Activity Labeler to Use Presence Data

**Files:**
- Modify: `aria/modules/activity_labeler.py` (`_context_to_features`)
- Modify: `tests/hub/test_activity_labeler.py`

**Step 1: Write the failing test**

```python
def test_context_to_features_with_presence(self, labeler):
    """Verify presence features are included when available."""
    ctx = make_context()
    ctx["presence_probability"] = 0.92
    ctx["occupied_room_count"] = 3
    features = labeler._context_to_features(ctx)
    # Features should now be 10 (was 8 + 2 presence)
    assert len(features) == 10
    assert features[8] == 0.92  # presence_probability
    assert features[9] == 3.0   # occupied_room_count
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_activity_labeler.py -k "presence" -v`
Expected: FAIL

**Step 3: Add presence features to _context_to_features**

In `activity_labeler.py`, extend the feature vector:
```python
def _context_to_features(self, ctx: dict) -> list:
    return [
        float(ctx.get("power_watts", 0)),
        float(ctx.get("lights_on", 0)),
        float(ctx.get("motion_room_count", 0) if isinstance(ctx.get("motion_rooms"), int) else len(ctx.get("motion_rooms", []) if isinstance(ctx.get("motion_rooms"), list) else [])),
        float(ctx.get("hour", 0)),
        1.0 if ctx.get("occupancy") == "home" else 0.0,
        float(ctx.get("correlated_entities_active", 0)),
        float(ctx.get("anomaly_nearby", 0)),
        float(ctx.get("active_appliance_count", 0)),
        # New: presence features
        float(ctx.get("presence_probability", 0)),
        float(ctx.get("occupied_room_count", 0)),
    ]
```

**CRITICAL:** This changes the feature count from 8 to 10. The existing classifier invalidation check in `initialize()` will auto-reset incompatible cached classifiers — this is by design.

**Step 4: Update existing tests for new feature count**

Update all `assert len(features) == 8` to `assert len(features) == 10`.
Add `assert features[8] == 0.0` and `assert features[9] == 0.0` to the defaults test.

**Step 5: Run all activity labeler tests**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/test_activity_labeler.py -v --timeout=120`
Expected: All tests pass

**Step 6: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/modules/activity_labeler.py tests/hub/test_activity_labeler.py
git commit -m "feat: add presence probability + room count to activity labeler features"
```

---

### Task 12: Feed Presence Into Activity Labeler Context

**Files:**
- Modify: `aria/modules/activity_labeler.py` (where context is assembled before prediction)

**Step 1: Find where sensor context is built**

Search for where `predict_activity` is called and how the context dict is assembled.

**Step 2: Add presence cache to context assembly**

```python
# When building sensor_context for prediction:
presence = await self.hub.get_cache(CACHE_PRESENCE)
if presence:
    sensor_context["presence_probability"] = max(
        (r.get("probability", 0) for r in presence.get("rooms", {}).values()),
        default=0
    )
    sensor_context["occupied_room_count"] = len(presence.get("occupied_rooms", []))
```

**Step 3: Run full test suite to confirm no regressions**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/hub/ -k "activity" -v --timeout=120`
Expected: All activity-related tests pass

**Step 4: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/modules/activity_labeler.py
git commit -m "feat: inject live presence data into activity labeler context"
```

---

## Phase 4: Dashboard + Operational (make it visible)

### Task 13: Add Presence Dashboard Card

**Files:**
- Create: `aria/dashboard/spa/src/components/PresenceCard.jsx`
- Modify: `aria/dashboard/spa/src/App.jsx` (add card to layout)
- Modify: `aria/dashboard/spa/src/hooks/useCache.js` (if needed for presence subscription)

**Step 1: Design the presence card**

The card should show:
- Per-room occupancy probability (color-coded: green >0.7, yellow 0.3-0.7, gray <0.3)
- Identified persons with room + last seen
- MQTT connection status
- Recent detections (last 5) with thumbnails
- Face recognition status (enabled/disabled, labeled face count)

**Step 2: Implement PresenceCard component**

Follow existing card patterns in the SPA (check other card components for structure).

```jsx
// aria/dashboard/spa/src/components/PresenceCard.jsx
import { h } from 'preact';
import { useCache } from '../hooks/useCache';

export function PresenceCard() {
    const presence = useCache('presence');
    if (!presence) return null;

    const rooms = Object.entries(presence.rooms || {})
        .filter(([name]) => name !== 'overall')
        .sort((a, b) => b[1].probability - a[1].probability);

    return (
        <div class="card">
            <h2>Presence</h2>
            <div class="presence-grid">
                {rooms.map(([room, data]) => (
                    <div class={`presence-room ${data.probability > 0.7 ? 'occupied' : data.probability > 0.3 ? 'maybe' : 'empty'}`}>
                        <span class="room-name">{room.replace(/_/g, ' ')}</span>
                        <span class="room-prob">{Math.round(data.probability * 100)}%</span>
                        {data.persons?.map(p => (
                            <span class="person-badge">{p.name}</span>
                        ))}
                    </div>
                ))}
            </div>
            {presence.mqtt_connected ? null : (
                <div class="warning">MQTT disconnected</div>
            )}
        </div>
    );
}
```

**Step 3: Build SPA**

Run: `cd ~/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`
Expected: `dist/bundle.js` rebuilt

**Step 4: Restart hub and verify**

Run: `systemctl --user restart aria-hub`
Then open: `http://127.0.0.1:8001/ui/`
Expected: Presence card visible with room data

**Step 5: Commit**

```bash
cd ~/Documents/projects/ha-aria
git add aria/dashboard/spa/src/components/PresenceCard.jsx aria/dashboard/spa/src/App.jsx
git commit -m "feat: add presence dashboard card with room occupancy + face recognition status"
```

---

### Task 14: Full Pipeline Verification (Dual-Axis)

**Files:**
- None (verification only)

**Step 1: Horizontal sweep — hit every presence-related endpoint**

```bash
# Presence cache
curl -s http://127.0.0.1:8001/api/cache/presence | python3 -m json.tool | head -5

# Latest snapshot should contain presence section
curl -s http://127.0.0.1:8001/api/cache/intelligence | python3 -m json.tool | grep -A 3 "presence"

# Dashboard serves
curl -s http://127.0.0.1:8001/ui/ | head -5
```

**Step 2: Vertical trace — one person detection through full stack**

```
Person walks past camera →
  Frigate detects person (check: curl http://127.0.0.1:5000/api/events?limit=1) →
    MQTT message published (check: mosquitto_sub) →
      PresenceModule._handle_frigate_event() (check: hub logs) →
        _room_signals updated →
          _flush_presence_state() (every 30s) →
            BayesianOccupancy._bayesian_fuse() →
              hub.set_cache(CACHE_PRESENCE, ...) →
                GET /api/cache/presence shows updated room →
                  WebSocket pushes to dashboard →
                    Dashboard PresenceCard renders
```

Verify each step with the corresponding curl/log check.

**Step 3: Vertical trace — presence into predictions**

```
aria snapshot-intraday →
  PresenceCollector reads presence cache →
    Snapshot JSON includes "presence" section →
      Engine feature vector includes presence_probability →
        Model training uses presence features →
          Predictions reflect occupancy patterns
```

**Step 4: Document results**

Record which signals are flowing, latencies, any gaps found.

**Step 5: Commit any fixes found during verification**

---

### Task 15: Run Full Test Suite

**Files:**
- None (verification)

**Step 1: Check available memory**

Run: `free -h | awk '/Mem:/{print $7}'`
If <4G, run by suite instead.

**Step 2: Run full suite**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/ -v --timeout=120 -q`
Expected: ~1111+ tests pass (some may be new from this work)

**Step 3: Fix any failures**

Address test failures. Common issues:
- Feature count mismatches (8 → 10 in activity labeler)
- Missing presence section in snapshot test fixtures
- Import errors for new modules

**Step 4: Final commit**

```bash
cd ~/Documents/projects/ha-aria
git add -A
git commit -m "chore: fix test suite after presence + prediction accuracy changes"
```

---

## Summary

| Phase | Tasks | What It Achieves |
|-------|-------|-----------------|
| **1: Finish Presence** | 1-4 | Camera + sensor → room occupancy working end-to-end |
| **2: Data Quality** | 5-7 | Corrupt snapshots caught, HA health checked before batch jobs |
| **3: Connect Presence → Predictions** | 8-12 | Presence data feeds into feature vectors + activity predictions |
| **4: Dashboard + Ops** | 13-15 | Visible, verified, tested |

**Dependencies:**
- Tasks 1-3 can run independently (verification of existing work)
- Task 4 requires user action (UniFi Protect UI)
- Tasks 5-7 are independent of presence (pure data quality)
- Tasks 8-12 require Phase 1 complete (presence flowing)
- Tasks 13-15 require all prior phases

**Estimated scope:** ~15 tasks, each 2-15 minutes of implementation. Phase 2 (data quality) can be done in parallel with Phase 1 verification since they don't share files.
