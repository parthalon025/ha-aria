# Home Assistant Intelligence Engine — Implementation Plan (v1)

> **Status: IMPLEMENTED.** Superseded by v2 ML design: `2026-02-10-ha-intelligence-ml-design.md`

**Goal:** Build a self-learning home intelligence engine that aggregates HA data with weather/calendar/holidays, detects anomalies, predicts device failures and behavior patterns, and improves its own accuracy over time — all running locally via Ollama.

**Architecture:** Python aggregator script collects daily snapshots from HA REST API + weather + calendar into structured JSON. Statistical engine computes baselines, anomalies, correlations, and predictions. Ollama interprets findings in natural language. Self-reinforcement loop tracks prediction accuracy and adjusts weights. Results feed into Telegram brief and Claude Code skills.

**Tech Stack:** Python 3.12 (stdlib + `math`/`statistics`), Ollama qwen2.5:7b (local LLM), HA REST API, wttr.in (weather), `gog` CLI (calendar), `holidays` Python library

---

## Actual System Inventory (from audit)

This plan is grounded in real data pulled from HA on 2026-02-10:

| Domain | Count | Predictable Signals |
|--------|-------|-------------------|
| sensor | 1,535 | Power (19 W sensors from USP PDU Pro), temperature (38 °F, 8 °C), battery (69 %), 1,058 network bandwidth sensors |
| device_tracker | 557 | 322 not_home, 62 home, 169 unavailable — mostly UniFi clients |
| switch | 289 | On/off patterns, 137 unavailable (offline UniFi devices) |
| light | 73 | 21 Hue (color/brightness), 52 other (recessed, spots, bars) — on/off + brightness patterns |
| binary_sensor | 95 | 13 motion, 7 door, 2 window, 22 running, 13 problem |
| lock | 9 | 4 house locks (battery: 57-92%), Tesla locks |
| climate | 4 | Bedroom (cool, 72°F→68°F target), Entryway (heat_cool), 2 Tesla |
| camera | 6 | Reolink I91BF (2), G4 Doorbell, G4 Instants, M4308-PLE, PTZ |
| automation | 31 | 13 on, 14 off, 4 unavailable — departure, arrival, doorbell, TTS, lighting |
| media_player | 18 | Sonos multiroom — living room, atrium |
| cover | 8 | Tesla doors/trunk/frunk/windows |
| person | 6 | Justin, Lisa, Patrick, dashboard, Test, Claude Code Analysis |
| vacuum | 3 | Roborock — status, missions |

**Key assets for prediction:**
- **USP PDU Pro**: 20 outlets with real-time wattage → energy prediction
- **2 Teslas** (TARS active): charging rate, battery %, range, departure times → EV prediction
- **Hue motion sensors**: closet motion = 38 events/day → occupancy proxy
- **Person tracking**: Justin + Lisa home/not_home → occupancy ground truth
- **10 cameras**: motion events → activity patterns
- **Weather**: wttr.in (already in brief) → HVAC/lighting correlation
- **Calendar**: gog CLI (already in brief) → occupancy prediction

**Logbook stats (24h):**
- 11,216 events total, but 8,718 are clock sensor updates (sensor.date_time_*)
- Actual useful events: ~2,500/day
- Top activity: device_tracker (829), switch (347), light (230), binary_sensor (160)
- Hourly distribution: spike at 18:00 (1,482 events), otherwise 360-570/hr

---

## File Structure

```
~/.local/bin/ha-intelligence          # Main orchestrator (Python, ~800 lines)
~/ha-logs/intelligence/
  ├── daily/YYYY-MM-DD.json           # Daily aggregated snapshots
  ├── baselines.json                  # Computed per-day-of-week baselines
  ├── predictions.json                # Active predictions for today/tomorrow
  ├── accuracy.json                   # Historical prediction accuracy scores
  ├── correlations.json               # Cross-correlation matrix
  └── insights/YYYY-MM-DD.json        # Daily Ollama-generated insights
~/Documents/tests/test_ha_intelligence.py  # Tests
~/.claude/skills/ha-predict/SKILL.md  # Prediction skill
~/.claude/skills/ha-learn/SKILL.md    # Learning feedback skill
```

---

## Task 1: Daily Snapshot Aggregator

**Files:**
- Create: `~/Documents/tests/test_ha_intelligence.py`
- Create: `~/.local/bin/ha-intelligence`

The aggregator collects one daily snapshot that all other components consume. It pulls from HA REST API, weather, and calendar into a single JSON structure.

**Step 1: Write the failing test for snapshot schema**

```python
# ~/Documents/tests/test_ha_intelligence.py
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the script as a module
sys.path.insert(0, os.path.expanduser("~/.local/bin"))

class TestDailySnapshot(unittest.TestCase):
    def test_snapshot_schema_has_required_keys(self):
        """Daily snapshot must contain all data sections."""
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        snapshot = ha.build_empty_snapshot("2026-02-10")
        required_keys = [
            "date", "day_of_week", "is_weekend", "is_holiday",
            "weather", "calendar_events",
            "entities", "power", "occupancy", "climate",
            "locks", "lights", "motion", "automations",
            "ev", "logbook_summary"
        ]
        for key in required_keys:
            self.assertIn(key, snapshot, f"Missing key: {key}")

    def test_snapshot_date_metadata(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        snapshot = ha.build_empty_snapshot("2026-02-10")
        self.assertEqual(snapshot["date"], "2026-02-10")
        self.assertEqual(snapshot["day_of_week"], "Tuesday")
        self.assertFalse(snapshot["is_weekend"])

if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python3 ~/Documents/tests/test_ha_intelligence.py -v`
Expected: FAIL — module not found or function missing

**Step 3: Write the snapshot builder**

```python
#!/usr/bin/env python3
"""Home Assistant Intelligence Engine.

Collects daily snapshots, computes baselines, detects anomalies,
generates predictions, and tracks accuracy over time.

Usage:
  ha-intelligence --snapshot       # Collect today's snapshot
  ha-intelligence --analyze        # Run analysis on latest snapshot
  ha-intelligence --predict        # Generate predictions for tomorrow
  ha-intelligence --score          # Score yesterday's predictions
  ha-intelligence --report         # Full Ollama insight report
  ha-intelligence --brief          # One-liner for telegram-brief
  ha-intelligence --dry-run        # Print instead of saving
"""
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta

# === Config ===
HA_URL = os.environ.get("HA_URL", "http://192.168.1.35:8123")
HA_TOKEN = os.environ.get("HA_TOKEN", "")
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"
DATA_DIR = os.path.expanduser("~/ha-logs/intelligence")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
INSIGHTS_DIR = os.path.join(DATA_DIR, "insights")
BASELINES_PATH = os.path.join(DATA_DIR, "baselines.json")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.json")
ACCURACY_PATH = os.path.join(DATA_DIR, "accuracy.json")
CORRELATIONS_PATH = os.path.join(DATA_DIR, "correlations.json")
WEATHER_LOCATION = "Shalimar+FL"
LOGBOOK_PATH = os.path.expanduser("~/ha-logs/current.json")

# Entities to exclude from unavailable counts (normally unavailable)
UNAVAILABLE_EXCLUDE_DOMAINS = {"update", "tts", "stt"}

# Safety-critical entities (excluded from predictions that suggest changes)
SAFETY_ENTITIES = {"lock.", "alarm_", "camera."}

# US holidays (Florida)
try:
    import holidays as holidays_lib
    US_HOLIDAYS = holidays_lib.US(years=range(2025, 2028))
except ImportError:
    US_HOLIDAYS = {}


def ensure_dirs():
    for d in [DATA_DIR, DAILY_DIR, INSIGHTS_DIR]:
        os.makedirs(d, exist_ok=True)


def build_empty_snapshot(date_str):
    """Build an empty snapshot with metadata filled in."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return {
        "date": date_str,
        "day_of_week": dt.strftime("%A"),
        "is_weekend": dt.weekday() >= 5,
        "is_holiday": date_str in US_HOLIDAYS if US_HOLIDAYS else False,
        "holiday_name": US_HOLIDAYS.get(date_str, None) if US_HOLIDAYS else None,
        "weather": {},
        "calendar_events": [],
        "entities": {"total": 0, "unavailable": 0, "by_domain": {}},
        "power": {"total_watts": 0.0, "outlets": {}},
        "occupancy": {"people_home": [], "people_away": [], "device_count_home": 0},
        "climate": [],
        "locks": [],
        "lights": {"on": 0, "off": 0, "unavailable": 0, "total_brightness": 0},
        "motion": {"events_24h": 0, "sensors": {}},
        "automations": {"on": 0, "off": 0, "unavailable": 0, "fired_24h": 0},
        "ev": {},
        "logbook_summary": {"total_events": 0, "useful_events": 0, "by_domain": {}, "hourly": {}},
    }
```

**Step 4: Run test to verify it passes**

Run: `python3 ~/Documents/tests/test_ha_intelligence.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ~/Documents/tests/test_ha_intelligence.py ~/.local/bin/ha-intelligence
git commit -m "feat(ha-intelligence): scaffold with empty snapshot builder"
```

---

## Task 2: HA API Data Collection

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Populate the snapshot from the HA REST API. This is the most data-dense task — we extract power, occupancy, climate, locks, lights, motion, automations, and EV data from the 3,065 entity states.

**Step 1: Write failing test for entity extraction**

```python
class TestEntityExtraction(unittest.TestCase):
    SAMPLE_STATES = [
        {"entity_id": "sensor.usp_pdu_pro_ac_power_consumption", "state": "156.5",
         "attributes": {"unit_of_measurement": "W", "friendly_name": "USP PDU Pro AC Power Consumption"}},
        {"entity_id": "person.justin", "state": "home",
         "attributes": {"friendly_name": "Justin", "source": "device_tracker.ipad_pro"}},
        {"entity_id": "person.lisa", "state": "not_home",
         "attributes": {"friendly_name": "Lisa"}},
        {"entity_id": "climate.bedroom", "state": "cool",
         "attributes": {"current_temperature": 72, "temperature": 68, "friendly_name": "Bedroom"}},
        {"entity_id": "lock.back_door", "state": "unlocked",
         "attributes": {"battery_level": 58, "friendly_name": "Back Door"}},
        {"entity_id": "light.atrium", "state": "on",
         "attributes": {"brightness": 143, "friendly_name": "Atrium"}},
        {"entity_id": "light.office", "state": "off", "attributes": {"friendly_name": "Office"}},
        {"entity_id": "automation.arrive_justin", "state": "on",
         "attributes": {"friendly_name": "Arrive Justin", "last_triggered": "2026-02-10T22:23:00"}},
        {"entity_id": "sensor.luda_battery", "state": "71",
         "attributes": {"unit_of_measurement": "%", "friendly_name": "TARS Battery"}},
        {"entity_id": "sensor.luda_charger_power", "state": "4.0",
         "attributes": {"unit_of_measurement": "kW", "friendly_name": "TARS Charger power"}},
        {"entity_id": "sensor.luda_range", "state": "199.3",
         "attributes": {"unit_of_measurement": "mi", "friendly_name": "TARS Range"}},
        {"entity_id": "binary_sensor.hue_motion_sensor_2_motion", "state": "off",
         "attributes": {"device_class": "motion", "friendly_name": "Closet motion Motion"}},
        {"entity_id": "device_tracker.iphonea17", "state": "home",
         "attributes": {"friendly_name": "iPhonea17"}},
    ]

    def test_extract_power(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()
        snapshot = ha.build_empty_snapshot("2026-02-10")
        ha.extract_power(snapshot, self.SAMPLE_STATES)
        self.assertAlmostEqual(snapshot["power"]["total_watts"], 156.5, places=1)

    def test_extract_occupancy(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()
        snapshot = ha.build_empty_snapshot("2026-02-10")
        ha.extract_occupancy(snapshot, self.SAMPLE_STATES)
        self.assertIn("Justin", snapshot["occupancy"]["people_home"])
        self.assertIn("Lisa", snapshot["occupancy"]["people_away"])
        self.assertGreater(snapshot["occupancy"]["device_count_home"], 0)

    def test_extract_climate(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()
        snapshot = ha.build_empty_snapshot("2026-02-10")
        ha.extract_climate(snapshot, self.SAMPLE_STATES)
        self.assertEqual(len(snapshot["climate"]), 1)
        self.assertEqual(snapshot["climate"][0]["name"], "Bedroom")
        self.assertEqual(snapshot["climate"][0]["current_temp"], 72)

    def test_extract_lights(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()
        snapshot = ha.build_empty_snapshot("2026-02-10")
        ha.extract_lights(snapshot, self.SAMPLE_STATES)
        self.assertEqual(snapshot["lights"]["on"], 1)
        self.assertEqual(snapshot["lights"]["off"], 1)

    def test_extract_ev(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()
        snapshot = ha.build_empty_snapshot("2026-02-10")
        ha.extract_ev(snapshot, self.SAMPLE_STATES)
        self.assertIn("TARS", snapshot["ev"])
        self.assertEqual(snapshot["ev"]["TARS"]["battery_pct"], 71)
        self.assertAlmostEqual(snapshot["ev"]["TARS"]["charger_power_kw"], 4.0)
```

**Step 2: Run tests to verify they fail**

Run: `python3 ~/Documents/tests/test_ha_intelligence.py -v`
Expected: FAIL — extract_* functions don't exist

**Step 3: Implement extraction functions**

Add to `ha-intelligence`:

```python
def extract_power(snapshot, states):
    """Extract USP PDU Pro power data."""
    total = 0.0
    outlets = {}
    for s in states:
        eid = s["entity_id"]
        if "usp_pdu_pro" in eid and "outlet" in eid and "power" in eid:
            name = s.get("attributes", {}).get("friendly_name", eid)
            try:
                watts = float(s["state"])
                outlets[name] = watts
                total += watts
            except (ValueError, TypeError):
                pass
        elif eid == "sensor.usp_pdu_pro_ac_power_consumption":
            try:
                total = float(s["state"])
            except (ValueError, TypeError):
                pass
    snapshot["power"]["total_watts"] = total
    snapshot["power"]["outlets"] = outlets


def extract_occupancy(snapshot, states):
    """Extract person and device tracker occupancy."""
    home = []
    away = []
    device_home = 0
    for s in states:
        eid = s["entity_id"]
        state = s.get("state", "")
        name = s.get("attributes", {}).get("friendly_name", eid)
        if eid.startswith("person."):
            if state == "home":
                home.append(name)
            elif state == "not_home":
                away.append(name)
        elif eid.startswith("device_tracker.") and state == "home":
            device_home += 1
    snapshot["occupancy"]["people_home"] = home
    snapshot["occupancy"]["people_away"] = away
    snapshot["occupancy"]["device_count_home"] = device_home


def extract_climate(snapshot, states):
    """Extract climate/thermostat data."""
    zones = []
    for s in states:
        if s["entity_id"].startswith("climate."):
            attrs = s.get("attributes", {})
            name = attrs.get("friendly_name", s["entity_id"])
            # Skip Tesla HVAC
            if "tars" in name.lower() or "tessy" in name.lower():
                continue
            zones.append({
                "name": name,
                "state": s.get("state", "unknown"),
                "current_temp": attrs.get("current_temperature"),
                "target_temp": attrs.get("temperature"),
                "hvac_action": attrs.get("hvac_action", ""),
            })
    snapshot["climate"] = zones


def extract_lights(snapshot, states):
    """Extract light state summary."""
    on = off = unavail = 0
    total_brightness = 0
    for s in states:
        if s["entity_id"].startswith("light."):
            state = s.get("state", "")
            if state == "on":
                on += 1
                total_brightness += s.get("attributes", {}).get("brightness", 0) or 0
            elif state == "off":
                off += 1
            elif state == "unavailable":
                unavail += 1
    snapshot["lights"]["on"] = on
    snapshot["lights"]["off"] = off
    snapshot["lights"]["unavailable"] = unavail
    snapshot["lights"]["total_brightness"] = total_brightness


def extract_locks(snapshot, states):
    """Extract lock states and battery levels."""
    locks = []
    for s in states:
        if s["entity_id"].startswith("lock."):
            attrs = s.get("attributes", {})
            name = attrs.get("friendly_name", s["entity_id"])
            locks.append({
                "name": name,
                "state": s.get("state", "unknown"),
                "battery": attrs.get("battery_level"),
            })
    snapshot["locks"] = locks


def extract_automations(snapshot, states):
    """Extract automation summary."""
    on = off = unavail = 0
    for s in states:
        if s["entity_id"].startswith("automation."):
            state = s.get("state", "")
            if state == "on":
                on += 1
            elif state == "off":
                off += 1
            elif state == "unavailable":
                unavail += 1
    snapshot["automations"]["on"] = on
    snapshot["automations"]["off"] = off
    snapshot["automations"]["unavailable"] = unavail


def extract_motion(snapshot, states):
    """Extract motion sensor data."""
    sensors = {}
    for s in states:
        if s["entity_id"].startswith("binary_sensor."):
            dc = s.get("attributes", {}).get("device_class", "")
            if dc == "motion":
                name = s.get("attributes", {}).get("friendly_name", s["entity_id"])
                sensors[name] = s.get("state", "off")
    snapshot["motion"]["sensors"] = sensors


def extract_ev(snapshot, states):
    """Extract Tesla/EV data. Maps luda_* entities to TARS."""
    ev_data = {}
    for s in states:
        eid = s["entity_id"]
        state_val = s.get("state", "")
        attrs = s.get("attributes", {})
        # TARS (luda_ prefix in HA)
        if "luda_battery" in eid and attrs.get("unit_of_measurement") == "%":
            ev_data.setdefault("TARS", {})["battery_pct"] = _safe_float(state_val)
        elif "luda_charger_power" in eid:
            ev_data.setdefault("TARS", {})["charger_power_kw"] = _safe_float(state_val)
        elif "luda_range" in eid and "mi" in str(attrs.get("unit_of_measurement", "")):
            ev_data.setdefault("TARS", {})["range_miles"] = _safe_float(state_val)
        elif "luda_charging_rate" in eid:
            ev_data.setdefault("TARS", {})["charging_rate_mph"] = _safe_float(state_val)
        elif "luda_energy_added" in eid:
            ev_data.setdefault("TARS", {})["energy_added_kwh"] = _safe_float(state_val)
    snapshot["ev"] = ev_data


def extract_entities_summary(snapshot, states):
    """Extract high-level entity counts."""
    total = len(states)
    unavail = 0
    by_domain = {}
    for s in states:
        eid = s["entity_id"]
        domain = eid.split(".")[0] if "." in eid else "unknown"
        by_domain[domain] = by_domain.get(domain, 0) + 1
        if s.get("state") == "unavailable" and domain not in UNAVAILABLE_EXCLUDE_DOMAINS:
            unavail += 1
    snapshot["entities"]["total"] = total
    snapshot["entities"]["unavailable"] = unavail
    snapshot["entities"]["by_domain"] = by_domain


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
```

**Step 4: Run tests to verify they pass**

Run: `python3 ~/Documents/tests/test_ha_intelligence.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ~/.local/bin/ha-intelligence ~/Documents/tests/test_ha_intelligence.py
git commit -m "feat(ha-intelligence): entity extraction for power, occupancy, climate, lights, locks, motion, EV"
```

---

## Task 3: Weather + Calendar + Logbook Collection

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

**Step 1: Write failing test for weather/calendar/logbook**

```python
class TestExternalData(unittest.TestCase):
    def test_parse_weather(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()
        raw = "Partly cloudy +62°F 70% →11mph"
        result = ha.parse_weather(raw)
        self.assertEqual(result["temp_f"], 62)
        self.assertEqual(result["humidity_pct"], 70)
        self.assertIn("cloudy", result["condition"].lower())

    def test_parse_logbook_summary(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()
        entries = [
            {"entity_id": "sensor.time", "when": "2026-02-10T01:02:00+00:00"},
            {"entity_id": "sensor.time", "when": "2026-02-10T01:03:00+00:00"},
            {"entity_id": "light.atrium", "when": "2026-02-10T18:30:00+00:00"},
            {"entity_id": "lock.front_door", "when": "2026-02-10T18:35:00+00:00"},
        ]
        result = ha.summarize_logbook(entries)
        self.assertEqual(result["total_events"], 4)
        # Exclude date_time sensors from useful count
        self.assertEqual(result["useful_events"], 2)
        self.assertEqual(result["by_domain"]["sensor"], 2)
        self.assertEqual(result["by_domain"]["light"], 1)
        self.assertIn("18", result["hourly"])
```

**Step 2: Run to verify fails**

Run: `python3 ~/Documents/tests/test_ha_intelligence.py -v`

**Step 3: Implement**

```python
# Clock sensors to exclude from "useful events" count
CLOCK_SENSORS = {
    "sensor.date_time_utc", "sensor.date_time_iso", "sensor.time_date",
    "sensor.time_utc", "sensor.time", "sensor.date_time",
}


def fetch_weather():
    """Fetch weather from wttr.in, return raw string."""
    try:
        url = f"https://wttr.in/{WEATHER_LOCATION}?format=%C+%t+%h+%w"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def parse_weather(raw):
    """Parse wttr.in compact format into structured dict."""
    result = {"raw": raw, "condition": "", "temp_f": None, "humidity_pct": None, "wind_mph": None}
    if not raw:
        return result
    # Temperature: +62°F or -5°F
    m = re.search(r'([+-]?\d+)\s*°F', raw)
    if m:
        result["temp_f"] = int(m.group(1))
    # Humidity: 70%
    m = re.search(r'(\d+)%', raw)
    if m:
        result["humidity_pct"] = int(m.group(1))
    # Wind: →11mph or ↑5mph
    m = re.search(r'[→←↑↓↗↘↙↖]?\s*(\d+)\s*mph', raw)
    if m:
        result["wind_mph"] = int(m.group(1))
    # Condition: everything before the temperature
    m = re.match(r'^(.+?)\s*[+-]?\d+°', raw)
    if m:
        result["condition"] = m.group(1).strip()
    return result


def fetch_calendar_events():
    """Fetch today's calendar events via gog CLI."""
    try:
        result = subprocess.run(
            ["gog", "calendar", "list", "--today", "--all", "--plain"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:
            return []
        events = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 4:
                events.append({"start": parts[1], "end": parts[2], "summary": parts[3]})
            elif len(parts) >= 5:
                events.append({"start": parts[2], "end": parts[3], "summary": parts[4]})
        return events
    except Exception:
        return []


def summarize_logbook(entries):
    """Summarize logbook entries into counts by domain and hour."""
    total = len(entries)
    useful = 0
    by_domain = {}
    hourly = {}
    for e in entries:
        eid = e.get("entity_id", "")
        domain = eid.split(".")[0] if "." in eid else e.get("domain", "unknown")
        by_domain[domain] = by_domain.get(domain, 0) + 1
        # Count useful events (exclude clock sensors)
        if eid not in CLOCK_SENSORS:
            useful += 1
        # Hourly
        when = e.get("when", "")
        if len(when) >= 13:
            h = when[11:13]
            hourly[h] = hourly.get(h, 0) + 1
    return {
        "total_events": total,
        "useful_events": useful,
        "by_domain": by_domain,
        "hourly": hourly,
    }
```

**Step 4: Run tests**

Run: `python3 ~/Documents/tests/test_ha_intelligence.py -v`

**Step 5: Commit**

```bash
git add ~/.local/bin/ha-intelligence ~/Documents/tests/test_ha_intelligence.py
git commit -m "feat(ha-intelligence): weather parsing, calendar fetch, logbook summarization"
```

---

## Task 4: Full Snapshot Assembly + CLI

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Wire all extraction functions together. The `--snapshot` command fetches live data and saves to `~/ha-logs/intelligence/daily/`.

**Step 1: Write failing test for full snapshot assembly**

```python
class TestSnapshotAssembly(unittest.TestCase):
    @patch("ha_intelligence.fetch_ha_states")
    @patch("ha_intelligence.fetch_weather")
    @patch("ha_intelligence.fetch_calendar_events")
    @patch("ha_intelligence.load_logbook")
    def test_build_snapshot_assembles_all_sections(self, mock_log, mock_cal, mock_weather, mock_states):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        mock_states.return_value = TestEntityExtraction.SAMPLE_STATES
        mock_weather.return_value = "Clear +75°F 50% →5mph"
        mock_cal.return_value = [{"start": "2026-02-10T09:00:00", "end": "2026-02-10T10:00:00", "summary": "Meeting"}]
        mock_log.return_value = [{"entity_id": "light.atrium", "when": "2026-02-10T18:00:00+00:00"}]

        snapshot = ha.build_snapshot("2026-02-10")

        # Verify all sections populated
        self.assertEqual(snapshot["weather"]["temp_f"], 75)
        self.assertEqual(len(snapshot["calendar_events"]), 1)
        self.assertGreater(snapshot["entities"]["total"], 0)
        self.assertGreater(snapshot["power"]["total_watts"], 0)
        self.assertGreater(len(snapshot["occupancy"]["people_home"]), 0)
```

**Step 2: Run to fail**

**Step 3: Implement**

```python
def fetch_ha_states():
    """Fetch all entity states from HA REST API."""
    if not HA_TOKEN:
        return []
    try:
        req = urllib.request.Request(
            f"{HA_URL}/api/states",
            headers={"Authorization": f"Bearer {HA_TOKEN}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception:
        return []


def load_logbook():
    """Load current logbook from synced JSON file."""
    if not os.path.isfile(LOGBOOK_PATH):
        return []
    try:
        with open(LOGBOOK_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def build_snapshot(date_str=None):
    """Build a complete daily snapshot from all sources."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    snapshot = build_empty_snapshot(date_str)

    # HA entities
    states = fetch_ha_states()
    if states:
        extract_entities_summary(snapshot, states)
        extract_power(snapshot, states)
        extract_occupancy(snapshot, states)
        extract_climate(snapshot, states)
        extract_lights(snapshot, states)
        extract_locks(snapshot, states)
        extract_motion(snapshot, states)
        extract_automations(snapshot, states)
        extract_ev(snapshot, states)

    # Weather
    weather_raw = fetch_weather()
    snapshot["weather"] = parse_weather(weather_raw)

    # Calendar
    snapshot["calendar_events"] = fetch_calendar_events()

    # Logbook
    entries = load_logbook()
    snapshot["logbook_summary"] = summarize_logbook(entries)

    return snapshot


def save_snapshot(snapshot):
    """Save snapshot to daily directory."""
    ensure_dirs()
    path = os.path.join(DAILY_DIR, f"{snapshot['date']}.json")
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return path


def load_snapshot(date_str):
    """Load a previously saved snapshot."""
    path = os.path.join(DAILY_DIR, f"{date_str}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_recent_snapshots(days=30):
    """Load up to N days of recent snapshots."""
    snapshots = []
    today = datetime.now()
    for i in range(days):
        date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        snap = load_snapshot(date_str)
        if snap:
            snapshots.append(snap)
    return snapshots
```

**Step 4: Run tests**

**Step 5: Commit**

```bash
git commit -m "feat(ha-intelligence): full snapshot assembly with HA, weather, calendar, logbook"
```

---

## Task 5: Baseline Builder

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Compute per-day-of-week baselines from accumulated daily snapshots. After 7+ days of data, we know what "normal Tuesday" looks like.

**Step 1: Write failing test**

```python
class TestBaselines(unittest.TestCase):
    def _make_snapshot(self, date_str, power=150, lights_on=30, devices_home=50):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()
        snap = ha.build_empty_snapshot(date_str)
        snap["power"]["total_watts"] = power
        snap["lights"]["on"] = lights_on
        snap["occupancy"]["device_count_home"] = devices_home
        return snap

    def test_compute_baselines_groups_by_day_of_week(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        snapshots = [
            self._make_snapshot("2026-02-03", power=100),  # Tuesday
            self._make_snapshot("2026-02-10", power=200),  # Tuesday
            self._make_snapshot("2026-02-04", power=150),  # Wednesday
        ]
        baselines = ha.compute_baselines(snapshots)
        self.assertIn("Tuesday", baselines)
        self.assertAlmostEqual(baselines["Tuesday"]["power_watts"]["mean"], 150.0)
        self.assertIn("Wednesday", baselines)

    def test_baseline_includes_stddev(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        snapshots = [
            self._make_snapshot("2026-02-03", power=100),
            self._make_snapshot("2026-02-10", power=200),
        ]
        baselines = ha.compute_baselines(snapshots)
        # With 2 samples, stddev should be calculable
        self.assertGreater(baselines["Tuesday"]["power_watts"]["stddev"], 0)
```

**Step 2: Run to fail**

**Step 3: Implement**

```python
def compute_baselines(snapshots):
    """Compute per-day-of-week baselines from historical snapshots.

    Returns dict keyed by day name, each containing metric means and stddevs.
    """
    by_day = {}
    for snap in snapshots:
        dow = snap.get("day_of_week", "Unknown")
        by_day.setdefault(dow, []).append(snap)

    baselines = {}
    for dow, snaps in by_day.items():
        metrics = {
            "power_watts": [s["power"]["total_watts"] for s in snaps],
            "lights_on": [s["lights"]["on"] for s in snaps],
            "lights_off": [s["lights"]["off"] for s in snaps],
            "devices_home": [s["occupancy"]["device_count_home"] for s in snaps],
            "unavailable": [s["entities"]["unavailable"] for s in snaps],
            "useful_events": [s["logbook_summary"].get("useful_events", 0) for s in snaps],
        }

        baseline = {"sample_count": len(snaps)}
        for metric_name, values in metrics.items():
            if len(values) >= 2:
                baseline[metric_name] = {
                    "mean": statistics.mean(values),
                    "stddev": statistics.stdev(values),
                    "min": min(values),
                    "max": max(values),
                }
            elif len(values) == 1:
                baseline[metric_name] = {
                    "mean": values[0],
                    "stddev": 0,
                    "min": values[0],
                    "max": values[0],
                }
        baselines[dow] = baseline

    return baselines


def save_baselines(baselines):
    ensure_dirs()
    with open(BASELINES_PATH, "w") as f:
        json.dump(baselines, f, indent=2)


def load_baselines():
    if not os.path.isfile(BASELINES_PATH):
        return {}
    with open(BASELINES_PATH) as f:
        return json.load(f)
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git commit -m "feat(ha-intelligence): per-day-of-week baseline computation"
```

---

## Task 6: Anomaly Detection

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Z-score deviation from baselines. >2σ = anomaly.

**Step 1: Write failing test**

```python
class TestAnomalyDetection(unittest.TestCase):
    def test_detects_high_power_anomaly(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        baselines = {
            "Tuesday": {
                "power_watts": {"mean": 150, "stddev": 10},
                "lights_on": {"mean": 30, "stddev": 5},
                "devices_home": {"mean": 50, "stddev": 10},
                "unavailable": {"mean": 900, "stddev": 20},
                "useful_events": {"mean": 2500, "stddev": 300},
            }
        }
        snapshot = TestBaselines._make_snapshot(self, "2026-02-10", power=300)  # 15σ above
        snapshot["logbook_summary"] = {"useful_events": 2500}
        snapshot["entities"]["unavailable"] = 900

        anomalies = ha.detect_anomalies(snapshot, baselines)
        power_anomalies = [a for a in anomalies if a["metric"] == "power_watts"]
        self.assertTrue(len(power_anomalies) > 0)
        self.assertGreater(power_anomalies[0]["z_score"], 2.0)

    def test_no_anomaly_within_normal_range(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        baselines = {
            "Tuesday": {
                "power_watts": {"mean": 150, "stddev": 10},
                "lights_on": {"mean": 30, "stddev": 5},
                "devices_home": {"mean": 50, "stddev": 10},
                "unavailable": {"mean": 900, "stddev": 20},
                "useful_events": {"mean": 2500, "stddev": 300},
            }
        }
        snapshot = TestBaselines._make_snapshot(self, "2026-02-10", power=155)
        snapshot["logbook_summary"] = {"useful_events": 2500}
        snapshot["entities"]["unavailable"] = 900

        anomalies = ha.detect_anomalies(snapshot, baselines)
        power_anomalies = [a for a in anomalies if a["metric"] == "power_watts"]
        self.assertEqual(len(power_anomalies), 0)
```

**Step 2: Run to fail**

**Step 3: Implement**

```python
ANOMALY_THRESHOLD = 2.0  # z-score above which we flag anomaly


def detect_anomalies(snapshot, baselines):
    """Detect z-score anomalies vs day-of-week baseline."""
    dow = snapshot.get("day_of_week", "Unknown")
    baseline = baselines.get(dow, {})
    if not baseline:
        return []

    # Map snapshot values to baseline metric names
    current_values = {
        "power_watts": snapshot["power"]["total_watts"],
        "lights_on": snapshot["lights"]["on"],
        "devices_home": snapshot["occupancy"]["device_count_home"],
        "unavailable": snapshot["entities"]["unavailable"],
        "useful_events": snapshot["logbook_summary"].get("useful_events", 0),
    }

    anomalies = []
    for metric, current in current_values.items():
        bl = baseline.get(metric, {})
        mean = bl.get("mean")
        stddev = bl.get("stddev")
        if mean is None or stddev is None or stddev == 0:
            continue
        z = abs(current - mean) / stddev
        if z > ANOMALY_THRESHOLD:
            direction = "above" if current > mean else "below"
            anomalies.append({
                "metric": metric,
                "current": current,
                "mean": mean,
                "stddev": stddev,
                "z_score": round(z, 2),
                "direction": direction,
                "description": f"{metric} is {z:.1f}σ {direction} normal ({current} vs {mean:.0f}±{stddev:.0f})",
            })

    return anomalies
```

**Step 4: Run tests**

**Step 5: Commit**

```bash
git commit -m "feat(ha-intelligence): z-score anomaly detection against baselines"
```

---

## Task 7: Device Reliability Tracker

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Track unavailable/available transitions per device over time. Predict which devices are degrading.

**Step 1: Write failing test**

```python
class TestDeviceReliability(unittest.TestCase):
    def test_reliability_score_decreases_with_more_outages(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        # 7 days of snapshots, device appears unavailable on 3 of them
        snapshots = []
        unavail_days = {"2026-02-04", "2026-02-06", "2026-02-09"}
        for i in range(7):
            date = f"2026-02-{4+i:02d}"
            snap = ha.build_empty_snapshot(date)
            if date in unavail_days:
                snap["entities"]["unavailable_list"] = ["sensor.flaky_device"]
            else:
                snap["entities"]["unavailable_list"] = []
            snapshots.append(snap)

        scores = ha.compute_device_reliability(snapshots)
        self.assertIn("sensor.flaky_device", scores)
        self.assertLess(scores["sensor.flaky_device"]["score"], 100)
        self.assertEqual(scores["sensor.flaky_device"]["outage_days"], 3)

    def test_healthy_device_gets_100_score(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        snapshots = []
        for i in range(7):
            snap = ha.build_empty_snapshot(f"2026-02-{4+i:02d}")
            snap["entities"]["unavailable_list"] = []
            snapshots.append(snap)

        scores = ha.compute_device_reliability(snapshots)
        # No devices should appear (or all should be 100)
        for eid, data in scores.items():
            self.assertEqual(data["score"], 100)
```

**Step 2: Run to fail**

**Step 3: Implement**

```python
def extract_unavailable_list(states):
    """Get list of entity_ids that are unavailable."""
    return [
        s["entity_id"] for s in states
        if s.get("state") == "unavailable"
        and s["entity_id"].split(".")[0] not in UNAVAILABLE_EXCLUDE_DOMAINS
    ]


def compute_device_reliability(snapshots):
    """Compute reliability score per device from historical snapshots.

    Score = 100 * (days_available / total_days). Tracks trend direction.
    Only reports devices that were unavailable at least once.
    """
    device_outages = {}
    total_days = len(snapshots)
    if total_days == 0:
        return {}

    for snap in snapshots:
        unavail_list = snap.get("entities", {}).get("unavailable_list", [])
        for eid in unavail_list:
            device_outages.setdefault(eid, []).append(snap["date"])

    scores = {}
    for eid, outage_dates in device_outages.items():
        outage_days = len(outage_dates)
        score = round(100 * (1 - outage_days / total_days))
        # Trend: are outages increasing? Compare first half to second half
        mid = total_days // 2
        early_dates = set(s["date"] for s in snapshots[:mid])
        late_dates = set(s["date"] for s in snapshots[mid:])
        early_outages = len([d for d in outage_dates if d in early_dates])
        late_outages = len([d for d in outage_dates if d in late_dates])
        if late_outages > early_outages:
            trend = "degrading"
        elif late_outages < early_outages:
            trend = "improving"
        else:
            trend = "stable"

        scores[eid] = {
            "score": score,
            "outage_days": outage_days,
            "total_days": total_days,
            "trend": trend,
            "last_outage": max(outage_dates),
        }

    return scores
```

Note: `extract_unavailable_list` must be called during snapshot creation (Task 2) and stored in `snapshot["entities"]["unavailable_list"]`. Add this to `extract_entities_summary`:

```python
# Add at the end of extract_entities_summary:
snapshot["entities"]["unavailable_list"] = [
    s["entity_id"] for s in states
    if s.get("state") == "unavailable"
    and s["entity_id"].split(".")[0] not in UNAVAILABLE_EXCLUDE_DOMAINS
]
```

**Step 4: Run tests**

**Step 5: Commit**

```bash
git commit -m "feat(ha-intelligence): device reliability scoring with trend detection"
```

---

## Task 8: Cross-Correlation Engine

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Discover which metrics correlate with each other and with external factors (weather, calendar, day-of-week).

**Step 1: Write failing test**

```python
class TestCorrelation(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r = ha.pearson_r(x, y)
        self.assertAlmostEqual(r, 1.0, places=5)

    def test_no_correlation(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        x = [1, 2, 3, 4, 5]
        y = [5, 1, 4, 2, 3]
        r = ha.pearson_r(x, y)
        self.assertLess(abs(r), 0.5)

    def test_cross_correlate_finds_weather_power_link(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        # Simulate: hot days → more power (HVAC)
        snapshots = []
        for i, temp in enumerate([60, 70, 80, 90, 95, 65, 75, 85, 92, 88]):
            snap = ha.build_empty_snapshot(f"2026-02-{i+1:02d}")
            snap["weather"]["temp_f"] = temp
            snap["power"]["total_watts"] = 100 + (temp - 60) * 3  # Linear relationship
            snap["lights"]["on"] = 30
            snap["occupancy"]["device_count_home"] = 50
            snap["entities"]["unavailable"] = 900
            snap["logbook_summary"]["useful_events"] = 2500
            snapshots.append(snap)

        corrs = ha.cross_correlate(snapshots)
        # Should find strong correlation between temp and power
        temp_power = [c for c in corrs if c["x"] == "weather_temp" and c["y"] == "power_watts"]
        self.assertTrue(len(temp_power) > 0)
        self.assertGreater(temp_power[0]["r"], 0.9)
```

**Step 2: Run to fail**

**Step 3: Implement**

```python
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two sequences."""
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def cross_correlate(snapshots, min_r=0.5):
    """Find significant correlations between all tracked metrics.

    Returns list of {x, y, r, strength, description} dicts, filtered by |r| >= min_r.
    """
    if len(snapshots) < 5:
        return []

    # Extract time series
    series = {}
    for snap in snapshots:
        series.setdefault("power_watts", []).append(snap["power"]["total_watts"])
        series.setdefault("lights_on", []).append(snap["lights"]["on"])
        series.setdefault("devices_home", []).append(snap["occupancy"]["device_count_home"])
        series.setdefault("unavailable", []).append(snap["entities"]["unavailable"])
        series.setdefault("useful_events", []).append(snap["logbook_summary"].get("useful_events", 0))
        series.setdefault("weather_temp", []).append(snap.get("weather", {}).get("temp_f") or 0)
        series.setdefault("calendar_count", []).append(len(snap.get("calendar_events", [])))
        series.setdefault("is_weekend", []).append(1 if snap.get("is_weekend") else 0)
        # EV
        ev = snap.get("ev", {}).get("TARS", {})
        series.setdefault("ev_battery", []).append(ev.get("battery_pct", 0))
        series.setdefault("ev_power", []).append(ev.get("charger_power_kw", 0))

    # All pairs
    keys = list(series.keys())
    results = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            r = pearson_r(series[keys[i]], series[keys[j]])
            if abs(r) >= min_r:
                strength = "strong" if abs(r) >= 0.8 else "moderate"
                direction = "positive" if r > 0 else "negative"
                results.append({
                    "x": keys[i],
                    "y": keys[j],
                    "r": round(r, 3),
                    "strength": strength,
                    "direction": direction,
                    "description": f"{keys[i]} ↔ {keys[j]}: r={r:.2f} ({strength} {direction})",
                })

    results.sort(key=lambda c: -abs(c["r"]))
    return results


def save_correlations(correlations):
    ensure_dirs()
    with open(CORRELATIONS_PATH, "w") as f:
        json.dump(correlations, f, indent=2)
```

**Step 4: Run tests**

**Step 5: Commit**

```bash
git commit -m "feat(ha-intelligence): cross-correlation engine with Pearson-r"
```

---

## Task 9: Prediction Engine

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Generate predictions for tomorrow based on day-of-week baselines, weather forecast, and calendar.

**Step 1: Write failing test**

```python
class TestPredictions(unittest.TestCase):
    def test_predict_uses_baseline_mean(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        baselines = {
            "Wednesday": {
                "sample_count": 4,
                "power_watts": {"mean": 160, "stddev": 15},
                "lights_on": {"mean": 35, "stddev": 5},
                "devices_home": {"mean": 55, "stddev": 8},
                "unavailable": {"mean": 905, "stddev": 10},
                "useful_events": {"mean": 2400, "stddev": 200},
            }
        }
        predictions = ha.generate_predictions("2026-02-11", baselines, correlations=[], weather_forecast=None)
        self.assertEqual(predictions["target_date"], "2026-02-11")
        self.assertAlmostEqual(predictions["power_watts"]["predicted"], 160, delta=20)
        self.assertIn("confidence", predictions["power_watts"])

    def test_predict_adjusts_for_weather(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        baselines = {
            "Wednesday": {
                "sample_count": 4,
                "power_watts": {"mean": 160, "stddev": 15},
                "lights_on": {"mean": 35, "stddev": 5},
                "devices_home": {"mean": 55, "stddev": 8},
                "unavailable": {"mean": 905, "stddev": 10},
                "useful_events": {"mean": 2400, "stddev": 200},
            }
        }
        # Hot weather should increase power prediction (HVAC)
        hot_corrs = [{"x": "weather_temp", "y": "power_watts", "r": 0.85}]
        pred_hot = ha.generate_predictions("2026-02-11", baselines, hot_corrs, {"temp_f": 95})
        pred_normal = ha.generate_predictions("2026-02-11", baselines, [], None)

        # Hot day should predict more power
        self.assertGreater(pred_hot["power_watts"]["predicted"], pred_normal["power_watts"]["predicted"])
```

**Step 2: Run to fail**

**Step 3: Implement**

```python
def generate_predictions(target_date, baselines, correlations=None, weather_forecast=None):
    """Generate predictions for a target date.

    Uses day-of-week baseline as starting point, then adjusts based on
    correlations with weather and other factors.
    """
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    dow = dt.strftime("%A")
    baseline = baselines.get(dow, {})

    predictions = {
        "target_date": target_date,
        "day_of_week": dow,
        "generated_at": datetime.now().isoformat(),
    }

    metrics = ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]

    for metric in metrics:
        bl = baseline.get(metric, {})
        mean = bl.get("mean", 0)
        stddev = bl.get("stddev", 0)
        sample_count = baseline.get("sample_count", 0)

        predicted = mean
        adjustments = []

        # Weather-based adjustment
        if weather_forecast and correlations:
            temp = weather_forecast.get("temp_f")
            if temp is not None:
                for corr in correlations:
                    if corr["x"] == "weather_temp" and corr["y"] == metric:
                        # Adjust proportionally to correlation strength
                        # Assume normal temp is 72°F; deviation drives adjustment
                        temp_deviation = (temp - 72) / 30  # normalized
                        adjustment = predicted * temp_deviation * abs(corr["r"]) * 0.2
                        predicted += adjustment
                        adjustments.append(f"weather({temp}°F): {'+' if adjustment > 0 else ''}{adjustment:.0f}")

        # Confidence based on sample count and stddev
        if sample_count >= 7:
            confidence = "high"
        elif sample_count >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        predictions[metric] = {
            "predicted": round(predicted, 1),
            "baseline_mean": mean,
            "baseline_stddev": stddev,
            "confidence": confidence,
            "adjustments": adjustments,
        }

    return predictions


def save_predictions(predictions):
    ensure_dirs()
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(predictions, f, indent=2)


def load_predictions():
    if not os.path.isfile(PREDICTIONS_PATH):
        return {}
    with open(PREDICTIONS_PATH) as f:
        return json.load(f)
```

**Step 4: Run tests**

**Step 5: Commit**

```bash
git commit -m "feat(ha-intelligence): prediction engine with weather-adjusted forecasts"
```

---

## Task 10: Self-Reinforcement Loop

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Compare yesterday's predictions to actual data. Score accuracy. Adjust weights over time.

**Step 1: Write failing test**

```python
class TestSelfReinforcement(unittest.TestCase):
    def test_score_prediction_perfect(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        prediction = {"power_watts": {"predicted": 150, "baseline_mean": 150, "baseline_stddev": 10}}
        actual = {"power": {"total_watts": 150}}
        score = ha.score_prediction("power_watts", prediction, actual)
        self.assertEqual(score["accuracy"], 100)

    def test_score_prediction_off_by_one_sigma(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        prediction = {"power_watts": {"predicted": 150, "baseline_mean": 150, "baseline_stddev": 10}}
        actual = {"power": {"total_watts": 160}}  # 1σ off
        score = ha.score_prediction("power_watts", prediction, actual)
        # Should be less than 100 but still decent
        self.assertGreater(score["accuracy"], 50)
        self.assertLess(score["accuracy"], 100)

    def test_accuracy_history_tracks_trend(self):
        from importlib.machinery import SourceFileLoader
        ha = SourceFileLoader("ha_intelligence", os.path.expanduser("~/.local/bin/ha-intelligence")).load_module()

        history = {
            "scores": [
                {"date": "2026-02-05", "overall": 70},
                {"date": "2026-02-06", "overall": 75},
                {"date": "2026-02-07", "overall": 80},
                {"date": "2026-02-08", "overall": 82},
                {"date": "2026-02-09", "overall": 85},
            ]
        }
        trend = ha.accuracy_trend(history)
        self.assertEqual(trend, "improving")
```

**Step 2: Run to fail**

**Step 3: Implement**

```python
# Mapping from prediction metric to snapshot path
METRIC_TO_ACTUAL = {
    "power_watts": lambda s: s["power"]["total_watts"],
    "lights_on": lambda s: s["lights"]["on"],
    "devices_home": lambda s: s["occupancy"]["device_count_home"],
    "unavailable": lambda s: s["entities"]["unavailable"],
    "useful_events": lambda s: s["logbook_summary"].get("useful_events", 0),
}


def score_prediction(metric, predictions, actual_snapshot):
    """Score a single prediction against actual data.

    Returns {accuracy: 0-100, predicted, actual, error}.
    Accuracy = max(0, 100 - |error| / stddev * 25).
    A prediction within 1σ scores 75+, within 2σ scores 50+.
    """
    pred_data = predictions.get(metric, {})
    predicted = pred_data.get("predicted", 0)
    stddev = pred_data.get("baseline_stddev", 1) or 1

    actual_fn = METRIC_TO_ACTUAL.get(metric)
    if actual_fn is None:
        return {"accuracy": 0, "error": None}
    actual = actual_fn(actual_snapshot)

    error = abs(predicted - actual)
    sigma_error = error / stddev if stddev > 0 else error
    accuracy = max(0, round(100 - sigma_error * 25))

    return {
        "accuracy": accuracy,
        "predicted": predicted,
        "actual": actual,
        "error": round(error, 1),
        "sigma_error": round(sigma_error, 2),
    }


def score_all_predictions(predictions, actual_snapshot):
    """Score all predictions and return overall accuracy."""
    metrics = ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]
    scores = {}
    accuracies = []
    for metric in metrics:
        result = score_prediction(metric, predictions, actual_snapshot)
        scores[metric] = result
        if result["accuracy"] is not None:
            accuracies.append(result["accuracy"])

    overall = round(statistics.mean(accuracies)) if accuracies else 0
    return {
        "date": predictions.get("target_date", ""),
        "overall": overall,
        "metrics": scores,
    }


def accuracy_trend(history):
    """Determine if accuracy is improving, degrading, or stable."""
    scores = history.get("scores", [])
    if len(scores) < 3:
        return "insufficient_data"
    recent = [s["overall"] for s in scores[-3:]]
    earlier = [s["overall"] for s in scores[-6:-3]] if len(scores) >= 6 else [s["overall"] for s in scores[:3]]
    recent_avg = statistics.mean(recent)
    earlier_avg = statistics.mean(earlier)
    if recent_avg > earlier_avg + 3:
        return "improving"
    elif recent_avg < earlier_avg - 3:
        return "degrading"
    return "stable"


def update_accuracy_history(new_score):
    """Append score to accuracy history and save."""
    ensure_dirs()
    if os.path.isfile(ACCURACY_PATH):
        with open(ACCURACY_PATH) as f:
            history = json.load(f)
    else:
        history = {"scores": []}
    history["scores"].append(new_score)
    # Keep last 90 days
    history["scores"] = history["scores"][-90:]
    history["trend"] = accuracy_trend(history)
    with open(ACCURACY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    return history
```

**Step 4: Run tests**

**Step 5: Commit**

```bash
git commit -m "feat(ha-intelligence): self-reinforcement loop with accuracy tracking"
```

---

## Task 11: Ollama Insight Generation

**Files:**
- Modify: `~/.local/bin/ha-intelligence`

No unit test for this — it's LLM output. Test manually with `--report --dry-run`.

**Step 1: Implement the Ollama interpretation layer**

```python
def ollama_chat(prompt, timeout=60):
    """Send prompt to local Ollama and return response."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        return result.get("message", {}).get("content", "")
    except Exception:
        return ""


def generate_insight_report(snapshot, anomalies, predictions, reliability, correlations, accuracy_history):
    """Generate natural language insight report via Ollama."""
    # Build structured context for the LLM
    context = {
        "date": snapshot["date"],
        "day": snapshot["day_of_week"],
        "weather": snapshot.get("weather", {}),
        "power_watts": snapshot["power"]["total_watts"],
        "lights_on": snapshot["lights"]["on"],
        "people_home": snapshot["occupancy"]["people_home"],
        "devices_home": snapshot["occupancy"]["device_count_home"],
        "ev": snapshot.get("ev", {}),
        "anomalies": [a["description"] for a in anomalies],
        "predictions_tomorrow": {k: v for k, v in predictions.items() if isinstance(v, dict) and "predicted" in v},
        "degrading_devices": [eid for eid, data in reliability.items() if data.get("trend") == "degrading"],
        "top_correlations": [c["description"] for c in (correlations or [])[:5]],
        "accuracy_trend": accuracy_history.get("trend", "unknown"),
        "recent_accuracy": [s["overall"] for s in accuracy_history.get("scores", [])[-7:]],
    }

    prompt = f"""You are a home intelligence analyst. Analyze this smart home data and provide insights.

DATA:
{json.dumps(context, indent=2)}

Provide a concise report with these sections:
1. TODAY'S SUMMARY (2-3 sentences: what happened, any anomalies)
2. PREDICTIONS (what to expect tomorrow, with confidence)
3. DEVICE HEALTH (any degrading devices, recommended actions)
4. PATTERNS DISCOVERED (interesting correlations)
5. SELF-ASSESSMENT (how accurate have predictions been, what's improving)

Rules:
- Be specific: use actual numbers and device names
- If there are no anomalies, say so briefly
- Predictions should state confidence level
- Device health should suggest specific actions
- Keep total output under 300 words
"""
    return ollama_chat(prompt, timeout=90)


def generate_brief_line(snapshot, anomalies, predictions, accuracy_history):
    """Generate a single-line intelligence summary for telegram-brief."""
    parts = []
    # Anomaly count
    if anomalies:
        parts.append(f"{len(anomalies)} anomalies")
    else:
        parts.append("normal")
    # Accuracy
    scores = accuracy_history.get("scores", [])
    if scores:
        parts.append(f"accuracy:{scores[-1]['overall']}%")
    # Top prediction
    preds = {k: v for k, v in predictions.items() if isinstance(v, dict) and "predicted" in v}
    if preds.get("power_watts"):
        parts.append(f"tmrw power:{preds['power_watts']['predicted']:.0f}W")
    return f"Intelligence: {' | '.join(parts)}"
```

**Step 2: Test manually**

Run: `source ~/.env && ha-intelligence --report --dry-run`

**Step 3: Commit**

```bash
git commit -m "feat(ha-intelligence): Ollama insight report generation"
```

---

## Task 12: CLI Main + Cron Integration

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/.local/bin/telegram-brief` (add intelligence line)
- Modify crontab

**Step 1: Implement CLI main**

```python
def cmd_snapshot():
    """Collect and save today's snapshot."""
    snapshot = build_snapshot()
    path = save_snapshot(snapshot)
    print(f"Snapshot saved: {path} ({snapshot['entities']['total']} entities)")
    return snapshot


def cmd_analyze():
    """Run full analysis on latest data."""
    today = datetime.now().strftime("%Y-%m-%d")
    snapshot = load_snapshot(today) or build_snapshot()

    # Baselines
    recent = load_recent_snapshots(30)
    baselines = compute_baselines(recent)
    save_baselines(baselines)

    # Anomalies
    anomalies = detect_anomalies(snapshot, baselines)

    # Reliability
    reliability = compute_device_reliability(recent)

    # Correlations
    correlations = cross_correlate(recent)
    save_correlations(correlations)

    print(f"Analysis complete: {len(anomalies)} anomalies, {len(correlations)} correlations")
    if anomalies:
        for a in anomalies:
            print(f"  ! {a['description']}")
    return anomalies, correlations, reliability


def cmd_predict():
    """Generate predictions for tomorrow."""
    baselines = load_baselines()
    correlations = json.load(open(CORRELATIONS_PATH)) if os.path.isfile(CORRELATIONS_PATH) else []
    weather = parse_weather(fetch_weather())  # Current weather as proxy for tomorrow
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    predictions = generate_predictions(tomorrow, baselines, correlations, weather)
    save_predictions(predictions)
    print(f"Predictions for {tomorrow}:")
    for k, v in predictions.items():
        if isinstance(v, dict) and "predicted" in v:
            print(f"  {k}: {v['predicted']} ({v['confidence']} confidence)")
    return predictions


def cmd_score():
    """Score yesterday's predictions against actual data."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    predictions = load_predictions()
    actual = load_snapshot(yesterday)
    if not actual:
        print(f"No snapshot for {yesterday}, cannot score.")
        return None
    if predictions.get("target_date") != yesterday:
        print(f"No predictions for {yesterday}.")
        return None
    result = score_all_predictions(predictions, actual)
    history = update_accuracy_history(result)
    print(f"Accuracy for {yesterday}: {result['overall']}% (trend: {history['trend']})")
    for metric, data in result.get("metrics", {}).items():
        print(f"  {metric}: predicted={data['predicted']}, actual={data['actual']}, accuracy={data['accuracy']}%")
    return result


def cmd_report(dry_run=False):
    """Generate full Ollama insight report."""
    today = datetime.now().strftime("%Y-%m-%d")
    snapshot = load_snapshot(today)
    if not snapshot:
        snapshot = build_snapshot()
        save_snapshot(snapshot)

    recent = load_recent_snapshots(30)
    baselines = compute_baselines(recent)
    anomalies = detect_anomalies(snapshot, baselines)
    reliability = compute_device_reliability(recent)
    correlations = json.load(open(CORRELATIONS_PATH)) if os.path.isfile(CORRELATIONS_PATH) else []
    predictions = load_predictions()
    accuracy = json.load(open(ACCURACY_PATH)) if os.path.isfile(ACCURACY_PATH) else {"scores": []}

    report = generate_insight_report(snapshot, anomalies, predictions, reliability, correlations, accuracy)
    if dry_run:
        print(report)
    else:
        ensure_dirs()
        path = os.path.join(INSIGHTS_DIR, f"{today}.json")
        with open(path, "w") as f:
            json.dump({"date": today, "report": report}, f, indent=2)
        print(f"Report saved: {path}")
    return report


def cmd_brief():
    """Print one-liner for telegram-brief integration."""
    today = datetime.now().strftime("%Y-%m-%d")
    snapshot = load_snapshot(today) or build_snapshot()
    baselines = load_baselines()
    anomalies = detect_anomalies(snapshot, baselines)
    predictions = load_predictions()
    accuracy = json.load(open(ACCURACY_PATH)) if os.path.isfile(ACCURACY_PATH) else {"scores": []}
    print(generate_brief_line(snapshot, anomalies, predictions, accuracy))


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    ensure_dirs()

    if "--snapshot" in args:
        cmd_snapshot()
    elif "--analyze" in args:
        cmd_analyze()
    elif "--predict" in args:
        cmd_predict()
    elif "--score" in args:
        cmd_score()
    elif "--report" in args:
        cmd_report(dry_run=dry_run)
    elif "--brief" in args:
        cmd_brief()
    elif "--full" in args:
        # Full daily pipeline: snapshot → score yesterday → analyze → predict → report
        cmd_snapshot()
        cmd_score()
        cmd_analyze()
        cmd_predict()
        cmd_report(dry_run=dry_run)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
```

**Step 2: Make executable and test**

```bash
chmod +x ~/.local/bin/ha-intelligence
source ~/.env && ha-intelligence --snapshot
source ~/.env && ha-intelligence --analyze
source ~/.env && ha-intelligence --predict
source ~/.env && ha-intelligence --report --dry-run
```

**Step 3: Add cron jobs**

```bash
# Add to crontab:
# HA Intelligence: daily snapshot at 11pm, full pipeline at 11:30pm
0 23 * * * . /home/justin/.env && /home/justin/.local/bin/ha-intelligence --snapshot >> /home/justin/.local/log/ha-intelligence.log 2>&1
30 23 * * * . /home/justin/.env && /home/justin/.local/bin/ha-intelligence --full >> /home/justin/.local/log/ha-intelligence.log 2>&1
```

**Step 4: Add intelligence line to telegram-brief**

In `telegram-brief`, add to `section_ha_health()`:

```python
# Add intelligence one-liner if available
try:
    intel = subprocess.run(
        ["ha-intelligence", "--brief"],
        capture_output=True, text=True, timeout=30,
        env={**os.environ},
    )
    if intel.returncode == 0 and intel.stdout.strip():
        lines.append(f"  {intel.stdout.strip()}")
except Exception:
    pass
```

**Step 5: Commit**

```bash
git commit -m "feat(ha-intelligence): CLI, cron integration, telegram-brief intelligence line"
```

---

## Task 13: Claude Code Skills

**Files:**
- Create: `~/.claude/skills/ha-predict/SKILL.md`
- Create: `~/.claude/skills/ha-learn/SKILL.md`

**Step 1: Create prediction skill**

```markdown
# ~/.claude/skills/ha-predict/SKILL.md
---
name: ha-predict
description: View HA intelligence predictions, anomalies, and correlations
---

**Show Home Assistant intelligence predictions and analysis.**

## Execute: Run analysis

Run the intelligence pipeline now:

1. `source ~/.env && ha-intelligence --snapshot`
2. `source ~/.env && ha-intelligence --analyze`
3. `source ~/.env && ha-intelligence --predict`

## Execute: Read results

Read the prediction and analysis files:

1. `Read ~/ha-logs/intelligence/predictions.json`
2. `Read ~/ha-logs/intelligence/correlations.json`
3. `Read ~/ha-logs/intelligence/baselines.json`

## Execute: Generate insight report

`source ~/.env && ha-intelligence --report --dry-run`

## Present results

Summarize: what's predicted for tomorrow, any anomalies detected today, interesting correlations found, current prediction accuracy trend.
```

**Step 2: Create learning feedback skill**

```markdown
# ~/.claude/skills/ha-learn/SKILL.md
---
name: ha-learn
description: Review HA intelligence accuracy, score predictions, analyze learning trend
---

**Review how well the HA intelligence engine is learning.**

## Execute: Score yesterday

`source ~/.env && ha-intelligence --score`

## Execute: Read accuracy history

`Read ~/ha-logs/intelligence/accuracy.json`

## Execute: Read recent daily snapshots

`ls ~/ha-logs/intelligence/daily/ | tail -7`

Read the most recent snapshots to show trends.

## Present results

Show: accuracy score history, trend direction (improving/degrading/stable), which metrics are predicted well vs poorly, suggestions for improving prediction accuracy.
```

**Step 3: Commit**

```bash
git commit -m "feat(ha-intelligence): Claude Code skills for prediction and learning review"
```

---

## Task 14: Install holidays library

**Step 1: Install dependency**

```bash
pip3 install holidays
```

**Step 2: Verify**

```bash
python3 -c "import holidays; h = holidays.US(years=[2026]); print('2026-07-04' in h)"
```

Expected: `True`

**Step 3: Commit** (no code change, just dependency)

---

## Execution Pipeline Summary

```
Daily at 11:00 PM:
  ha-intelligence --snapshot     → Collect today's data

Daily at 11:30 PM:
  ha-intelligence --full         → Score yesterday → Analyze → Predict → Report

Every 15 minutes (existing):
  ha-log-sync                    → Keep logbook data fresh

Daily briefs (existing):
  telegram-brief                 → Includes HA health + intelligence one-liner
```

**Self-reinforcement cycle:**
```
Day 1-7:   Collect snapshots, build initial baselines
Day 8+:    Start predictions, no scoring yet (no prior predictions)
Day 9+:    Score Day 8 predictions, start accuracy tracking
Day 14+:   Enough data for meaningful correlations
Day 30+:   Full baseline (4+ samples per day-of-week), high confidence predictions
```

---

## What This Does NOT Do (Future Work)

- **Train ML models** — starts with statistics, graduates to sklearn if patterns warrant
- **Control devices** — read-only in this phase, no write actions
- **Real-time streaming** — snapshot-based (once daily), not event-driven
- **Multi-home support** — single HA instance only
- **Custom notification triggers** — predictions are passive, not action-generating
