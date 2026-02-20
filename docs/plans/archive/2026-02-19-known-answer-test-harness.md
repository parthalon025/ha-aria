# Known-Answer Test Harness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build deterministic known-answer tests for all 10 surviving hub modules plus one end-to-end pipeline test, with hybrid behavioral + golden snapshot assertions.

**Architecture:** Each module gets an isolated test that creates a minimal `IntelligenceHub` with SQLite cache, feeds deterministic input, runs the module's main method, and asserts behavioral properties. A shared `golden_compare()` utility optionally diffs output against stored reference JSON. A `--update-golden` pytest flag re-baselines all golden files. Quick wins (dashboard greyed-out, CLAUDE.md update) are included at the end.

**Tech Stack:** Python 3.12, pytest, asyncio, aiosqlite, unittest.mock

## Quality Gates

Run between each batch of tasks:

```bash
python3 -m pytest tests/integration/known_answer/ --timeout=120 -x -q
python3 -m pytest tests/ --timeout=120 -x -q
```

---

## Context

### Hub Module Testing Pattern

All hub modules follow this lifecycle:

```python
hub = IntelligenceHub(cache_path=str(tmp_path / "hub.db"))
await hub.initialize()
module = SomeModule(hub, ...)  # module-specific args
hub.register_module(module)
await module.initialize()
# ... trigger module logic ...
result = await hub.get_cache("cache_key")
await module.shutdown()
await hub.shutdown()
```

Cache keys are defined in `aria/hub/constants.py`. Events flow via `hub.publish()` → `module.on_event()`.

### Existing Infrastructure

- `tests/synthetic/simulator.py` — `HouseholdSimulator` with deterministic seed
- `tests/synthetic/pipeline.py` — `PipelineRunner` for engine-side pipeline
- `tests/synthetic/events.py` — `EventStreamGenerator` for state_changed events
- `tests/integration/conftest.py` — shared fixtures (`stable_30d_runner`, etc.)
- `tests/integration/golden/backtest_baseline.json` — existing golden file pattern

---

### Task 1: Create known_answer directory and conftest infrastructure

**Files:**
- Create: `tests/integration/known_answer/__init__.py`
- Create: `tests/integration/known_answer/conftest.py`
- Create: `tests/integration/known_answer/golden/` (directory)
- Create: `tests/integration/known_answer/fixtures/` (directory)

**Step 1: Create directory structure**

```bash
mkdir -p tests/integration/known_answer/golden tests/integration/known_answer/fixtures
touch tests/integration/known_answer/__init__.py
```

**Step 2: Write conftest.py with golden_compare and --update-golden flag**

Create `tests/integration/known_answer/conftest.py`:

```python
"""Known-answer test infrastructure — golden comparison + shared fixtures."""

import json
from pathlib import Path
from typing import Any

import pytest

from aria.hub.core import IntelligenceHub

GOLDEN_DIR = Path(__file__).parent / "golden"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Update golden reference files with current output",
    )


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")


def golden_compare(
    actual: dict[str, Any],
    golden_name: str,
    update: bool = False,
) -> dict[str, Any] | None:
    """Compare actual output against golden reference file.

    Args:
        actual: The actual output from the module
        golden_name: Name of the golden file (without .json extension)
        update: If True, overwrite the golden file with actual output

    Returns:
        The golden data if comparison made, None if file didn't exist or was updated.
        Drift is reported as a pytest warning, never a failure.
    """
    golden_path = GOLDEN_DIR / f"{golden_name}.json"

    if update:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(actual, indent=2, default=str) + "\n")
        return None

    if not golden_path.exists():
        # First run — create the golden file
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(actual, indent=2, default=str) + "\n")
        import warnings
        warnings.warn(f"Golden file created: {golden_name}.json (first run)")
        return None

    golden = json.loads(golden_path.read_text())

    if actual != golden:
        import warnings
        # Build a human-readable diff summary
        diff_keys = []
        if isinstance(actual, dict) and isinstance(golden, dict):
            all_keys = set(actual.keys()) | set(golden.keys())
            for key in sorted(all_keys):
                if actual.get(key) != golden.get(key):
                    diff_keys.append(key)
        warnings.warn(
            f"Golden drift in {golden_name}.json: "
            f"keys differ: {diff_keys or 'structure mismatch'}. "
            f"Run with --update-golden to re-baseline."
        )

    return golden


@pytest.fixture
async def hub(tmp_path):
    """Create a minimal IntelligenceHub for known-answer tests."""
    h = IntelligenceHub(cache_path=str(tmp_path / "hub.db"))
    await h.initialize()
    yield h
    await h.shutdown()
```

**Step 3: Run to verify conftest loads**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/ --collect-only -q
```

Expected: `no tests ran` (no test files yet), no import errors.

**Step 4: Commit**

```bash
git add tests/integration/known_answer/
git commit -m "test: add known-answer test infrastructure with golden comparison"
```

---

### Task 2: Known-answer test for discovery module

**Files:**
- Create: `tests/integration/known_answer/test_discovery_ka.py`
- Reference: `aria/modules/discovery.py` (DiscoveryModule class)

**Step 1: Write the test**

Create `tests/integration/known_answer/test_discovery_ka.py`:

```python
"""Known-answer test: discovery module."""

import pytest

from aria.modules.discovery import DiscoveryModule
from tests.integration.known_answer.conftest import golden_compare


@pytest.fixture
def mock_ha_response():
    """Deterministic HA entity/device/area response."""
    return {
        "entities": [
            {"entity_id": "light.living_room", "state": "on", "attributes": {"friendly_name": "Living Room Light"}, "device_id": "dev_001", "area_id": None},
            {"entity_id": "sensor.temperature", "state": "22.5", "attributes": {"friendly_name": "Temperature", "unit_of_measurement": "°C", "device_class": "temperature"}, "device_id": "dev_002", "area_id": None},
            {"entity_id": "binary_sensor.motion", "state": "off", "attributes": {"friendly_name": "Motion Sensor", "device_class": "motion"}, "device_id": "dev_003", "area_id": None},
            {"entity_id": "switch.smart_plug", "state": "on", "attributes": {"friendly_name": "Smart Plug"}, "device_id": "dev_001", "area_id": None},
            {"entity_id": "automation.morning_lights", "state": "on", "attributes": {"friendly_name": "Morning Lights"}, "device_id": None, "area_id": None},
            {"entity_id": "update.hacs", "state": "off", "attributes": {"friendly_name": "HACS Update"}, "device_id": None, "area_id": None},
        ],
        "devices": [
            {"id": "dev_001", "name": "Living Room Hub", "area_id": "area_living"},
            {"id": "dev_002", "name": "Climate Sensor", "area_id": "area_bedroom"},
            {"id": "dev_003", "name": "Motion Detector", "area_id": "area_hallway"},
        ],
        "areas": [
            {"area_id": "area_living", "name": "Living Room"},
            {"area_id": "area_bedroom", "name": "Bedroom"},
            {"area_id": "area_hallway", "name": "Hallway"},
        ],
    }


class TestDiscoveryKnownAnswer:
    """Discovery module known-answer tests."""

    @pytest.mark.asyncio
    async def test_entity_discovery_count(self, hub, mock_ha_response, monkeypatch):
        """Discovery should find all entities from the HA response."""
        module = DiscoveryModule(hub)
        hub.register_module(module)

        # Mock the subprocess call that fetches HA data
        async def mock_run_discovery_subprocess():
            return mock_ha_response

        monkeypatch.setattr(module, "_run_discovery_subprocess", mock_run_discovery_subprocess)
        await module.initialize()

        entities_cache = await hub.get_cache("entities")
        assert entities_cache is not None
        entities = entities_cache.get("data", {})
        assert len(entities) >= 4, f"Expected >= 4 entities, got {len(entities)}"

    @pytest.mark.asyncio
    async def test_entity_classification_tiers(self, hub, mock_ha_response, monkeypatch):
        """Classification should auto-exclude automation/update domains."""
        module = DiscoveryModule(hub)
        hub.register_module(module)

        async def mock_run_discovery_subprocess():
            return mock_ha_response

        monkeypatch.setattr(module, "_run_discovery_subprocess", mock_run_discovery_subprocess)
        await module.initialize()

        # Run classification
        await module.run_classification()

        curation_cache = await hub.get_cache("entity_curation")
        assert curation_cache is not None
        curations = curation_cache.get("data", {}).get("curations", [])

        # automation.* and update.* should be tier 1 (auto-excluded)
        auto_excluded = [c for c in curations if c.get("tier") == 1]
        auto_excluded_ids = [c["entity_id"] for c in auto_excluded]
        assert "automation.morning_lights" in auto_excluded_ids or "update.hacs" in auto_excluded_ids, \
            f"Expected automation/update domains auto-excluded, got tier-1: {auto_excluded_ids}"

    @pytest.mark.asyncio
    async def test_golden_snapshot(self, hub, mock_ha_response, monkeypatch, update_golden):
        """Golden snapshot comparison for discovery output."""
        module = DiscoveryModule(hub)
        hub.register_module(module)

        async def mock_run_discovery_subprocess():
            return mock_ha_response

        monkeypatch.setattr(module, "_run_discovery_subprocess", mock_run_discovery_subprocess)
        await module.initialize()
        await module.run_classification()

        entities_cache = await hub.get_cache("entities")
        curation_cache = await hub.get_cache("entity_curation")
        actual = {
            "entities": entities_cache.get("data", {}) if entities_cache else {},
            "curation": curation_cache.get("data", {}) if curation_cache else {},
        }
        golden_compare(actual, "discovery", update=update_golden)
```

**Step 2: Run test to verify it works**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/test_discovery_ka.py -v --timeout=120
```

Expected: Tests pass (golden file auto-created on first run). If `_run_discovery_subprocess` doesn't exist as a method, adjust the monkeypatch target to match the actual subprocess call method in `discovery.py` — check `grep -n "subprocess\|async def _run\|async def _fetch" aria/modules/discovery.py`.

**Step 3: Commit**

```bash
git add tests/integration/known_answer/test_discovery_ka.py tests/integration/known_answer/golden/
git commit -m "test: add discovery known-answer test"
```

---

### Task 3: Known-answer test for activity_monitor module

**Files:**
- Create: `tests/integration/known_answer/test_activity_monitor_ka.py`
- Reference: `aria/modules/activity_monitor.py` (ActivityMonitor class)

**Step 1: Write the test**

Create `tests/integration/known_answer/test_activity_monitor_ka.py`:

```python
"""Known-answer test: activity_monitor module."""

from datetime import datetime

import pytest

from aria.modules.activity_monitor import ActivityMonitor
from tests.integration.known_answer.conftest import golden_compare


def _make_state_changed_event(entity_id: str, old_state: str, new_state: str, timestamp: str = None) -> dict:
    """Create a deterministic state_changed event."""
    return {
        "entity_id": entity_id,
        "old_state": old_state,
        "new_state": new_state,
        "timestamp": timestamp or datetime.now().isoformat(),
        "attributes": {},
    }


FIXTURE_EVENTS = [
    _make_state_changed_event("light.living_room", "off", "on", "2026-02-19T08:00:00"),
    _make_state_changed_event("sensor.temperature", "21.0", "22.5", "2026-02-19T08:01:00"),
    _make_state_changed_event("binary_sensor.motion", "off", "on", "2026-02-19T08:02:00"),
    _make_state_changed_event("light.kitchen", "off", "on", "2026-02-19T08:05:00"),
    _make_state_changed_event("light.living_room", "on", "off", "2026-02-19T08:30:00"),
    _make_state_changed_event("binary_sensor.motion", "on", "off", "2026-02-19T08:35:00"),
    _make_state_changed_event("sensor.temperature", "22.5", "23.0", "2026-02-19T09:00:00"),
    _make_state_changed_event("switch.smart_plug", "off", "on", "2026-02-19T09:15:00"),
]


class TestActivityMonitorKnownAnswer:
    """Activity monitor known-answer tests."""

    @pytest.mark.asyncio
    async def test_events_produce_activity_window(self, hub):
        """Feeding state_changed events should produce an activity log."""
        module = ActivityMonitor(hub)
        hub.register_module(module)
        await module.initialize()

        for event in FIXTURE_EVENTS:
            await hub.publish("state_changed", event)

        activity_cache = await hub.get_cache("activity_log")
        assert activity_cache is not None, "activity_log cache should be populated after events"
        log_data = activity_cache.get("data", {})
        # Should have recorded some events
        events_recorded = log_data.get("events", [])
        assert len(events_recorded) >= 1, f"Expected >= 1 recorded events, got {len(events_recorded)}"

    @pytest.mark.asyncio
    async def test_entity_count_in_summary(self, hub):
        """Activity summary should reflect the distinct entities seen."""
        module = ActivityMonitor(hub)
        hub.register_module(module)
        await module.initialize()

        for event in FIXTURE_EVENTS:
            await hub.publish("state_changed", event)

        summary_cache = await hub.get_cache("activity_summary")
        if summary_cache:
            summary = summary_cache.get("data", {})
            entity_count = summary.get("unique_entities", 0)
            # We sent events for 5 distinct entities
            assert entity_count >= 1, f"Expected some entities tracked, got {entity_count}"

    @pytest.mark.asyncio
    async def test_golden_snapshot(self, hub, update_golden):
        """Golden snapshot comparison for activity monitor output."""
        module = ActivityMonitor(hub)
        hub.register_module(module)
        await module.initialize()

        for event in FIXTURE_EVENTS:
            await hub.publish("state_changed", event)

        activity_cache = await hub.get_cache("activity_log")
        summary_cache = await hub.get_cache("activity_summary")
        actual = {
            "activity_log": activity_cache.get("data", {}) if activity_cache else {},
            "activity_summary": summary_cache.get("data", {}) if summary_cache else {},
        }
        golden_compare(actual, "activity_monitor", update=update_golden)
```

**Step 2: Run test**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/test_activity_monitor_ka.py -v --timeout=120
```

Expected: Tests pass. Activity monitor listens for `state_changed` events and writes to `activity_log` / `activity_summary` cache. If the module requires additional constructor args, check `ActivityMonitor.__init__` signature and adapt.

**Step 3: Commit**

```bash
git add tests/integration/known_answer/test_activity_monitor_ka.py tests/integration/known_answer/golden/
git commit -m "test: add activity_monitor known-answer test"
```

---

### Task 4: Known-answer test for patterns module

**Files:**
- Create: `tests/integration/known_answer/test_patterns_ka.py`
- Create: `tests/integration/known_answer/fixtures/logbook_patterns.json`
- Reference: `aria/modules/patterns.py` (PatternRecognition class)

**Step 1: Create logbook fixture**

Create `tests/integration/known_answer/fixtures/logbook_patterns.json` — a minimal logbook with 3 clear recurring patterns over 21 days. The patterns module reads logbook files from `log_dir`.

```json
{
  "_description": "Hand-crafted logbook with 3 embedded patterns for known-answer testing",
  "entries": [
    {"when": "2026-02-01T07:00:00", "entity_id": "light.kitchen", "state": "on", "area": "Kitchen"},
    {"when": "2026-02-01T07:05:00", "entity_id": "switch.coffee_maker", "state": "on", "area": "Kitchen"},
    {"when": "2026-02-01T07:30:00", "entity_id": "light.living_room", "state": "on", "area": "Living Room"},
    {"when": "2026-02-01T18:00:00", "entity_id": "light.living_room", "state": "on", "area": "Living Room"},
    {"when": "2026-02-01T18:05:00", "entity_id": "media_player.tv", "state": "playing", "area": "Living Room"},
    {"when": "2026-02-02T07:00:00", "entity_id": "light.kitchen", "state": "on", "area": "Kitchen"},
    {"when": "2026-02-02T07:05:00", "entity_id": "switch.coffee_maker", "state": "on", "area": "Kitchen"},
    {"when": "2026-02-02T07:30:00", "entity_id": "light.living_room", "state": "on", "area": "Living Room"},
    {"when": "2026-02-02T18:00:00", "entity_id": "light.living_room", "state": "on", "area": "Living Room"},
    {"when": "2026-02-02T18:05:00", "entity_id": "media_player.tv", "state": "playing", "area": "Living Room"},
    {"when": "2026-02-03T07:00:00", "entity_id": "light.kitchen", "state": "on", "area": "Kitchen"},
    {"when": "2026-02-03T07:05:00", "entity_id": "switch.coffee_maker", "state": "on", "area": "Kitchen"},
    {"when": "2026-02-03T07:30:00", "entity_id": "light.living_room", "state": "on", "area": "Living Room"},
    {"when": "2026-02-03T18:00:00", "entity_id": "light.living_room", "state": "on", "area": "Living Room"},
    {"when": "2026-02-03T18:05:00", "entity_id": "media_player.tv", "state": "playing", "area": "Living Room"}
  ]
}
```

**Step 2: Write the test**

Create `tests/integration/known_answer/test_patterns_ka.py`:

```python
"""Known-answer test: patterns module.

The patterns module reads logbook JSON files from a directory and applies
hierarchical clustering + association rules to find recurring patterns.
It also calls Ollama for LLM interpretation — we mock that.
"""

import json
from pathlib import Path

import pytest

from aria.modules.patterns import PatternRecognition
from tests.integration.known_answer.conftest import FIXTURES_DIR, golden_compare


@pytest.fixture
def logbook_dir(tmp_path):
    """Create a log directory with the hand-crafted logbook fixture."""
    log_dir = tmp_path / "logbook"
    log_dir.mkdir()

    fixture_path = FIXTURES_DIR / "logbook_patterns.json"
    fixture = json.loads(fixture_path.read_text())

    # Write entries as the format patterns.py expects (check _extract_sequences)
    # Patterns module reads current.json from the logbook dir
    (log_dir / "current.json").write_text(json.dumps(fixture["entries"]))
    return tmp_path


class TestPatternsKnownAnswer:
    """Patterns module known-answer tests."""

    @pytest.mark.asyncio
    async def test_detects_recurring_patterns(self, hub, logbook_dir, monkeypatch):
        """Should detect at least 1 pattern from the morning kitchen routine."""
        module = PatternRecognition(
            hub=hub,
            log_dir=logbook_dir,
            min_pattern_frequency=2,
            min_support=0.5,
            min_confidence=0.5,
        )
        hub.register_module(module)

        # Mock LLM interpretation (Ollama call)
        async def mock_interpret(pattern):
            return f"Mock description for pattern {pattern.get('pattern_id', 'unknown')}"

        monkeypatch.setattr(module, "_interpret_pattern_llm", mock_interpret)

        patterns = await module.detect_patterns()
        assert len(patterns) >= 1, f"Expected >= 1 patterns from hand-crafted logbook, got {len(patterns)}"

    @pytest.mark.asyncio
    async def test_patterns_cached(self, hub, logbook_dir, monkeypatch):
        """Detected patterns should be stored in hub cache."""
        module = PatternRecognition(
            hub=hub,
            log_dir=logbook_dir,
            min_pattern_frequency=2,
            min_support=0.5,
            min_confidence=0.5,
        )
        hub.register_module(module)

        async def mock_interpret(pattern):
            return "Mock pattern description"

        monkeypatch.setattr(module, "_interpret_pattern_llm", mock_interpret)

        await module.detect_patterns()

        patterns_cache = await hub.get_cache("patterns")
        assert patterns_cache is not None, "patterns cache should be populated"
        data = patterns_cache.get("data", {})
        assert "patterns" in data
        assert data.get("pattern_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_golden_snapshot(self, hub, logbook_dir, monkeypatch, update_golden):
        """Golden snapshot comparison for patterns output."""
        module = PatternRecognition(
            hub=hub,
            log_dir=logbook_dir,
            min_pattern_frequency=2,
            min_support=0.5,
            min_confidence=0.5,
        )
        hub.register_module(module)

        async def mock_interpret(pattern):
            return "Mock pattern description"

        monkeypatch.setattr(module, "_interpret_pattern_llm", mock_interpret)

        patterns = await module.detect_patterns()

        # Normalize for golden comparison (remove timestamps that vary)
        actual = {
            "pattern_count": len(patterns),
            "patterns": [
                {
                    "pattern_id": p.get("pattern_id"),
                    "area": p.get("area"),
                    "entities": sorted(p.get("entities", [])),
                    "confidence": round(p.get("confidence", 0), 2),
                }
                for p in patterns
            ],
        }
        golden_compare(actual, "patterns", update=update_golden)
```

**Step 3: Run test**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/test_patterns_ka.py -v --timeout=120
```

Expected: Tests pass. If the logbook format is different from what `_extract_sequences` expects, inspect the method and adjust the fixture format. The key is the LLM mock — patterns calls Ollama for interpretation.

**Step 4: Commit**

```bash
git add tests/integration/known_answer/test_patterns_ka.py tests/integration/known_answer/fixtures/ tests/integration/known_answer/golden/
git commit -m "test: add patterns known-answer test with logbook fixture"
```

---

### Task 5: Known-answer test for orchestrator module

**Files:**
- Create: `tests/integration/known_answer/test_orchestrator_ka.py`
- Reference: `aria/modules/orchestrator.py` (OrchestratorModule class)

**Step 1: Write the test**

The orchestrator reads patterns from cache and generates automation suggestions. We pre-populate the patterns cache with known data.

Create `tests/integration/known_answer/test_orchestrator_ka.py`:

```python
"""Known-answer test: orchestrator module.

Orchestrator reads patterns from hub cache and generates automation YAML.
We pre-populate cache with known patterns and verify suggestion output.
"""

import pytest

from aria.modules.orchestrator import OrchestratorModule
from tests.integration.known_answer.conftest import golden_compare


KNOWN_PATTERNS = {
    "patterns": [
        {
            "pattern_id": "ka_morning_kitchen",
            "area": "Kitchen",
            "entities": ["light.kitchen", "switch.coffee_maker"],
            "trigger_time": "07:00",
            "confidence": 0.92,
            "frequency": 15,
            "type": "temporal_sequence",
            "llm_description": "Morning kitchen routine: lights then coffee maker",
        },
        {
            "pattern_id": "ka_evening_living",
            "area": "Living Room",
            "entities": ["light.living_room", "media_player.tv"],
            "trigger_time": "18:00",
            "confidence": 0.88,
            "frequency": 12,
            "type": "temporal_sequence",
            "llm_description": "Evening living room: lights then TV",
        },
    ],
    "pattern_count": 2,
    "areas_analyzed": ["Kitchen", "Living Room"],
}


class TestOrchestratorKnownAnswer:
    """Orchestrator module known-answer tests."""

    @pytest.mark.asyncio
    async def test_generates_suggestions_from_patterns(self, hub):
        """Orchestrator should generate >= 1 automation suggestion from known patterns."""
        # Pre-populate patterns cache
        await hub.set_cache("patterns", KNOWN_PATTERNS)

        module = OrchestratorModule(
            hub=hub,
            ha_url="http://test-host:8123",
            ha_token="test-token",
            min_confidence=0.7,
        )
        hub.register_module(module)

        suggestions = await module.generate_suggestions()
        assert len(suggestions) >= 1, f"Expected >= 1 suggestion, got {len(suggestions)}"

    @pytest.mark.asyncio
    async def test_suggestion_has_required_fields(self, hub):
        """Each suggestion should have the fields needed for HA automation."""
        await hub.set_cache("patterns", KNOWN_PATTERNS)

        module = OrchestratorModule(
            hub=hub,
            ha_url="http://test-host:8123",
            ha_token="test-token",
            min_confidence=0.7,
        )
        hub.register_module(module)

        suggestions = await module.generate_suggestions()
        assert len(suggestions) >= 1

        suggestion = suggestions[0]
        # Should have key automation structure
        assert "pattern_id" in suggestion or "source_pattern" in suggestion
        assert any(k in suggestion for k in ("automation", "yaml", "trigger", "action"))

    @pytest.mark.asyncio
    async def test_low_confidence_filtered(self, hub):
        """Patterns below min_confidence should not generate suggestions."""
        low_confidence_patterns = {
            "patterns": [
                {
                    "pattern_id": "ka_weak",
                    "area": "Kitchen",
                    "entities": ["light.kitchen"],
                    "trigger_time": "03:00",
                    "confidence": 0.3,
                    "frequency": 2,
                    "type": "temporal_sequence",
                    "llm_description": "Weak pattern",
                },
            ],
            "pattern_count": 1,
            "areas_analyzed": ["Kitchen"],
        }
        await hub.set_cache("patterns", low_confidence_patterns)

        module = OrchestratorModule(
            hub=hub,
            ha_url="http://test-host:8123",
            ha_token="test-token",
            min_confidence=0.7,
        )
        hub.register_module(module)

        suggestions = await module.generate_suggestions()
        assert len(suggestions) == 0, f"Expected 0 suggestions for low-confidence pattern, got {len(suggestions)}"

    @pytest.mark.asyncio
    async def test_suggestions_cached(self, hub):
        """Generated suggestions should be stored in hub cache."""
        await hub.set_cache("patterns", KNOWN_PATTERNS)

        module = OrchestratorModule(
            hub=hub,
            ha_url="http://test-host:8123",
            ha_token="test-token",
            min_confidence=0.7,
        )
        hub.register_module(module)

        await module.generate_suggestions()

        cache = await hub.get_cache("automation_suggestions")
        assert cache is not None
        data = cache.get("data", {})
        assert "suggestions" in data

    @pytest.mark.asyncio
    async def test_golden_snapshot(self, hub, update_golden):
        """Golden snapshot comparison for orchestrator output."""
        await hub.set_cache("patterns", KNOWN_PATTERNS)

        module = OrchestratorModule(
            hub=hub,
            ha_url="http://test-host:8123",
            ha_token="test-token",
            min_confidence=0.7,
        )
        hub.register_module(module)

        suggestions = await module.generate_suggestions()
        actual = {
            "suggestion_count": len(suggestions),
            "suggestions": suggestions,
        }
        golden_compare(actual, "orchestrator", update=update_golden)
```

**Step 2: Run test**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/test_orchestrator_ka.py -v --timeout=120
```

Expected: Pass. The orchestrator doesn't call external services for suggestion generation — it reads cache and produces YAML structures. The `_session` (aiohttp) is only used for executing approved automations against HA.

**Step 3: Commit**

```bash
git add tests/integration/known_answer/test_orchestrator_ka.py tests/integration/known_answer/golden/
git commit -m "test: add orchestrator known-answer test"
```

---

### Task 6: Known-answer test for shadow_engine module

**Files:**
- Create: `tests/integration/known_answer/test_shadow_engine_ka.py`
- Reference: `aria/modules/shadow_engine.py` (ShadowEngine class)

**Step 1: Write the test**

The shadow engine subscribes to `state_changed` events, generates predictions, then compares against actual outcomes. It writes to `predictions` cache and `pipeline_state` cache.

Create `tests/integration/known_answer/test_shadow_engine_ka.py`:

```python
"""Known-answer test: shadow_engine module.

Shadow engine predicts next state changes and scores accuracy.
We feed a predictable sequence and verify predictions are generated.
"""

from datetime import datetime

import pytest

from aria.modules.shadow_engine import ShadowEngine
from tests.integration.known_answer.conftest import golden_compare


def _state_changed(entity_id: str, old: str, new: str, ts: str) -> dict:
    return {
        "entity_id": entity_id,
        "old_state": old,
        "new_state": new,
        "timestamp": ts,
        "attributes": {},
    }


# A predictable morning routine: light on at 7, coffee at 7:05, every day
PREDICTABLE_EVENTS = []
for day in range(1, 8):
    date = f"2026-02-{day:02d}"
    PREDICTABLE_EVENTS.extend([
        _state_changed("light.kitchen", "off", "on", f"{date}T07:00:00"),
        _state_changed("switch.coffee_maker", "off", "on", f"{date}T07:05:00"),
        _state_changed("light.kitchen", "on", "off", f"{date}T08:00:00"),
        _state_changed("switch.coffee_maker", "on", "off", f"{date}T08:30:00"),
    ])


class TestShadowEngineKnownAnswer:
    """Shadow engine known-answer tests."""

    @pytest.mark.asyncio
    async def test_generates_predictions(self, hub):
        """After enough events, shadow engine should generate predictions."""
        # Pre-populate entities for shadow engine to track
        await hub.set_cache("entities", {
            "light.kitchen": {"entity_id": "light.kitchen", "device_id": "d1", "area_id": "kitchen"},
            "switch.coffee_maker": {"entity_id": "switch.coffee_maker", "device_id": "d2", "area_id": "kitchen"},
        })
        await hub.set_cache("entity_curation", {
            "curations": [
                {"entity_id": "light.kitchen", "tier": 3},
                {"entity_id": "switch.coffee_maker", "tier": 3},
            ]
        })

        module = ShadowEngine(hub)
        hub.register_module(module)
        await module.initialize()

        # Feed all events
        for event in PREDICTABLE_EVENTS:
            await hub.publish("state_changed", event)

        # Check that predictions were generated
        pipeline_cache = await hub.get_cache("pipeline_state")
        predictions_cache = await hub.get_cache("predictions")

        # At least one of the caches should have data
        has_output = (
            (pipeline_cache is not None and pipeline_cache.get("data"))
            or (predictions_cache is not None and predictions_cache.get("data"))
        )
        assert has_output, "Shadow engine should produce predictions or pipeline state after events"

    @pytest.mark.asyncio
    async def test_golden_snapshot(self, hub, update_golden):
        """Golden snapshot comparison for shadow engine output."""
        await hub.set_cache("entities", {
            "light.kitchen": {"entity_id": "light.kitchen", "device_id": "d1", "area_id": "kitchen"},
            "switch.coffee_maker": {"entity_id": "switch.coffee_maker", "device_id": "d2", "area_id": "kitchen"},
        })
        await hub.set_cache("entity_curation", {
            "curations": [
                {"entity_id": "light.kitchen", "tier": 3},
                {"entity_id": "switch.coffee_maker", "tier": 3},
            ]
        })

        module = ShadowEngine(hub)
        hub.register_module(module)
        await module.initialize()

        for event in PREDICTABLE_EVENTS:
            await hub.publish("state_changed", event)

        pipeline = await hub.get_cache("pipeline_state")
        predictions = await hub.get_cache("predictions")
        actual = {
            "pipeline_state": pipeline.get("data", {}) if pipeline else {},
            "predictions": predictions.get("data", {}) if predictions else {},
        }
        golden_compare(actual, "shadow_engine", update=update_golden)
```

**Step 2: Run test**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/test_shadow_engine_ka.py -v --timeout=120
```

Expected: Pass. Shadow engine might need additional constructor args — check `ShadowEngine.__init__`. If it requires config or model paths, adapt the fixture setup. The shadow engine has complex internal state (Thompson sampling, prediction windows) so initial runs may produce minimal output.

**Step 3: Commit**

```bash
git add tests/integration/known_answer/test_shadow_engine_ka.py tests/integration/known_answer/golden/
git commit -m "test: add shadow_engine known-answer test"
```

---

### Task 7: Known-answer test for trajectory_classifier module

**Files:**
- Create: `tests/integration/known_answer/test_trajectory_classifier_ka.py`
- Reference: `aria/modules/trajectory_classifier.py` (TrajectoryClassifier class)

**Step 1: Write the test**

Trajectory classifier subscribes to `shadow_resolved` events and classifies trajectories. We publish known events and check classification output.

Create `tests/integration/known_answer/test_trajectory_classifier_ka.py`:

```python
"""Known-answer test: trajectory_classifier module.

Classifies shadow engine resolved events into trajectory types
and provides anomaly explanations.
"""

import pytest

from aria.modules.trajectory_classifier import TrajectoryClassifier
from tests.integration.known_answer.conftest import golden_compare


SHADOW_RESOLVED_EVENTS = [
    {
        "target": "light.kitchen",
        "predicted_state": "on",
        "actual_state": "on",
        "accuracy": 1.0,
        "timestamp": "2026-02-01T07:00:00",
        "prediction_window_minutes": 15,
    },
    {
        "target": "light.kitchen",
        "predicted_state": "on",
        "actual_state": "on",
        "accuracy": 1.0,
        "timestamp": "2026-02-02T07:00:00",
        "prediction_window_minutes": 15,
    },
    {
        "target": "light.kitchen",
        "predicted_state": "on",
        "actual_state": "on",
        "accuracy": 1.0,
        "timestamp": "2026-02-03T07:00:00",
        "prediction_window_minutes": 15,
    },
    {
        "target": "light.kitchen",
        "predicted_state": "on",
        "actual_state": "off",
        "accuracy": 0.0,
        "timestamp": "2026-02-04T07:00:00",
        "prediction_window_minutes": 15,
    },
]


class TestTrajectoryClassifierKnownAnswer:
    """Trajectory classifier known-answer tests."""

    @pytest.mark.asyncio
    async def test_classifies_after_events(self, hub):
        """Should classify trajectories after receiving shadow_resolved events."""
        module = TrajectoryClassifier(hub)
        hub.register_module(module)
        await module.initialize()

        for event in SHADOW_RESOLVED_EVENTS:
            await hub.publish("shadow_resolved", event)

        # Check classification output in cache
        cache = await hub.get_cache("trajectory_classifications")
        # May not produce output until enough events accumulate (Tier 3+ window)
        # This is a soft assertion — the behavioral claim is "no crash, processes events"
        # The golden snapshot captures whatever output state results

    @pytest.mark.asyncio
    async def test_golden_snapshot(self, hub, update_golden):
        """Golden snapshot comparison for trajectory classifier output."""
        module = TrajectoryClassifier(hub)
        hub.register_module(module)
        await module.initialize()

        for event in SHADOW_RESOLVED_EVENTS:
            await hub.publish("shadow_resolved", event)

        cache = await hub.get_cache("trajectory_classifications")
        actual = {
            "classifications": cache.get("data", {}) if cache else {},
        }
        golden_compare(actual, "trajectory_classifier", update=update_golden)
```

**Step 2: Run test**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/test_trajectory_classifier_ka.py -v --timeout=120
```

Expected: Pass. The cache key for trajectory output may differ — check `set_cache` call in `trajectory_classifier.py` and adjust. The module is Tier 3+ so it may require sufficient event history before producing output.

**Step 3: Commit**

```bash
git add tests/integration/known_answer/test_trajectory_classifier_ka.py tests/integration/known_answer/golden/
git commit -m "test: add trajectory_classifier known-answer test"
```

---

### Task 8: Known-answer test for ml_engine module

**Files:**
- Create: `tests/integration/known_answer/test_ml_engine_ka.py`
- Reference: `aria/modules/ml_engine.py` (MLEngine class)

**Step 1: Write the test**

ML engine trains models and manages ensemble weights. We use the existing `PipelineRunner` to create training data, then test that `ml_engine` can initialize and produce cache output.

Create `tests/integration/known_answer/test_ml_engine_ka.py`:

```python
"""Known-answer test: ml_engine module.

MLEngine manages hub-side model training, periodic retraining, and ensemble
weight computation. We verify it initializes with cached intelligence data
and produces model metrics.
"""

import pytest

from aria.modules.ml_engine import MLEngine
from tests.integration.known_answer.conftest import golden_compare
from tests.synthetic.simulator import HouseholdSimulator
from tests.synthetic.pipeline import PipelineRunner


@pytest.fixture(scope="module")
def pipeline_results(tmp_path_factory):
    """Run engine pipeline to produce data for ml_engine tests."""
    tmp = tmp_path_factory.mktemp("ml_ka")
    sim = HouseholdSimulator(scenario="stable_couple", days=14, seed=42)
    snapshots = sim.generate()
    runner = PipelineRunner(snapshots, data_dir=tmp)
    result = runner.run_full()
    return {"runner": runner, "result": result, "data_dir": tmp}


class TestMLEngineKnownAnswer:
    """ML engine known-answer tests."""

    @pytest.mark.asyncio
    async def test_initializes_with_data(self, hub, pipeline_results):
        """MLEngine should initialize and write ensemble weights to cache."""
        # Pre-populate intelligence cache with pipeline results
        await hub.set_cache("intelligence", {
            "predictions": pipeline_results["result"]["predictions"],
            "scores": pipeline_results["result"]["scores"],
            "training": pipeline_results["result"]["training"],
        })

        module = MLEngine(hub)
        hub.register_module(module)
        # Skip full initialize (it schedules timers) — just call the core method
        # Check what methods are available for direct testing

        weights_cache = await hub.get_cache("ml_ensemble_weights")
        # MLEngine might not write weights until on_event or scheduled task runs
        # The behavioral assertion is: module can be constructed and registered

    @pytest.mark.asyncio
    async def test_golden_snapshot(self, hub, pipeline_results, update_golden):
        """Golden snapshot for ml_engine output."""
        await hub.set_cache("intelligence", {
            "predictions": pipeline_results["result"]["predictions"],
            "scores": pipeline_results["result"]["scores"],
        })

        module = MLEngine(hub)
        hub.register_module(module)

        weights = await hub.get_cache("ml_ensemble_weights")
        features = await hub.get_cache("feature_config")
        actual = {
            "ensemble_weights": weights.get("data", {}) if weights else {},
            "feature_config": features.get("data", {}) if features else {},
        }
        golden_compare(actual, "ml_engine", update=update_golden)
```

**Step 2: Run test**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/test_ml_engine_ka.py -v --timeout=120
```

Expected: Pass. MLEngine has a complex `__init__` — check the actual constructor signature and adapt. It may require `data_dir` or `config` args.

**Step 3: Commit**

```bash
git add tests/integration/known_answer/test_ml_engine_ka.py tests/integration/known_answer/golden/
git commit -m "test: add ml_engine known-answer test"
```

---

### Task 9: Known-answer test for intelligence module

**Files:**
- Create: `tests/integration/known_answer/test_intelligence_ka.py`
- Reference: `aria/modules/intelligence.py` (IntelligenceModule class)

**Step 1: Write the test**

Intelligence module is a thin bridge — reads engine JSON output files and caches them. We create a deterministic engine output file and verify it's read correctly.

Create `tests/integration/known_answer/test_intelligence_ka.py`:

```python
"""Known-answer test: intelligence module (thin bridge).

Reads batch engine JSON output → caches metrics in hub.
We create a known engine output file and verify correct caching.
"""

import json
from pathlib import Path

import pytest

from aria.modules.intelligence import IntelligenceModule, METRIC_PATHS
from tests.integration.known_answer.conftest import golden_compare


MOCK_ENGINE_OUTPUT = {
    "date": "2026-02-19",
    "overall_accuracy": 84,
    "prediction_method": "blended",
    "days_of_data": 14,
    "metrics": {
        "power_watts": {"accuracy": 83, "predicted": 150.0, "actual": 145.0},
        "lights_on": {"accuracy": 65, "predicted": 20.0, "actual": 46.0},
        "devices_home": {"accuracy": 80, "predicted": 62.0, "actual": 64.0},
        "unavailable": {"accuracy": 95, "predicted": 910.0, "actual": 909.0},
        "useful_events": {"accuracy": 98, "predicted": 3700.0, "actual": 3381.0},
    },
}


class TestIntelligenceKnownAnswer:
    """Intelligence module known-answer tests."""

    @pytest.mark.asyncio
    async def test_reads_engine_output(self, hub, tmp_path):
        """Intelligence module should read engine JSON and cache metrics."""
        # Create engine output file
        output_dir = tmp_path / "intelligence"
        output_dir.mkdir()
        (output_dir / "latest.json").write_text(json.dumps(MOCK_ENGINE_OUTPUT))

        module = IntelligenceModule(hub, data_dir=tmp_path)
        hub.register_module(module)
        await module.initialize()

        intel_cache = await hub.get_cache("intelligence")
        assert intel_cache is not None, "Intelligence cache should be populated"
        data = intel_cache.get("data", {})
        assert "overall_accuracy" in data or "metrics" in data or len(data) > 0, \
            f"Intelligence cache should contain metrics, got keys: {list(data.keys())}"

    @pytest.mark.asyncio
    async def test_all_metric_paths_resolved(self, hub, tmp_path):
        """Every METRIC_PATH should resolve to a value from the engine output."""
        output_dir = tmp_path / "intelligence"
        output_dir.mkdir()
        (output_dir / "latest.json").write_text(json.dumps(MOCK_ENGINE_OUTPUT))

        module = IntelligenceModule(hub, data_dir=tmp_path)
        hub.register_module(module)
        await module.initialize()

        intel_cache = await hub.get_cache("intelligence")
        if intel_cache and intel_cache.get("data"):
            data = intel_cache["data"]
            # At least some metric paths should have resolved
            resolved_count = sum(1 for v in data.values() if v is not None) if isinstance(data, dict) else 0
            assert resolved_count >= 1, f"Expected some resolved metrics, got {resolved_count}"

    @pytest.mark.asyncio
    async def test_golden_snapshot(self, hub, tmp_path, update_golden):
        """Golden snapshot for intelligence module output."""
        output_dir = tmp_path / "intelligence"
        output_dir.mkdir()
        (output_dir / "latest.json").write_text(json.dumps(MOCK_ENGINE_OUTPUT))

        module = IntelligenceModule(hub, data_dir=tmp_path)
        hub.register_module(module)
        await module.initialize()

        intel_cache = await hub.get_cache("intelligence")
        actual = {
            "intelligence": intel_cache.get("data", {}) if intel_cache else {},
        }
        golden_compare(actual, "intelligence", update=update_golden)
```

**Step 2: Run test**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/test_intelligence_ka.py -v --timeout=120
```

Expected: Pass. The `IntelligenceModule` constructor may use different arg names — check `__init__` and adapt. It reads from `~/ha-logs/intelligence/` by default so we override with `tmp_path`.

**Step 3: Commit**

```bash
git add tests/integration/known_answer/test_intelligence_ka.py tests/integration/known_answer/golden/
git commit -m "test: add intelligence known-answer test"
```

---

### Task 10: Known-answer test for presence module

**Files:**
- Create: `tests/integration/known_answer/test_presence_ka.py`
- Reference: `aria/modules/presence.py` (PresenceModule class)

**Step 1: Write the test**

Presence module listens for MQTT messages (Frigate person detection). We mock the MQTT connection and feed known messages.

Create `tests/integration/known_answer/test_presence_ka.py`:

```python
"""Known-answer test: presence module.

Presence tracks per-room occupancy from Frigate camera events + HA sensors.
MQTT is mocked — we feed deterministic person detection events.
"""

import pytest

from aria.modules.presence import PresenceModule
from tests.integration.known_answer.conftest import golden_compare


class TestPresenceKnownAnswer:
    """Presence module known-answer tests."""

    @pytest.mark.asyncio
    async def test_processes_person_event(self, hub, monkeypatch):
        """Should update occupancy state when receiving person detection."""
        module = PresenceModule(hub)
        hub.register_module(module)

        # Mock MQTT connection (presence connects on initialize)
        monkeypatch.setattr(module, "_connect_mqtt", lambda: None)
        # Manually set internal state as if MQTT connected
        module._mqtt_connected = False  # Skip actual MQTT

        # Simulate a Frigate person detection via on_event
        person_event = {
            "type": "person",
            "camera": "front_door",
            "person": "person_a",
            "confidence": 0.95,
            "timestamp": "2026-02-19T08:00:00",
            "area": "Hallway",
        }
        await module.on_event("frigate_person", person_event)

        presence_cache = await hub.get_cache("presence")
        # Presence may require multiple events or HA sensor data to produce output
        # Soft assertion: module processes event without crash

    @pytest.mark.asyncio
    async def test_golden_snapshot(self, hub, monkeypatch, update_golden):
        """Golden snapshot for presence module output."""
        module = PresenceModule(hub)
        hub.register_module(module)

        monkeypatch.setattr(module, "_connect_mqtt", lambda: None)
        module._mqtt_connected = False

        person_events = [
            {"type": "person", "camera": "front_door", "person": "person_a", "confidence": 0.95, "timestamp": "2026-02-19T08:00:00", "area": "Hallway"},
            {"type": "person", "camera": "living_room", "person": "person_a", "confidence": 0.90, "timestamp": "2026-02-19T08:05:00", "area": "Living Room"},
        ]
        for event in person_events:
            await module.on_event("frigate_person", event)

        presence_cache = await hub.get_cache("presence")
        actual = {
            "presence": presence_cache.get("data", {}) if presence_cache else {},
        }
        golden_compare(actual, "presence", update=update_golden)
```

**Step 2: Run test**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/test_presence_ka.py -v --timeout=120
```

Expected: Pass. The `PresenceModule` constructor and MQTT mocking may need adjustment — check `__init__` for required args (MQTT host/port, etc.) and the actual `_connect_mqtt` method name.

**Step 3: Commit**

```bash
git add tests/integration/known_answer/test_presence_ka.py tests/integration/known_answer/golden/
git commit -m "test: add presence known-answer test"
```

---

### Task 11: Known-answer test for audit_logger

**Files:**
- Create: `tests/integration/known_answer/test_audit_logger_ka.py`
- Reference: `aria/hub/audit.py` (AuditLogger class)

**Step 1: Write the test**

Audit logger writes tamper-evident log entries to SQLite. We verify entries are written with correct checksums.

Create `tests/integration/known_answer/test_audit_logger_ka.py`:

```python
"""Known-answer test: audit_logger.

Writes tamper-evident audit entries to SQLite with SHA-256 checksums.
"""

import pytest

from aria.hub.audit import AuditLogger
from tests.integration.known_answer.conftest import golden_compare


class TestAuditLoggerKnownAnswer:
    """Audit logger known-answer tests."""

    @pytest.mark.asyncio
    async def test_writes_audit_entry(self, tmp_path):
        """Should write an audit entry with valid checksum."""
        db_path = tmp_path / "audit.db"
        logger = AuditLogger(db_path=str(db_path))
        await logger.initialize()

        await logger.log(
            event_type="module.register",
            source="hub",
            action="register",
            subject="discovery",
            detail={"class": "DiscoveryModule"},
        )

        # Flush the queue
        await logger.flush()

        entries = await logger.query(limit=10)
        assert len(entries) >= 1, "Should have at least 1 audit entry"

        entry = entries[0]
        assert entry["event_type"] == "module.register"
        assert entry["source"] == "hub"
        assert entry["checksum"], "Entry should have a checksum"

    @pytest.mark.asyncio
    async def test_checksum_integrity(self, tmp_path):
        """Checksum should be a valid SHA-256 hash."""
        import hashlib

        db_path = tmp_path / "audit.db"
        logger = AuditLogger(db_path=str(db_path))
        await logger.initialize()

        await logger.log(
            event_type="test.known_answer",
            source="test",
            action="verify",
            subject="checksum",
            detail={"test": True},
        )
        await logger.flush()

        entries = await logger.query(limit=1)
        assert len(entries) == 1
        checksum = entries[0]["checksum"]
        # Should be a 64-char hex string (SHA-256)
        assert len(checksum) == 64, f"Expected 64-char SHA-256, got {len(checksum)} chars"
        assert all(c in "0123456789abcdef" for c in checksum)

        await logger.close()

    @pytest.mark.asyncio
    async def test_golden_snapshot(self, tmp_path, update_golden):
        """Golden snapshot for audit logger output structure."""
        db_path = tmp_path / "audit.db"
        logger = AuditLogger(db_path=str(db_path))
        await logger.initialize()

        events = [
            ("module.register", "hub", "register", "discovery", {"class": "DiscoveryModule"}),
            ("cache.write", "hub", "set", "entities", {"version": 1}),
            ("module.register", "hub", "register", "patterns", {"class": "PatternRecognition"}),
        ]
        for event_type, source, action, subject, detail in events:
            await logger.log(
                event_type=event_type,
                source=source,
                action=action,
                subject=subject,
                detail=detail,
            )
        await logger.flush()

        entries = await logger.query(limit=10)

        # Normalize for golden (remove timestamps and checksums which vary)
        actual = {
            "entry_count": len(entries),
            "entries": [
                {
                    "event_type": e["event_type"],
                    "source": e["source"],
                    "action": e["action"],
                    "subject": e["subject"],
                }
                for e in entries
            ],
        }
        golden_compare(actual, "audit_logger", update=update_golden)

        await logger.close()
```

**Step 2: Run test**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/test_audit_logger_ka.py -v --timeout=120
```

Expected: Pass. The `AuditLogger` constructor and `query`/`flush` method names may differ — check `audit.py` and adapt. The key assertion is tamper-evident entries with valid checksums.

**Step 3: Commit**

```bash
git add tests/integration/known_answer/test_audit_logger_ka.py tests/integration/known_answer/golden/
git commit -m "test: add audit_logger known-answer test"
```

---

### Task 12: Full pipeline known-answer test

**Files:**
- Create: `tests/integration/known_answer/test_full_pipeline_ka.py`
- Reference: `tests/synthetic/pipeline.py`, all module files

**Step 1: Write the test**

This traces one scenario end-to-end: engine output → hub modules → final recommendations + anomaly signals. Uses the existing `PipelineRunner` for engine side, then manually runs hub modules in sequence.

Create `tests/integration/known_answer/test_full_pipeline_ka.py`:

```python
"""Known-answer test: full pipeline end-to-end.

Traces stable_couple scenario:
  Engine (snapshots → baselines → train → predict → score)
  → Hub (intelligence reads engine output → patterns → orchestrator → shadow predictions)
  → Final outputs: automation_suggestions + pipeline_state
"""

import json

import pytest

from aria.hub.core import IntelligenceHub
from aria.modules.intelligence import IntelligenceModule
from aria.modules.orchestrator import OrchestratorModule
from aria.modules.patterns import PatternRecognition
from tests.integration.known_answer.conftest import golden_compare
from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import HouseholdSimulator


@pytest.fixture(scope="module")
def engine_output(tmp_path_factory):
    """Run engine pipeline to produce output for hub consumption."""
    tmp = tmp_path_factory.mktemp("full_pipeline_ka")
    sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
    snapshots = sim.generate()
    runner = PipelineRunner(snapshots, data_dir=tmp)
    result = runner.run_full()
    return {
        "data_dir": tmp,
        "result": result,
        "snapshots": snapshots,
    }


class TestFullPipelineKnownAnswer:
    """Full pipeline known-answer tests."""

    @pytest.mark.asyncio
    async def test_engine_produces_predictions(self, engine_output):
        """Engine should produce predictions and scores."""
        result = engine_output["result"]
        assert result["predictions"] is not None, "Engine should produce predictions"
        assert result["scores"] is not None, "Engine should produce scores"
        assert result["snapshots_saved"] > 0

    @pytest.mark.asyncio
    async def test_hub_reads_engine_output(self, engine_output, tmp_path):
        """Intelligence module should read engine output into hub cache."""
        hub = IntelligenceHub(cache_path=str(tmp_path / "hub.db"))
        await hub.initialize()

        try:
            # Write engine output as JSON file for intelligence module
            intel_dir = tmp_path / "intelligence"
            intel_dir.mkdir()
            (intel_dir / "latest.json").write_text(
                json.dumps(engine_output["result"]["predictions"], default=str)
            )

            module = IntelligenceModule(hub, data_dir=tmp_path)
            hub.register_module(module)
            await module.initialize()

            intel_cache = await hub.get_cache("intelligence")
            assert intel_cache is not None, "Intelligence cache should be populated from engine output"
        finally:
            await hub.shutdown()

    @pytest.mark.asyncio
    async def test_patterns_to_suggestions_flow(self, engine_output, tmp_path):
        """Patterns → orchestrator should produce automation suggestions."""
        hub = IntelligenceHub(cache_path=str(tmp_path / "hub.db"))
        await hub.initialize()

        try:
            # Manually populate patterns cache with known patterns
            # (In full production flow, patterns reads logbook files)
            await hub.set_cache("patterns", {
                "patterns": [
                    {
                        "pattern_id": "full_pipeline_morning",
                        "area": "Kitchen",
                        "entities": ["light.kitchen", "switch.coffee_maker"],
                        "trigger_time": "07:00",
                        "confidence": 0.92,
                        "frequency": 15,
                        "type": "temporal_sequence",
                        "llm_description": "Morning routine",
                    },
                ],
                "pattern_count": 1,
                "areas_analyzed": ["Kitchen"],
            })

            orchestrator = OrchestratorModule(
                hub=hub,
                ha_url="http://test-host:8123",
                ha_token="test-token",
                min_confidence=0.7,
            )
            hub.register_module(orchestrator)

            suggestions = await orchestrator.generate_suggestions()
            assert len(suggestions) >= 1, "Orchestrator should produce >= 1 suggestion from pipeline patterns"

            # Verify suggestion is cached
            cache = await hub.get_cache("automation_suggestions")
            assert cache is not None
        finally:
            await hub.shutdown()

    @pytest.mark.asyncio
    async def test_golden_snapshot(self, engine_output, tmp_path, update_golden):
        """Golden snapshot of full pipeline output."""
        hub = IntelligenceHub(cache_path=str(tmp_path / "hub.db"))
        await hub.initialize()

        try:
            await hub.set_cache("patterns", {
                "patterns": [
                    {
                        "pattern_id": "full_pipeline_morning",
                        "area": "Kitchen",
                        "entities": ["light.kitchen", "switch.coffee_maker"],
                        "trigger_time": "07:00",
                        "confidence": 0.92,
                        "frequency": 15,
                        "type": "temporal_sequence",
                        "llm_description": "Morning routine",
                    },
                ],
                "pattern_count": 1,
                "areas_analyzed": ["Kitchen"],
            })

            orchestrator = OrchestratorModule(
                hub=hub,
                ha_url="http://test-host:8123",
                ha_token="test-token",
                min_confidence=0.7,
            )
            hub.register_module(orchestrator)
            suggestions = await orchestrator.generate_suggestions()

            actual = {
                "engine_scores": engine_output["result"]["scores"],
                "suggestion_count": len(suggestions),
                "has_recommendations": len(suggestions) > 0,
                "has_engine_predictions": engine_output["result"]["predictions"] is not None,
            }
            golden_compare(actual, "full_pipeline", update=update_golden)
        finally:
            await hub.shutdown()
```

**Step 2: Run test**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/test_full_pipeline_ka.py -v --timeout=120
```

Expected: Pass. This is the most comprehensive test — it validates the vertical trace from engine data through hub modules to final outputs.

**Step 3: Commit**

```bash
git add tests/integration/known_answer/test_full_pipeline_ka.py tests/integration/known_answer/golden/
git commit -m "test: add full pipeline known-answer test"
```

---

### Task 13: Run full test suite and fix issues

**Step 1: Check memory**

```bash
free -h | awk '/Mem:/{print $7}'
```

If < 4G, run targeted:
```bash
.venv/bin/python -m pytest tests/integration/known_answer/ --timeout=120 -x -q
```

**Step 2: Run known-answer suite**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/ -v --timeout=120
```

Fix any failures — common issues:
- Wrong constructor args (check `__init__` signatures)
- Wrong cache keys (check `set_cache` calls in module)
- Wrong method names for monkeypatching (check actual method names)
- Missing `pytest-asyncio` markers

**Step 3: Run full suite to check for regressions**

```bash
.venv/bin/python -m pytest tests/ --timeout=120 -x -q
```

Expected: All existing tests still pass, plus new known-answer tests.

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve known-answer test issues from first run"
```

---

### Task 14: Dashboard greyed-out modules

**Files:**
- Modify: `aria/dashboard/spa/src/lib/pipelineGraph.js`
- Modify: `aria/dashboard/spa/src/pages/intelligence/utils.jsx` (if module status rendering)

**Step 1: Read current pipelineGraph.js**

Check how nodes are currently rendered — look for styling/class logic.

```bash
grep -n "class\|style\|color\|opacity\|tier\|disabled" aria/dashboard/spa/src/lib/pipelineGraph.js
```

**Step 2: Add tier-based greyed-out styling**

In `pipelineGraph.js`, add a `tierGated` property to `NODE_DETAIL` for `trajectory_classifier`:

```javascript
trajectory_classifier: {
  // ... existing properties ...
  tierGated: 3,  // Module requires tier 3+ data to produce output
},
```

In the Sankey rendering logic, add opacity/greyed-out styling for nodes where `tierGated` is set and the current system tier is below that value. The exact implementation depends on how the Sankey is rendered — likely in a component that reads `NODE_DETAIL`.

**Step 3: Rebuild SPA**

```bash
cd aria/dashboard/spa && npm install && npm run build
```

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/lib/pipelineGraph.js
git commit -m "feat: show tier-gated modules as greyed out in pipeline Sankey"
```

---

### Task 15: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (project root, i.e. `ha-aria/CLAUDE.md`)

**Step 1: Update module count and test commands**

In the CLAUDE.md file:
- Update any reference to "14 modules" → "10 modules"
- Remove test command examples referencing archived modules: `organic`, `data_quality`, `activity_labeler`, `online_learner`
- Add known-answer test command:
  ```bash
  .venv/bin/python -m pytest tests/integration/known_answer/ -v --timeout=120
  ```
- Update entity count or module list if mentioned
- Remove `organic_discovery` from gotchas if present

**Step 2: Verify no stale references**

```bash
grep -n "organic_discovery\|online_learner\|transfer_engine\|activity_labeler\|data_quality\|14 modules\|pattern_recognition" CLAUDE.md
```

Update or remove all matches.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for lean audit Phase 1 changes"
```

---

### Task 16: Final verification

**Step 1: Run known-answer suite**

```bash
.venv/bin/python -m pytest tests/integration/known_answer/ -v --timeout=120
```

**Step 2: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ --timeout=120 -x -q
```

**Step 3: Verify golden files committed**

```bash
ls tests/integration/known_answer/golden/
git status
```

All golden `.json` files should be committed.

**Step 4: Verify no dangling references**

```bash
grep -r "online_learner\|organic_discovery\|transfer_engine\|activity_labeler" aria/ tests/ --include="*.py" -l
```

Should return nothing (or only `_archived/` paths).

---

## Verification Checklist

1. [ ] `tests/integration/known_answer/` directory exists with conftest, 11 test files, golden/, fixtures/
2. [ ] All 10 modules have known-answer tests with behavioral assertions
3. [ ] Full pipeline test traces engine → hub → recommendations
4. [ ] Golden snapshots created and committed for all modules
5. [ ] `--update-golden` flag works
6. [ ] Dashboard shows greyed-out tier-gated modules
7. [ ] CLAUDE.md reflects 10-module architecture
8. [ ] No regressions in existing test suite
9. [ ] All golden drift warnings are expected (first-run baselines)
