# Phase 4: I&W Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build behavioral state detection via indicator chains, lifecycle management, backtesting, organic composite discovery, and synthetic testing.

**Architecture:** Three-layer data model (Indicator → BehavioralStateDefinition → BehavioralStateTracker) with real-time event-subscriber detection, batch discovery from patterns.py + gap_analyzer output, lifecycle state machine with backtest gate, and synthetic test framework. New `aria/iw/` package.

**Tech Stack:** Python 3.12, aiosqlite, pytest, existing aria.hub.core.Module base class, existing EventStore/EntityGraph/patterns/co_occurrence/gap_analyzer infrastructure.

**Design Doc:** `docs/plans/2026-02-21-phase4-iw-framework-design.md`
**PRD:** `tasks/prd.json` (P4-01 through P4-15)

---

## Quality Gates

Run between every batch:

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/ --timeout=120 -x -q
```

Additional checks:
- `ruff check aria/iw/` — no lint violations
- Config defaults count check (current: 148, expected after P4-03: 168)

---

## Batch 1: Data Models + Storage (P4-01, P4-02, P4-03)

Foundation layer — all subsequent batches depend on these.

### Task 1: Data models (P4-01)

**Files:**
- Create: `aria/iw/__init__.py`
- Create: `aria/iw/models.py`
- Create: `tests/iw/__init__.py`
- Create: `tests/iw/test_models.py`

**Step 1: Create package and write model tests**

Create `aria/iw/__init__.py` (empty) and `tests/iw/__init__.py` (empty).

Write `tests/iw/test_models.py` testing:

```python
"""Tests for aria.iw.models — behavioral state data models."""
import pytest
from aria.iw.models import Indicator, BehavioralStateDefinition, BehavioralStateTracker, ActiveState


class TestIndicator:
    def test_state_change_mode(self):
        ind = Indicator(entity_id="binary_sensor.bedroom_motion", role="trigger",
                       mode="state_change", expected_state="on")
        assert ind.mode == "state_change"
        assert ind.entity_id == "binary_sensor.bedroom_motion"

    def test_quiet_period_mode(self):
        ind = Indicator(entity_id="binary_sensor.bedroom_motion", role="trigger",
                       mode="quiet_period", quiet_seconds=14400)
        assert ind.quiet_seconds == 14400

    def test_threshold_mode(self):
        ind = Indicator(entity_id="sensor.bedroom_illuminance", role="confirming",
                       mode="threshold", threshold_value=50.0,
                       threshold_direction="below", max_delay_seconds=300)
        assert ind.threshold_direction == "below"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError, match="role"):
            Indicator(entity_id="x", role="invalid", mode="state_change")

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            Indicator(entity_id="x", role="trigger", mode="invalid")

    def test_frozen(self):
        ind = Indicator(entity_id="x", role="trigger", mode="state_change")
        with pytest.raises(AttributeError):
            ind.entity_id = "y"

    def test_json_roundtrip(self):
        ind = Indicator(entity_id="light.kitchen", role="confirming",
                       mode="state_change", expected_state="on",
                       max_delay_seconds=600, confidence=0.85)
        data = ind.to_dict()
        restored = Indicator.from_dict(data)
        assert restored == ind


class TestBehavioralStateDefinition:
    def test_create_with_indicators(self):
        trigger = Indicator(entity_id="binary_sensor.bedroom_motion",
                           role="trigger", mode="state_change", expected_state="on")
        confirming = [
            Indicator(entity_id="binary_sensor.bathroom_motion",
                     role="confirming", mode="state_change",
                     expected_state="on", max_delay_seconds=600),
        ]
        defn = BehavioralStateDefinition(
            id="morning_bedroom_workday",
            name="Morning Routine (Bedroom, Workday)",
            trigger=trigger,
            trigger_preconditions=[],
            confirming=confirming,
            deviations=[],
            areas=frozenset(["bedroom", "bathroom"]),
            day_types=frozenset(["workday"]),
            person_attribution="person.justin",
            typical_duration_minutes=45.0,
            expected_outcomes=({"entity_id": "binary_sensor.front_door", "state": "on", "probability": 0.85},),
        )
        assert defn.id == "morning_bedroom_workday"
        assert len(defn.confirming) == 1

    def test_frozen(self):
        trigger = Indicator(entity_id="x", role="trigger", mode="state_change")
        defn = BehavioralStateDefinition(
            id="test", name="Test", trigger=trigger,
            trigger_preconditions=[], confirming=[], deviations=[],
            areas=frozenset(), day_types=frozenset(),
            person_attribution=None, typical_duration_minutes=30.0,
            expected_outcomes=(),
        )
        with pytest.raises(AttributeError):
            defn.name = "changed"

    def test_json_roundtrip(self):
        trigger = Indicator(entity_id="binary_sensor.bedroom_motion",
                           role="trigger", mode="state_change", expected_state="on")
        defn = BehavioralStateDefinition(
            id="test_rt", name="Test Roundtrip", trigger=trigger,
            trigger_preconditions=[], confirming=[], deviations=[],
            areas=frozenset(["bedroom"]), day_types=frozenset(["workday"]),
            person_attribution=None, typical_duration_minutes=30.0,
            expected_outcomes=(),
        )
        data = defn.to_dict()
        restored = BehavioralStateDefinition.from_dict(data)
        assert restored.id == defn.id
        assert restored.trigger == defn.trigger


class TestBehavioralStateTracker:
    def test_create_default(self):
        tracker = BehavioralStateTracker(definition_id="test_def")
        assert tracker.lifecycle == "seed"
        assert tracker.observation_count == 0

    def test_record_observation(self):
        tracker = BehavioralStateTracker(definition_id="test_def")
        tracker.record_observation("2026-02-21T08:00:00Z", match_ratio=0.8)
        assert tracker.observation_count == 1
        assert tracker.last_seen == "2026-02-21T08:00:00Z"

    def test_json_roundtrip(self):
        tracker = BehavioralStateTracker(definition_id="test_def")
        tracker.record_observation("2026-02-21T08:00:00Z", match_ratio=0.8)
        data = tracker.to_dict()
        restored = BehavioralStateTracker.from_dict(data)
        assert restored.observation_count == 1


class TestActiveState:
    def test_match_ratio(self):
        active = ActiveState(
            definition_id="test",
            trigger_time="2026-02-21T07:00:00Z",
            matched_confirming=["entity_a"],
            pending_confirming=["entity_b", "entity_c"],
            window_expires="2026-02-21T07:30:00Z",
        )
        assert abs(active.match_ratio - 1/3) < 0.01

    def test_match_ratio_all_matched(self):
        active = ActiveState(
            definition_id="test",
            trigger_time="2026-02-21T07:00:00Z",
            matched_confirming=["entity_a", "entity_b"],
            pending_confirming=[],
            window_expires="2026-02-21T07:30:00Z",
        )
        assert active.match_ratio == 1.0

    def test_match_ratio_empty(self):
        active = ActiveState(
            definition_id="test",
            trigger_time="2026-02-21T07:00:00Z",
            matched_confirming=[],
            pending_confirming=[],
            window_expires="2026-02-21T07:30:00Z",
        )
        assert active.match_ratio == 0.0
```

**Step 2: Run tests — verify they fail**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_models.py -v --timeout=30
```

Expected: FAIL (ImportError — module doesn't exist yet)

**Step 3: Implement models**

Create `aria/iw/models.py`:

```python
"""Behavioral state data models for the I&W framework.

Three-layer design:
  Layer 1: Indicator — atomic detection unit (frozen)
  Layer 2: BehavioralStateDefinition — indicator chain pattern (frozen)
  Layer 3: BehavioralStateTracker — runtime observations + lifecycle (mutable)
  Plus: ActiveState — real-time partial match tracking (mutable, in-memory only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_ROLES = frozenset({"trigger", "confirming", "deviation"})
VALID_MODES = frozenset({"state_change", "quiet_period", "threshold"})
VALID_LIFECYCLES = frozenset({"seed", "emerging", "confirmed", "mature", "dormant", "retired"})
VALID_THRESHOLD_DIRECTIONS = frozenset({"above", "below"})


@dataclass(frozen=True)
class Indicator:
    """Atomic detection unit — one entity condition to watch."""

    entity_id: str
    role: str           # trigger | confirming | deviation
    mode: str           # state_change | quiet_period | threshold

    expected_state: str | None = None
    quiet_seconds: int | None = None
    threshold_value: float | None = None
    threshold_direction: str | None = None  # above | below
    max_delay_seconds: int = 0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}, got {self.role!r}")
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {self.mode!r}")
        if self.threshold_direction is not None and self.threshold_direction not in VALID_THRESHOLD_DIRECTIONS:
            raise ValueError(f"threshold_direction must be one of {sorted(VALID_THRESHOLD_DIRECTIONS)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id, "role": self.role, "mode": self.mode,
            "expected_state": self.expected_state, "quiet_seconds": self.quiet_seconds,
            "threshold_value": self.threshold_value,
            "threshold_direction": self.threshold_direction,
            "max_delay_seconds": self.max_delay_seconds, "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Indicator:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class BehavioralStateDefinition:
    """Immutable indicator chain pattern — describes a behavioral state."""

    id: str
    name: str
    trigger: Indicator
    trigger_preconditions: list[Indicator]
    confirming: list[Indicator]
    deviations: list[Indicator]
    areas: frozenset[str]
    day_types: frozenset[str]
    person_attribution: str | None
    typical_duration_minutes: float
    expected_outcomes: tuple[dict, ...]
    composite_of: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name,
            "trigger": self.trigger.to_dict(),
            "trigger_preconditions": [i.to_dict() for i in self.trigger_preconditions],
            "confirming": [i.to_dict() for i in self.confirming],
            "deviations": [i.to_dict() for i in self.deviations],
            "areas": sorted(self.areas), "day_types": sorted(self.day_types),
            "person_attribution": self.person_attribution,
            "typical_duration_minutes": self.typical_duration_minutes,
            "expected_outcomes": list(self.expected_outcomes),
            "composite_of": list(self.composite_of),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BehavioralStateDefinition:
        return cls(
            id=data["id"], name=data["name"],
            trigger=Indicator.from_dict(data["trigger"]),
            trigger_preconditions=[Indicator.from_dict(i) for i in data.get("trigger_preconditions", [])],
            confirming=[Indicator.from_dict(i) for i in data.get("confirming", [])],
            deviations=[Indicator.from_dict(i) for i in data.get("deviations", [])],
            areas=frozenset(data.get("areas", [])),
            day_types=frozenset(data.get("day_types", [])),
            person_attribution=data.get("person_attribution"),
            typical_duration_minutes=data.get("typical_duration_minutes", 30.0),
            expected_outcomes=tuple(data.get("expected_outcomes", ())),
            composite_of=tuple(data.get("composite_of", ())),
        )


@dataclass
class BehavioralStateTracker:
    """Mutable runtime tracking for a behavioral state definition."""

    definition_id: str
    lifecycle: str = "seed"
    observation_count: int = 0
    consistency: float = 0.0
    first_seen: str = ""
    last_seen: str = ""
    lifecycle_history: list[dict] = field(default_factory=list)
    backtest_result: dict | None = None
    user_feedback: str | None = None
    automation_suggestion_id: str | None = None
    automation_status: str | None = None

    def __post_init__(self) -> None:
        if self.lifecycle not in VALID_LIFECYCLES:
            raise ValueError(f"lifecycle must be one of {sorted(VALID_LIFECYCLES)}")

    def record_observation(self, timestamp: str, match_ratio: float) -> None:
        self.observation_count += 1
        self.last_seen = timestamp
        if not self.first_seen:
            self.first_seen = timestamp
        # Running average of match_ratio as consistency proxy
        alpha = 1.0 / self.observation_count
        self.consistency = self.consistency * (1 - alpha) + match_ratio * alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition_id": self.definition_id, "lifecycle": self.lifecycle,
            "observation_count": self.observation_count, "consistency": self.consistency,
            "first_seen": self.first_seen, "last_seen": self.last_seen,
            "lifecycle_history": self.lifecycle_history,
            "backtest_result": self.backtest_result,
            "user_feedback": self.user_feedback,
            "automation_suggestion_id": self.automation_suggestion_id,
            "automation_status": self.automation_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BehavioralStateTracker:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ActiveState:
    """In-memory partial match tracking for real-time detection."""

    definition_id: str
    trigger_time: str
    matched_confirming: list[str]
    pending_confirming: list[str]
    window_expires: str

    @property
    def match_ratio(self) -> float:
        total = len(self.matched_confirming) + len(self.pending_confirming)
        return len(self.matched_confirming) / total if total > 0 else 0.0
```

**Step 4: Run tests — verify they pass**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_models.py -v --timeout=30
```

Expected: all tests PASS

**Step 5: Commit**

```bash
git add aria/iw/__init__.py aria/iw/models.py tests/iw/__init__.py tests/iw/test_models.py
git commit -m "feat(iw): add behavioral state data models — Indicator, Definition, Tracker, ActiveState (P4-01)"
```

---

### Task 2: Hub.db schema + BehavioralStateStore (P4-02)

**Files:**
- Create: `aria/iw/store.py`
- Create: `tests/iw/test_store.py`

**Step 1: Write store tests**

Write `tests/iw/test_store.py` testing:
- `save_definition()` and `get_definition()` — JSON roundtrip through SQLite
- `list_definitions()` — returns all definitions
- `save_tracker()` and `get_tracker()` — mutable tracker persistence
- `list_trackers(lifecycle_filter=)` — filter by lifecycle stage
- `record_co_activation(state_a_id, state_b_id)` — increment counter
- `get_co_activations(min_count=)` — return pairs above threshold
- `delete_definition()` — cascade deletes tracker and co-activations

Use `aiosqlite` with temporary database (`:memory:` or `tmp_path`).

**Step 2: Run tests — verify they fail**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_store.py -v --timeout=30
```

**Step 3: Implement BehavioralStateStore**

Create `aria/iw/store.py` with:
- `BehavioralStateStore` class accepting an `aiosqlite.Connection`
- `initialize()` creates 3 tables: `behavioral_state_definitions` (id TEXT PK, name TEXT, data_json TEXT), `behavioral_state_trackers` (definition_id TEXT PK FK, data_json TEXT), `state_co_activations` (state_a_id TEXT, state_b_id TEXT, count INTEGER, PK(state_a_id, state_b_id))
- CRUD methods: `save_definition`, `get_definition`, `list_definitions`, `delete_definition`, `save_tracker`, `get_tracker`, `list_trackers`, `record_co_activation`, `get_co_activations`
- JSON serialization via model `to_dict()`/`from_dict()`

**Step 4: Run tests — verify they pass**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_store.py -v --timeout=30
```

**Step 5: Commit**

```bash
git add aria/iw/store.py tests/iw/test_store.py
git commit -m "feat(iw): add BehavioralStateStore — SQLite persistence for definitions, trackers, co-activations (P4-02)"
```

---

### Task 3: Config entries (P4-03)

**Files:**
- Modify: `aria/hub/config_defaults.py` — add 20 `iw.*` entries
- Modify: existing config defaults test (update count assertion)

**Step 1: Add config entries**

Add 20 entries to `CONFIG_DEFAULTS` in `config_defaults.py` under a new `"I&W Framework"` category. Each entry follows the existing pattern with `key`, `default_value`, `value_type`, `label`, `description`, `description_layman`, `description_technical`, `category`, `min_value`, `max_value`. Keys:

`iw.discovery_interval_hours` (6), `iw.min_discovery_confidence` (0.60), `iw.min_match_ratio` (0.50), `iw.min_observations_seed` (3), `iw.min_observations_emerging` (7), `iw.min_consistency_emerging` (0.60), `iw.min_observations_confirmed` (15), `iw.min_consistency_confirmed` (0.70), `iw.min_observations_mature` (30), `iw.min_consistency_mature` (0.80), `iw.min_density_emerging` (0.3), `iw.min_density_confirmed` (0.5), `iw.dormant_days` (30), `iw.retired_days` (90), `iw.max_composites` (20), `iw.backtest_days` (90), `iw.backtest_holdout_ratio` (0.30), `iw.backtest_min_f1` (0.65), `iw.detector_window_seconds` (60), `iw.cold_start_replay_minutes` (60).

**Step 2: Update config count test**

Find the test that asserts config count (grep for `148` or `len(CONFIG_DEFAULTS)`). Update from 148 to 168.

**Step 3: Run tests**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/ -k "config_defaults" -v --timeout=30
```

**Step 4: Verify count**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -c "from aria.hub.config_defaults import CONFIG_DEFAULTS; iw = [c for c in CONFIG_DEFAULTS if c['key'].startswith('iw.')]; assert len(iw) >= 20, f'Only {len(iw)}'; print(f'{len(iw)} iw entries, {len(CONFIG_DEFAULTS)} total')"
```

**Step 5: Commit**

```bash
git add aria/hub/config_defaults.py tests/
git commit -m "feat(iw): add 20 iw.* config entries with layman/technical descriptions (P4-03)"
```

---

**Run quality gate after Batch 1:**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/ --timeout=120 -x -q
```

---

## Batch 2: Discovery Engine (P4-04)

### Task 4: Discovery engine

**Files:**
- Create: `aria/iw/discovery.py`
- Create: `tests/iw/test_discovery.py`

**Step 1: Write discovery tests**

Test the three-stage pipeline:
- `test_discover_from_patterns` — mock `patterns.detect_patterns()` returning enriched patterns with `entity_chain`, `trigger_entity`, `area`, `day_type`, `confidence`. Assert `BehavioralStateDefinition` created with correct trigger/confirming indicators.
- `test_discover_from_gap_analyzer` — mock `anomaly_gap.analyze_gaps()` returning `DetectionResult` objects. Assert definitions created for solo toggles.
- `test_indicator_chain_construction` — verify trigger = first entity, confirming = remaining with `max_delay_seconds` from co_occurrence timing.
- `test_precondition_quiet_period` — when events show >4h gap before trigger, assert `trigger_preconditions` includes `quiet_period` indicator.
- `test_deduplication_merge` — two patterns with >60% indicator overlap → merged into one definition (higher confidence wins).
- `test_deduplication_distinct` — two patterns with <60% overlap → two separate definitions.
- `test_deterministic_id` — same indicators in same area/day_type → same ID regardless of run order.
- `test_empty_patterns` — no patterns → no definitions, no error.

**Step 2: Implement DiscoveryEngine**

Create `aria/iw/discovery.py` with:
- `DiscoveryEngine.__init__(self, hub)` — stores hub reference
- `async def discover(self) -> list[BehavioralStateDefinition]` — runs 3-stage pipeline
- Stage 1: `_gather_sources()` — calls `patterns.detect_patterns()` and `gap_analyzer.analyze_gaps()`
- Stage 2: `_build_indicator_chains(patterns, gaps)` — constructs definitions
- Stage 3: `_deduplicate(new_definitions, existing_definitions)` — merge or create

The discovery engine reads config values via `hub.cache.get_config_value()`. Uses `co_occurrence.compute_adaptive_window()` for timing. Uses `event_normalizer` filtering for noise exclusion.

**Step 3: Run tests, commit**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_discovery.py -v --timeout=60
git add aria/iw/discovery.py tests/iw/test_discovery.py
git commit -m "feat(iw): add discovery engine — patterns + gaps → indicator chains with dedup/merge (P4-04)"
```

---

**Run quality gate after Batch 2.**

---

## Batch 3: Real-Time Detector + Cold Start (P4-05, P4-06)

### Task 5: Real-time detector

**Files:**
- Create: `aria/iw/detector.py`
- Create: `tests/iw/test_detector.py`

**Step 1: Write detector tests**

- `test_trigger_creates_active_state` — state_changed event matching a trigger indicator → ActiveState created
- `test_confirming_updates_active_state` — subsequent event matching confirming indicator → `matched_confirming` updated
- `test_window_expiry_records_observation` — ActiveState past window with sufficient match_ratio → observation recorded in tracker
- `test_window_expiry_discards_low_match` — ActiveState past window with low match_ratio → discarded, no observation
- `test_entity_index_lookup` — events for non-indexed entities are ignored (O(1) skip)
- `test_person_departure_terminates` — person.X goes away → person-attributed ActiveStates terminated
- `test_domain_filter` — detector only subscribes to domains present in definitions
- `test_multiple_active_states` — two overlapping definitions can be active simultaneously
- `test_definition_refresh` — after `refresh_definitions()`, new definitions are picked up

**Step 2: Implement IWDetector**

Create `aria/iw/detector.py`:
- `class IWDetector(Module)` with `module_id = "iw_detector"`
- `CAPABILITIES` list with appropriate Capability declaration
- `initialize()`: load definitions from store, build entity_index, build domain set, subscribe to state_changed, start timer
- `shutdown()`: unsubscribe callback (store ref on self — lesson #37), cancel timer
- `_on_state_changed(data)`: entity_index lookup, trigger/confirming/person handling
- `_check_expiry()`: timer callback, evaluate and expire ActiveStates
- `refresh_definitions()`: reload from store, rebuild indexes

**Step 3: Run tests, commit**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_detector.py -v --timeout=60
git add aria/iw/detector.py tests/iw/test_detector.py
git commit -m "feat(iw): add real-time detector — event subscriber with entity-indexed O(1) lookup (P4-05)"
```

### Task 6: Cold-start replay (P4-06)

**Files:**
- Modify: `aria/iw/detector.py` — add `_cold_start_replay()` to `initialize()`
- Modify: `tests/iw/test_detector.py` — add cold-start tests

**Step 1: Write cold-start tests**

- `test_cold_start_replays_recent_events` — detector initializes with events in EventStore from last N minutes → ActiveStates reconstructed
- `test_cold_start_empty_store` — no events → no error, no ActiveStates
- `test_cold_start_respects_config` — uses `iw.cold_start_replay_minutes` config value

**Step 2: Implement cold-start replay**

In `initialize()`, after loading definitions and building indexes, call `_cold_start_replay()`:
- Query EventStore for events in last `cold_start_replay_minutes`
- Feed each event through `_on_state_changed()` in timestamp order
- Log count of replayed events and any ActiveStates created

**Step 3: Run tests, commit**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_detector.py -k cold_start -v --timeout=30
git add aria/iw/detector.py tests/iw/test_detector.py
git commit -m "feat(iw): add cold-start replay from EventStore on detector init (P4-06)"
```

---

**Run quality gate after Batch 3.**

---

## Batch 4: Lifecycle Manager (P4-07)

### Task 7: Lifecycle manager

**Files:**
- Create: `aria/iw/lifecycle.py`
- Create: `tests/iw/test_lifecycle.py`

**Step 1: Write lifecycle tests**

- `test_seed_to_emerging` — 7+ observations, ≥0.60 consistency, ≥0.3 density → promotes
- `test_seed_stays_below_threshold` — 5 observations → stays at seed
- `test_emerging_to_confirmed_requires_backtest` — meets count/consistency but no backtest → stays emerging
- `test_confirmed_to_mature_by_user_approval` — user approved → promote regardless of count
- `test_confirmed_to_mature_by_observation` — 30+ observations, ≥0.80 consistency → promote
- `test_dormancy_detection` — <3 observations in 30 active days → dormant
- `test_dormancy_excludes_vacation` — vacation days not counted in dormancy window (mock day_classifier)
- `test_revived_from_dormant` — 3+ new observations while dormant → emerging
- `test_retired_after_90_days_dormant` — 90 days dormant → retired
- `test_rejection_penalty_first` — first automation rejection → +10% threshold increase
- `test_rejection_penalty_second` — second rejection → demote to seed
- `test_density_check` — 15 observations over 60 days with only 10 matching-day-type days → density 1.5 (passes) vs 15 over 60 days with 50 matching days → density 0.3 (fails for confirmed)
- `test_lifecycle_history_recorded` — each transition appends to history

**Step 2: Implement LifecycleManager**

Create `aria/iw/lifecycle.py`:
- `LifecycleManager.__init__(self, hub, store)` — hub for config, store for persistence
- `async def evaluate(self, tracker) -> str | None` — returns new lifecycle stage or None
- `_check_promotion(tracker)` — promotion rules with density check
- `_check_demotion(tracker)` — dormancy with vacation exclusion via `day_classifier.classify_days()`
- `_compute_density(tracker, day_types)` — observations / active-day-type-days in window
- `_apply_rejection_penalty(tracker)` — threshold adjustment
- `async def evaluate_and_persist(self, tracker)` — evaluate + save + record history

**Step 3: Run tests, commit**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_lifecycle.py -v --timeout=30
git add aria/iw/lifecycle.py tests/iw/test_lifecycle.py
git commit -m "feat(iw): add lifecycle manager — promotion/demotion with density checks and vacation exclusion (P4-07)"
```

---

**Run quality gate after Batch 4.**

---

## Batch 5: Backtest Engine (P4-08)

### Task 8: Backtest engine

**Files:**
- Create: `aria/iw/backtest.py`
- Create: `tests/iw/test_backtest.py`

**Step 1: Write backtest tests**

- `test_historical_replay_detects_known_pattern` — inject events with a known pattern, replay → F1 > 0
- `test_historical_replay_no_false_positives` — inject events with no matching pattern → precision = 0, recall = 0
- `test_holdout_stratified_split` — verify train/test sets have proportional day_type representation
- `test_holdout_consistency_drift` — stable pattern → low drift score; drifting pattern → high drift
- `test_counterfactual_uses_template_engine` — verify Phase 3 template_engine.generate() called for each outcome
- `test_pass_criteria_all_must_pass` — fail one criterion → overall FAIL
- `test_adaptive_f1_threshold` — state with 0.80 consistency → F1 threshold = max(0.65, 0.70) = 0.70
- `test_backtest_result_stored` — after run, tracker.backtest_result populated

**Step 2: Implement BacktestEngine**

Create `aria/iw/backtest.py`:
- `BacktestEngine.__init__(self, hub, store)` — references EventStore, gap_analyzer for ground truth
- `async def run_backtest(self, definition, tracker) -> dict` — runs all 3 tests, returns results dict
- `_historical_replay(definition, events)` — replay through fresh detector
- `_holdout_validation(definition, events, day_classifier)` — stratified split + consistency comparison
- `_counterfactual_test(definition, events)` — generate automation YAML, simulate against events
- `_evaluate_pass(results, consistency)` — adaptive threshold check

**Step 3: Run tests, commit**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_backtest.py -v --timeout=60
git add aria/iw/backtest.py tests/iw/test_backtest.py
git commit -m "feat(iw): add backtest engine — historical replay, holdout validation, counterfactual test (P4-08)"
```

---

**Run quality gate after Batch 5.**

---

## Batch 6: Composite Detection (P4-09)

### Task 9: Composite state detection

**Files:**
- Create: `aria/iw/composite.py`
- Create: `tests/iw/test_composite.py`

**Step 1: Write composite tests**

- `test_co_activation_tracking` — two states active in overlapping windows → co-activation recorded
- `test_composite_proposed` — co-activation ≥ 5 and rate ≥ 60% → composite candidate
- `test_composite_with_ordering` — discovery runs on activation events → composite has consistent ordering
- `test_max_composites_pruning` — exceeding max_composites → oldest/lowest-confidence pruned
- `test_composite_enters_seed` — new composite starts at seed lifecycle
- `test_no_composite_below_threshold` — co-activation < 5 → no composite

**Step 2: Implement CompositeDetector**

Create `aria/iw/composite.py`:
- `CompositeDetector.__init__(self, hub, store)` — hub for config/event_store, store for persistence
- `async def check_co_activations(self, activated_state_id, active_states)` — record co-activations
- `async def propose_composites(self) -> list[BehavioralStateDefinition]` — check thresholds, cluster, build
- `_prune_excess(definitions)` — enforce max_composites

**Step 3: Run tests, commit**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_composite.py -v --timeout=30
git add aria/iw/composite.py tests/iw/test_composite.py
git commit -m "feat(iw): add composite state detection — co-activation tracking with LRU pruning (P4-09)"
```

---

**Run quality gate after Batch 6.**

---

## Batch 7: Synthetic Testing (P4-10, P4-11)

### Task 10: Event simulator

**Files:**
- Create: `aria/iw/synthetic.py`
- Create: `tests/iw/test_synthetic.py`

**Step 1: Write simulator tests**

- `test_inject_pattern_basic` — inject 20 repeats at 80% consistency → ~16 occurrences in output events
- `test_gaussian_jitter` — timing varies around mean (stddev check)
- `test_noise_profiles` — random, periodic, bursty noise all produce events between pattern instances
- `test_day_type_filtering` — workday-only pattern → events only on simulated workdays
- `test_empty_pattern` — 0 repeats → only noise events

**Step 2: Implement EventSimulator**

Create `aria/iw/synthetic.py`:
- `EventSimulator` class
- `inject_pattern()` → generates events with Gaussian jitter, consistency probability, noise
- `_generate_noise(profile, count)` — three noise modes
- Uses temporary EventStore for isolation

### Task 11: Hyperparameter sweep

**Step 1: Write sweep tests**

- `test_sweep_returns_results` — sweep over 3 values → 3 result dicts with metric scores
- `test_sweep_includes_baseline` — current production config always in results
- `test_sweep_isolation` — each run uses fresh detector (no cross-contamination)

**Step 2: Implement HyperparameterSweep**

Add to `aria/iw/synthetic.py`:
- `HyperparameterSweep.__init__(self, hub)` — hub for config access
- `async def sweep(param_key, values, event_source, quality_metric)` → results with baseline comparison

**Step 3: Run tests, commit**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_synthetic.py -v --timeout=60
git add aria/iw/synthetic.py tests/iw/test_synthetic.py
git commit -m "feat(iw): add synthetic event simulator and hyperparameter sweep framework (P4-10, P4-11)"
```

---

**Run quality gate after Batch 7.**

---

## Batch 8: Hub Integration (P4-12)

### Task 12: Wire into hub core

**Files:**
- Modify: `aria/hub/core.py` — import and register IWDetector, schedule discovery, add behavioral_states cache
- Create: `tests/iw/test_hub_integration.py`

**Step 1: Write integration tests**

- `test_iw_detector_registered` — after hub start, `hub.modules["iw_detector"]` exists
- `test_discovery_scheduled` — discovery task scheduled on configurable interval
- `test_lifecycle_triggered_on_observation` — when detector records observation → lifecycle manager evaluates
- `test_automation_generated_on_confirmed` — when state reaches confirmed and backtest passes → automation suggestion created via Phase 3 generator
- `test_behavioral_states_cache` — hub cache has `behavioral_states` key with current states

**Step 2: Implement hub wiring**

In `aria/hub/core.py`:
- Import `IWDetector` and `BehavioralStateStore`
- In `_initialize_modules()` or equivalent: create `BehavioralStateStore`, initialize it on hub.db connection, create `IWDetector(hub)`, register it
- Schedule discovery: `hub.schedule_task("iw_discovery", discovery_engine.discover, interval=timedelta(hours=config_interval))`
- Wire detector observation callback → lifecycle evaluate → automation generator (for confirmed states)
- Add `behavioral_states` cache key updated after each lifecycle evaluation

**Important patterns to follow:**
- Subscribe lifecycle: `initialize()` not `__init__()` (lesson #28, #37)
- Store callback ref on `self` for `shutdown()` unsubscribe
- Domain filter before async work (lesson #39)
- `create_task` done_callback for error visibility (lesson #43)

**Step 3: Run tests, commit**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_hub_integration.py -v --timeout=60
git add aria/hub/core.py aria/iw/ tests/iw/test_hub_integration.py
git commit -m "feat(iw): wire detector + discovery + lifecycle into hub core (P4-12)"
```

---

**Run quality gate after Batch 8.**

---

## Batch 9: API Endpoints (P4-13)

### Task 13: API endpoints

**Files:**
- Create: `aria/iw/routes.py` (or add `_register_iw_routes` to `aria/hub/api.py`)
- Modify: `aria/hub/api.py` — register IW routes
- Create: `tests/iw/test_api.py`

**Step 1: Write API tests**

- `test_list_behavioral_states` — GET /api/behavioral-states → list of states with lifecycle
- `test_list_with_lifecycle_filter` — GET /api/behavioral-states?lifecycle=confirmed → filtered
- `test_get_behavioral_state` — GET /api/behavioral-states/{id} → definition + tracker detail
- `test_get_active_states` — GET /api/behavioral-states/active → current ActiveStates
- `test_trigger_backtest` — POST /api/behavioral-states/{id}/backtest → runs backtest, returns result
- `test_submit_feedback` — POST /api/behavioral-states/{id}/feedback → approve/reject, updates tracker
- `test_not_found` — GET /api/behavioral-states/nonexistent → 404

**Step 2: Implement routes**

Create `_register_iw_routes(router, hub)` following existing pattern (see `_register_event_store_routes` as template):
- GET `/api/behavioral-states` — query store, optional lifecycle filter
- GET `/api/behavioral-states/active` — read detector active_states
- GET `/api/behavioral-states/{state_id}` — definition + tracker joined
- POST `/api/behavioral-states/{state_id}/backtest` — trigger backtest engine
- POST `/api/behavioral-states/{state_id}/feedback` — body `{"feedback": "approved"|"rejected"}`

Register in `create_app()`: `_register_iw_routes(router, hub)`

**Step 3: Run tests, commit**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/iw/test_api.py -v --timeout=30
git add aria/iw/routes.py aria/hub/api.py tests/iw/test_api.py
git commit -m "feat(iw): add behavioral state API endpoints — list, detail, active, backtest, feedback (P4-13)"
```

---

**Run quality gate after Batch 9.**

---

## Batch 10: Integration Test + Regression (P4-14, P4-15)

### Task 14: End-to-end integration test

**Files:**
- Create: `tests/integration/test_iw_flow.py`

**Step 1: Write integration test**

End-to-end flow:
1. Create EventSimulator, inject a known pattern (bedroom_motion → bathroom_motion → kitchen_motion, 20 repeats, 80% consistency)
2. Write events to test EventStore
3. Run DiscoveryEngine → assert ≥1 BehavioralStateDefinition created
4. Feed events through IWDetector → assert observations recorded
5. Run LifecycleManager → assert state promoted from seed toward emerging
6. Assert BehavioralStateTracker has correct observation count and consistency
7. Assert behavioral_states cache populated

**Step 2: Run integration test**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/integration/test_iw_flow.py -v --timeout=120
```

**Step 3: Commit**

```bash
git add tests/integration/test_iw_flow.py
git commit -m "test(iw): add end-to-end integration test — discovery→detector→lifecycle flow (P4-14)"
```

### Task 15: Full regression

**Step 1: Run full suite**

```bash
cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/ --timeout=120 -x -q
```

Expected: all tests pass (current 2089 + new ~100+ IW tests).

**Step 2: Fix any regressions**

Common issues:
- Config defaults count assertions (update from 148 → 168)
- Module count assertions if any test checks registered module count
- Import ordering (ruff will catch)

**Step 3: Final commit**

```bash
git add -A
git commit -m "test: full regression passes with Phase 4 I&W framework (P4-15)"
```

---

## Final Integration Wiring Batch (Batch 11)

Per Code Factory rule (#53): plans with 3+ batches MUST include a final integration wiring batch.

### Task: Verify all components wire together

**Step 1: Vertical trace**

Start hub → discovery runs → definitions created → events flow → detector activates → lifecycle evaluates → cache updated → API returns data.

Mock HA WebSocket in test, inject events, trace through every layer.

**Step 2: Bottom-up + top-down verification (lesson #56)**

- Bottom-up: each unit test suite passes independently
- Top-down: integration test traces one behavioral state from discovery to API response

**Step 3: Update progress.txt with final status**

Append completion summary with test counts, files created, PRD task pass status.

---

## Summary

| Batch | Tasks | PRD IDs | Estimated Tests |
|-------|-------|---------|-----------------|
| 1 | Models + Store + Config | P4-01, P4-02, P4-03 | ~25 |
| 2 | Discovery Engine | P4-04 | ~10 |
| 3 | Detector + Cold Start | P4-05, P4-06 | ~15 |
| 4 | Lifecycle Manager | P4-07 | ~15 |
| 5 | Backtest Engine | P4-08 | ~10 |
| 6 | Composite Detection | P4-09 | ~8 |
| 7 | Synthetic Testing | P4-10, P4-11 | ~10 |
| 8 | Hub Integration | P4-12 | ~8 |
| 9 | API Endpoints | P4-13 | ~8 |
| 10 | Integration + Regression | P4-14, P4-15 | ~5 |
| 11 | Final Wiring Verification | — | ~3 |
| **Total** | **15 tasks, 11 batches** | **P4-01 – P4-15** | **~117** |
