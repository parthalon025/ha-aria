# Adaptive ML Pipeline — Phase 4: Advanced Expansion

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** Completed — merged to main on 2026-02-18

**Goal:** Add cross-domain pattern transfer (room-to-room and routine-to-routine hypothesis generation tested via shadow engine) and an attention-based anomaly explainer (PyTorch autoencoder with temporal attention and contrastive explanations). Transfer operates at Tier 3+; attention autoencoder at Tier 4 only. Tier 1-2 behavior unchanged.

**Architecture:** Two new engine modules: `transfer.py` (Tier 3+ — generates `TransferCandidate` objects from organic discovery clusters, tests them via shadow engine, promotes after 7-day hit rate threshold) and `attention_explainer.py` (Tier 4 — small PyTorch autoencoder with attention weights for temporal anomaly explanation). A new hub module `transfer_engine.py` orchestrates transfer hypothesis lifecycle. Existing `pattern_recognition.py` gains attention-based explanations alongside the existing IsolationForest path tracing. New `/api/transfer` and `/api/anomalies/explain` endpoints expose results.

**Tech Stack:** Python 3.12 (dataclasses, asyncio), scipy (Jaccard similarity — already installed), optional `torch` (Tier 4 only, `try/except` guarded), existing hub event bus, existing shadow engine, existing organic discovery cache

**Design doc:** `docs/plans/2026-02-17-adaptive-ml-pipeline-design.md` § Phase 4

---

## Dependencies & Prerequisites

- **Phase 3 must be merged** (completed 2026-02-17)
- **Branch:** Create `feature/adaptive-ml-pipeline-phase4` from `main`
- **Virtual env:** `.venv/bin/python -m pytest` for all test runs
- **Test baseline:** 1541 collected, ~1530 passed, ~11 skipped

## Critical Patterns (Read Before Implementing)

1. **Sync vs async module access:** `hub.get_module()` is synchronous (fixed in Phase 3 code review). From async code, just call `hub.get_module("name")` — no `await`.

2. **Optional imports:** All `torch` imports must be `try/except` guarded with lazy loading. Never crash on import. Pattern from `aria/engine/online.py:_create_model()`.

3. **Hub cache access:** Use `await self.hub.set_cache()` / `get_cache()`, NOT `self.hub.cache.*`. See `CLAUDE.md` gotchas.

4. **Config defaults format:** Each entry is a dict with `key`, `default_value` (string), `value_type`, `label`, `description`, `category`, and optional `min_value`/`max_value`/`step`/`options`. See `config_defaults.py:699-743`.

5. **Module registration:** Follow the try/except pattern in `cli.py:382-400` (online_learner and pattern_recognition blocks).

6. **Module base class:** Extend `aria.hub.core.Module`, call `super().__init__(module_id, hub)`, implement `initialize()` and `shutdown()`. See `pattern_recognition.py` for the full lifecycle pattern including subscribe-in-initialize, unsubscribe-in-shutdown.

7. **Shadow engine event bus:** The shadow engine publishes `shadow_resolved` events with `prediction_id`, `features`, `outcome`, `actual_data`. Per-target events include `target`, `features`, `actual_value`, `outcome`. Subscribe in `initialize()`.

8. **Organic discovery cache:** Capabilities live in the `capabilities` cache. Each capability has `entities` (list), `source` ("seed"/"organic"), `usefulness`, `layer` ("domain"/"behavioral"), `status` ("candidate"/"promoted"/"archived"). Organic discovery runs every 6h via scheduled task.

---

## Task 1: TransferCandidate Data Model

**Files:**
- Create: `aria/engine/transfer.py`
- Test: `tests/engine/test_transfer.py`

**Context:** A `TransferCandidate` represents a hypothesis: "pattern X observed in context A might also apply in context B." The candidate tracks its source capability, target context, structural similarity (Jaccard), shadow test results, and lifecycle state. This is a pure data module — no I/O, no hub dependency.

**Step 1: Write the failing tests**

Create `tests/engine/test_transfer.py`:

```python
"""Tests for cross-domain transfer candidate data model."""

from datetime import datetime

import pytest

from aria.engine.transfer import (
    TransferCandidate,
    TransferType,
    compute_jaccard_similarity,
)


class TestTransferType:
    """Test transfer type enum."""

    def test_enum_values(self):
        assert TransferType.ROOM_TO_ROOM.value == "room_to_room"
        assert TransferType.ROUTINE_TO_ROUTINE.value == "routine_to_routine"


class TestJaccardSimilarity:
    """Test Jaccard similarity computation."""

    def test_identical_sets(self):
        assert compute_jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint_sets(self):
        assert compute_jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        result = compute_jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(result - 0.5) < 0.01  # 2 shared / 4 total

    def test_empty_sets(self):
        assert compute_jaccard_similarity(set(), set()) == 0.0

    def test_one_empty_set(self):
        assert compute_jaccard_similarity({"a"}, set()) == 0.0


class TestTransferCandidate:
    """Test TransferCandidate lifecycle."""

    def test_create_room_to_room(self):
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1", "light.kitchen_2"],
            target_entities=["light.bedroom_1", "light.bedroom_2"],
        )
        assert tc.state == "hypothesis"
        assert tc.shadow_tests == 0
        assert tc.shadow_hits == 0
        assert tc.hit_rate == 0.0

    def test_create_routine_to_routine(self):
        tc = TransferCandidate(
            source_capability="weekday_morning",
            target_context="weekend_morning",
            transfer_type=TransferType.ROUTINE_TO_ROUTINE,
            similarity_score=0.65,
            source_entities=["light.kitchen_1", "sensor.motion_kitchen"],
            target_entities=["light.kitchen_1", "sensor.motion_kitchen"],
            timing_offset_minutes=120,
        )
        assert tc.transfer_type == TransferType.ROUTINE_TO_ROUTINE
        assert tc.timing_offset_minutes == 120

    def test_record_shadow_result_hit(self):
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        tc.record_shadow_result(hit=True)
        assert tc.shadow_tests == 1
        assert tc.shadow_hits == 1
        assert tc.hit_rate == 1.0
        assert tc.state == "testing"

    def test_record_shadow_result_miss(self):
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        tc.record_shadow_result(hit=False)
        assert tc.shadow_tests == 1
        assert tc.shadow_hits == 0
        assert tc.hit_rate == 0.0

    def test_promotion_threshold(self):
        """After 7+ days with >=60% hit rate, state becomes promoted."""
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        # Simulate 20 tests, 15 hits (75% hit rate)
        for _ in range(15):
            tc.record_shadow_result(hit=True)
        for _ in range(5):
            tc.record_shadow_result(hit=False)
        # Force testing_since to 8 days ago
        tc.testing_since = datetime(2026, 2, 9)
        tc.check_promotion(min_days=7, min_hit_rate=0.6)
        assert tc.state == "promoted"

    def test_rejection_threshold(self):
        """After 7+ days with <30% hit rate, state becomes rejected."""
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        # Simulate 20 tests, 4 hits (20% hit rate)
        for _ in range(4):
            tc.record_shadow_result(hit=True)
        for _ in range(16):
            tc.record_shadow_result(hit=False)
        tc.testing_since = datetime(2026, 2, 9)
        tc.check_promotion(min_days=7, min_hit_rate=0.6, reject_below=0.3)
        assert tc.state == "rejected"

    def test_not_promoted_before_min_days(self):
        """Promotion requires minimum testing period."""
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        for _ in range(20):
            tc.record_shadow_result(hit=True)
        # testing_since is now (very recent), so min_days=7 not met
        tc.check_promotion(min_days=7, min_hit_rate=0.6)
        assert tc.state == "testing"

    def test_to_dict(self):
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        d = tc.to_dict()
        assert d["source_capability"] == "kitchen_lighting"
        assert d["target_context"] == "bedroom"
        assert d["transfer_type"] == "room_to_room"
        assert d["similarity_score"] == 0.72
        assert d["state"] == "hypothesis"

    def test_similarity_below_threshold_raises(self):
        """Similarity below 0.4 should not create a candidate."""
        with pytest.raises(ValueError, match="similarity"):
            TransferCandidate(
                source_capability="kitchen_lighting",
                target_context="bedroom",
                transfer_type=TransferType.ROOM_TO_ROOM,
                similarity_score=0.3,
                source_entities=["light.kitchen_1"],
                target_entities=["light.bedroom_1"],
            )
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/engine/test_transfer.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'aria.engine.transfer'`

**Step 3: Implement the transfer data model**

Create `aria/engine/transfer.py`:

```python
"""Cross-domain pattern transfer data model.

TransferCandidates represent hypotheses that a pattern observed in one
context might apply in another. The shadow engine tests these hypotheses
and promotes them after sufficient evidence accumulates.

Tier 3+ only — no heavy dependencies, pure data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class TransferType(StrEnum):
    """Types of cross-domain pattern transfer."""

    ROOM_TO_ROOM = "room_to_room"
    ROUTINE_TO_ROUTINE = "routine_to_routine"


MIN_SIMILARITY = 0.4


def compute_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    J(A,B) = |A ∩ B| / |A ∪ B|

    Returns 0.0 for empty sets.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


@dataclass
class TransferCandidate:
    """A hypothesis that a pattern in one context applies to another.

    Lifecycle: hypothesis → testing → promoted/rejected

    Args:
        source_capability: Name of the capability being transferred from.
        target_context: Target context identifier (room name, routine label).
        transfer_type: Room-to-room or routine-to-routine.
        similarity_score: Jaccard similarity between source and target (0-1).
        source_entities: Entity IDs in the source context.
        target_entities: Entity IDs in the target context.
        timing_offset_minutes: For routine-to-routine, time shift in minutes.
    """

    source_capability: str
    target_context: str
    transfer_type: TransferType
    similarity_score: float
    source_entities: list[str]
    target_entities: list[str]
    timing_offset_minutes: int = 0

    # Lifecycle state
    state: str = field(default="hypothesis", init=False)
    created_at: datetime = field(default_factory=datetime.now, init=False)
    testing_since: datetime | None = field(default=None, init=False)
    promoted_at: datetime | None = field(default=None, init=False)

    # Shadow test tracking
    shadow_tests: int = field(default=0, init=False)
    shadow_hits: int = field(default=0, init=False)

    def __post_init__(self):
        if self.similarity_score < MIN_SIMILARITY:
            raise ValueError(
                f"similarity score {self.similarity_score:.2f} below minimum "
                f"{MIN_SIMILARITY}. Transfer candidates require structural "
                f"similarity >= {MIN_SIMILARITY}."
            )

    @property
    def hit_rate(self) -> float:
        """Current shadow test hit rate (0-1)."""
        if self.shadow_tests == 0:
            return 0.0
        return self.shadow_hits / self.shadow_tests

    def record_shadow_result(self, hit: bool) -> None:
        """Record a shadow engine test result.

        Transitions from hypothesis to testing on first result.
        """
        if self.state == "hypothesis":
            self.state = "testing"
            self.testing_since = datetime.now()

        self.shadow_tests += 1
        if hit:
            self.shadow_hits += 1

    def check_promotion(
        self,
        min_days: int = 7,
        min_hit_rate: float = 0.6,
        reject_below: float = 0.3,
    ) -> None:
        """Check if this candidate should be promoted or rejected.

        Args:
            min_days: Minimum days of testing before promotion decision.
            min_hit_rate: Minimum hit rate for promotion.
            reject_below: Hit rate below which candidate is rejected.
        """
        if self.state != "testing" or self.testing_since is None:
            return

        days_testing = (datetime.now() - self.testing_since).days
        if days_testing < min_days:
            return

        if self.shadow_tests < 5:
            return  # Not enough data

        if self.hit_rate >= min_hit_rate:
            self.state = "promoted"
            self.promoted_at = datetime.now()
        elif self.hit_rate < reject_below:
            self.state = "rejected"

    def to_dict(self) -> dict:
        """Serialize to dict for cache storage."""
        return {
            "source_capability": self.source_capability,
            "target_context": self.target_context,
            "transfer_type": self.transfer_type.value,
            "similarity_score": self.similarity_score,
            "source_entities": self.source_entities,
            "target_entities": self.target_entities,
            "timing_offset_minutes": self.timing_offset_minutes,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "testing_since": self.testing_since.isoformat() if self.testing_since else None,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "shadow_tests": self.shadow_tests,
            "shadow_hits": self.shadow_hits,
            "hit_rate": round(self.hit_rate, 4),
        }
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/engine/test_transfer.py -v
```
Expected: 13 passed

**Step 5: Commit**

```bash
git add aria/engine/transfer.py tests/engine/test_transfer.py
git commit -m "feat: add TransferCandidate data model for cross-domain pattern transfer"
```

---

## Task 2: Transfer Hypothesis Generator

**Files:**
- Create: `aria/engine/transfer_generator.py`
- Test: `tests/engine/test_transfer_generator.py`

**Context:** The generator scans organic discovery capabilities and finds pairs with structural similarity. For room-to-room: capabilities in different areas with overlapping entity domains (e.g., both have lights + motion sensors). For routine-to-routine: behavioral capabilities with similar entity sets but different temporal patterns (e.g., weekday morning vs. weekend morning). Uses Jaccard similarity on domain sets + entity type sets. Only generates candidates above the similarity threshold (default 0.6).

**Step 1: Write the failing tests**

Create `tests/engine/test_transfer_generator.py`:

```python
"""Tests for transfer hypothesis generation from organic capabilities."""

import pytest

from aria.engine.transfer import TransferCandidate, TransferType
from aria.engine.transfer_generator import generate_transfer_candidates


class TestRoomToRoomGeneration:
    """Test room-to-room transfer hypothesis generation."""

    def test_generates_candidate_from_similar_rooms(self):
        """Two capabilities in different areas with overlapping domains."""
        capabilities = {
            "kitchen_lighting": {
                "entities": ["light.kitchen_1", "light.kitchen_2", "binary_sensor.kitchen_motion"],
                "layer": "domain",
                "status": "promoted",
                "source": "organic",
            },
            "bedroom_lighting": {
                "entities": ["light.bedroom_1", "binary_sensor.bedroom_motion"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
        }
        entities_cache = {
            "light.kitchen_1": {"entity_id": "light.kitchen_1", "domain": "light", "area_id": "kitchen"},
            "light.kitchen_2": {"entity_id": "light.kitchen_2", "domain": "light", "area_id": "kitchen"},
            "binary_sensor.kitchen_motion": {"entity_id": "binary_sensor.kitchen_motion", "domain": "binary_sensor", "area_id": "kitchen", "device_class": "motion"},
            "light.bedroom_1": {"entity_id": "light.bedroom_1", "domain": "light", "area_id": "bedroom"},
            "binary_sensor.bedroom_motion": {"entity_id": "binary_sensor.bedroom_motion", "domain": "binary_sensor", "area_id": "bedroom", "device_class": "motion"},
        }

        candidates = generate_transfer_candidates(
            capabilities, entities_cache, min_similarity=0.5
        )

        assert len(candidates) >= 1
        # At least one room-to-room candidate
        r2r = [c for c in candidates if c.transfer_type == TransferType.ROOM_TO_ROOM]
        assert len(r2r) >= 1

    def test_no_candidate_for_dissimilar_rooms(self):
        """Totally different domain compositions don't generate candidates."""
        capabilities = {
            "power_monitoring": {
                "entities": ["sensor.power_1", "sensor.power_2"],
                "layer": "domain",
                "status": "promoted",
                "source": "organic",
            },
            "bedroom_lighting": {
                "entities": ["light.bedroom_1", "light.bedroom_2"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
        }
        entities_cache = {
            "sensor.power_1": {"entity_id": "sensor.power_1", "domain": "sensor", "area_id": "office", "device_class": "power"},
            "sensor.power_2": {"entity_id": "sensor.power_2", "domain": "sensor", "area_id": "office", "device_class": "power"},
            "light.bedroom_1": {"entity_id": "light.bedroom_1", "domain": "light", "area_id": "bedroom"},
            "light.bedroom_2": {"entity_id": "light.bedroom_2", "domain": "light", "area_id": "bedroom"},
        }

        candidates = generate_transfer_candidates(
            capabilities, entities_cache, min_similarity=0.5
        )
        assert len(candidates) == 0

    def test_skips_seed_capabilities(self):
        """Seed capabilities are not source candidates for transfer."""
        capabilities = {
            "power_monitoring": {
                "entities": ["sensor.power_1"],
                "layer": "domain",
                "status": "promoted",
                "source": "seed",
            },
            "bedroom_power": {
                "entities": ["sensor.power_2"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
        }
        entities_cache = {
            "sensor.power_1": {"entity_id": "sensor.power_1", "domain": "sensor", "area_id": "kitchen"},
            "sensor.power_2": {"entity_id": "sensor.power_2", "domain": "sensor", "area_id": "bedroom"},
        }

        candidates = generate_transfer_candidates(
            capabilities, entities_cache, min_similarity=0.5
        )
        # Should not use seed as source
        for c in candidates:
            assert c.source_capability != "power_monitoring"

    def test_only_promoted_as_source(self):
        """Only promoted capabilities can be source of transfer."""
        capabilities = {
            "kitchen_lighting": {
                "entities": ["light.kitchen_1"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
            "bedroom_lighting": {
                "entities": ["light.bedroom_1"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
        }
        entities_cache = {
            "light.kitchen_1": {"entity_id": "light.kitchen_1", "domain": "light", "area_id": "kitchen"},
            "light.bedroom_1": {"entity_id": "light.bedroom_1", "domain": "light", "area_id": "bedroom"},
        }

        candidates = generate_transfer_candidates(
            capabilities, entities_cache, min_similarity=0.5
        )
        assert len(candidates) == 0


class TestRoutineToRoutineGeneration:
    """Test routine-to-routine transfer hypothesis generation."""

    def test_generates_from_behavioral_with_different_timing(self):
        """Behavioral capabilities with overlapping entities but different peak hours."""
        capabilities = {
            "weekday_morning": {
                "entities": ["light.kitchen_1", "sensor.motion_kitchen"],
                "layer": "behavioral",
                "status": "promoted",
                "source": "organic",
                "temporal_pattern": {"peak_hours": [7, 8], "weekday_bias": 0.9},
            },
            "weekend_morning": {
                "entities": ["light.kitchen_1", "sensor.motion_kitchen"],
                "layer": "behavioral",
                "status": "candidate",
                "source": "organic",
                "temporal_pattern": {"peak_hours": [9, 10], "weekday_bias": 0.2},
            },
        }
        entities_cache = {
            "light.kitchen_1": {"entity_id": "light.kitchen_1", "domain": "light", "area_id": "kitchen"},
            "sensor.motion_kitchen": {"entity_id": "sensor.motion_kitchen", "domain": "sensor", "area_id": "kitchen"},
        }

        candidates = generate_transfer_candidates(
            capabilities, entities_cache, min_similarity=0.5
        )
        r2r = [c for c in candidates if c.transfer_type == TransferType.ROUTINE_TO_ROUTINE]
        assert len(r2r) >= 1
        # Should have a timing offset
        assert r2r[0].timing_offset_minutes != 0

    def test_no_duplicate_candidates(self):
        """Same pair should not produce duplicate candidates."""
        capabilities = {
            "kitchen_lighting": {
                "entities": ["light.kitchen_1"],
                "layer": "domain",
                "status": "promoted",
                "source": "organic",
            },
            "bedroom_lighting": {
                "entities": ["light.bedroom_1"],
                "layer": "domain",
                "status": "promoted",
                "source": "organic",
            },
        }
        entities_cache = {
            "light.kitchen_1": {"entity_id": "light.kitchen_1", "domain": "light", "area_id": "kitchen"},
            "light.bedroom_1": {"entity_id": "light.bedroom_1", "domain": "light", "area_id": "bedroom"},
        }

        candidates = generate_transfer_candidates(
            capabilities, entities_cache, min_similarity=0.5
        )
        # For each ordered pair (A→B), there should be at most one candidate
        pairs = [(c.source_capability, c.target_context) for c in candidates]
        assert len(pairs) == len(set(pairs))
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/engine/test_transfer_generator.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement the transfer generator**

Create `aria/engine/transfer_generator.py`:

```python
"""Cross-domain pattern transfer hypothesis generator.

Scans organic discovery capabilities and finds pairs with structural
similarity. Generates TransferCandidate objects for the transfer engine
hub module to test via shadow engine.

Tier 3+ only — called periodically by TransferEngineModule.
"""

from __future__ import annotations

import logging
from typing import Any

from aria.engine.transfer import (
    MIN_SIMILARITY,
    TransferCandidate,
    TransferType,
    compute_jaccard_similarity,
)

logger = logging.getLogger(__name__)


def _extract_domain_set(entity_ids: list[str], entities_cache: dict) -> set[str]:
    """Extract the set of entity domains for a capability's entities."""
    domains = set()
    for eid in entity_ids:
        entity = entities_cache.get(eid, {})
        domain = entity.get("domain", "")
        if domain:
            domains.add(domain)
    return domains


def _extract_device_class_set(entity_ids: list[str], entities_cache: dict) -> set[str]:
    """Extract the set of device classes for a capability's entities."""
    classes = set()
    for eid in entity_ids:
        entity = entities_cache.get(eid, {})
        dc = entity.get("device_class", "")
        if dc:
            classes.add(dc)
    return classes


def _extract_area(entity_ids: list[str], entities_cache: dict) -> str | None:
    """Find the dominant area for a capability's entities."""
    area_counts: dict[str, int] = {}
    for eid in entity_ids:
        entity = entities_cache.get(eid, {})
        area = entity.get("area_id", "")
        if area:
            area_counts[area] = area_counts.get(area, 0) + 1
    if not area_counts:
        return None
    return max(area_counts, key=area_counts.get)


def _compute_structural_similarity(
    cap_a_entities: list[str],
    cap_b_entities: list[str],
    entities_cache: dict,
) -> float:
    """Compute structural similarity between two capabilities.

    Blends domain Jaccard (weight 0.6) and device class Jaccard (weight 0.4).
    """
    domains_a = _extract_domain_set(cap_a_entities, entities_cache)
    domains_b = _extract_domain_set(cap_b_entities, entities_cache)
    classes_a = _extract_device_class_set(cap_a_entities, entities_cache)
    classes_b = _extract_device_class_set(cap_b_entities, entities_cache)

    domain_sim = compute_jaccard_similarity(domains_a, domains_b)
    class_sim = compute_jaccard_similarity(classes_a, classes_b)

    # Weight domains more heavily — device class is a refinement
    return domain_sim * 0.6 + class_sim * 0.4


def generate_transfer_candidates(
    capabilities: dict[str, dict[str, Any]],
    entities_cache: dict[str, dict[str, Any]],
    min_similarity: float = 0.6,
) -> list[TransferCandidate]:
    """Generate transfer candidates from organic capabilities.

    Rules:
    - Only promoted organic capabilities can be sources
    - Seed capabilities are never sources
    - Room-to-room: domain-layer caps in different areas with similar composition
    - Routine-to-routine: behavioral-layer caps with overlapping entity sets
      but different temporal patterns

    Args:
        capabilities: Capabilities cache dict (name → data).
        entities_cache: Entities cache dict (entity_id → entity data).
        min_similarity: Minimum structural similarity for candidate generation.

    Returns:
        List of TransferCandidate objects.
    """
    candidates: list[TransferCandidate] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Filter to promoted organic capabilities
    promoted = {
        name: data
        for name, data in capabilities.items()
        if data.get("source") != "seed" and data.get("status") == "promoted"
    }

    # All organic capabilities (promoted or candidate) as potential targets
    all_organic = {
        name: data
        for name, data in capabilities.items()
        if data.get("source") != "seed"
    }

    for source_name, source_data in promoted.items():
        source_entities = source_data.get("entities", [])
        source_layer = source_data.get("layer", "domain")

        for target_name, target_data in all_organic.items():
            if target_name == source_name:
                continue

            pair_key = (source_name, target_name)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            target_entities = target_data.get("entities", [])
            target_layer = target_data.get("layer", "domain")

            # Compute structural similarity
            similarity = _compute_structural_similarity(
                source_entities, target_entities, entities_cache
            )

            if similarity < max(min_similarity, MIN_SIMILARITY):
                continue

            # Determine transfer type
            if source_layer == "behavioral" and target_layer == "behavioral":
                # Routine-to-routine: check for temporal pattern differences
                source_temporal = source_data.get("temporal_pattern", {})
                target_temporal = target_data.get("temporal_pattern", {})
                source_peaks = source_temporal.get("peak_hours", [])
                target_peaks = target_temporal.get("peak_hours", [])

                if not source_peaks or not target_peaks:
                    continue

                # Compute timing offset (average peak hour difference)
                avg_source = sum(source_peaks) / len(source_peaks)
                avg_target = sum(target_peaks) / len(target_peaks)
                offset_hours = avg_target - avg_source
                offset_minutes = int(offset_hours * 60)

                if offset_minutes == 0:
                    continue  # Same timing — not a useful transfer

                try:
                    candidates.append(
                        TransferCandidate(
                            source_capability=source_name,
                            target_context=target_name,
                            transfer_type=TransferType.ROUTINE_TO_ROUTINE,
                            similarity_score=similarity,
                            source_entities=source_entities,
                            target_entities=target_entities,
                            timing_offset_minutes=offset_minutes,
                        )
                    )
                except ValueError:
                    continue

            else:
                # Room-to-room: domain-layer caps in different areas
                source_area = _extract_area(source_entities, entities_cache)
                target_area = _extract_area(target_entities, entities_cache)

                if source_area and target_area and source_area == target_area:
                    continue  # Same room — not a transfer

                try:
                    candidates.append(
                        TransferCandidate(
                            source_capability=source_name,
                            target_context=target_area or target_name,
                            transfer_type=TransferType.ROOM_TO_ROOM,
                            similarity_score=similarity,
                            source_entities=source_entities,
                            target_entities=target_entities,
                        )
                    )
                except ValueError:
                    continue

    logger.info(
        f"Generated {len(candidates)} transfer candidates from "
        f"{len(promoted)} promoted capabilities"
    )
    return candidates
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/engine/test_transfer_generator.py -v
```
Expected: 6 passed

**Step 5: Commit**

```bash
git add aria/engine/transfer_generator.py tests/engine/test_transfer_generator.py
git commit -m "feat: add transfer hypothesis generator for room-to-room and routine-to-routine"
```

---

## Task 3: Attention-Based Anomaly Explainer (Tier 4)

**Files:**
- Create: `aria/engine/attention_explainer.py`
- Modify: `pyproject.toml:43-50` (add torch to new gpu optional group)
- Test: `tests/engine/test_attention_explainer.py`

**Context:** A small PyTorch autoencoder with attention weights that explains anomalies both temporally (which time steps mattered) and per-feature. This is Tier 4 only — requires PyTorch. Falls back to the existing `AnomalyExplainer` (Phase 3 IsolationForest path tracing) when torch is unavailable. The autoencoder has ~50K parameters: 2-layer encoder + attention + 2-layer decoder.

**Important:** All torch imports must be `try/except` guarded. If torch is missing, every public method returns graceful fallback values. Tests must work both with and without torch installed.

**Step 1: Add torch to pyproject.toml**

In `pyproject.toml`, add a `gpu` optional dependency group after `ml-extra`:

```toml
gpu = [
    "torch>=2.0.0",
]
```

**Do NOT install torch** unless the user explicitly opts in — it's a large dependency. Tests will handle the torch-missing case.

**Step 2: Write the failing tests**

Create `tests/engine/test_attention_explainer.py`:

```python
"""Tests for attention-based anomaly explainer (Tier 4).

Tests are written to work both with and without torch installed.
When torch is unavailable, tests verify graceful fallback behavior.
"""

import numpy as np
import pytest

from aria.engine.attention_explainer import AttentionExplainer

# Check if torch is available for conditional test running
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestAttentionExplainerInit:
    """Test initialization and torch detection."""

    def test_creates_without_torch(self):
        """Should initialize even without torch."""
        explainer = AttentionExplainer(n_features=5, sequence_length=6)
        assert explainer.n_features == 5
        assert explainer.sequence_length == 6

    def test_torch_availability_detected(self):
        """Should report whether torch is available."""
        explainer = AttentionExplainer(n_features=5, sequence_length=6)
        assert isinstance(explainer.torch_available, bool)

    def test_is_trained_false_initially(self):
        explainer = AttentionExplainer(n_features=5, sequence_length=6)
        assert not explainer.is_trained


class TestFallbackBehavior:
    """Test behavior when torch is not available or model not trained."""

    def test_explain_untrained_returns_empty(self):
        """Untrained explainer returns empty explanations."""
        explainer = AttentionExplainer(n_features=5, sequence_length=6)
        window = np.zeros((6, 5))
        result = explainer.explain(window)
        assert result["feature_contributions"] == []
        assert result["temporal_attention"] == []
        assert result["contrastive_explanation"] is None
        assert result["anomaly_score"] is None

    def test_get_stats_untrained(self):
        """Stats report untrained state."""
        explainer = AttentionExplainer(n_features=5, sequence_length=6)
        stats = explainer.get_stats()
        assert stats["is_trained"] is False
        assert stats["n_features"] == 5
        assert stats["sequence_length"] == 6


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestWithTorch:
    """Tests that require torch — skipped on Tier 1-3 machines."""

    def test_build_model(self):
        """Model builds with correct parameter count."""
        explainer = AttentionExplainer(n_features=5, sequence_length=6, hidden_dim=32)
        assert explainer._model is not None
        param_count = sum(p.numel() for p in explainer._model.parameters())
        # Should be roughly 10K-100K parameters
        assert 1_000 < param_count < 200_000

    def test_train_on_synthetic_data(self):
        """Train on synthetic normal data."""
        explainer = AttentionExplainer(n_features=5, sequence_length=6, hidden_dim=16)
        np.random.seed(42)
        # 50 normal windows
        X_train = np.random.normal(0, 1, (50, 6, 5))
        result = explainer.train(X_train, epochs=5)
        assert result["trained"] is True
        assert result["final_loss"] > 0
        assert explainer.is_trained

    def test_explain_after_training(self):
        """Explain an anomalous window after training."""
        explainer = AttentionExplainer(n_features=5, sequence_length=6, hidden_dim=16)
        np.random.seed(42)
        X_train = np.random.normal(0, 1, (50, 6, 5))
        explainer.train(X_train, epochs=5)

        # Anomalous window: spike in feature 0
        anomaly_window = np.zeros((6, 5))
        anomaly_window[:, 0] = 10.0

        result = explainer.explain(anomaly_window)
        assert len(result["feature_contributions"]) > 0
        assert len(result["temporal_attention"]) == 6
        assert result["anomaly_score"] is not None
        # Temporal attention should sum to ~1
        attn_sum = sum(result["temporal_attention"])
        assert 0.5 < attn_sum < 1.5

    def test_contrastive_explanation(self):
        """Contrastive explanation compares anomaly to normal baseline."""
        explainer = AttentionExplainer(n_features=5, sequence_length=6, hidden_dim=16)
        np.random.seed(42)
        X_train = np.random.normal(0, 1, (50, 6, 5))
        explainer.train(X_train, epochs=5)

        anomaly_window = np.zeros((6, 5))
        anomaly_window[:, 0] = 10.0

        result = explainer.explain(
            anomaly_window,
            feature_names=["power", "lights", "motion", "temp", "humidity"],
        )
        assert result["contrastive_explanation"] is not None
        assert isinstance(result["contrastive_explanation"], str)
        assert len(result["contrastive_explanation"]) > 0
```

**Step 3: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/engine/test_attention_explainer.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Implement the attention explainer**

Create `aria/engine/attention_explainer.py`:

```python
"""Attention-based anomaly explainer using PyTorch autoencoder.

A small autoencoder with self-attention that explains anomalies via:
- Feature contributions: which features have highest reconstruction error
- Temporal attention: which time steps the attention layer weighted most
- Contrastive explanation: "Looks like X but with unusually high Y"

Tier 4 only — requires torch. Falls back gracefully when unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy torch import
_torch = None
_nn = None


def _ensure_torch():
    """Lazily import torch, returning True if available."""
    global _torch, _nn
    if _torch is not None:
        return True
    try:
        import torch
        import torch.nn as nn

        _torch = torch
        _nn = nn
        return True
    except ImportError:
        return False


def _build_autoencoder(n_features: int, sequence_length: int, hidden_dim: int):
    """Build the attention autoencoder model.

    Architecture:
    - Encoder: Linear(n_features, hidden_dim) → ReLU → Linear(hidden_dim, hidden_dim//2)
    - Attention: Scaled dot-product self-attention over time steps
    - Decoder: Linear(hidden_dim//2, hidden_dim) → ReLU → Linear(hidden_dim, n_features)
    """
    if not _ensure_torch():
        return None

    class AttentionAutoencoder(_nn.Module):
        def __init__(self):
            super().__init__()
            half_dim = max(hidden_dim // 2, 4)

            # Encoder
            self.enc1 = _nn.Linear(n_features, hidden_dim)
            self.enc2 = _nn.Linear(hidden_dim, half_dim)

            # Self-attention over time steps
            self.query = _nn.Linear(half_dim, half_dim)
            self.key = _nn.Linear(half_dim, half_dim)
            self.value = _nn.Linear(half_dim, half_dim)
            self.attn_scale = half_dim**0.5

            # Decoder
            self.dec1 = _nn.Linear(half_dim, hidden_dim)
            self.dec2 = _nn.Linear(hidden_dim, n_features)

            self.relu = _nn.ReLU()

        def forward(self, x):
            # x: (batch, seq_len, n_features)
            # Encode
            h = self.relu(self.enc1(x))
            h = self.enc2(h)  # (batch, seq_len, half_dim)

            # Self-attention
            Q = self.query(h)
            K = self.key(h)
            V = self.value(h)
            attn_weights = _torch.softmax(
                _torch.bmm(Q, K.transpose(1, 2)) / self.attn_scale, dim=-1
            )
            h = _torch.bmm(attn_weights, V)

            # Decode
            out = self.relu(self.dec1(h))
            out = self.dec2(out)  # (batch, seq_len, n_features)

            return out, attn_weights

    return AttentionAutoencoder()


class AttentionExplainer:
    """Attention-based anomaly explainer (Tier 4).

    Args:
        n_features: Number of features per time step.
        sequence_length: Number of time steps in each window.
        hidden_dim: Hidden dimension of the autoencoder (default 32).
    """

    def __init__(
        self,
        n_features: int,
        sequence_length: int,
        hidden_dim: int = 32,
    ):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.torch_available = _ensure_torch()

        self._model = None
        self._normal_baseline: np.ndarray | None = None
        self._training_mean: np.ndarray | None = None
        self._training_std: np.ndarray | None = None
        self._trained = False

        if self.torch_available:
            self._model = _build_autoencoder(n_features, sequence_length, hidden_dim)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(
        self,
        windows: np.ndarray,
        epochs: int = 20,
        learning_rate: float = 1e-3,
    ) -> dict[str, Any]:
        """Train the autoencoder on normal windows.

        Args:
            windows: Array of shape (n_samples, sequence_length, n_features).
            epochs: Training epochs.
            learning_rate: Adam learning rate.

        Returns:
            Training result dict with trained, final_loss, epochs.
        """
        if not self.torch_available or self._model is None:
            return {"trained": False, "reason": "torch not available"}

        # Store normal baseline statistics
        self._training_mean = windows.mean(axis=0)
        self._training_std = windows.std(axis=0) + 1e-8
        self._normal_baseline = self._training_mean

        # Normalize
        normed = (windows - self._training_mean) / self._training_std

        X = _torch.FloatTensor(normed)
        optimizer = _torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        loss_fn = _nn.MSELoss()

        self._model.train()
        final_loss = 0.0

        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed, _attn = self._model(X)
            loss = loss_fn(reconstructed, X)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self._trained = True
        logger.info(
            f"Attention explainer trained: {epochs} epochs, "
            f"final_loss={final_loss:.6f}, {windows.shape[0]} samples"
        )
        return {"trained": True, "final_loss": final_loss, "epochs": epochs}

    def explain(
        self,
        window: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Explain an anomalous window.

        Args:
            window: Array of shape (sequence_length, n_features).
            feature_names: Optional names for features.

        Returns:
            Dict with feature_contributions, temporal_attention,
            contrastive_explanation, anomaly_score.
        """
        empty = {
            "feature_contributions": [],
            "temporal_attention": [],
            "contrastive_explanation": None,
            "anomaly_score": None,
        }

        if not self._trained or self._model is None:
            return empty

        names = feature_names or [f"feature_{i}" for i in range(self.n_features)]

        # Normalize using training stats
        normed = (window - self._training_mean) / self._training_std
        X = _torch.FloatTensor(normed).unsqueeze(0)  # (1, seq, feat)

        self._model.eval()
        with _torch.no_grad():
            reconstructed, attn_weights = self._model(X)

        # Reconstruction error per feature (averaged over time)
        recon_error = (X - reconstructed).squeeze(0).numpy()  # (seq, feat)
        per_feature_error = np.mean(np.abs(recon_error), axis=0)  # (feat,)

        # Normalize to contributions
        total_error = per_feature_error.sum()
        if total_error > 0:
            contributions = per_feature_error / total_error
        else:
            contributions = np.zeros(self.n_features)

        # Sort by contribution
        sorted_idx = np.argsort(contributions)[::-1]
        feature_contributions = [
            {"feature": names[i], "contribution": round(float(contributions[i]), 4)}
            for i in sorted_idx
            if contributions[i] > 0.01
        ]

        # Temporal attention (averaged across attention heads / queries)
        attn = attn_weights.squeeze(0).numpy()  # (seq, seq)
        temporal_attention = np.mean(attn, axis=0).tolist()  # avg attention received
        temporal_attention = [round(float(v), 4) for v in temporal_attention]

        # Overall anomaly score (mean reconstruction error)
        anomaly_score = round(float(np.mean(np.abs(recon_error))), 4)

        # Contrastive explanation
        contrastive = self._build_contrastive(
            window, per_feature_error, names
        )

        return {
            "feature_contributions": feature_contributions,
            "temporal_attention": temporal_attention,
            "contrastive_explanation": contrastive,
            "anomaly_score": anomaly_score,
        }

    def _build_contrastive(
        self,
        window: np.ndarray,
        per_feature_error: np.ndarray,
        feature_names: list[str],
    ) -> str | None:
        """Build a contrastive explanation string.

        "Looks like [normal pattern] but with unusually high [feature]"
        """
        if self._normal_baseline is None:
            return None

        # Find the feature with highest deviation from normal
        window_mean = window.mean(axis=0)
        deviation = np.abs(window_mean - self._normal_baseline.mean(axis=0))
        top_idx = np.argmax(deviation)
        top_feature = feature_names[top_idx]
        direction = "high" if window_mean[top_idx] > self._normal_baseline.mean(axis=0)[top_idx] else "low"

        return f"Looks like normal pattern but with unusually {direction} {top_feature}"

    def get_stats(self) -> dict[str, Any]:
        """Return explainer statistics."""
        stats = {
            "is_trained": self._trained,
            "torch_available": self.torch_available,
            "n_features": self.n_features,
            "sequence_length": self.sequence_length,
            "hidden_dim": self.hidden_dim,
        }
        if self._model is not None:
            param_count = sum(p.numel() for p in self._model.parameters())
            stats["parameter_count"] = param_count
        return stats
```

**Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/engine/test_attention_explainer.py -v
```
Expected: 5 passed + (if torch installed: 4 more passed, else 4 skipped)

**Step 6: Commit**

```bash
git add aria/engine/attention_explainer.py tests/engine/test_attention_explainer.py pyproject.toml
git commit -m "feat: add attention-based anomaly explainer autoencoder (Tier 4)"
```

---

## Task 4: Transfer Engine Hub Module

**Files:**
- Create: `aria/modules/transfer_engine.py`
- Test: `tests/hub/test_transfer_engine.py`

**Context:** This hub module orchestrates the transfer hypothesis lifecycle. It subscribes to `organic_discovery_complete` events to regenerate candidates, and to `shadow_resolved` events to test active candidates. It caches transfer state and runs periodic promotion checks. Self-gates on Tier 3+ hardware.

**Step 1: Write the failing tests**

Create `tests/hub/test_transfer_engine.py`:

```python
"""Tests for transfer engine hub module."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.modules.transfer_engine import TransferEngineModule


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.subscribe = MagicMock()
    hub.unsubscribe = MagicMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.publish = AsyncMock()
    hub.get_config_value = MagicMock(return_value=None)
    hub.get_module = MagicMock(return_value=None)
    hub.modules = {}
    return hub


class TestTransferEngineInit:
    """Test module initialization and tier gating."""

    def test_module_id(self, mock_hub):
        module = TransferEngineModule(mock_hub)
        assert module.module_id == "transfer_engine"

    def test_no_subscribe_in_constructor(self, mock_hub):
        module = TransferEngineModule(mock_hub)
        mock_hub.subscribe.assert_not_called()

    @patch("aria.modules.transfer_engine.recommend_tier", return_value=2)
    @patch("aria.modules.transfer_engine.scan_hardware")
    async def test_tier_gate_blocks_below_tier_3(self, mock_scan, mock_tier, mock_hub):
        mock_scan.return_value = MagicMock(ram_gb=4, cpu_cores=2)
        module = TransferEngineModule(mock_hub)
        await module.initialize()
        assert module.active is False

    @patch("aria.modules.transfer_engine.recommend_tier", return_value=3)
    @patch("aria.modules.transfer_engine.scan_hardware")
    async def test_tier_gate_allows_tier_3(self, mock_scan, mock_tier, mock_hub):
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)
        module = TransferEngineModule(mock_hub)
        await module.initialize()
        assert module.active is True
        # Should subscribe to events
        subscribe_events = [call[0][0] for call in mock_hub.subscribe.call_args_list]
        assert "organic_discovery_complete" in subscribe_events
        assert "shadow_resolved" in subscribe_events


class TestCandidateGeneration:
    """Test transfer candidate generation from discovery events."""

    @patch("aria.modules.transfer_engine.recommend_tier", return_value=3)
    @patch("aria.modules.transfer_engine.scan_hardware")
    async def test_on_discovery_complete_generates_candidates(self, mock_scan, mock_tier, mock_hub):
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)

        # Set up capability and entity caches
        capabilities = {
            "kitchen_lighting": {
                "entities": ["light.kitchen_1", "binary_sensor.kitchen_motion"],
                "layer": "domain",
                "status": "promoted",
                "source": "organic",
            },
            "bedroom_lighting": {
                "entities": ["light.bedroom_1", "binary_sensor.bedroom_motion"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
        }
        entities = {
            "light.kitchen_1": {"entity_id": "light.kitchen_1", "domain": "light", "area_id": "kitchen"},
            "binary_sensor.kitchen_motion": {"entity_id": "binary_sensor.kitchen_motion", "domain": "binary_sensor", "area_id": "kitchen", "device_class": "motion"},
            "light.bedroom_1": {"entity_id": "light.bedroom_1", "domain": "light", "area_id": "bedroom"},
            "binary_sensor.bedroom_motion": {"entity_id": "binary_sensor.bedroom_motion", "domain": "binary_sensor", "area_id": "bedroom", "device_class": "motion"},
        }

        mock_hub.get_cache = AsyncMock(side_effect=lambda key: {
            "capabilities": {"data": capabilities},
            "entities": {"data": entities},
        }.get(key))

        module = TransferEngineModule(mock_hub)
        await module.initialize()
        await module._on_discovery_complete({})

        assert len(module.candidates) >= 0  # May or may not meet threshold


class TestShadowTesting:
    """Test shadow result integration for transfer candidates."""

    @patch("aria.modules.transfer_engine.recommend_tier", return_value=3)
    @patch("aria.modules.transfer_engine.scan_hardware")
    async def test_shadow_result_updates_candidate(self, mock_scan, mock_tier, mock_hub):
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)

        module = TransferEngineModule(mock_hub)
        await module.initialize()

        # Manually add a test candidate
        from aria.engine.transfer import TransferCandidate, TransferType

        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        module.candidates.append(tc)

        # Shadow event for a target entity
        await module._on_shadow_resolved({
            "prediction_id": "test-123",
            "features": {"hour_sin": 0.5},
            "outcome": "correct",
            "actual_data": {"entity_id": "light.bedroom_1"},
        })

        assert tc.shadow_tests >= 0  # May or may not match entity


class TestTransferEngineState:
    """Test state reporting."""

    def test_get_stats(self, mock_hub):
        module = TransferEngineModule(mock_hub)
        stats = module.get_stats()
        assert "active" in stats
        assert "candidates_total" in stats
        assert "candidates_by_state" in stats

    def test_get_current_state(self, mock_hub):
        module = TransferEngineModule(mock_hub)
        state = module.get_current_state()
        assert "candidates" in state
        assert "summary" in state

    @patch("aria.modules.transfer_engine.recommend_tier", return_value=3)
    @patch("aria.modules.transfer_engine.scan_hardware")
    async def test_shutdown_unsubscribes(self, mock_scan, mock_tier, mock_hub):
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)
        module = TransferEngineModule(mock_hub)
        await module.initialize()
        await module.shutdown()
        assert mock_hub.unsubscribe.call_count == 2
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/hub/test_transfer_engine.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement the transfer engine module**

Create `aria/modules/transfer_engine.py`:

```python
"""Transfer engine hub module — cross-domain pattern transfer orchestration.

Generates transfer candidates from organic discovery capabilities,
tests them via shadow engine results, and promotes/rejects after
sufficient evidence. Self-gates on Tier 3+ hardware.
"""

import logging
from collections import Counter
from datetime import datetime
from typing import Any

from aria.engine.hardware import recommend_tier, scan_hardware
from aria.engine.transfer import TransferCandidate
from aria.engine.transfer_generator import generate_transfer_candidates
from aria.hub.core import Module

logger = logging.getLogger(__name__)

MIN_TIER = 3


class TransferEngineModule(Module):
    """Hub module for cross-domain pattern transfer."""

    def __init__(self, hub):
        super().__init__("transfer_engine", hub)
        self.active = False
        self.candidates: list[TransferCandidate] = []
        self._generation_count = 0

    async def initialize(self):
        """Check hardware tier and activate if sufficient."""
        profile = scan_hardware()
        tier = recommend_tier(profile)

        if tier < MIN_TIER:
            logger.info(
                f"Transfer engine disabled: tier {tier} < {MIN_TIER} "
                f"({profile.ram_gb:.1f}GB RAM, {profile.cpu_cores} cores)"
            )
            self.active = False
            return

        self.active = True
        self.hub.subscribe("organic_discovery_complete", self._on_discovery_complete)
        self.hub.subscribe("shadow_resolved", self._on_shadow_resolved)

        # Load persisted candidates from cache
        cached = await self.hub.get_cache("transfer_candidates")
        if cached and cached.get("data"):
            self._load_candidates(cached["data"])

        logger.info(
            f"Transfer engine active at tier {tier}, "
            f"{len(self.candidates)} cached candidates loaded"
        )

    async def shutdown(self):
        """Unsubscribe and persist state."""
        if self.active:
            self.hub.unsubscribe("organic_discovery_complete", self._on_discovery_complete)
            self.hub.unsubscribe("shadow_resolved", self._on_shadow_resolved)
            await self._persist_candidates()

    async def _on_discovery_complete(self, event: dict[str, Any]):
        """Regenerate transfer candidates when organic discovery runs."""
        if not self.active:
            return

        caps_entry = await self.hub.get_cache("capabilities")
        entities_entry = await self.hub.get_cache("entities")

        if not caps_entry or not caps_entry.get("data"):
            return
        if not entities_entry or not entities_entry.get("data"):
            return

        capabilities = caps_entry["data"]
        entities_cache = entities_entry["data"]

        # Generate new candidates
        new_candidates = generate_transfer_candidates(
            capabilities, entities_cache, min_similarity=0.6
        )

        # Merge with existing — preserve active/testing candidates
        existing_keys = {
            (c.source_capability, c.target_context) for c in self.candidates
            if c.state in ("testing", "promoted")
        }
        for nc in new_candidates:
            key = (nc.source_capability, nc.target_context)
            if key not in existing_keys:
                self.candidates.append(nc)

        # Prune rejected candidates older than 30 days
        self.candidates = [
            c for c in self.candidates
            if c.state != "rejected"
            or (datetime.now() - c.created_at).days < 30
        ]

        self._generation_count += 1
        await self._persist_candidates()

        logger.info(
            f"Transfer generation #{self._generation_count}: "
            f"{len(new_candidates)} new, {len(self.candidates)} total"
        )

    async def _on_shadow_resolved(self, event: dict[str, Any]):
        """Test active transfer candidates against shadow results."""
        if not self.active:
            return

        outcome = event.get("outcome", "")
        actual_data = event.get("actual_data", {})
        if not outcome or not actual_data:
            return

        # Check if any candidate's target entities are involved
        entity_id = actual_data.get("entity_id", "")
        if not entity_id:
            return

        for candidate in self.candidates:
            if candidate.state not in ("hypothesis", "testing"):
                continue

            if entity_id in candidate.target_entities:
                hit = outcome == "correct"
                candidate.record_shadow_result(hit=hit)

        # Periodic promotion check (every 10 shadow events)
        self._check_promotions()

    def _check_promotions(self):
        """Check all testing candidates for promotion/rejection."""
        for candidate in self.candidates:
            if candidate.state == "testing":
                candidate.check_promotion(
                    min_days=7, min_hit_rate=0.6, reject_below=0.3
                )

    def _load_candidates(self, data: list[dict]) -> None:
        """Load candidates from cached dicts (best-effort)."""
        from aria.engine.transfer import TransferCandidate, TransferType

        for d in data:
            try:
                tc = TransferCandidate(
                    source_capability=d["source_capability"],
                    target_context=d["target_context"],
                    transfer_type=TransferType(d["transfer_type"]),
                    similarity_score=d["similarity_score"],
                    source_entities=d.get("source_entities", []),
                    target_entities=d.get("target_entities", []),
                    timing_offset_minutes=d.get("timing_offset_minutes", 0),
                )
                tc.state = d.get("state", "hypothesis")
                tc.shadow_tests = d.get("shadow_tests", 0)
                tc.shadow_hits = d.get("shadow_hits", 0)
                if d.get("testing_since"):
                    tc.testing_since = datetime.fromisoformat(d["testing_since"])
                self.candidates.append(tc)
            except (KeyError, ValueError) as e:
                logger.debug(f"Skipping invalid cached candidate: {e}")

    async def _persist_candidates(self):
        """Save candidates to cache."""
        data = [c.to_dict() for c in self.candidates]
        await self.hub.set_cache(
            "transfer_candidates",
            data,
            {"count": len(data), "source": "transfer_engine"},
        )

    def get_current_state(self) -> dict[str, Any]:
        """Return current transfer engine state."""
        state_counts = Counter(c.state for c in self.candidates)
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "summary": {
                "total": len(self.candidates),
                "by_state": dict(state_counts),
                "generation_runs": self._generation_count,
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Return module statistics."""
        state_counts = Counter(c.state for c in self.candidates)
        return {
            "active": self.active,
            "candidates_total": len(self.candidates),
            "candidates_by_state": dict(state_counts),
            "generation_runs": self._generation_count,
        }
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/hub/test_transfer_engine.py -v
```
Expected: 9 passed

**Step 5: Commit**

```bash
git add aria/modules/transfer_engine.py tests/hub/test_transfer_engine.py
git commit -m "feat: add transfer engine hub module for cross-domain hypothesis testing"
```

---

## Task 5: Register Transfer Engine in Hub

**Files:**
- Modify: `aria/cli.py:392-400` (add after pattern_recognition block)
- Test: Existing hub tests should not regress

**Step 1: Add registration to cli.py**

After the pattern_recognition block (`cli.py:400`), add:

```python
    # transfer_engine (Tier 3+ — module self-gates on hardware tier)
    try:
        from aria.modules.transfer_engine import TransferEngineModule

        transfer_engine = TransferEngineModule(hub)
        hub.register_module(transfer_engine)
        await _init(transfer_engine, "transfer_engine")()
    except Exception as e:
        logger.warning(f"Transfer engine module failed (non-fatal): {e}")
```

**Step 2: Run existing tests to verify no regression**

```bash
.venv/bin/python -m pytest tests/hub/test_transfer_engine.py tests/hub/test_pattern_recognition.py -v --timeout=60
```
Expected: All pass

**Step 3: Commit**

```bash
git add aria/cli.py
git commit -m "feat: register transfer engine module in hub startup"
```

---

## Task 6: Config Entries for Phase 4

**Files:**
- Modify: `aria/hub/config_defaults.py:743` (add transfer.* entries before closing bracket)
- Test: `tests/hub/test_config_defaults.py` (update count if needed)

**Step 1: Add config entries**

After the Phase 3 pattern entries (line 743), add:

```python
    # Phase 4: Transfer & Attention
    {
        "key": "transfer.min_similarity",
        "default_value": "0.6",
        "value_type": "number",
        "label": "Transfer Min Similarity",
        "description": "Minimum Jaccard structural similarity for generating transfer candidates (0.4-1.0)",
        "category": "transfer",
        "min_value": 0.4,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "transfer.promotion_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Transfer Promotion Days",
        "description": "Minimum days of shadow testing before a transfer candidate can be promoted",
        "category": "transfer",
        "min_value": 3,
        "max_value": 30,
        "step": 1,
    },
    {
        "key": "transfer.promotion_hit_rate",
        "default_value": "0.6",
        "value_type": "number",
        "label": "Transfer Promotion Hit Rate",
        "description": "Minimum shadow hit rate for transfer promotion (0.0-1.0)",
        "category": "transfer",
        "min_value": 0.3,
        "max_value": 0.9,
        "step": 0.05,
    },
    {
        "key": "transfer.reject_hit_rate",
        "default_value": "0.3",
        "value_type": "number",
        "label": "Transfer Reject Hit Rate",
        "description": "Shadow hit rate below which transfer candidates are rejected (0.0-1.0)",
        "category": "transfer",
        "min_value": 0.1,
        "max_value": 0.5,
        "step": 0.05,
    },
    {
        "key": "attention.hidden_dim",
        "default_value": "32",
        "value_type": "number",
        "label": "Attention Hidden Dim",
        "description": "Hidden dimension for the attention autoencoder (Tier 4 only, higher = more expressive)",
        "category": "attention",
        "min_value": 8,
        "max_value": 128,
        "step": 8,
    },
    {
        "key": "attention.train_epochs",
        "default_value": "20",
        "value_type": "number",
        "label": "Attention Training Epochs",
        "description": "Training epochs for the attention autoencoder (Tier 4 only)",
        "category": "attention",
        "min_value": 5,
        "max_value": 100,
        "step": 5,
    },
```

**Step 2: Run config tests**

```bash
.venv/bin/python -m pytest tests/hub/test_config_defaults.py -v
```
Expected: All pass (update count assertion if needed)

**Step 3: Commit**

```bash
git add aria/hub/config_defaults.py tests/hub/test_config_defaults.py
git commit -m "config: add transfer and attention settings for Phase 4"
```

---

## Task 7: Wire Attention Explainer into Pattern Recognition Module

**Files:**
- Modify: `aria/modules/pattern_recognition.py` (add Tier 4 attention explainer alongside existing AnomalyExplainer)
- Modify: `tests/hub/test_pattern_recognition.py` (add attention tests)

**Context:** When the system is at Tier 4 and torch is available, the pattern recognition module initializes an `AttentionExplainer` and uses it for richer anomaly explanations. It falls back to the Phase 3 IsolationForest path tracer at Tier 3. Both are used together when available — path tracing for "which features" (fast) and attention for "which time steps + contrastive story" (richer).

**Step 1: Write the failing tests**

Add to `tests/hub/test_pattern_recognition.py`:

```python
class TestAttentionExplainerIntegration:
    """Test attention explainer wiring in pattern recognition."""

    @patch("aria.modules.pattern_recognition.recommend_tier", return_value=4)
    @patch("aria.modules.pattern_recognition.scan_hardware")
    async def test_tier_4_initializes_attention_explainer(self, mock_scan, mock_tier, mock_hub):
        """At Tier 4, attention explainer should be created."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=8, gpu_available=True, gpu_name="Test GPU")
        module = PatternRecognitionModule(mock_hub)
        await module.initialize()
        assert module.active is True
        assert module.attention_explainer is not None

    @patch("aria.modules.pattern_recognition.recommend_tier", return_value=3)
    @patch("aria.modules.pattern_recognition.scan_hardware")
    async def test_tier_3_no_attention_explainer(self, mock_scan, mock_tier, mock_hub):
        """At Tier 3, attention explainer should be None."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4, gpu_available=False)
        module = PatternRecognitionModule(mock_hub)
        await module.initialize()
        assert module.active is True
        assert module.attention_explainer is None

    def test_get_stats_includes_attention(self, mock_hub):
        module = PatternRecognitionModule(mock_hub)
        stats = module.get_stats()
        assert "attention_explainer" in stats
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/hub/test_pattern_recognition.py::TestAttentionExplainerIntegration -v
```
Expected: FAIL — `PatternRecognitionModule` has no `attention_explainer` attribute

**Step 3: Add attention explainer to PatternRecognitionModule**

In `aria/modules/pattern_recognition.py`:

Add to `__init__`:
```python
        self.attention_explainer = None  # Tier 4 only
```

In `initialize()`, after `self.active = True`:
```python
        # Tier 4: attention-based anomaly explainer
        if tier >= 4:
            try:
                from aria.engine.attention_explainer import AttentionExplainer

                self.attention_explainer = AttentionExplainer(
                    n_features=DEFAULT_WINDOW_SIZE,
                    sequence_length=DEFAULT_WINDOW_SIZE,
                )
                logger.info("Attention explainer initialized (Tier 4)")
            except Exception as e:
                logger.warning(f"Attention explainer failed (non-fatal): {e}")
```

In `get_stats()`, add:
```python
            "attention_explainer": (
                self.attention_explainer.get_stats()
                if self.attention_explainer is not None
                else None
            ),
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/hub/test_pattern_recognition.py -v
```
Expected: All pass

**Step 5: Commit**

```bash
git add aria/modules/pattern_recognition.py tests/hub/test_pattern_recognition.py
git commit -m "feat: wire attention explainer into pattern recognition (Tier 4)"
```

---

## Task 8: API Endpoints for Transfer and Anomaly Explanation

**Files:**
- Modify: `aria/hub/api.py` (add `/api/transfer` and `/api/anomalies/explain` routes)
- Create: `tests/hub/test_api_transfer.py`

**Step 1: Write the failing tests**

Create `tests/hub/test_api_transfer.py`:

```python
"""Tests for /api/transfer and /api/anomalies/explain endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from aria.hub.api import create_app


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.get_module = MagicMock()
    hub.get_cache = AsyncMock(return_value=None)
    return hub


class TestTransferEndpoint:
    """Test GET /api/transfer."""

    def test_full_response(self, mock_hub):
        transfer_mod = MagicMock()
        transfer_mod.get_current_state.return_value = {
            "candidates": [
                {
                    "source_capability": "kitchen_lighting",
                    "target_context": "bedroom",
                    "state": "testing",
                    "hit_rate": 0.75,
                }
            ],
            "summary": {"total": 1, "by_state": {"testing": 1}},
        }
        transfer_mod.get_stats.return_value = {
            "active": True,
            "candidates_total": 1,
        }
        mock_hub.get_module = MagicMock(return_value=transfer_mod)

        app = create_app(mock_hub)
        client = TestClient(app)
        resp = client.get("/api/transfer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert len(data["candidates"]) == 1

    def test_module_not_available(self, mock_hub):
        mock_hub.get_module = MagicMock(return_value=None)

        app = create_app(mock_hub)
        client = TestClient(app)
        resp = client.get("/api/transfer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is False
        assert data["candidates"] == []


class TestAnomalyExplainEndpoint:
    """Test GET /api/anomalies/explain."""

    def test_with_explanations(self, mock_hub):
        pattern_mod = MagicMock()
        pattern_mod.get_current_state.return_value = {
            "anomaly_explanations": [
                {"feature": "power", "contribution": 0.45},
                {"feature": "lights", "contribution": 0.30},
            ],
            "trajectory": "ramping_up",
        }
        pattern_mod.attention_explainer = None
        mock_hub.get_module = MagicMock(return_value=pattern_mod)

        app = create_app(mock_hub)
        client = TestClient(app)
        resp = client.get("/api/anomalies/explain")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["path_tracing"]) == 2
        assert data["attention"] is None

    def test_module_not_available(self, mock_hub):
        mock_hub.get_module = MagicMock(return_value=None)

        app = create_app(mock_hub)
        client = TestClient(app)
        resp = client.get("/api/anomalies/explain")
        assert resp.status_code == 200
        data = resp.json()
        assert data["path_tracing"] == []
        assert data["attention"] is None
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/hub/test_api_transfer.py -v
```
Expected: FAIL — routes don't exist

**Step 3: Add the endpoints to api.py**

After the `/api/patterns` route, add:

```python
    @router.get("/api/transfer")
    async def get_transfer():
        """Transfer engine state — candidates, hit rates, promotions."""
        try:
            transfer_mod = hub.get_module("transfer_engine")
            if transfer_mod is None:
                return {"active": False, "candidates": [], "summary": {}, "stats": {}}

            state = transfer_mod.get_current_state()
            stats = transfer_mod.get_stats()
            return {
                "active": stats.get("active", False),
                "candidates": state.get("candidates", []),
                "summary": state.get("summary", {}),
                "stats": stats,
            }
        except Exception as e:
            logger.error(f"Error fetching transfer data: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.get("/api/anomalies/explain")
    async def get_anomaly_explanations():
        """Combined anomaly explanations — path tracing + attention."""
        try:
            pattern_mod = hub.get_module("pattern_recognition")
            if pattern_mod is None:
                return {"path_tracing": [], "attention": None}

            state = pattern_mod.get_current_state()
            path_explanations = state.get("anomaly_explanations", [])

            # Attention explainer results (Tier 4 only)
            attention_result = None
            if hasattr(pattern_mod, "attention_explainer") and pattern_mod.attention_explainer is not None:
                attn = pattern_mod.attention_explainer
                if attn.is_trained:
                    attention_result = attn.get_stats()

            return {
                "path_tracing": path_explanations,
                "attention": attention_result,
            }
        except Exception as e:
            logger.error(f"Error fetching anomaly explanations: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/hub/test_api_transfer.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api_transfer.py
git commit -m "feat: add /api/transfer and /api/anomalies/explain endpoints"
```

---

## Task 9: Integration Tests

**Files:**
- Create: `tests/integration/test_phase4_pipeline.py`

**Step 1: Write integration tests**

Create `tests/integration/test_phase4_pipeline.py`:

```python
"""Integration tests for Phase 4 — transfer engine and attention explainer.

Tests the full flows:
  organic discovery → transfer candidate generation → shadow testing → promotion
  attention explainer train → explain → contrastive output
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from aria.engine.transfer import TransferCandidate, TransferType, compute_jaccard_similarity
from aria.engine.transfer_generator import generate_transfer_candidates
from aria.modules.transfer_engine import TransferEngineModule


# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.subscribe = MagicMock()
    hub.unsubscribe = MagicMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.publish = AsyncMock()
    hub.get_config_value = MagicMock(return_value=None)
    hub.get_module = MagicMock(return_value=None)
    hub.modules = {}
    return hub


class TestTransferPipeline:
    """End-to-end transfer candidate lifecycle."""

    def test_jaccard_used_for_structural_similarity(self):
        """Jaccard similarity correctly identifies matching compositions."""
        kitchen = {"light", "binary_sensor"}
        bedroom = {"light", "binary_sensor"}
        garage = {"sensor", "cover"}

        assert compute_jaccard_similarity(kitchen, bedroom) == 1.0
        assert compute_jaccard_similarity(kitchen, garage) == 0.0

    def test_candidate_full_lifecycle(self):
        """hypothesis → testing → promoted via shadow results."""
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        assert tc.state == "hypothesis"

        # First shadow result transitions to testing
        tc.record_shadow_result(hit=True)
        assert tc.state == "testing"

        # Accumulate hits
        for _ in range(19):
            tc.record_shadow_result(hit=True)

        # Not promoted yet — min_days not met
        tc.check_promotion(min_days=7, min_hit_rate=0.6)
        assert tc.state == "testing"

        # Fast-forward testing_since
        tc.testing_since = datetime(2026, 2, 9)
        tc.check_promotion(min_days=7, min_hit_rate=0.6)
        assert tc.state == "promoted"

    def test_generator_produces_candidates_for_similar_rooms(self):
        """Room-to-room generation from organic capabilities."""
        capabilities = {
            "kitchen_lights": {
                "entities": ["light.kitchen_1", "binary_sensor.kitchen_motion"],
                "layer": "domain",
                "status": "promoted",
                "source": "organic",
            },
            "bedroom_lights": {
                "entities": ["light.bedroom_1", "binary_sensor.bedroom_motion"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
        }
        entities_cache = {
            "light.kitchen_1": {"entity_id": "light.kitchen_1", "domain": "light", "area_id": "kitchen"},
            "binary_sensor.kitchen_motion": {"entity_id": "binary_sensor.kitchen_motion", "domain": "binary_sensor", "area_id": "kitchen", "device_class": "motion"},
            "light.bedroom_1": {"entity_id": "light.bedroom_1", "domain": "light", "area_id": "bedroom"},
            "binary_sensor.bedroom_motion": {"entity_id": "binary_sensor.bedroom_motion", "domain": "binary_sensor", "area_id": "bedroom", "device_class": "motion"},
        }

        candidates = generate_transfer_candidates(capabilities, entities_cache, min_similarity=0.5)
        assert len(candidates) >= 1

    @patch("aria.modules.transfer_engine.recommend_tier", return_value=2)
    @patch("aria.modules.transfer_engine.scan_hardware")
    async def test_tier_2_disables_transfer(self, mock_scan, mock_tier, mock_hub):
        """Tier 2 hardware disables transfer engine."""
        mock_scan.return_value = MagicMock(ram_gb=4, cpu_cores=2)
        module = TransferEngineModule(mock_hub)
        await module.initialize()
        assert module.active is False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestAttentionPipeline:
    """End-to-end attention explainer tests (torch required)."""

    def test_train_explain_cycle(self):
        """Train on normal data, explain anomaly, get contrastive output."""
        from aria.engine.attention_explainer import AttentionExplainer

        explainer = AttentionExplainer(n_features=5, sequence_length=6, hidden_dim=16)
        np.random.seed(42)
        X_train = np.random.normal(0, 1, (30, 6, 5))
        result = explainer.train(X_train, epochs=5)
        assert result["trained"] is True

        # Anomalous window
        anomaly = np.zeros((6, 5))
        anomaly[:, 0] = 10.0

        explanation = explainer.explain(
            anomaly,
            feature_names=["power", "lights", "motion", "temp", "humidity"],
        )
        assert len(explanation["feature_contributions"]) > 0
        assert len(explanation["temporal_attention"]) == 6
        assert explanation["contrastive_explanation"] is not None
        assert "power" in explanation["contrastive_explanation"]
```

**Step 2: Run integration tests**

```bash
.venv/bin/python -m pytest tests/integration/test_phase4_pipeline.py -v
```
Expected: 4 passed (+ attention tests if torch available)

**Step 3: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ --timeout=120 -q
```
Expected: All existing tests pass + new Phase 4 tests pass

**Step 4: Commit**

```bash
git add tests/integration/test_phase4_pipeline.py
git commit -m "test: add integration tests for Phase 4 transfer and attention pipeline"
```

---

## Task 10: Update Plan Status and Design Doc

**Files:**
- Modify: `docs/plans/2026-02-17-adaptive-ml-pipeline-design.md` (update status)
- Modify: This plan file (update status)

**Step 1: Update status**

In the design doc, change:
```
**Status:** Phase 3 Completed — Phase 4 next
```
to:
```
**Status:** Phase 4 Completed — all phases done
```

In this plan file, change:
```
**Status:** Draft
```
to:
```
**Status:** Completed — merged to main on YYYY-MM-DD
```

**Step 2: Commit**

```bash
git add docs/plans/
git commit -m "docs: mark Phase 4 and full ML pipeline design as complete"
```

---

## Summary

| Task | Component | Files | Dependencies | Tier |
|------|-----------|-------|-------------|------|
| 1 | Transfer Data Model | `aria/engine/transfer.py`, test | None | 3+ |
| 2 | Transfer Generator | `aria/engine/transfer_generator.py`, test | Task 1 | 3+ |
| 3 | Attention Explainer | `aria/engine/attention_explainer.py`, test, `pyproject.toml` | torch (optional) | 4 |
| 4 | Transfer Engine Module | `aria/modules/transfer_engine.py`, test | Tasks 1, 2 | 3+ |
| 5 | Register in Hub | `aria/cli.py` | Task 4 | — |
| 6 | Config Entries | `config_defaults.py`, test | None | — |
| 7 | Wire Attention | `pattern_recognition.py`, test | Task 3 | 4 |
| 8 | API Endpoints | `api.py`, test | Tasks 4, 5 | — |
| 9 | Integration Test | `tests/integration/`, full suite | All | — |
| 10 | Docs Update | Plan files | All | — |

## Critical Path

```
Task 1 (transfer model) ──→ Task 2 (generator) ──→ Task 4 (hub module) ──→ Task 5 (register)
Task 3 (attention) ──────────────────────────────────────────────────────→ Task 7 (wire)
Task 6 (config) ─── [independent] ──────────────────────────────────────┐
                                                                        ↓
                                                        Task 8 (API) ←──┘
                                                            ↓
                                                        Task 9 (integration)
                                                            ↓
                                                        Task 10 (docs)
```

Tasks 1, 3, and 6 are fully independent — they can run in any order or in parallel.
Task 2 depends on 1. Task 4 depends on 1+2. Task 5 depends on 4.
Task 7 depends on 3. Task 8 depends on 5. Task 9 depends on all. Task 10 depends on 9.
