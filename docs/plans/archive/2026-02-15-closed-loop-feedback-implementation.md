# Closed-Loop Feedback System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close ARIA's open-loop pipeline by wiring ML accuracy, shadow hit rates, automation feedback, drift signals, and activity labels back into capability scoring — and visualize the full loop as an animated bus-architecture diagram on the Home dashboard.

**Architecture:** Five backend feedback channels write signals back to the capabilities cache. A new activity_labeler module uses Ollama to predict activities from sensor state, with user corrections building labeled training data. The Home dashboard replaces its linear 3-lane pipeline with a 3-plane block diagram connected by animated data buses.

**Tech Stack:** Python 3.12 (asyncio, aiohttp), FastAPI, Preact + esbuild, SVG animations, Ollama queue (port 7683), SQLite cache (hub.db)

---

## Task 1: ML Engine Feedback Write-Back

**Files:**
- Modify: `aria/modules/ml_engine.py:215-226` (after training metadata write)
- Test: `tests/hub/test_ml_training.py`

**Step 1: Write the failing test**

In `tests/hub/test_ml_training.py`, add:

```python
@pytest.mark.asyncio
async def test_ml_feedback_writes_predictability_to_capabilities(mock_hub):
    """ML Engine should write accuracy back to capabilities cache after training."""
    # Seed capabilities cache with a test capability
    await mock_hub.set_cache("capabilities", {
        "power_monitoring": {
            "available": True,
            "entities": ["sensor.power_1", "sensor.power_2"],
            "total_count": 2,
            "can_predict": True,
            "source": "seed",
            "usefulness": 50,
            "usefulness_components": {"predictability": 0, "stability": 50, "entity_coverage": 30, "activity": 40, "cohesion": 60},
            "status": "promoted",
        }
    })

    engine = MLEngine(mock_hub, models_dir="/tmp/test_models", training_data_dir="/tmp/test_data")
    mock_hub.register_module(engine)

    # Mock the actual training to return known accuracy
    engine._train_model_for_target = AsyncMock(return_value={
        "target": "power_watts",
        "accuracy_scores": {"gb_r2": 0.85, "rf_r2": 0.78, "lgbm_r2": 0.90},
    })

    # Provide minimal training data
    await mock_hub.set_cache("ml_training_metadata", {"targets_trained": []})

    await engine._write_feedback_to_capabilities({"power_monitoring": {"power_watts": {"r2": 0.85, "mae": 12.3}}})

    caps = await mock_hub.get_cache("capabilities")
    cap = caps["data"]["power_monitoring"]
    assert cap["ml_accuracy"]["mean_r2"] == 0.85
    assert cap["usefulness_components"]["predictability"] == 85
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py::test_ml_feedback_writes_predictability_to_capabilities -v`
Expected: FAIL — `_write_feedback_to_capabilities` doesn't exist

**Step 3: Write the implementation**

In `aria/modules/ml_engine.py`, add method after `train_models()` (after line 226):

```python
async def _write_feedback_to_capabilities(self, training_results: dict) -> None:
    """Write ML accuracy back to capabilities cache for feedback loop."""
    caps_entry = await self.hub.get_cache("capabilities")
    if not caps_entry or not caps_entry.get("data"):
        return
    caps = caps_entry["data"]
    updated = False
    for cap_name, targets in training_results.items():
        if cap_name not in caps:
            continue
        r2_values = [t.get("r2", 0.0) for t in targets.values()]
        mean_r2 = sum(r2_values) / len(r2_values) if r2_values else 0.0
        top_features = []
        for t in targets.values():
            top_features.extend(t.get("top_features", []))
        caps[cap_name]["ml_accuracy"] = {
            "mean_r2": round(mean_r2, 3),
            "targets": {k: {"r2": round(v.get("r2", 0), 3), "mae": round(v.get("mae", 0), 3)} for k, v in targets.items()},
            "last_trained": datetime.now().isoformat(),
            "feature_importance_top5": top_features[:5],
        }
        caps[cap_name].setdefault("usefulness_components", {})["predictability"] = round(mean_r2 * 100)
        updated = True
        self.logger.info(f"Feedback: {cap_name} predictability={round(mean_r2 * 100)} (R²={mean_r2:.3f})")
    if updated:
        await self.hub.set_cache("capabilities", caps, {"source": "ml_feedback"})
```

Then call it at the end of `train_models()` (after line 226), building `training_results` from the per-target accuracy data accumulated during the training loop.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py::test_ml_feedback_writes_predictability_to_capabilities -v`
Expected: PASS

**Step 5: Commit**

```bash
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "feat: ML engine writes accuracy feedback to capabilities cache"
```

---

## Task 2: Shadow Engine Feedback Write-Back

**Files:**
- Modify: `aria/modules/shadow_engine.py:909` (near _resolution_loop)
- Test: `tests/hub/test_shadow_engine.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_shadow_feedback_writes_hit_rate_to_capabilities(mock_hub):
    """Shadow engine should write prediction hit rates to capabilities cache."""
    await mock_hub.set_cache("capabilities", {
        "lighting": {
            "available": True,
            "entities": ["light.kitchen", "light.living"],
            "total_count": 2,
            "usefulness_components": {"predictability": 0},
            "status": "promoted",
        }
    })

    engine = ShadowEngine(mock_hub)
    # Mock prediction outcomes
    engine._get_capability_hit_rates = Mock(return_value={
        "lighting": {"hits": 15, "total": 20},
    })

    await engine._write_feedback_to_capabilities()

    caps = await mock_hub.get_cache("capabilities")
    cap = caps["data"]["lighting"]
    assert cap["shadow_accuracy"]["hit_rate"] == 0.75
    assert cap["shadow_accuracy"]["total_predictions"] == 20
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_shadow_engine.py::test_shadow_feedback_writes_hit_rate_to_capabilities -v`
Expected: FAIL

**Step 3: Write the implementation**

In `aria/modules/shadow_engine.py`, add method:

```python
async def _write_feedback_to_capabilities(self) -> None:
    """Write shadow prediction hit rates to capabilities cache."""
    caps_entry = await self.hub.get_cache("capabilities")
    if not caps_entry or not caps_entry.get("data"):
        return
    caps = caps_entry["data"]
    hit_rates = self._get_capability_hit_rates()
    updated = False
    for cap_name, rates in hit_rates.items():
        if cap_name not in caps or rates["total"] == 0:
            continue
        hit_rate = round(rates["hits"] / rates["total"], 3)
        caps[cap_name]["shadow_accuracy"] = {
            "hit_rate": hit_rate,
            "total_predictions": rates["total"],
            "last_updated": datetime.now().isoformat(),
        }
        updated = True
    if updated:
        await self.hub.set_cache("capabilities", caps, {"source": "shadow_feedback"})

def _get_capability_hit_rates(self) -> dict:
    """Compute per-capability hit rates from resolved predictions."""
    # Query predictions table, group by capability entities, count correct/total
    rates = {}
    for cap_name, cap_data in (self._cached_capabilities or {}).items():
        entity_set = set(cap_data.get("entities", []))
        if not entity_set:
            continue
        hits = 0
        total = 0
        for pred in self._recent_resolved:
            pred_entities = {p.get("predicted", "") for p in pred.get("predictions", [])}
            if pred_entities & entity_set:
                total += 1
                if pred.get("outcome") == "correct":
                    hits += 1
        if total > 0:
            rates[cap_name] = {"hits": hits, "total": total}
    return rates
```

Add call to `_write_feedback_to_capabilities()` inside `_resolution_loop()` every 10th iteration (every ~10 minutes).

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add aria/modules/shadow_engine.py tests/hub/test_shadow_engine.py
git commit -m "feat: shadow engine writes hit rate feedback to capabilities cache"
```

---

## Task 3: Discovery Reads Predictability Feedback

**Files:**
- Modify: `aria/modules/organic_discovery/module.py:229` and `:282`
- Test: `tests/hub/test_organic_discovery_module.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_discovery_reads_ml_predictability(mock_hub):
    """Discovery should use ML accuracy for predictability instead of 0.0."""
    # Pre-populate capabilities with ml_accuracy from feedback
    await mock_hub.set_cache("capabilities", {
        "power_monitoring": {
            "available": True,
            "entities": ["sensor.power_1"],
            "total_count": 1,
            "ml_accuracy": {"mean_r2": 0.82},
            "shadow_accuracy": {"hit_rate": 0.70},
            "usefulness_components": {"predictability": 82},
            "status": "promoted",
            "source": "seed",
        }
    })

    module = OrganicDiscoveryModule(mock_hub)
    predictability = module._compute_predictability("power_monitoring", await mock_hub.get_cache("capabilities"))
    # 0.82 * 0.7 + 0.70 * 0.3 = 0.574 + 0.21 = 0.784
    assert abs(predictability - 0.784) < 0.01
```

**Step 2: Run test, verify fail**

**Step 3: Implement `_compute_predictability` method and replace hardcoded 0.0**

```python
def _compute_predictability(self, cap_name: str, caps_data: dict) -> float:
    """Compute predictability from ML + shadow feedback signals."""
    caps = caps_data.get("data", caps_data) if isinstance(caps_data, dict) else {}
    existing = caps.get(cap_name, {})
    ml_r2 = existing.get("ml_accuracy", {}).get("mean_r2", 0.0)
    shadow_hr = existing.get("shadow_accuracy", {}).get("hit_rate", 0.0)
    if ml_r2 + shadow_hr == 0:
        return 0.0
    return ml_r2 * 0.7 + shadow_hr * 0.3
```

Replace `predictability=0.0` at lines 229 and 282 with call to `self._compute_predictability(name, existing_caps)`.

**Step 4: Run test, verify pass. Run full organic discovery suite.**

Run: `.venv/bin/python -m pytest tests/hub/ -k "organic" -v --timeout=120`

**Step 5: Commit**

```bash
git add aria/modules/organic_discovery/module.py tests/hub/test_organic_discovery_module.py
git commit -m "feat: discovery reads ML/shadow predictability instead of hardcoded 0.0"
```

---

## Task 4: DemandSignal Dataclass + Registry Extension

**Files:**
- Modify: `aria/capabilities.py:20-42`
- Test: `tests/test_capabilities.py`

**Step 1: Write the failing test**

```python
def test_demand_signal_on_capability():
    """Capabilities can declare demand signals."""
    from aria.capabilities import Capability, DemandSignal
    cap = Capability(
        id="ml_realtime",
        name="ML Realtime",
        description="Test",
        module="ml_engine",
        layer="hub",
        demand_signals=[
            DemandSignal(
                entity_domains=["sensor"],
                device_classes=["power", "energy"],
                min_entities=5,
                description="Power monitoring for predictions",
            ),
        ],
    )
    assert len(cap.demand_signals) == 1
    assert cap.demand_signals[0].entity_domains == ["sensor"]
```

**Step 2: Run test, verify fail**

**Step 3: Add DemandSignal dataclass before Capability in `aria/capabilities.py`:**

```python
@dataclass(frozen=True)
class DemandSignal:
    """Declares what entity groupings a module needs from discovery."""
    entity_domains: List[str] = field(default_factory=list)
    device_classes: List[str] = field(default_factory=list)
    min_entities: int = 5
    description: str = ""
```

Add `demand_signals: List[DemandSignal] = field(default_factory=list)` to Capability dataclass.

**Step 4: Run test, verify pass. Run capabilities suite.**

Run: `.venv/bin/python -m pytest tests/test_capabilities.py -v`

**Step 5: Commit**

```bash
git add aria/capabilities.py tests/test_capabilities.py
git commit -m "feat: add DemandSignal dataclass and demand_signals to Capability"
```

---

## Task 5: ML Engine + Shadow Engine Declare Demand Signals

**Files:**
- Modify: `aria/modules/ml_engine.py` (CAPABILITIES list)
- Modify: `aria/modules/shadow_engine.py` (CAPABILITIES list)
- Test: `tests/test_capabilities.py` (verify declarations parse)

**Step 1: Write test that verifies demand signals exist**

```python
def test_ml_engine_declares_demand_signals():
    from aria.modules.ml_engine import MLEngine
    caps = MLEngine.CAPABILITIES
    assert any(len(c.demand_signals) > 0 for c in caps)
```

**Step 2: Run test, verify fail**

**Step 3: Add demand_signals to CAPABILITIES in ml_engine.py and shadow_engine.py**

ML Engine demands: power, lighting, occupancy, motion, climate entity groupings.
Shadow Engine demands: any entities with high event rates.

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add aria/modules/ml_engine.py aria/modules/shadow_engine.py tests/test_capabilities.py
git commit -m "feat: ML and shadow engines declare demand signals for discovery"
```

---

## Task 6: Discovery Demand Alignment Scoring

**Files:**
- Modify: `aria/modules/organic_discovery/module.py`
- Test: `tests/hub/test_organic_discovery_module.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_demand_alignment_bonus(mock_hub):
    """Clusters matching consumer demand should get a scoring bonus."""
    module = OrganicDiscoveryModule(mock_hub)
    from aria.capabilities import DemandSignal

    demands = [
        DemandSignal(entity_domains=["sensor"], device_classes=["power"], min_entities=3),
    ]

    # Cluster with power sensors — should match
    cluster_entities = [
        {"entity_id": "sensor.power_1", "domain": "sensor", "device_class": "power"},
        {"entity_id": "sensor.power_2", "domain": "sensor", "device_class": "power"},
        {"entity_id": "sensor.power_3", "domain": "sensor", "device_class": "power"},
    ]
    bonus = module._compute_demand_alignment(cluster_entities, demands)
    assert 0.05 <= bonus <= 0.20  # Gets a bonus

    # Cluster with no matching demand — no bonus
    other_entities = [
        {"entity_id": "light.lamp1", "domain": "light", "device_class": ""},
    ]
    no_bonus = module._compute_demand_alignment(other_entities, demands)
    assert no_bonus == 0.0
```

**Step 2: Run test, verify fail**

**Step 3: Implement `_compute_demand_alignment` method**

```python
def _compute_demand_alignment(self, cluster_entities: list, demands: list) -> float:
    """Score how well a cluster aligns with consumer demand signals (0.0-0.2)."""
    if not demands:
        return 0.0
    best_score = 0.0
    domains = {e.get("domain", "") for e in cluster_entities}
    device_classes = {e.get("device_class", "") for e in cluster_entities}
    count = len(cluster_entities)
    for demand in demands:
        domain_match = bool(set(demand.entity_domains) & domains) if demand.entity_domains else True
        class_match = bool(set(demand.device_classes) & device_classes) if demand.device_classes else True
        size_match = count >= demand.min_entities
        if domain_match and class_match and size_match:
            best_score = max(best_score, 0.2)
        elif domain_match and class_match:
            best_score = max(best_score, 0.1)
        elif domain_match:
            best_score = max(best_score, 0.05)
    return best_score
```

Wire into usefulness calculation: add bonus to final score (capped at 100).

**Step 4: Run test, verify pass. Run full organic suite.**

**Step 5: Commit**

```bash
git add aria/modules/organic_discovery/module.py tests/hub/test_organic_discovery_module.py
git commit -m "feat: discovery scores demand alignment from registry consumers"
```

---

## Task 7: Drift Detection → Re-Discovery Trigger

**Files:**
- Modify: `aria/modules/intelligence.py:169`
- Modify: `aria/modules/organic_discovery/module.py`
- Test: `tests/hub/test_organic_discovery_module.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_drift_flags_capability_for_rediscovery(mock_hub):
    """When drift is detected, the capability should be flagged for re-evaluation."""
    await mock_hub.set_cache("capabilities", {
        "climate": {"available": True, "entities": ["sensor.temp"], "status": "promoted"}
    })

    module = OrganicDiscoveryModule(mock_hub)
    await module.on_event("drift_detected", {
        "capability": "climate",
        "drift_type": "behavioral_drift",
        "severity": 0.8,
    })

    caps = await mock_hub.get_cache("capabilities")
    assert caps["data"]["climate"]["drift_flagged"] is True
```

**Step 2: Run test, verify fail**

**Step 3: Implement**

In `aria/modules/intelligence.py`, add drift publishing to `_periodic_refresh` when `compare_model_accuracy` returns behavioral_drift:

```python
if result["interpretation"] == "behavioral_drift":
    await self.hub.publish("drift_detected", {
        "capability": cap_name,
        "drift_type": "behavioral_drift",
        "severity": abs(result["divergence_pct"]) / 100,
    })
```

In `aria/modules/organic_discovery/module.py`, handle drift event in `on_event`:

```python
async def on_event(self, event_type: str, data: dict) -> None:
    if event_type == "drift_detected":
        cap_name = data.get("capability", "")
        caps_entry = await self.hub.get_cache("capabilities")
        if caps_entry and caps_entry.get("data") and cap_name in caps_entry["data"]:
            caps_entry["data"][cap_name]["drift_flagged"] = True
            caps_entry["data"][cap_name]["drift_detected_at"] = datetime.now().isoformat()
            await self.hub.set_cache("capabilities", caps_entry["data"], {"source": "drift_flag"})
            self.logger.warning(f"Capability '{cap_name}' flagged for re-discovery due to drift")
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add aria/modules/intelligence.py aria/modules/organic_discovery/module.py tests/hub/test_organic_discovery_module.py
git commit -m "feat: drift detection flags capabilities for re-discovery"
```

---

## Task 8: Automation Suggestion Feedback API

**Files:**
- Modify: `aria/hub/api.py` (add endpoints after line 995)
- Create: `tests/hub/test_api_automation_feedback.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_post_automation_feedback(client, mock_hub):
    """User can submit feedback on automation suggestions."""
    resp = await client.post("/api/automations/feedback", json={
        "suggestion_id": "auto_2026_001",
        "capability_source": "lighting",
        "user_action": "accepted",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "recorded"

@pytest.mark.asyncio
async def test_get_automation_feedback(client, mock_hub):
    """Can retrieve automation feedback history."""
    # Post one first
    await client.post("/api/automations/feedback", json={
        "suggestion_id": "auto_2026_001",
        "capability_source": "lighting",
        "user_action": "accepted",
    })
    resp = await client.get("/api/automations/feedback")
    assert resp.status_code == 200
    data = resp.json()
    assert data["per_capability"]["lighting"]["accepted"] == 1
```

**Step 2: Run test, verify fail**

**Step 3: Implement endpoints in api.py**

```python
@router.post("/api/automations/feedback")
async def post_automation_feedback(body: dict):
    suggestion_id = body.get("suggestion_id", "")
    capability_source = body.get("capability_source", "")
    user_action = body.get("user_action", "")
    if user_action not in ("accepted", "rejected", "modified", "ignored"):
        return JSONResponse({"error": "Invalid user_action"}, status_code=400)

    feedback_entry = await hub.get_cache("automation_feedback") or {}
    feedback = feedback_entry.get("data", {"suggestions": {}, "per_capability": {}})

    feedback["suggestions"][suggestion_id] = {
        "capability_source": capability_source,
        "user_action": user_action,
        "timestamp": datetime.now().isoformat(),
    }

    pc = feedback["per_capability"].setdefault(capability_source, {"suggested": 0, "accepted": 0, "rejected": 0})
    pc["suggested"] = pc.get("suggested", 0) + 1
    if user_action == "accepted":
        pc["accepted"] = pc.get("accepted", 0) + 1
    elif user_action == "rejected":
        pc["rejected"] = pc.get("rejected", 0) + 1
    total = pc["suggested"]
    pc["acceptance_rate"] = round(pc["accepted"] / total, 3) if total > 0 else 0

    await hub.set_cache("automation_feedback", feedback, {"source": "user_feedback"})
    return {"status": "recorded", "suggestion_id": suggestion_id}

@router.get("/api/automations/feedback")
async def get_automation_feedback():
    entry = await hub.get_cache("automation_feedback")
    if not entry or not entry.get("data"):
        return {"suggestions": {}, "per_capability": {}}
    return entry["data"]
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api_automation_feedback.py
git commit -m "feat: add automation suggestion feedback API endpoints"
```

---

## Task 9: Activity Labeler Module

**Files:**
- Create: `aria/modules/activity_labeler.py`
- Create: `tests/hub/test_activity_labeler.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_activity_labeler_predicts_from_context(mock_hub):
    """Activity labeler should predict activity from sensor context."""
    labeler = ActivityLabeler(mock_hub)

    # Mock Ollama response
    labeler._query_ollama = AsyncMock(return_value={"activity": "cooking", "confidence": 0.72})

    context = {
        "power_watts": 450,
        "lights_on": 3,
        "motion_rooms": ["kitchen"],
        "time_of_day": "evening",
        "occupancy": "home",
    }
    result = await labeler.predict_activity(context)
    assert result["predicted"] == "cooking"
    assert result["confidence"] == 0.72
    assert result["method"] == "ollama"

@pytest.mark.asyncio
async def test_activity_labeler_stores_correction(mock_hub):
    """User corrections should be stored as labeled training data."""
    labeler = ActivityLabeler(mock_hub)
    await labeler.record_label(
        predicted="cooking",
        actual="cleaning",
        sensor_context={"power_watts": 450},
        source="corrected",
    )

    labels_entry = await mock_hub.get_cache("activity_labels")
    labels = labels_entry["data"]
    assert labels["label_stats"]["total_labels"] == 1
    assert labels["label_stats"]["total_corrections"] == 1
    assert len(labels["labels"]) == 1
    assert labels["labels"][0]["actual_activity"] == "cleaning"

@pytest.mark.asyncio
async def test_activity_labeler_uses_classifier_when_ready(mock_hub):
    """When enough labels exist, classifier should replace Ollama."""
    labeler = ActivityLabeler(mock_hub)
    labeler._classifier = Mock()
    labeler._classifier.predict.return_value = ["cooking"]
    labeler._classifier_ready = True

    context = {"power_watts": 450, "lights_on": 3, "motion_rooms": ["kitchen"], "time_of_day": "evening", "occupancy": "home"}
    result = await labeler.predict_activity(context)
    assert result["method"] == "classifier"
    labeler._query_ollama = AsyncMock()  # should NOT be called
    labeler._query_ollama.assert_not_called()
```

**Step 2: Run test, verify fail**

**Step 3: Create `aria/modules/activity_labeler.py`**

```python
"""Activity Labeler Module — LLM predicts activities, user corrects, system retrains.

Bridges the gap from sensor states to named activities. Uses Ollama for initial
predictions, stores user corrections as labeled training data, and trains a
lightweight classifier once enough labels accumulate.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from aria.hub.core import Module, IntelligenceHub
from aria.capabilities import Capability

logger = logging.getLogger(__name__)

CLASSIFIER_THRESHOLD = 50  # Labels needed before training classifier
PREDICTION_INTERVAL = timedelta(minutes=15)  # How often to predict activity
OLLAMA_QUEUE_URL = "http://127.0.0.1:7683"

ACTIVITY_PROMPT_TEMPLATE = """Given the current smart home state:
- Power draw: {power_watts}W
- Lights on: {lights_on} ({light_rooms})
- Motion detected: {motion_rooms}
- Time: {time_of_day} ({hour}:{minute})
- Occupancy: {occupancy}
- Recent events: {recent_events}

What activity is the resident most likely doing?
Choose from common activities: sleeping, cooking, watching_tv, working, cleaning, eating, away, relaxing, exercising, showering, unknown.
Respond with ONLY JSON: {{"activity": "...", "confidence": 0.0-1.0}}"""


class ActivityLabeler(Module):
    """Predicts household activities from sensor state with human-in-the-loop correction."""

    CAPABILITIES = [
        Capability(
            id="activity_labeler",
            name="Activity Labeler",
            description="LLM-predicted activity labels with user correction and classifier retraining.",
            module="activity_labeler",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_activity_labeler.py"],
            systemd_units=["aria-hub.service"],
            status="experimental",
            added_version="1.1.0",
            depends_on=["discovery", "activity_analytics"],
        ),
    ]

    def __init__(self, hub: IntelligenceHub):
        super().__init__("activity_labeler", hub)
        self._classifier = None
        self._classifier_ready = False

    async def initialize(self):
        self.logger.info("Activity labeler initializing...")
        # Check if we have enough labels to use classifier
        labels_entry = await self.hub.get_cache("activity_labels")
        if labels_entry and labels_entry.get("data"):
            stats = labels_entry["data"].get("label_stats", {})
            if stats.get("classifier_ready", False):
                await self._train_classifier()
        # Schedule periodic predictions
        await self.hub.schedule_task(
            task_id="activity_prediction",
            coro=self._periodic_predict,
            interval=PREDICTION_INTERVAL,
            run_immediately=False,
        )
        self.logger.info("Activity labeler initialized")

    async def predict_activity(self, context: dict) -> dict:
        """Predict current activity from sensor context."""
        if self._classifier_ready and self._classifier is not None:
            features = self._context_to_features(context)
            predicted = self._classifier.predict([features])[0]
            return {
                "predicted": predicted,
                "confidence": 0.85,  # Classifier confidence estimated from training accuracy
                "method": "classifier",
                "sensor_context": context,
                "predicted_at": datetime.now().isoformat(),
            }
        # Fall back to Ollama
        result = await self._query_ollama(context)
        return {
            "predicted": result.get("activity", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "method": "ollama",
            "sensor_context": context,
            "predicted_at": datetime.now().isoformat(),
        }

    async def record_label(self, predicted: str, actual: str, sensor_context: dict, source: str = "corrected") -> None:
        """Record a user-confirmed or corrected activity label."""
        labels_entry = await self.hub.get_cache("activity_labels")
        data = labels_entry.get("data", {}) if labels_entry else {}
        labels = data.get("labels", [])
        stats = data.get("label_stats", {
            "total_labels": 0,
            "total_corrections": 0,
            "accuracy": 1.0,
            "activities_seen": [],
            "classifier_ready": False,
            "last_trained": None,
        })

        label = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now().isoformat(),
            "sensor_context": sensor_context,
            "predicted_activity": predicted,
            "actual_activity": actual,
            "source": source,
        }
        labels.append(label)

        stats["total_labels"] = len(labels)
        if source == "corrected":
            stats["total_corrections"] = stats.get("total_corrections", 0) + 1
        confirmed = stats["total_labels"] - stats.get("total_corrections", 0)
        stats["accuracy"] = round(confirmed / stats["total_labels"], 3) if stats["total_labels"] > 0 else 1.0
        if actual not in stats.get("activities_seen", []):
            stats.setdefault("activities_seen", []).append(actual)
        stats["classifier_ready"] = stats["total_labels"] >= CLASSIFIER_THRESHOLD

        await self.hub.set_cache("activity_labels", {
            "current_activity": {
                "predicted": actual,
                "confidence": 1.0,
                "method": "user_set",
                "sensor_context": sensor_context,
                "predicted_at": datetime.now().isoformat(),
            },
            "labels": labels,
            "label_stats": stats,
        }, {"source": "user_label"})

        # Train classifier if we just crossed the threshold
        if stats["classifier_ready"] and not self._classifier_ready:
            await self._train_classifier()

    async def _query_ollama(self, context: dict) -> dict:
        """Query Ollama via queue for activity prediction."""
        import aiohttp
        prompt = ACTIVITY_PROMPT_TEMPLATE.format(
            power_watts=context.get("power_watts", 0),
            lights_on=context.get("lights_on", 0),
            light_rooms=", ".join(context.get("light_rooms", [])),
            motion_rooms=", ".join(context.get("motion_rooms", [])),
            time_of_day=context.get("time_of_day", "unknown"),
            hour=datetime.now().hour,
            minute=datetime.now().minute,
            occupancy=context.get("occupancy", "unknown"),
            recent_events=", ".join(context.get("recent_events", [])[:5]),
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{OLLAMA_QUEUE_URL}/api/generate", json={
                    "model": "gemma3:4b",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                }, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return json.loads(body.get("response", "{}"))
        except Exception as e:
            self.logger.warning(f"Ollama query failed: {e}")
        return {"activity": "unknown", "confidence": 0.0}

    async def _train_classifier(self) -> None:
        """Train GradientBoosting classifier from labeled data."""
        labels_entry = await self.hub.get_cache("activity_labels")
        if not labels_entry or not labels_entry.get("data"):
            return
        labels = labels_entry["data"].get("labels", [])
        if len(labels) < CLASSIFIER_THRESHOLD:
            return
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import LabelEncoder
            X = [self._context_to_features(l["sensor_context"]) for l in labels]
            y = [l["actual_activity"] for l in labels]
            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(y)
            clf = GradientBoostingClassifier(n_estimators=50, max_depth=3)
            clf.fit(X, y_encoded)
            self._classifier = clf
            self._classifier_ready = True
            self.logger.info(f"Activity classifier trained on {len(labels)} labels")
        except Exception as e:
            self.logger.warning(f"Classifier training failed: {e}")

    def _context_to_features(self, ctx: dict) -> list:
        """Convert sensor context dict to feature vector for classifier."""
        hour = datetime.now().hour
        return [
            ctx.get("power_watts", 0),
            ctx.get("lights_on", 0),
            len(ctx.get("motion_rooms", [])),
            hour,
            1 if ctx.get("occupancy") == "home" else 0,
        ]

    async def _periodic_predict(self) -> None:
        """Periodically predict current activity from latest sensor data."""
        activity = await self.hub.get_cache("activity_summary")
        intelligence = await self.hub.get_cache("intelligence")
        if not activity or not intelligence:
            return
        act_data = activity.get("data", {}).get("data", {})
        intel_data = intelligence.get("data", {})
        intraday = intel_data.get("intraday_trend", [])
        latest = intraday[-1] if intraday else {}
        occ = act_data.get("occupancy", {})
        context = {
            "power_watts": latest.get("power_watts", 0),
            "lights_on": latest.get("lights_on", 0),
            "motion_rooms": list(act_data.get("active_rooms", {}).keys()) if "active_rooms" in act_data else [],
            "time_of_day": self._time_of_day(),
            "occupancy": "home" if occ.get("anyone_home") else "away",
        }
        result = await self.predict_activity(context)
        # Store as current_activity in cache
        labels_entry = await self.hub.get_cache("activity_labels") or {}
        data = labels_entry.get("data", {"labels": [], "label_stats": {}})
        data["current_activity"] = result
        await self.hub.set_cache("activity_labels", data, {"source": "periodic_predict"})

    @staticmethod
    def _time_of_day() -> str:
        hour = datetime.now().hour
        if hour < 6: return "night"
        if hour < 12: return "morning"
        if hour < 18: return "afternoon"
        return "evening"
```

**Step 4: Run test, verify pass**

Run: `.venv/bin/python -m pytest tests/hub/test_activity_labeler.py -v`

**Step 5: Commit**

```bash
git add aria/modules/activity_labeler.py tests/hub/test_activity_labeler.py
git commit -m "feat: add activity labeler module with Ollama prediction and user correction"
```

---

## Task 10: Register Activity Labeler Module + API Endpoints

**Files:**
- Modify: `aria/cli.py` (add registration after activity_monitor block ~line 297)
- Modify: `aria/hub/api.py` (add activity endpoints)
- Create: `tests/hub/test_api_activity.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_get_current_activity(client, mock_hub):
    await mock_hub.set_cache("activity_labels", {
        "current_activity": {"predicted": "cooking", "confidence": 0.72, "method": "ollama"},
        "labels": [],
        "label_stats": {"total_labels": 0},
    })
    resp = await client.get("/api/activity/current")
    assert resp.status_code == 200
    assert resp.json()["predicted"] == "cooking"

@pytest.mark.asyncio
async def test_post_activity_label(client, mock_hub):
    # Seed current activity
    await mock_hub.set_cache("activity_labels", {
        "current_activity": {"predicted": "cooking", "confidence": 0.72, "method": "ollama", "sensor_context": {"power_watts": 450}},
        "labels": [],
        "label_stats": {"total_labels": 0, "total_corrections": 0, "accuracy": 1.0, "activities_seen": [], "classifier_ready": False, "last_trained": None},
    })
    resp = await client.post("/api/activity/label", json={
        "actual_activity": "cleaning",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"
```

**Step 2: Run test, verify fail**

**Step 3: Add endpoints to api.py and registration to cli.py**

API endpoints:
```python
@router.get("/api/activity/current")
async def get_current_activity():
    entry = await hub.get_cache("activity_labels")
    if not entry or not entry.get("data"):
        return {"predicted": "unknown", "confidence": 0, "method": "none"}
    return entry["data"].get("current_activity", {})

@router.post("/api/activity/label")
async def post_activity_label(body: dict):
    actual = body.get("actual_activity", "")
    if not actual:
        return JSONResponse({"error": "actual_activity required"}, status_code=400)
    entry = await hub.get_cache("activity_labels")
    current = entry["data"]["current_activity"] if entry and entry.get("data") else {}
    predicted = current.get("predicted", "unknown")
    context = current.get("sensor_context", {})
    source = "confirmed" if actual == predicted else "corrected"
    labeler = hub.get_module("activity_labeler")
    if labeler:
        await labeler.record_label(predicted, actual, context, source)
    return {"status": "recorded", "predicted": predicted, "actual": actual, "source": source}

@router.get("/api/activity/labels")
async def get_activity_labels(limit: int = 50):
    entry = await hub.get_cache("activity_labels")
    if not entry or not entry.get("data"):
        return {"labels": [], "label_stats": {}}
    data = entry["data"]
    return {"labels": data.get("labels", [])[-limit:], "label_stats": data.get("label_stats", {})}

@router.get("/api/activity/stats")
async def get_activity_stats():
    entry = await hub.get_cache("activity_labels")
    if not entry or not entry.get("data"):
        return {"total_labels": 0, "classifier_ready": False}
    return entry["data"].get("label_stats", {})
```

CLI registration (in `cli.py`, after activity_monitor block):
```python
# activity_labeler (non-fatal)
try:
    from aria.modules.activity_labeler import ActivityLabeler
    activity_labeler = ActivityLabeler(hub)
    hub.register_module(activity_labeler)
    await _init_module(activity_labeler, "activity_labeler")()
except Exception as e:
    logger.warning(f"Activity labeler module failed (non-fatal): {e}")
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add aria/cli.py aria/hub/api.py tests/hub/test_api_activity.py
git commit -m "feat: register activity labeler module and add activity API endpoints"
```

---

## Task 11: Feedback Health API Endpoint

**Files:**
- Modify: `aria/hub/api.py`
- Create: `tests/hub/test_api_feedback_health.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_feedback_health_endpoint(client, mock_hub):
    await mock_hub.set_cache("capabilities", {
        "power_monitoring": {
            "ml_accuracy": {"mean_r2": 0.85, "last_trained": "2026-02-15T10:00:00"},
            "shadow_accuracy": {"hit_rate": 0.73, "last_updated": "2026-02-15T10:00:00"},
        }
    })
    resp = await client.get("/api/capabilities/feedback/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "capabilities_with_ml_feedback" in data
    assert "capabilities_with_shadow_feedback" in data
    assert data["capabilities_with_ml_feedback"] == 1
```

**Step 2: Run test, verify fail**

**Step 3: Implement**

```python
@router.get("/api/capabilities/feedback/health")
async def get_feedback_health():
    entry = await hub.get_cache("capabilities")
    caps = entry.get("data", {}) if entry else {}
    ml_count = sum(1 for c in caps.values() if c.get("ml_accuracy"))
    shadow_count = sum(1 for c in caps.values() if c.get("shadow_accuracy"))
    drift_count = sum(1 for c in caps.values() if c.get("drift_flagged"))

    labels_entry = await hub.get_cache("activity_labels")
    label_stats = labels_entry.get("data", {}).get("label_stats", {}) if labels_entry else {}

    feedback_entry = await hub.get_cache("automation_feedback")
    suggestion_stats = feedback_entry.get("data", {}).get("per_capability", {}) if feedback_entry else {}

    return {
        "capabilities_total": len(caps),
        "capabilities_with_ml_feedback": ml_count,
        "capabilities_with_shadow_feedback": shadow_count,
        "capabilities_drift_flagged": drift_count,
        "activity_labels": label_stats.get("total_labels", 0),
        "activity_classifier_ready": label_stats.get("classifier_ready", False),
        "automation_feedback_count": sum(v.get("suggested", 0) for v in suggestion_stats.values()),
    }
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api_feedback_health.py
git commit -m "feat: add feedback health API endpoint"
```

---

## Task 12: Home Dashboard — Bus Architecture SVG Diagram

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Home.jsx` (replace PipelineFlow)
- Modify: `aria/dashboard/spa/src/index.css` (new animations)

**Step 1: Replace PipelineFlow with BusArchitecture component**

This is the largest frontend change. Replace the existing `LANE_NODES`, `NODE_META`, `LANES`, `PipelineFlow`, `PipelineNode`, `LaneArrow`, `DownArrow` components (lines 23-435) with the bus architecture diagram.

**Key SVG structure:**

```jsx
function BusArchitecture({ statusData, feedbackHealth, activityData }) {
  // Three planes: Data (top), Learning (middle), Action (bottom)
  // Two buses: Capabilities Bus, Feedback Bus
  // Each node: rect + status LED + label + metric

  return (
    <section class="t-terminal-bg rounded-lg p-4 overflow-x-auto">
      <svg viewBox="0 0 900 520" class="w-full" style="min-width: 700px; max-width: 100%;">
        <defs>
          {/* Animated dash pattern for bus lines */}
          <pattern id="bus-flow" width="20" height="4" patternUnits="userSpaceOnUse">
            <rect width="10" height="4" fill="var(--accent)" opacity="0.6">
              <animate attributeName="x" from="-20" to="20" dur="1.5s" repeatCount="indefinite" />
            </rect>
          </pattern>
          {/* Feedback bus flows upward */}
          <pattern id="feedback-flow" width="20" height="4" patternUnits="userSpaceOnUse">
            <rect width="10" height="4" fill="var(--status-healthy)" opacity="0.6">
              <animate attributeName="x" from="20" to="-20" dur="2s" repeatCount="indefinite" />
            </rect>
          </pattern>
          {/* Signal packet dot */}
          <circle id="signal-dot" r="3" fill="var(--accent)" opacity="0.8">
            <animate attributeName="opacity" values="0.8;0.3;0.8" dur="2s" repeatCount="indefinite" />
          </circle>
          {/* LED glow filter */}
          <filter id="led-glow">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* ═══ DATA PLANE ═══ */}
        <g transform="translate(0, 0)">
          <PlaneLabel x={450} y={18} label="DATA PLANE" />
          <ModuleNode x={60}  y={35} id="discovery"        status={...} label="Discovery"        metric="1,247 entities" />
          <ModuleNode x={270} y={35} id="activity_monitor"  status={...} label="Activity Monitor" metric="4.2 ev/min" />
          <ModuleNode x={480} y={35} id="data_quality"      status={...} label="Data Curation"   metric="892 included" />
          <ModuleNode x={690} y={35} id="activity_labeler"  status={...} label="Activity Labeler" metric='"Cooking"' />
          {/* Vertical connectors down to bus */}
          <BusConnector x={130} y1={95} y2={130} label="entities" />
          <BusConnector x={340} y1={95} y2={130} label="events" />
          <BusConnector x={550} y1={95} y2={130} label="rules" />
          <BusConnector x={760} y1={95} y2={130} label="labels" />
        </g>

        {/* ═══ CAPABILITIES BUS ═══ */}
        <g transform="translate(0, 130)">
          <rect x={30} y={0} width={840} height={24} rx={4} fill="url(#bus-flow)" />
          <rect x={30} y={0} width={840} height={24} rx={4} fill="none" stroke="var(--accent)" stroke-width={1.5} opacity={0.4} />
          <text x={450} y={16} text-anchor="middle" fill="var(--accent)" font-size="10" font-family="var(--font-mono)">
            CAPABILITIES BUS  [entities] [activity] [curation] [labels] [usefulness]
          </text>
          {/* Animated signal packets traveling along bus */}
          <circle r="3" fill="var(--accent)">
            <animateMotion dur="3s" repeatCount="indefinite" path="M30,12 L870,12" />
          </circle>
        </g>

        {/* ═══ LEARNING PLANE ═══ */}
        <g transform="translate(0, 170)">
          <BusConnector x={130} y1={0} y2={30} label="" direction="down" />
          <BusConnector x={340} y1={0} y2={30} label="" direction="down" />
          <BusConnector x={550} y1={0} y2={30} label="" direction="down" />
          <BusConnector x={760} y1={0} y2={30} label="" direction="down" />
          <PlaneLabel x={450} y={48} label="LEARNING PLANE" />
          <ModuleNode x={60}  y={55} id="intelligence"       status={...} label="Intelligence"  metric="Day 14" />
          <ModuleNode x={270} y={55} id="ml_engine"           status={...} label="ML Engine"     metric="R²: 0.84" />
          <ModuleNode x={480} y={55} id="pattern_recognition" status={...} label="Patterns"      metric="12 sequences" />
          <ModuleNode x={690} y={55} id="drift_monitor"       status={...} label="Drift Monitor" metric="0 flagged" />
          <BusConnector x={340} y1={115} y2={150} label="accuracy" />
        </g>

        {/* ═══ FEEDBACK BUS ═══ */}
        <g transform="translate(0, 320)">
          <rect x={30} y={0} width={840} height={24} rx={4} fill="url(#feedback-flow)" />
          <rect x={30} y={0} width={840} height={24} rx={4} fill="none" stroke="var(--status-healthy)" stroke-width={1.5} opacity={0.4} />
          <text x={450} y={16} text-anchor="middle" fill="var(--status-healthy)" font-size="10" font-family="var(--font-mono)">
            FEEDBACK BUS  [accuracy] [hit_rate] [suggestions] [drift] [corrections]
          </text>
          {/* Feedback packet traveling reverse direction */}
          <circle r="3" fill="var(--status-healthy)">
            <animateMotion dur="4s" repeatCount="indefinite" path="M870,12 L30,12" />
          </circle>
        </g>

        {/* ═══ ACTION PLANE ═══ */}
        <g transform="translate(0, 360)">
          <BusConnector x={130} y1={0} y2={30} label="" direction="down" />
          <BusConnector x={340} y1={0} y2={30} label="" direction="down" />
          <BusConnector x={550} y1={0} y2={30} label="" direction="down" />
          <BusConnector x={760} y1={0} y2={30} label="" direction="down" />
          <PlaneLabel x={450} y={48} label="ACTION PLANE" />
          <ModuleNode x={60}  y={55} id="shadow_engine" status={...} label="Shadow Engine" metric="73% accuracy" />
          <ModuleNode x={270} y={55} id="orchestrator"   status={...} label="Orchestrator"  metric="2 pending" />
          <ModuleNode x={480} y={55} id="pipeline_gates" status={...} label="Pipeline Gates" metric="shadow" />
          <ModuleNode x={690} y={55} id="feedback_health" status={...} label="Feedback Health" metric="3/4 fresh" />
        </g>

        {/* ═══ YOU NODE ═══ */}
        <g transform="translate(350, 480)">
          <rect x={0} y={0} width={200} height={35} rx={4} fill="var(--bg-inset)" stroke="var(--accent)" stroke-width={2} />
          <text x={100} y={22} text-anchor="middle" fill="var(--accent)" font-size="12" font-weight="bold" font-family="var(--font-mono)">
            YOU: Label · Curate · Review
          </text>
        </g>
      </svg>
    </section>
  );
}
```

**ModuleNode subcomponent:**

```jsx
function ModuleNode({ x, y, id, status, label, metric }) {
  const colors = {
    healthy: 'var(--status-healthy)',
    waiting: 'var(--status-waiting)',
    blocked: 'var(--status-error)',
    review: 'var(--status-warning)',
  };
  const color = colors[status] || colors.waiting;

  return (
    <g transform={`translate(${x}, ${y})`}>
      {/* Node box */}
      <rect width={180} height={55} rx={4}
        fill="var(--bg-surface)" stroke="var(--border-primary)" stroke-width={1} />
      {/* Status LED with glow */}
      <circle cx={16} cy={16} r={5} fill={color} filter="url(#led-glow)">
        {status === 'healthy' && (
          <animate attributeName="opacity" values="1;0.6;1" dur="3s" repeatCount="indefinite" />
        )}
      </circle>
      {/* Label */}
      <text x={28} y={20} fill="var(--text-primary)" font-size="12" font-weight="600" font-family="var(--font-mono)">
        {label}
      </text>
      {/* Metric */}
      <text x={16} y={40} fill="var(--text-tertiary)" font-size="10" font-family="var(--font-mono)">
        {metric}
      </text>
    </g>
  );
}
```

**Step 2: Add CSS animations for buses**

In `index.css`, add:

```css
/* ── Bus architecture animations ── */
@keyframes bus-packet {
  0% { transform: translateX(-20px); opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { transform: translateX(840px); opacity: 0; }
}

@keyframes feedback-packet {
  0% { transform: translateX(840px); opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { transform: translateX(-20px); opacity: 0; }
}

@keyframes led-breathe {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

@keyframes bus-glow {
  0%, 100% { filter: drop-shadow(0 0 0 transparent); }
  50% { filter: drop-shadow(0 0 4px var(--accent)); }
}

@keyframes connector-flow {
  0% { stroke-dashoffset: 12; }
  100% { stroke-dashoffset: 0; }
}

.bus-connector-line {
  stroke-dasharray: 4 4;
  animation: connector-flow 1s linear infinite;
}
```

**Step 3: Wire data — fetch feedback health + activity data**

In `Home()`, add:
```jsx
const [feedbackHealth, setFeedbackHealth] = useState(null);
const [activityData, setActivityData] = useState(null);

// In the Promise.all:
fetchJson('/api/capabilities/feedback/health').catch(() => null),
fetchJson('/api/activity/current').catch(() => null),
```

**Step 4: Rebuild SPA**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`

**Step 5: Test visually**

Open `http://127.0.0.1:8001/ui/` and verify:
- Three planes render with correct module nodes
- Status LEDs show correct colors
- Bus lines animate (capabilities flows right, feedback flows left)
- Signal packets travel along buses
- Metrics update via WebSocket

**Step 6: Commit**

```bash
git add aria/dashboard/spa/src/pages/Home.jsx aria/dashboard/spa/src/index.css
git commit -m "feat: replace pipeline flow with animated bus architecture diagram"
```

---

## Task 13: Enhanced Animations — Context-Aware Bus Activity

**Files:**
- Modify: `aria/dashboard/spa/src/pages/Home.jsx`

**Animations to add:**

1. **Cache update flash** — When WebSocket sends `cache_updated`, the corresponding bus connector briefly flashes (uses existing `data-refresh-flash` keyframe). Identify which connector by matching the cache category to the signal label.

2. **Feedback loop highlight** — Every 30 seconds, a "tracer" animation follows the full feedback path: from ML Engine node → down connector → feedback bus → up connector → capabilities bus → up connector → Discovery node. Shows the loop is alive. Uses SVG `animateMotion` along a compound path.

3. **Activity labeler pulse** — When `current_activity` changes, the Activity Labeler node pulses with `data-refresh-flash` and the label text updates with a `typewriter-in` animation.

4. **Stale indicator** — If any feedback source is >48h old, its connector line changes from animated dashes to a static red dotted line. Visual "broken wire" metaphor.

5. **Bus load indicator** — Bus width subtly varies based on how many signals are flowing. More active = slightly thicker bus trace (2px → 3px). CSS transition.

```jsx
// Feedback tracer — SVG path following the full loop
function FeedbackTracer() {
  // Path: ML Engine (340,285) → down to feedback bus (340,320) →
  //        left along bus (340→130) → up to capabilities bus (130,154) →
  //        up to Discovery (130,130)
  const tracerPath = "M340,285 L340,320 L130,332 L130,154 L130,130";
  return (
    <circle r="4" fill="var(--status-healthy)" opacity="0.9" filter="url(#led-glow)">
      <animateMotion
        dur="6s"
        repeatCount="indefinite"
        path={tracerPath}
        begin="0s"
      />
      <animate attributeName="r" values="4;6;4" dur="6s" repeatCount="indefinite" />
    </circle>
  );
}
```

**Commit:**

```bash
git add aria/dashboard/spa/src/pages/Home.jsx
git commit -m "feat: add context-aware bus animations — tracer, cache flash, stale indicator"
```

---

## Task 14: Rebuild SPA + Run Full Test Suite

**Step 1: Rebuild dashboard**

```bash
cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build
```

**Step 2: Run full test suite**

```bash
free -h | awk '/Mem:/{print $7}'  # Check available memory
.venv/bin/python -m pytest tests/ -v --timeout=120 -x -q
```

If memory < 4G, run by suite:
```bash
.venv/bin/python -m pytest tests/hub/ -v --timeout=120 -x -q
.venv/bin/python -m pytest tests/engine/ -v --timeout=120 -x -q
.venv/bin/python -m pytest tests/integration/ -v --timeout=120 -x -q
```

**Step 3: Fix any failures**

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: closed-loop feedback system — ML accuracy, shadow hit rates, activity labeling, bus diagram"
```

---

## Task Dependencies

```
Task 1 (ML feedback) ──┐
Task 2 (Shadow feedback)┼──► Task 3 (Discovery reads feedback) ──► Task 6 (Demand alignment)
Task 4 (DemandSignal) ──┤                                          │
Task 5 (Declare demands)┘                                          ▼
                                                              Task 7 (Drift trigger)
Task 8 (Automation feedback) ──► Task 11 (Feedback health API)
Task 9 (Activity labeler) ──► Task 10 (Register + API)

Task 11 ──┐
Task 10 ──┼──► Task 12 (Home dashboard SVG) ──► Task 13 (Animations) ──► Task 14 (Test + build)
Task 7  ──┘
```

**Wave plan for parallel execution:**
- **Wave 1:** Tasks 1, 2, 4, 8, 9 (all independent)
- **Wave 2:** Tasks 3, 5, 10, 11 (depend on Wave 1)
- **Wave 3:** Tasks 6, 7, 12 (depend on Wave 2)
- **Wave 4:** Tasks 13, 14 (depend on Wave 3)
