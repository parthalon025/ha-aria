# Closed-Loop Feedback System Design

**Date:** 2026-02-15
**Status:** Approved
**Depends on:** Capability Propagation Framework (2026-02-14), Organic Discovery (2026-02-14)

## Problem

ARIA's pipeline is open-loop. Discovery clusters entities, ML trains models, shadow validates predictions — but no downstream signal feeds back to improve discovery or scoring. The `predictability` component of usefulness scoring is hardcoded to 0.0 (`module.py:229`). Capabilities auto-promote/archive based on incomplete evidence.

Research validates this gap: closed-loop ML pipelines outperform open-loop by 5-10% on accuracy, and the gap widens over time because open-loop systems can't adapt to drift (Martin Fowler CD4ML, Self-Healing ML Pipelines 2025).

Additionally, ARIA knows sensor states ("power is 450W, 3 lights on") but not activities ("cooking dinner"). The EL-HARP framework (Sensors 2025) shows activity labeling via LLM prediction + user correction produces high-accuracy classifiers with minimal user effort.

## Solution

Close the loop with five feedback channels and a new activity labeling system, visualized on the Home dashboard as a bus-architecture block diagram.

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         DATA PLANE                                  │
  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐      │
  │  │ ◉ Discover │ │ ◉ Activity │ │ ◉ Data     │ │ ◉ Activity │      │
  │  │            │ │   Monitor  │ │   Curation │ │   Labeler  │      │
  │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘      │
  │     entities       events         rules          labels            │
  └────────┼──────────────┼──────────────┼──────────────┼──────────────┘
           ▼              ▼              ▼              ▼
  ═════════════════════════════════════════════════════════════════════════
   CAPABILITIES BUS [entities] [activity] [curation] [labels] [usefulness]
  ═════════════════════════════════════════════════════════════════════════
           │              │              │              │
  ┌────────┼──────────────┼──────────────┼──────────────┼──────────────┐
  │        ▼              ▼              ▼              ▼              │
  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐     │
  │  │ ◉ Intel.   │ │ ◉ ML Eng.  │ │ ◉ Patterns │ │ ◉ Drift    │     │
  │  └────────────┘ └─────┬──────┘ └────────────┘ └────────────┘     │
  │                 LEARNING PLANE                                     │
  └───────────────────────┼────────────────────────────────────────────┘
                       accuracy
                          ▼
  ═════════════════════════════════════════════════════════════════════════
   FEEDBACK BUS [accuracy] [hit_rate] [suggestions] [drift] [corrections]
  ═════════════════════════════════════════════════════════════════════════
           │              │              │              │
  ┌────────┼──────────────┼──────────────┼──────────────┼──────────────┐
  │        ▼              ▼              ▼              ▼              │
  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐     │
  │  │ ◉ Shadow   │ │ ◉ Orchest. │ │ ◉ Pipeline │ │ ◉ Feedback │     │
  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘     │
  │                  ACTION PLANE                                      │
  └───────────────────────────────────────────────────────────────────┘
                          │
                    ┌── YOU ───┐
                    │ Label    │
                    │ Curate   │
                    │ Review   │
                    │ Advance  │
                    └──────────┘
```

## Phase 1: ML Accuracy → Predictability Score

**Goal:** Fill the predictability=0.0 gap with real ML accuracy data.

### Backend Changes

**ml_engine.py** — After training all targets for a capability, write accuracy back:
```python
# In train_models(), after training loop completes:
capabilities = await self.hub.get_cache("capabilities")
for cap_name, targets in trained_results.items():
    if cap_name in capabilities:
        r2_values = [t["r2"] for t in targets.values() if "r2" in t]
        mean_r2 = sum(r2_values) / len(r2_values) if r2_values else 0.0
        capabilities[cap_name]["ml_accuracy"] = {
            "mean_r2": round(mean_r2, 3),
            "targets": targets,
            "last_trained": datetime.now().isoformat(),
            "feature_importance_top5": top_features[:5],
        }
        capabilities[cap_name]["usefulness_components"]["predictability"] = round(mean_r2 * 100)
await self.hub.set_cache("capabilities", capabilities, {"source": "ml_feedback"})
```

**shadow_engine.py** — Periodically write hit rates per capability:
```python
# In _periodic_feedback() (new method, called every hour):
capabilities = await self.hub.get_cache("capabilities")
for cap_name, cap_data in capabilities.items():
    entity_ids = set(cap_data.get("entities", []))
    # Count predictions involving this capability's entities
    hits, total = self._count_hits_for_entities(entity_ids)
    if total > 0:
        cap_data["shadow_accuracy"] = {
            "hit_rate": round(hits / total, 3),
            "total_predictions": total,
            "last_updated": datetime.now().isoformat(),
        }
await self.hub.set_cache("capabilities", capabilities, {"source": "shadow_feedback"})
```

**module.py:229** — Replace hardcoded predictability:
```python
# Before: predictability=0.0
# After:
ml_acc = existing_cap.get("ml_accuracy", {})
shadow_acc = existing_cap.get("shadow_accuracy", {})
ml_r2 = ml_acc.get("mean_r2", 0.0)
shadow_hr = shadow_acc.get("hit_rate", 0.0)
predictability = (ml_r2 * 0.7 + shadow_hr * 0.3) if (ml_r2 + shadow_hr) > 0 else 0.0
```

## Phase 2: Registry Demand Signals

**Goal:** Discovery becomes consumer-aware without constraining clustering.

### Backend Changes

**capabilities.py** — New dataclass:
```python
@dataclass(frozen=True)
class DemandSignal:
    entity_domains: List[str]
    device_classes: List[str]
    min_entities: int = 5
    description: str = ""
```

Add `demand_signals` field to Capability dataclass.

**ML Engine, Shadow Engine CAPABILITIES** — Declare what entity groupings they need.

**module.py** — After clustering, compute demand alignment bonus (0-20 points additive to usefulness).

## Phase 3: Automation Suggestion Feedback

**Goal:** Track whether users accept/reject suggested automations.

### New API Endpoints

- `POST /api/automations/feedback` — Record accept/reject/modify
- `GET /api/automations/feedback` — Get feedback history

### New Cache Category: `automation_feedback`

```python
{
    "per_capability": {
        "capability_name": {
            "suggested": int,
            "accepted": int,
            "rejected": int,
            "acceptance_rate": float,
        }
    }
}
```

Acceptance rate feeds into usefulness scoring as optional 6th component.

## Phase 4: Drift-Triggered Re-Discovery

**Goal:** When drift is detected, flag capabilities for re-clustering.

### Backend Changes

**intelligence.py** — Publish `drift_detected` event with capability name and severity.

**module.py** — Subscribe to `drift_detected`. Set `cap["drift_flagged"] = True`. Next discovery run re-evaluates flagged capabilities' entity membership.

## Phase 5: Activity Labeling Loop

**Goal:** Bridge the gap from sensor states to named activities. LLM predicts, user corrects, system retrains.

### Flow

1. Ollama receives sensor context → predicts activity ("cooking", "watching TV")
2. Dashboard shows prediction with confidence + one-tap confirm/correct
3. User corrections stored as labeled training data
4. After 50+ labels, train lightweight GradientBoosting classifier
5. Classifier replaces Ollama for known patterns (Ollama for novel states)
6. Activity labels enrich capability metadata

### New Cache Category: `activity_labels`

```python
{
    "current_activity": {
        "predicted": "cooking",
        "confidence": 0.72,
        "method": "ollama" | "classifier" | "user_set",
        "sensor_context": {
            "power_watts": 450,
            "lights_on": 3,
            "motion_rooms": ["kitchen"],
            "time_of_day": "evening",
            "occupancy": "home",
        },
        "predicted_at": "ISO",
    },
    "labels": [...],  # Training data
    "label_stats": {
        "total_labels": int,
        "total_corrections": int,
        "accuracy": float,
        "activities_seen": [],
        "classifier_ready": bool,
        "last_trained": "ISO" | null,
    }
}
```

### New API Endpoints

- `GET /api/activity/current` — Current predicted activity
- `POST /api/activity/label` — Confirm or correct
- `GET /api/activity/labels` — Label history
- `GET /api/activity/stats` — Accuracy, classifier status

### Ollama Integration

Submit through ollama-queue (port 7683). Prompt template:
```
Given the current smart home state:
- Power draw: {power_watts}W
- Lights on: {lights_on} ({rooms_with_lights})
- Motion detected: {motion_rooms}
- Time: {time_of_day} ({hour}:{minute})
- Occupancy: {occupancy_status}
- Recent events: {last_5_events}

What activity is the resident most likely doing?
Respond with JSON: {"activity": "...", "confidence": 0.0-1.0}
```

### Feed into Discovery

Capabilities get `activity_context`:
```python
cap["activity_context"] = {
    "associated_activities": ["cooking", "cleaning"],
    "activity_coverage": 0.8,
}
```

## Phase 6: Home Dashboard — Bus Architecture Diagram

**Goal:** Replace linear 3-lane pipeline with EE-style block diagram showing feedback loops.

### Visual Design

Three planes (Data, Learning, Action) separated by labeled capability and feedback buses. Each module node has:
- Status LED (green/amber/red circle)
- Module name
- Key metric (entity count, accuracy, event rate)

### Animations

1. **Bus pulse** — Dashed lines animate along bus traces showing data flow direction (down through capabilities bus, up through feedback bus)
2. **Signal packets** — Small dots travel along bus lines when cache updates occur (triggered by WebSocket `cache_updated` events)
3. **LED pulse** — Status LEDs glow/pulse when module is actively processing
4. **Feedback glow** — When feedback writes back to capabilities, the feedback bus briefly glows brighter
5. **Activity label highlight** — When user confirms/corrects activity, a brief animation shows the label flowing down through the system

### Mobile Layout

Planes stack vertically. Buses become horizontal dividers with signal names. Nodes within each plane arrange in a 2-column grid.

## Research Validation

| Phase | Validated By |
|-------|-------------|
| ML feedback loop | Martin Fowler CD4ML, Self-Healing ML Pipelines (2025) |
| Activity labeling | EL-HARP (Sensors 2025), LLMs for Activity Recognition (ACM ToIT 2025) |
| Shadow deployment | AWS SageMaker Shadow, Neptune.ai deployment strategies |
| Drift detection | Meta-ADD (ScienceDirect), OTACON (DMKD 2025) |
| Ensemble prediction | MDPI Sensor Benchmark (2025), Springer WIDECOM (2024) |
| User feedback RL | Automation of Smart Homes (arXiv 2024) |

## Files Modified

### Backend (Python)
- `aria/modules/ml_engine.py` — Write accuracy back to capabilities
- `aria/modules/shadow_engine.py` — Write hit rates to capabilities
- `aria/modules/organic_discovery/module.py` — Read predictability, demand alignment
- `aria/modules/organic_discovery/scoring.py` — (no changes, reads components as-is)
- `aria/modules/intelligence.py` — Publish drift_detected events
- `aria/capabilities.py` — Add DemandSignal, demand_signals field
- `aria/hub/api.py` — New endpoints for activity, automations feedback, feedback health
- `aria/modules/activity_labeler.py` — New module

### Frontend (Preact JSX)
- `aria/dashboard/spa/src/pages/Home.jsx` — Bus architecture diagram
- `aria/dashboard/spa/src/pages/Capabilities.jsx` — Evidence display per capability
