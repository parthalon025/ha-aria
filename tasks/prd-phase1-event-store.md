# PRD: Phase 1 — Event Store + Entity Graph

**Feature:** Persistent event storage and centralized entity hierarchy for ARIA Roadmap 2.0
**Design Doc:** `docs/plans/2026-02-20-aria-roadmap-2-design.md`
**Implementation Plan:** `docs/plans/2026-02-20-phase1-event-store-entity-graph.md`

## Tasks

| ID | Title | Acceptance Criterion |
|----|-------|---------------------|
| P1-01 | EventStore class with SQLite schema | `from aria.shared.event_store import EventStore` succeeds |
| P1-02 | EventStore write and read methods | `pytest tests/shared/test_event_store.py` passes |
| P1-03 | EventStore retention and pruning | `pytest tests/shared/test_event_store.py::test_prune_events` passes |
| P1-04 | EntityGraph class with resolution | `pytest tests/shared/test_entity_graph.py` passes |
| P1-05 | Wire EntityGraph into hub core | `pytest tests/hub/test_hub_entity_graph.py` passes |
| P1-06 | Wire EventStore into hub core | `pytest tests/hub/test_hub_event_store.py` passes |
| P1-07 | Activity monitor persists events | `pytest tests/hub/test_activity_event_persistence.py` passes |
| P1-08 | EventStore pruning timer | `pytest tests/hub/test_hub_event_store.py::test_pruning_timer` passes |
| P1-09 | EventStore API endpoints | `pytest tests/hub/test_api_events.py` passes |
| P1-10 | Integration test: full event flow | `pytest tests/integration/test_event_flow.py` passes |
| P1-11 | Existing test suite passes | `pytest tests/ --timeout=120 -x -q` all pass |
