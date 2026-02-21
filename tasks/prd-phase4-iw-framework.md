# PRD: Phase 4 — I&W Framework + Organic Capabilities + Synthetic Testing

**Design:** `docs/plans/2026-02-21-phase4-iw-framework-design.md`
**PRD:** `tasks/prd.json` (15 tasks, P4-01 through P4-15)

## Task Summary

| ID | Title | Dependencies |
|----|-------|-------------|
| P4-01 | Data models (Indicator, BehavioralStateDefinition, Tracker, ActiveState) | — |
| P4-02 | Hub.db schema + BehavioralStateStore CRUD | P4-01 |
| P4-03 | 20 iw.* config entries | — |
| P4-04 | Discovery engine (patterns + gap → indicator chains) | P4-01, P4-02, P4-03 |
| P4-05 | Real-time detector (event subscriber + entity index) | P4-01, P4-02 |
| P4-06 | Detector cold-start replay | P4-05 |
| P4-07 | Lifecycle manager (promotion/demotion/density/vacation) | P4-01, P4-02 |
| P4-08 | Backtest engine (replay + holdout + counterfactual) | P4-01, P4-04, P4-07 |
| P4-09 | Composite state detection | P4-01, P4-02, P4-07 |
| P4-10 | Synthetic event simulator | P4-01 |
| P4-11 | Hyperparameter sweep framework | P4-10, P4-05 |
| P4-12 | Wire into hub core | P4-04, P4-05, P4-07, P4-09 |
| P4-13 | API endpoints | P4-12 |
| P4-14 | Integration test (end-to-end flow) | P4-12 |
| P4-15 | Full regression | All |

## Acceptance Criteria

All criteria are shell commands — exit 0 = pass.

## Quality Gates

- `cd /home/justin/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/ --timeout=120 -x -q`
- Config defaults count must increase by 20 (iw.* entries)
- No ruff lint violations in new files
